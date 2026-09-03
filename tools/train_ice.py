#!/usr/bin/env python
"""Phase 4b: learn an icy level with a BACKWARD CURRICULUM over demonstration
snapshots (Salimans & Chen 2018), so momentum is trained on, not settled away.

Why not the existing reverse curriculum: it places the king through teleport(),
which resets his physics, settles him to REST and only then injects a velocity.
On non-ice that is harmless (slip == 0, so every grounded state converges to the
same rest point). On ice the state is (position, velocity, slide phase) and a
real arrival carries speed ~2-4 -- states a teleport pool never contains. The
policy is then trained on a start distribution the game does not produce.

Here every start is a SNAPSHOT of a state the game really reached, taken from
starts/demo_L<N>.json (see tools/ice_demo.py). Episodes begin at the LAST
snapshot -- one action from the exit -- and the start walks backwards through
the demonstration as the success rate allows, until the level is solved from its
real entry.

Reward is the env's own potential-based route shaping, with the demonstration's
own decision points as the waypoints, plus the level-transition reward.

    python tools/train_ice.py --level 36
    python tools/train_ice.py --level 36 --updates 400 --num-envs 6

Nothing here touches the verified 0->42 relay: checkpoints go to
checkpoints/L<N>_ice/, which FullRelay.latest() (checkpoints/L<N>/*.pt) cannot
pick up.
"""
import os, sys, json, time, copy, math, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
from JK_Env import JumpKingEnv, build_action_table
from PPO import PPO, RolloutBuffer, get_device
from Occupancy import resolve_channels

ICE_CHANNELS = ("solid", "slope")

# Set once from --wind-jitter. It has to be reachable from `greedy_rate`, which
# the seeding path calls without an `args` in scope, and every gate measurement
# must randomise the phase the same way training does -- a gate that always
# measures one gust timing would pass a policy that only works in that phase.
WIND_JITTER = False

# (dx, dv) applied at RUNG 0 ONLY, from --entry-jitter-dx/dv. Set globally for
# the same reason as WIND_JITTER.
ENTRY_JITTER = None


def rung_jitter(k, dx, dv):
    """The perturbation for a start at rung `k`.

    Rung 0 is the level's real ENTRY, and it is the only rung whose state the
    chain below actually decides. When the level below is replaced by a learned
    policy that takes a slightly different route, that arrival MOVES: measured,
    the learned L36-38 deliver into L39 at x=449.7 where the route below used
    to arrive at x=443.0, and a route aimed at 443 falls three screens from 449.
    So the entry is jittered wide (the model has to be a controller over an
    arrival BAND), while the mid-route rungs keep the small jitter -- those are
    states the policy itself produces, and displacing them by 10 px invents
    starts the level never reaches and can stall the ladder on them."""
    if k == 0 and ENTRY_JITTER is not None:
        return ENTRY_JITTER
    return (dx, dv)


def king_vel(speed, angle):
    """The king's velocity VECTOR. King.py moves by rect_x += sin(a) * speed and
    rect_y -= cos(a) * speed, so `speed` alone is a magnitude and `angle` carries
    the direction -- two states can agree on (x, y, speed) and be sliding 60
    degrees apart. Measured on L37 rung 2: a five-action shuffle ended 0.62 px
    and 0.07 speed from snapshot 3 -- deep inside any sub-goal box -- with
    velocity (-2.80, -0.07) against the target's (-1.29, +2.57). The policy sees
    the difference (vel_obs feeds it the direction) and picks a different action,
    so the reward was paying for a state the tail cannot use."""
    return math.sin(angle) * speed, -math.cos(angle) * speed


def load_demo(level, path=None):
    path = path or os.path.join("starts", f"demo_L{level}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run: python tools/ice_demo.py --levels {level}")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not d.get("crossed"):
        raise SystemExit(f"{path} does not cross the level; it cannot seed a curriculum")
    return d


def won(info):
    """Did the episode reach the level's GOAL?

    `info["success"]` is the env's own verdict (goal_level reached inside the
    goal region, or the babe -- `levels.ending`). The old `level > args.level`
    test cannot express either of the two cases this file now has to handle: a
    route that SPANS a screen (L23 -> 25, L40 -> 42), where merely leaving the
    level lands on the intermediate one and is NOT a win, and the summit (L42),
    where winning does not change the level number at all."""
    return bool(info.get("success"))


def waypoints(demo, tol):
    """The demo's own decision points as ordered route boxes (y, x_lo, x_hi).

    Only points below the goal count -- the last step reaches it, and that
    transition is already paid by the level-transition reward. A spanning demo
    (L23 climbs through L24) keeps the waypoints it lays down on the
    intermediate screen: those are still route, not the crossing."""
    out = []
    goal = int(demo.get("goal") or (demo["level"] + 1))
    for s in demo["steps"]:
        st = s["state_after"]
        if int(st["level"]) >= goal:
            break
        out.append((int(round(st["y"])), int(round(st["x"])) - tol,
                    int(round(st["x"])) + tol))
    return tuple(out)


def demo_goal(args, demo):
    """The level number that counts as won."""
    return int(args.goal_level or demo.get("goal") or (demo["level"] + 1))


def make_env(args, demo, wps):
    return JumpKingEnv(
        max_steps=args.max_steps,
        goal_level=demo_goal(args, demo),
        fine_walk_frames=args.fine_walk_frames,
        extra_charges=(22, 24, 28, 30),
        settle_action=args.settle_action,
        wind_jump=args.wind_jump,
        vel_obs=args.vel_obs, wind_obs=args.wind_obs,
        vel_encoding=args.vel_encoding,
        grid_channels=args.channels,
        step_cost=args.step_cost,
        route_dense=args.route_dense,
        fail_penalty=args.fail_penalty,
        stuck_limit=args.stuck_limit,
        goal_boxes=wps,
        seed=args.seed,
    )


class Curriculum:
    """Start index into the demonstration, walked backwards.

    `k` is the earliest snapshot unlocked. An episode starts at a uniformly
    chosen UNLOCKED snapshot rather than always at `k`: rehearsing the already
    solved tail is what stops the policy forgetting it as the ladder grows.
    Only episodes that actually started at `k` are allowed to advance the
    ladder, otherwise easy rehearsal episodes would inflate the rate."""

    def __init__(self, n_snaps, window, thresh, rehearse, enabled=True, guard=0.0):
        self.n = n_snaps
        self.enabled = bool(enabled)
        # last snapshot = one action from the exit; pinned to 0 (the real entry)
        # for the ablation arm that gets no backward curriculum at all
        self.k = (n_snaps - 1) if self.enabled else 0
        self.window, self.thresh, self.rehearse = window, thresh, rehearse
        self.guard = float(guard)
        self.recent = []
        self.solved_from_entry = False
        self.last_greedy = float("nan")
        self.passes = 0

    def sample(self, rng):
        if not self.enabled or self.k >= self.n - 1:
            return self.k
        # GUARD THE RUNG BELOW THE FRONTIER. Uniform rehearsal spreads its
        # budget over every unlocked rung, so the one that actually breaks --
        # always the one immediately above the frontier, because it shares
        # actions with what the frontier is learning -- gets only 1/(n-k) of it.
        # Measured on L37: rung 8 shares `jump right 32` with a rung-7 rollout
        # that falls off the level, so training rung 7 suppressed exactly the
        # action rung 8 needs, and the ladder cycled 8 -> 7 -> 8 -> 7.
        if self.guard > 0.0 and rng.random() < self.guard:
            return self.k + 1
        if rng.random() >= self.rehearse:
            return self.k
        return int(rng.integers(self.k, self.n))

    def report(self, success):
        """Record a FRONTIER episode. Logging only -- the ladder is gated on the
        greedy measurement, see advance()."""
        self.recent.append(1.0 if success else 0.0)
        if len(self.recent) > self.window:
            self.recent = self.recent[-self.window:]
        return False

    def demote(self):
        """Give a previously unlocked rung back.

        A rung is opened once and then never re-checked, while training moves on
        to the harder one below it. Measured on L36: rung 4 was opened at greedy
        0.55 and had decayed to 0.00 two hundred updates later, which silently
        turned rung 3 from a one-action discovery (1/38) into a two-action one
        (1/1444) -- the ladder was standing on a rung that had rotted. Walking
        back up until it holds again makes the curriculum two-way."""
        if self.k < self.n - 1:
            self.k += 1
            self.recent = []
            self.passes = 0
            return True
        return False

    def advance(self, greedy, hold=1):
        """Move the start back one snapshot if the GREEDY policy clears this rung
        on `hold` consecutive evaluations. Advancing on a single passing measure
        let rungs through on successes PPO immediately undid."""
        self.last_greedy = greedy
        if greedy < self.thresh:
            self.passes = 0
            return False
        self.passes = getattr(self, "passes", 0) + 1
        if self.passes < hold:
            return False
        self.passes = 0
        if self.enabled and self.k > 0:
            self.k -= 1
            self.recent = []
            return True
        self.solved_from_entry = True
        return False

    @property
    def rate(self):
        return float(np.mean(self.recent)) if self.recent else float("nan")


def demo_action_indices(demo, table):
    """The demo's action at each snapshot, as an index into `table`."""
    out = []
    for st in demo["steps"]:
        a = list(st["action"])
        if a[0] == "settle":
            want = ("settle", "none", 0)
        elif isinstance(a[1], (list, tuple)):
            # ("jump_wind", (bucket, direction), charge) -- the pair survives the
            # JSON round trip as a list and has to become a tuple again, or it
            # would never match a row of the table.
            want = (a[0], (int(a[1][0]), str(a[1][1])), int(a[2]))
        else:
            want = (a[0], a[1], int(a[2]))
        out.append(table.index(want) if want in table else -1)
    return out


def bc_init(agent, envs, snaps, acts, epochs, lr, device):
    """Behaviour-clone the demonstration before PPO starts.

    The ladder's hard rungs need a TWO-action discovery; with 38 actions that is
    1/1444 by random sampling, which is why a from-scratch run stalls partway up
    the demonstration. Cloning the demo's own action at each of its states gives
    the policy a prior exactly on the states the curriculum will visit -- the
    robustification step Go-Explore performs on its brittle trajectories.

    It clones ONE route; PPO under the backward curriculum is what then has to
    make it work from the states around it that the route never visits."""
    pairs = [(k, a) for k, a in enumerate(acts) if a >= 0]
    if not pairs:
        return 0.0, None
    e = envs[0]
    X = np.stack([e.restore(copy.deepcopy(snaps[k]), route_i=k - 1)[0] for k, _ in pairs])
    Y = np.array([a for _, a in pairs], dtype=np.int64)
    xb = torch.as_tensor(X, device=device).float()
    yb = torch.as_tensor(Y, device=device)

    # RESTARTS. These are a handful of samples the net can memorise exactly, but
    # a single full-batch run at a fixed lr sometimes collapses to a constant
    # policy -- measured on L36: accuracy 0.22 with the SAME action distribution
    # at every demo state, i.e. the scalars were being ignored. That silently
    # removes the prior the whole ladder depends on, so try again, slower, and
    # keep the best attempt.
    best_acc, best_state, best_loss = -1.0, None, float("nan")
    for attempt in range(4):
        if attempt:                       # fresh parameters, not a warm restart
            agent.net.apply(agent.net._init)
            torch.nn.init.orthogonal_(agent.net.policy_head.weight, 0.01)
        cur_lr = lr / (3.0 ** attempt)
        opt = torch.optim.Adam(agent.net.parameters(), lr=cur_lr)
        loss = torch.tensor(float("nan"))
        for _ in range(epochs):
            logits, _ = agent.net(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((agent.net(xb)[0].argmax(1) == yb).float().mean())
        if acc > best_acc:
            best_acc, best_loss = acc, loss.detach().item()
            best_state = copy.deepcopy(agent.net.state_dict())
        print(f"  BC attempt {attempt + 1} lr={cur_lr:.1e}: loss "
              f"{loss.detach().item():.4f} acc {acc:.2f}", flush=True)
        if acc >= 0.999:
            break
    agent.net.load_state_dict(best_state)
    print(f"BC init: {len(pairs)} demo pairs, {epochs} epochs, "
          f"final loss {best_loss:.4f}, action accuracy {best_acc:.2f}", flush=True)
    if best_acc < 0.9:
        print("  WARNING: the policy cannot separate the demonstration's own "
              "states -- the curriculum has no usable prior", flush=True)
    return best_acc, (xb, yb)


def jitter_snap(snap, dx, dv, wind=False, rng=None):
    """Perturb a demo snapshot: arrival position, carried speed, wind phase.

    The wind phase is half the state on the snow biome and `snapshot()` stores
    it, so a curriculum restored from one demonstration would train every
    episode on the SAME gust timing. Randomising it is what forces the policy to
    use the wind-timed macros (which wait for their own bucket) rather than
    memorise when to jump."""
    snap = copy.deepcopy(snap)
    if dx > 0.0:
        snap["king"]["rect_x"] = float(snap["king"]["rect_x"]) + float(
            rng.uniform(-dx, dx))
    if dv > 0.0:
        snap["king"]["speed"] = float(snap["king"]["speed"]) * float(
            rng.uniform(1.0 - dv, 1.0 + dv))
    if wind:
        snap["wind_var"] = float(rng.uniform(0.0, 2.0 * math.pi))
    return snap


def greedy_rate(net, env, snaps, k, episodes, max_steps, level, device, jitter,
                rng, wind=None):
    """Crossing rate of the DETERMINISTIC policy from rung k.

    The ladder must not be gated on the stochastic rollout rate. PPO explores by
    sampling, and the entropy floor deliberately keeps that sampling broad, so a
    rung needing m correct actions caps the rollout rate near p**m however good
    the policy is -- measured: every level plateaus around 0.37 and the ladder
    stops. The greedy policy is what gets deployed and evaluated, so it is what
    the ladder should be measured on."""
    wins = 0
    dx, dv = rung_jitter(k, *jitter)
    wind = WIND_JITTER if wind is None else wind
    for _ in range(episodes):
        obs, _ = env.restore(jitter_snap(snaps[k], dx, dv, wind, rng),
                             route_i=k - 1)
        for _ in range(max_steps):
            with torch.no_grad():
                a = int(net(torch.as_tensor(obs, device=device)
                            .float().unsqueeze(0))[0].argmax(1))
            obs, _, te, tr, info = env.step(a)
            if te or tr:
                wins += won(info)
                break
    return wins / max(1, episodes)


def handoff_actions(net, env, snaps, k, n_actions, level, budget, device):
    """Which single action at rung k hands off to a state the CURRENT policy
    finishes from -- and the whole trajectory it then takes.

    The rungs above k are already solved, so rung k does not need a plan: it
    needs one action that lands somewhere the policy can finish. That is a
    38-way question, not a 38**m one, and it always has at least one answer
    while the tail holds, because the demonstration's own action at k lands
    exactly on snapshot k+1. Measured on L37 rung 5 with a solved 6..11 tail:
    exactly 1 of 38 first actions crosses, and it is the demo's -- while the
    policy's own argmax there was a jump that falls off the level.

    Returns [(action, [(obs, action), ...]), ...]: the crossing action followed
    by the (state, action) pairs of the run that crossed. Those pairs are the
    thing to protect -- distilling the first action while protecting only the
    demo snapshots left L37 rung 5 argmax-correct and still scoring 0.00,
    because the states the seeded action actually leads THROUGH were nowhere in
    the preservation set."""
    hits = []
    for a in range(n_actions):
        env.max_steps = budget
        obs, _ = env.restore(copy.deepcopy(snaps[k]), route_i=k - 1)
        pairs = [(obs, a)]
        obs, _, te, tr, info = env.step(a)
        if not (te or tr):
            for _ in range(budget - 1):
                with torch.no_grad():
                    aa = int(net(torch.as_tensor(obs, device=device)
                                 .float().unsqueeze(0))[0].argmax(1))
                pairs.append((obs, aa))
                obs, _, te, tr, info = env.step(aa)
                if te or tr:
                    break
        if won(info):
            hits.append((a, pairs))
    return hits


def greedy_pairs(net, env, snaps, k, n_snaps, slack, device):
    """(observation, greedy action) along the policy's own rollout from rung k.

    These are the labels that must NOT move while the frontier is being seeded.
    Refitting the DEMONSTRATION over the solved tail would be worse than doing
    nothing: measured on L37, the learned tail leaves rung 6 by `walk right 3`
    then `jump right 24` straight to the exit ledge -- four actions where the
    demo takes six -- and cloning the demo back over it would throw that away."""
    env.max_steps = (n_snaps - k) + slack
    obs, _ = env.restore(copy.deepcopy(snaps[k]), route_i=k - 1)
    out = []
    for _ in range(env.max_steps):
        with torch.no_grad():
            a = int(net(torch.as_tensor(obs, device=device)
                        .float().unsqueeze(0))[0].argmax(1))
        out.append((obs, a))
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
    return out


def seed_frontier(agent, env, snaps, k, n_snaps, level, slack, n_actions,
                  device, steps, lr, check_every, rng, gate_jitter, episodes,
                  thresh, demo_act=None):
    """Give the frontier rung the one action that hands off to the solved tail,
    without disturbing the tail itself.

    A re-initialised head leaves the new rung a uniform 38-way lottery and PPO
    has to rediscover an action worth 1/38 an episode; a plain BC refit teaches
    the demonstration everywhere and overwrites what the policy improved on.
    This does neither: it searches for the action, then fits the CROSSING RUN it
    found -- first action included -- while holding the policy's own current
    choices fixed on the rungs already solved. The search is the expert and the
    policy is its own teacher everywhere else; PPO's job afterwards is to make
    the seeded action robust to arrivals the search never saw, not to discover
    it."""
    budget = (n_snaps - k) + slack
    hits = handoff_actions(agent.net, env, snaps, k, n_actions, level,
                           budget, device)
    if not hits:
        print(f"  seed: no single action at rung {k} hands off to the current "
              f"policy -- leaving it to PPO", flush=True)
        return False
    # prefer the demonstration's own action when it is among the winners: it is
    # the one the rungs below were recorded against
    win = next((h for h in hits if h[0] == demo_act), hits[0])
    a_star, traj = win
    # NEVER LEAVE THE RUNG WORSE THAN IT WAS. A distilled plan is fitted on the
    # snapshot's exact state, and the gate measures a JITTERED one -- measured on
    # L37 rung 5, distilling a verified 5-action crossing run took the gate score
    # from 0.30 to 0.00 because the fitted run is razor-thin off its own start.
    # Seeding is only ever an improvement if it is checked and rolled back.
    g_before = greedy_rate(agent.net, env, snaps, k, episodes, budget, level,
                           device, gate_jitter, rng)
    before = copy.deepcopy(agent.net.state_dict())
    keep = []
    for j in range(k + 1, n_snaps):
        keep.extend(greedy_pairs(agent.net, env, snaps, j, n_snaps, slack,
                                 device))
    X = np.stack([o for o, _ in traj] + [o for o, _ in keep])
    Y = np.array([a for _, a in traj] + [a for _, a in keep], dtype=np.int64)
    # weight the crossing run above the rehearsal set, but only mildly, and STOP
    # as soon as every label is argmax-correct: fitting past that point buys
    # nothing and costs the tail.
    w = torch.ones(len(Y), device=device)
    w[:len(traj)] = 4.0
    xb = torch.as_tensor(X, device=device).float()
    yb = torch.as_tensor(Y, device=device)
    opt = torch.optim.Adam(agent.net.parameters(), lr=lr)
    best_g, best_state = g_before, before
    fitted = None          # last weights where every label was argmax-correct
    for i in range(1, steps + 1):
        logits, _ = agent.net(xb)
        loss = (torch.nn.functional.cross_entropy(logits, yb, reduction="none")
                * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % check_every:
            continue
        with torch.no_grad():
            if not bool((agent.net(xb)[0].argmax(1) == yb).all()):
                continue
        fitted = copy.deepcopy(agent.net.state_dict())
        g = greedy_rate(agent.net, env, snaps, k, episodes, budget, level,
                        device, gate_jitter, rng)
        if g > best_g:
            best_g, best_state = g, copy.deepcopy(agent.net.state_dict())
        if g >= thresh:
            print(f"  seed: rung {k} <- {len(traj)}-action crossing run "
                  f"(first action {a_star}) distilled in {i} steps, "
                  f"greedy {g:.2f}", flush=True)
            agent.opt = torch.optim.Adam(agent.net.parameters(),
                                         lr=agent.opt.defaults["lr"])
            return True
    # WHAT TO KEEP. Rolling back whenever the gate does not improve was wrong:
    # a freshly distilled plan is razor-thin off its own start, so it scores 0
    # under the gate's jitter even though it has put the SEARCHED ACTION on top
    # at the frontier state -- which is the only thing PPO was missing. Measured
    # on L37 rung 5, exactly that seed scored 0.00 and PPO then took the rung
    # 0.00 -> 0.30 -> 0.67 in ninety updates and the ladder advanced. Measured on
    # rung 2 with the seed rolled back: 195 updates, not one frontier hit,
    # because the policy sat at p=0.99 on a wrong action and the exploration
    # floor alone is ~1 chance in 800 per episode.
    # So: keep a strict improvement; protect a rung that ALREADY worked; and
    # when the rung scores nothing either way, keep the prior -- there is
    # nothing to lose and a searched action to gain.
    if best_g > g_before:
        agent.net.load_state_dict(best_state)
        kept = "kept (improved)"
    elif g_before > 0.0:
        agent.net.load_state_dict(before)
        kept = "rolled back (the rung already worked)"
    elif fitted is not None:
        agent.net.load_state_dict(fitted)
        kept = "kept as a prior (rung scored 0 either way)"
    else:
        agent.net.load_state_dict(before)
        kept = "rolled back (never fitted)"
    print(f"  seed: rung {k} <- {len(traj)}-action crossing run (first action "
          f"{a_star}), {steps} steps, greedy {g_before:.2f} -> {best_g:.2f} "
          f"({kept})", flush=True)
    agent.opt = torch.optim.Adam(agent.net.parameters(),
                                 lr=agent.opt.defaults["lr"])
    return False


def cfg_payload(agent, args, demo, cur, n_snaps, step):
    return {"model": agent.net.state_dict(), "opt": agent.opt.state_dict(),
            "step": step,
            "action_cfg": {"fine_walk_frames": int(args.fine_walk_frames),
                           "extra_charges": [22, 24, 28, 30],
                           "wait_frames": [],
                           "wind_jump": list(args.wind_jump),
                           "approach_jump": [], "wind_combo": [],
                           "settle_action": bool(args.settle_action),
                           "wind_obs": bool(args.wind_obs),
                           "vel_obs": bool(args.vel_obs),
                           "vel_encoding": args.vel_encoding,
                           "grid_channels": list(resolve_channels(args.channels)),
                           "extra_conv": bool(args.extra_conv),
                           "scalar_embed": int(args.scalar_embed)},
            "ice_train": {"bc_init": bool(args.bc_init),
                          "curriculum": bool(args.curriculum),
                          "thresh": args.thresh, "ent_floor": args.ent_floor,
                          "jitter_dx": args.jitter_dx, "jitter_dv": args.jitter_dv,
                          "route_dense": args.route_dense,
                          "guard_prev": args.guard_prev,
                          "explore_eps": args.explore_eps,
                          "temper_after": args.temper_after,
                          "step_cost": args.step_cost,
                          "subgoal": bool(args.subgoal),
                          "grad_split": bool(args.grad_split),
                          "handoff_seed": bool(args.handoff_seed),
                          "fail_penalty": args.fail_penalty},
            "ice_curriculum": {"level": args.level, "k": cur.k,
                               "n_snapshots": n_snaps,
                               "solved_from_entry": bool(cur.solved_from_entry),
                               "demo_source": demo.get("source")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=6)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=24,
                    help="upper bound; the per-episode budget is set from the "
                         "curriculum rung, see --step-slack")
    ap.add_argument("--step-slack", type=int, default=2,
                    help="extra actions allowed beyond what the demonstration "
                         "needs from the current rung")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--waypoint-tol", type=int, default=10)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=10,
                    help="measure the greedy policy at the frontier this often")
    ap.add_argument("--eval-episodes", type=int, default=20)
    # Calibrated against a SOLVED policy measured under the gate's own jitter:
    # L38, verified 60/60 unjittered at every rung, scores 0.73 at rung 0 and
    # 0.90-0.92 at rungs 1-2, because a deep rung chains more actions and each
    # one is another chance for the perturbed start to knock it off. A 0.9 bar
    # would have rejected that level's own solution. Rot is caught by the demote
    # rule instead -- which is what the original loose 0.55 lacked.
    ap.add_argument("--thresh", type=float, default=0.7,
                    help="GREEDY crossing rate needed to move the start back one "
                         "snapshot. This gates on the deterministic policy, not "
                         "on the stochastic rollout rate, so the entropy floor "
                         "does not cap it. 0.55 was too loose: a rung opened at "
                         "0.55 on L36 had decayed to 0.00 two hundred updates "
                         "later, which is what the demote rule now catches.")
    ap.add_argument("--rehearse", type=float, default=0.5)
    # ON by default: the demonstration's own first step is a settle, so without
    # this primitive the curriculum's LAST rung (the real level entry, where the
    # king arrives sliding) has no action that reproduces the proven route. The
    # policy is free to skip it and ride the arrival momentum instead -- which
    # is exactly the behaviour this whole exercise is trying to make learnable.
    ap.add_argument("--settle-action", dest="settle_action",
                    action="store_true", default=True)
    ap.add_argument("--no-settle-action", dest="settle_action", action="store_false",
                    help="ablation: remove the momentum-killing settle primitive")
    ap.add_argument("--extra-conv", dest="extra_conv", action="store_true", default=True)
    ap.add_argument("--no-extra-conv", dest="extra_conv", action="store_false")
    ap.add_argument("--scalar-embed", type=int, default=64)
    ap.add_argument("--ent-floor", type=float, default=0.35,
                    help="push ent_coef up whenever measured entropy drops under "
                         "this. A curriculum rung solved by ONE action drives the "
                         "policy deterministic, and a collapsed policy can never "
                         "sample the two-action discovery the NEXT rung needs.")
    ap.add_argument("--ent-coef-max", type=float, default=0.2)
    ap.add_argument("--bc-init", dest="bc_init", action="store_true", default=True)
    ap.add_argument("--no-bc-init", dest="bc_init", action="store_false",
                    help="ablation: no behaviour cloning of the demonstration")
    ap.add_argument("--bc-epochs", type=int, default=1000)
    ap.add_argument("--bc-lr", type=float, default=3e-3)
    ap.add_argument("--bc-every", type=int, default=10,
                    help="re-fit the demo actions every N PPO updates (0 = off)")
    ap.add_argument("--bc-steps", type=int, default=30)
    ap.add_argument("--bc-refresh-lr", type=float, default=3e-4)
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                    default=True,
                    help="ablation: always start at the REAL entry, no backward "
                         "ladder (isolates the curriculum's contribution)")
    ap.add_argument("--channels", default=",".join(ICE_CHANNELS),
                    help="ablation: grid channels, e.g. 'solid,king,hazard' for "
                         "the legacy observation")
    ap.add_argument("--no-vel-obs", dest="vel_obs", action="store_false", default=True,
                    help="ablation: hide the king's velocity")
    ap.add_argument("--vel-encoding", default="xy", choices=("xy", "polar"),
                    help="'polar' reports (|v|, dir_x, dir_y): how much momentum, "
                         "separately from which way. Measured on the demo decision "
                         "points, the king is almost always sliding DOWN A RAMP, so "
                         "direction needs both components, while the charge-time "
                         "drift scales with magnitude alone.")
    ap.add_argument("--fail-penalty", type=float, default=0.0,
                    help="uniform cost for any episode that does not cross. Keep "
                         "it BELOW the crossing reward (~21) so exploring always "
                         "beats failing safely.")
    ap.add_argument("--route-dense", type=float, default=0.0,
                    help="weight on the within-leg route term. The default sparse "
                         "potential only moves when the king lands inside the next "
                         "waypoint box, which a policy with no prior never does. "
                         "Set it equal to the waypoint reward (5) for a continuous "
                         "potential.")
    ap.add_argument("--stuck-limit", type=int, default=2,
                    help="end an episode after this many consecutive actions "
                         "that move the king nowhere (0 = off)")
    ap.add_argument("--gate-jitter-dx", type=float, default=2.0,
                    help="position spread used by the LADDER GATE only. With no "
                         "jitter the greedy policy is deterministic, so 20 eval "
                         "episodes are 20 identical runs -- a one-sample test "
                         "that lets a rung through on a transient success the "
                         "next PPO update undoes. Measured: rung 2 on L38 was "
                         "'passed' at update 220 and no saved checkpoint after "
                         "it ever crossed from there.")
    ap.add_argument("--gate-jitter-dv", type=float, default=0.05,
                    help="speed spread for the ladder gate, as a fraction")
    ap.add_argument("--subgoal", action="store_true",
                    help="at the FRONTIER rung, end the episode successfully as "
                         "soon as the king reaches the next rung's state. The "
                         "rungs above are already solved, so that state is the "
                         "real task; requiring the whole crossing makes the "
                         "gradient depend on a stochastic tail.")
    ap.add_argument("--subgoal-tol", type=float, default=14.0)
    ap.add_argument("--subgoal-vtol", type=float, default=1.0,
                    help="how close the SPEED must also be. Position alone is "
                         "not the state on ice.")
    ap.add_argument("--subgoal-bonus", type=float, default=20.0)
    ap.add_argument("--step-cost", type=float, default=0.0,
                    help="cost per ACTION. fail_penalty prices the end of a failed "
                         "episode but not the time spent, so shuffling in place "
                         "stays free and there is no pressure to attempt anything "
                         "-- measured on L37 rung 6, the policy walked left eight "
                         "times for 0.00 each and then took the -10. A per-step "
                         "cost makes a short successful route beat a long safe one.")
    ap.add_argument("--temper-after", type=int, default=0,
                    help="if the ladder has not advanced for this many updates, "
                         "multiply the policy head by --temper-by. Scaling all "
                         "logits by a positive constant leaves argmax (and so "
                         "every solved rung) EXACTLY unchanged while softening "
                         "the distribution, which restores both sampling and the "
                         "gradient. Aimed at a saturated softmax: measured on L36 "
                         "rung 5 the policy sat at p=1.00 on an action that falls "
                         "three levels, and 120-odd sampled alternatives over 300 "
                         "updates could not shift it. 0 disables.")
    ap.add_argument("--temper-by", type=float, default=0.5)
    ap.add_argument("--temper-below", type=float, default=0.3,
                    help="only temper when the frontier greedy rate is under this. "
                         "Tempering a policy that is climbing knocks its confidence "
                         "back down: measured on L36, greedy sat at 0.70-0.80 "
                         "against a 0.9 gate and the stall timer fired three times "
                         "in a row, each time undoing the progress toward it.")
    ap.add_argument("--explore-eps", type=float, default=0.0,
                    help="probability floor mixed into the policy so no action is "
                         "ever sampled with probability 0. The entropy coefficient "
                         "only raises AVERAGE entropy; a policy can average 0.8 and "
                         "still be at p=1.00 on the one state that matters.")
    ap.add_argument("--guard-prev", type=float, default=0.0,
                    help="fraction of episodes started at the rung JUST ABOVE the "
                         "frontier -- the one that regresses, because it shares "
                         "actions with what the frontier is learning. 0 keeps the "
                         "plain uniform rehearsal.")
    ap.add_argument("--demote", dest="demote", action="store_true", default=True,
                    help="step the ladder back up when an already-unlocked rung "
                         "stops working (on by default)")
    ap.add_argument("--no-demote", dest="demote", action="store_false")
    ap.add_argument("--demote-hold", type=int, default=2,
                    help="consecutive failed checks before a rung is given back. "
                         "A rung sitting near the bar fluctuates across it from "
                         "sampling noise alone -- measured on L36 rung 4, the true "
                         "rate under the gate's jitter is 0.70 against a 0.70 bar -- "
                         "and demoting on one bad sample burns the run re-solving "
                         "a rung that was never actually lost.")
    ap.add_argument("--demote-thresh", type=float, default=0.45,
                    help="an unlocked rung falling below this is treated as lost. "
                         "Must sit well under --thresh, or a rung opened at the "
                         "bar is handed straight back on the next evaluation.")
    ap.add_argument("--gate-hold", type=int, default=2,
                    help="consecutive passing evaluations required to advance")
    ap.add_argument("--jitter-dx", type=float, default=0.0,
                    help="randomise the restored x by +/- this many pixels")
    ap.add_argument("--jitter-dv", type=float, default=0.0,
                    help="randomise the restored speed by +/- this fraction")
    ap.add_argument("--goal-level", type=int, default=None,
                    help="the level that counts as won; default is the demo's "
                         "own goal (level+1, or the far side of a spanning route)")
    ap.add_argument("--fine-walk-frames", type=int, default=3,
                    help="length of the fine walk action, in frames; the "
                         "demonstration's own table_cfg overrides this")
    ap.add_argument("--wind-obs", action="store_true",
                    help="put the wind phase in the observation (snow biome)")
    ap.add_argument("--wind-jump", default="",
                    help="wind buckets to build jump_wind macros for, e.g. 0,4 -- "
                         "the snow levels' demonstrations are made of them")
    ap.add_argument("--entry-jitter-dx", type=float, default=0.0,
                    help="position jitter applied AT RUNG 0 only (the real "
                         "arrival), where the chain below decides the state")
    ap.add_argument("--entry-jitter-dv", type=float, default=0.0,
                    help="speed jitter applied at rung 0 only")
    ap.add_argument("--wind-jitter", action="store_true",
                    help="randomise the wind phase at every episode start, so the "
                         "policy cannot memorise one gust timing")
    ap.add_argument("--tag", default=None, help="suffix for the checkpoint dir")
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint: weights, optimiser state "
                         "AND the curriculum rung it had reached")
    ap.add_argument("--start-rung", type=int, default=None,
                    help="with --resume, force the ladder back to this rung")
    ap.add_argument("--reset-head-on-advance", action="store_true",
                    help="re-initialise the policy head every time the ladder "
                         "opens a new rung, so each frontier starts from a "
                         "uniform prior at its own state")
    ap.add_argument("--reset-head", action="store_true",
                    help="with --resume, re-initialise the policy head while "
                         "keeping the trunk -- the only way found to un-stick an "
                         "action PPO has driven to the bottom at a state")
    ap.add_argument("--reset-opt", action="store_true",
                    help="with --resume, start Adam fresh instead of restoring it")
    ap.add_argument("--demo", default=None,
                    help="demonstration file to build the curriculum from "
                         "(default starts/demo_L<N>.json). Use this to train on "
                         "the arrival the RELAY really delivers -- see "
                         "tools/route_from_entry.py -- rather than one an "
                         "earlier route happened to start from.")
    ap.add_argument("--handoff-seed", action="store_true",
                    help="when a rung opens (and once at startup), search the "
                         "38 first actions for one that hands off to a state "
                         "the CURRENT policy finishes from, then distil that "
                         "single label at that single state while holding the "
                         "solved tail's own choices fixed. The demo's action "
                         "always qualifies while the tail holds, so the ladder "
                         "can always be extended by one rung; PPO's job becomes "
                         "making the seeded action robust, not discovering it.")
    ap.add_argument("--seed-steps", type=int, default=400)
    ap.add_argument("--seed-lr", type=float, default=1e-4)
    ap.add_argument("--seed-check-every", type=int, default=20)
    ap.add_argument("--grad-split", dest="grad_split", action="store_true",
                    default=True,
                    help="clip the policy and value gradients SEPARATELY before "
                         "summing them into the shared trunk. On by default: with "
                         "one global clip the value loss -- a squared return, "
                         "~485 on this reward scale -- owns 99.9%% of the gradient "
                         "norm, and clipping to 0.5 leaves the policy moving at "
                         "~1/800 of its learning rate. Measured at the stuck L37 "
                         "rung 5: policy-loss grad 0.55 against value-loss grad "
                         "443.6.")
    ap.add_argument("--no-grad-split", dest="grad_split", action="store_false",
                    help="ablation: one global gradient clip, as before")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()
    args.wind_jump = tuple(int(b) for b in str(args.wind_jump).split(",") if b.strip())
    global WIND_JITTER, ENTRY_JITTER
    WIND_JITTER = bool(args.wind_jitter)
    if args.entry_jitter_dx > 0.0 or args.entry_jitter_dv > 0.0:
        ENTRY_JITTER = (args.entry_jitter_dx, args.entry_jitter_dv)

    demo = load_demo(args.level, args.demo)
    wps = waypoints(demo, args.waypoint_tol)
    snaps = [s["snapshot"] for s in demo["steps"]]
    save_dir = args.save_dir or f"checkpoints/L{args.level}_ice{args.tag or ''}"
    os.makedirs(save_dir, exist_ok=True)

    device = get_device()
    rng = np.random.default_rng(args.seed)
    # THE TABLE COMES FROM THE DEMONSTRATION when it carries one; this has to
    # happen before make_env, which needs fine_walk_frames as well.
    tcfg = demo.get("table_cfg")
    if tcfg:
        args.settle_action = bool(tcfg.get("settle_action", args.settle_action))
        args.wind_jump = tuple(tcfg.get("wind_jump", args.wind_jump))
        args.fine_walk_frames = int(tcfg.get("fine_walk_frames",
                                             args.fine_walk_frames))
        print(f"action table from the demonstration: settle={args.settle_action} "
              f"wind_jump={list(args.wind_jump)} "
              f"fine_walk_frames={args.fine_walk_frames}", flush=True)
    envs = [make_env(args, demo, wps) for _ in range(args.num_envs)]
    table = build_action_table(fine_walk_frames=args.fine_walk_frames,
                               extra_charges=(22, 24, 28, 30),
                               wind_jump=args.wind_jump,
                               settle_action=args.settle_action)
    for e in envs:
        e.actions = table
        e.num_actions = len(table)
    e0 = envs[0]
    print(f"L{args.level}: {len(snaps)} demo snapshots, {len(wps)} waypoints, "
          f"{len(table)} actions, obs_dim {e0.obs_dim} "
          f"(channels {e0.grid_channels}, {e0.n_scalars} scalars)", flush=True)

    agent = PPO(e0.obs_dim, len(table), device=device, lr=args.lr,
                ent_coef=args.ent_coef,
                grid_shape=e0.grid_shape, n_scalars=e0.n_scalars,
                extra_conv=args.extra_conv, scalar_embed=args.scalar_embed,
                explore_eps=args.explore_eps, grad_split=args.grad_split)
    print(f"params={sum(p.numel() for p in agent.net.parameters()):,} "
          f"trunk_in={agent.net.trunk[0].in_features}", flush=True)

    demo_acts = demo_action_indices(demo, table)
    bc_data = None
    if args.resume:
        pass                      # the resumed weights ARE the prior
    elif args.bc_init:
        _, bc_data = bc_init(agent, envs, snaps, demo_acts,
                             args.bc_epochs, args.bc_lr, device)

    cur = Curriculum(len(snaps), args.window, args.thresh, args.rehearse,
                     enabled=args.curriculum, guard=args.guard_prev)

    resumed_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        sd = dict(ck["model"])
        # The action table may have shrunk since the checkpoint -- dropping the
        # settle primitive removes the LAST row of the policy head. Slice it
        # instead of re-initialising: the preferences learned for every other
        # action are still valid, and only the removed one goes.
        head_resized = False
        hw = sd.get("policy_head.weight")
        if hw is not None and hw.shape[0] != agent.net.policy_head.weight.shape[0]:
            head_resized = True
            n_new = agent.net.policy_head.weight.shape[0]
            if hw.shape[0] > n_new:
                sd["policy_head.weight"] = hw[:n_new]
                sd["policy_head.bias"] = sd["policy_head.bias"][:n_new]
                print(f"resume: policy head {hw.shape[0]} -> {n_new} actions "
                      f"(trailing rows dropped, the rest kept)", flush=True)
            else:
                raise SystemExit(f"checkpoint has {hw.shape[0]} actions but the "
                                 f"env has {n_new}; cannot grow a head here")
        agent.net.load_state_dict(sd)
        # Adam keeps per-parameter moments; a resized head makes the stored ones
        # the wrong shape, and load_state_dict does NOT check -- it fails later
        # inside the first opt.step().
        if head_resized:
            print("resume: head was resized -> starting with a fresh optimiser",
                  flush=True)
        elif not args.reset_opt and "opt" in ck:
            try:
                agent.opt.load_state_dict(ck["opt"])
            except Exception as ex:
                print(f"resume: optimiser state unusable ({ex}); starting fresh")
        resumed_step = int(ck.get("step", 0))
        icc = ck.get("ice_curriculum", {})
        # carry the LADDER too: restoring weights but restarting the curriculum
        # at the last rung would just re-solve what the policy already knows
        if args.curriculum and icc.get("n_snapshots") == len(snaps):
            cur.k = int(icc.get("k", cur.k))
        if args.start_rung is not None:
            cur.k = int(args.start_rung)     # the ladder may have over-advanced
            # ONLY inherit the finished flag if the ladder is actually AT the
            # entry. Carrying it while forcing the start back up ends the run on
            # the first update with a "SOLVED" line that means nothing --
            # measured on L38 resumed against a NEW demonstration (the real relay
            # arrival): the checkpoint had solved the OLD entry, so update 1
            # printed SOLVED and broke before the new rung 0 was ever trained.
            cur.solved_from_entry = (bool(icc.get("solved_from_entry", False))
                                     and cur.k == 0)
        if args.reset_head:
            # RE-INITIALISE THE POLICY HEAD, KEEP THE TRUNK. Once PPO drives an
            # action to the bottom at a state it cannot climb back: measured on
            # L37 rung 5, the one action that works sat at rank 38 of 38 with
            # p=1e-5 and stayed there through every remedy -- a lone sample in a
            # batch of 768, with the ratio clipped at 1.2, cannot lift it. A
            # fresh head restores a uniform starting point while the trunk keeps
            # what the level looks like. The rungs already solved must be
            # re-learned, which the sub-goal makes cheap.
            torch.nn.init.orthogonal_(agent.net.policy_head.weight, 0.01)
            torch.nn.init.constant_(agent.net.policy_head.bias, 0.0)
            agent.opt = torch.optim.Adam(agent.net.parameters(), lr=args.lr)
            print("resume: policy head re-initialised, trunk kept, optimiser fresh",
                  flush=True)
        print(f"resumed {os.path.basename(args.resume)} at step {resumed_step}, "
              f"rung k={cur.k}/{len(snaps) - 1}", flush=True)

    # SUB-GOAL TOLERANCES ARE PER RUNG. A fixed box is degenerate wherever the
    # demonstration's own step is shorter than the box: measured on L37 rung 3,
    # snapshot 3 sits 8.5 px and 0.60 speed from snapshot 4 against a 14 px /
    # 1.0 box, so the rung's OWN START already satisfied its sub-goal. PPO found
    # a two-action shuffle that ends 6 px from the target at the wrong speed,
    # banked the +20, and the frontier rate read 0.93 while the greedy crossing
    # sat at 0.00 for 755 updates. Capping each tolerance at 40% of the gap the
    # step actually covers keeps the sub-goal a statement about progress.
    def _sg_tol(k, trials=40):
        """Size the box from HOW PRECISELY THE CORRECT ACTION LANDS, not from a
        constant. Capping at a fraction of the gap fixed rung 3 and still left
        rung 2 open: there the gap in speed is 4.52, so the cap was the flat 1.0,
        and a four-action shuffle ending 5.2 px away at speed 2.21 (against the
        rung's 2.87) sat inside the box. PPO took the +20 and terminated, the
        frontier rate read 0.83, and the greedy crossing stayed 0.00 for 360
        updates -- because the tail cannot finish from that state.

        The demonstration's own action, replayed from starts jittered exactly
        the way training jitters them, lands within ~1 px and ~0.1 speed. Twice
        that spread is a box the right action always hits and a wandering
        sequence does not. The box only shortcuts the frontier decision; a
        policy that finds a different route is still paid by the real crossing
        reward, which is how the tail came to beat the demonstration."""
        if k + 1 >= len(snaps):
            return args.subgoal_tol, args.subgoal_tol, args.subgoal_vtol
        a_, b_ = snaps[k]["king"], snaps[k + 1]["king"]
        gx = abs(float(a_["rect_x"]) - float(b_["rect_x"]))
        gy = abs(float(a_["rect_y"]) - float(b_["rect_y"]))
        gvx_a, gvy_a = king_vel(float(a_["speed"]), float(a_["angle"]))
        gvx_b, gvy_b = king_vel(float(b_["speed"]), float(b_["angle"]))
        gv = max(abs(gvx_a - gvx_b), abs(gvy_a - gvy_b))
        ex = ey = ev = 0.0
        act = demo_acts[k]
        if act >= 0:
            e = envs[0]
            for _ in range(trials):
                sn = copy.deepcopy(snaps[k])
                if args.jitter_dx > 0.0:
                    sn["king"]["rect_x"] = (float(sn["king"]["rect_x"])
                                            + float(rng.uniform(-args.jitter_dx,
                                                                args.jitter_dx)))
                if args.jitter_dv > 0.0:
                    sn["king"]["speed"] = (float(sn["king"]["speed"])
                                           * float(rng.uniform(1.0 - args.jitter_dv,
                                                               1.0 + args.jitter_dv)))
                e.max_steps = 4
                e.restore(sn, route_i=k - 1)
                e.step(act)
                vx, vy = king_vel(float(e.king.speed), float(e.king.angle))
                ex = max(ex, abs(float(e.king.rect_x) - float(b_["rect_x"])))
                ey = max(ey, abs(float(e.king.rect_y) - float(b_["rect_y"])))
                ev = max(ev, abs(vx - gvx_b), abs(vy - gvy_b))
        tx = min(args.subgoal_tol, max(2.0, 2.0 * ex))
        ty = min(args.subgoal_tol, max(2.0, 2.0 * ey))
        tv = min(args.subgoal_vtol, max(0.15, 2.0 * ev))
        # and never wider than 40% of the gap the step actually covers, or the
        # rung's own start can satisfy its own sub-goal
        if gx > 0.0:
            tx = min(tx, max(2.0, 0.4 * gx))
        if gy > 0.0:
            ty = min(ty, max(2.0, 0.4 * gy))
        if gv > 0.0:
            tv = min(tv, max(0.15, 0.4 * gv))
        return tx, ty, tv

    sg_tol = [_sg_tol(k) for k in range(len(snaps))]
    if args.subgoal:
        for k in range(len(snaps) - 1):
            tx, ty, tv = sg_tol[k]
            print(f"  subgoal rung {k}: box {tx:.2f}/{ty:.2f} px, "
                  f"{tv:.2f} per velocity component", flush=True)

    starts = [0] * args.num_envs

    def reset_one(i):
        k = cur.sample(rng)
        starts[i] = k
        # BUDGET THE EPISODE TO THE RUNG. The demonstration needs (n - k) actions
        # from snapshot k, so anything past that is the agent flailing after the
        # decision that mattered. On ice that flailing is not neutral: one wrong
        # action wedges the king somewhere no action moves him, and every step
        # after it teaches "this action does nothing" -- including the winning
        # action, if it happens to be sampled there. Measured at L38 rung 5,
        # 15 of every 16 buffered steps came from the wedge and PPO drove the two
        # crossing actions to p=1e-4. Cutting the episode to the rung's own
        # length keeps the buffer about the decision.
        envs[i].max_steps = (len(snaps) - k) + args.step_slack
        snap = copy.deepcopy(snaps[k])
        # ARRIVAL RANDOMISATION. Trained only on the demonstration's exact
        # states, a policy has no reason to generalise -- measured: it matches
        # the open-loop macro from the exact entry and degrades with it under
        # perturbation, because both are effectively replaying one sequence.
        # Jittering the position and the carried speed at every rung is what
        # makes the policy a controller rather than a second copy of the script.
        snap = jitter_snap(snap, *rung_jitter(k, args.jitter_dx, args.jitter_dv),
                           wind=args.wind_jitter, rng=rng)
        # snapshot k is the state BEFORE demo step k, so waypoint k is the next
        # one to earn -- the env must not be left waiting for waypoint 0.
        obs, _ = envs[i].restore(snap, route_i=k - 1)
        return obs

    eval_env = make_env(args, demo, wps)      # separate: never disturbs a rollout
    eval_env.actions = table
    eval_env.num_actions = len(table)

    if args.handoff_seed:
        eval_env.max_steps = (len(snaps) - cur.k) + args.step_slack
        g0 = greedy_rate(agent.net, eval_env, snaps, cur.k, args.eval_episodes,
                         eval_env.max_steps, args.level, device,
                         (args.gate_jitter_dx, args.gate_jitter_dv), rng)
        if g0 < args.thresh:
            print(f"  seed: rung {cur.k} starts at greedy {g0:.2f}", flush=True)
            seed_frontier(agent, eval_env, snaps, cur.k, len(snaps), args.level,
                          args.step_slack, len(table), device, args.seed_steps,
                          args.seed_lr, args.seed_check_every, rng,
                          (args.gate_jitter_dx, args.gate_jitter_dv),
                          args.eval_episodes, args.thresh,
                          demo_acts[cur.k] if cur.k < len(demo_acts) else None)

    obs = np.stack([reset_one(i) for i in range(args.num_envs)])
    ep_ret = np.zeros(args.num_envs)
    hist_ret, hist_succ = [], []
    last_move = [0]            # update at which the ladder last moved
    fails = [0]                # consecutive failed checks of the rung below
    best_by_rung = {}          # rung -> best gate score ever seen there
    t0 = time.monotonic()      # wall clock can jump; rate must not
    step_count = resumed_step

    for update in range(1, args.updates + 1):
        buf = RolloutBuffer(args.rollout, args.num_envs, e0.obs_dim, device)
        for _ in range(args.rollout):
            ob = torch.as_tensor(obs, device=device).float()
            with torch.no_grad():
                action, logprob, value = agent.net.act(ob)
            a_np = action.cpu().numpy()
            rews = np.zeros(args.num_envs, np.float32)
            terms = np.zeros(args.num_envs, np.float32)
            truncs = np.zeros(args.num_envs, np.float32)
            nxt = []
            for i, e in enumerate(envs):
                o, r, te, tr, info = e.step(int(a_np[i]))
                # SUB-GOAL AT THE FRONTIER. Asking rung k to cross the whole
                # level makes its reward depend on 5-6 further actions that are
                # sampled stochastically, so the one action being learned is
                # credited only when the entire rest of the chain happens to go
                # right. Measured on L37 rung 6: the three actions that reach
                # the only useful platform sat at p=1e-5, and raising their
                # sampling 3000x did not move them, because the tail failed and
                # handed them a NEGATIVE advantage. Rungs above are already
                # solved, so reaching the next rung's state IS the task.
                if args.subgoal and starts[i] == cur.k and cur.k + 1 < len(snaps)                         and not (te or tr):
                    goal_king = snaps[cur.k + 1]["king"]
                    # POSITION IS NOT THE STATE. Matching only x,y let the
                    # policy collect the bonus while arriving with the wrong
                    # momentum -- measured on L37 rung 7: it reached x=307.8
                    # against the rung's 308.4 but at speed 4.41 against 1.80,
                    # banked the reward, and fell off the level two actions
                    # later. On ice the velocity is half the state; this is the
                    # same trap teleport() falls into.
                    tol_x, tol_y, tol_v = sg_tol[cur.k]
                    gvx, gvy = king_vel(float(goal_king["speed"]),
                                        float(goal_king["angle"]))
                    kvx, kvy = king_vel(float(e.king.speed), float(e.king.angle))
                    if (abs(float(e.king.rect_x) - float(goal_king["rect_x"])) <= tol_x
                            and abs(float(e.king.rect_y) - float(goal_king["rect_y"])) <= tol_y
                            and abs(kvx - gvx) <= tol_v and abs(kvy - gvy) <= tol_v
                            and e.levels.current_level == args.level
                            and e.move_available()):
                        r += args.subgoal_bonus
                        te = True
                        info = dict(info)
                        info["level"] = args.level + 1
                        info["success"] = True             # counts as success
                rews[i], terms[i], truncs[i] = r, float(te), float(tr)
                ep_ret[i] += r
                if te or tr:
                    success = won(info)
                    hist_ret.append(ep_ret[i])
                    hist_succ.append(1.0 if success else 0.0)
                    if starts[i] == cur.k:
                        cur.report(success)
                    ep_ret[i] = 0.0
                    o = reset_one(i)
                nxt.append(o)
            buf.add(ob, action, logprob,
                    torch.as_tensor(rews, device=device),
                    value,
                    torch.as_tensor(terms, device=device),
                    torch.as_tensor(truncs, device=device))
            obs = np.stack(nxt)
            step_count += args.num_envs

        with torch.no_grad():
            last_v = agent.net.forward(torch.as_tensor(obs, device=device).float())[1]
        logs = agent.update(buf, last_v)

        # DAPG-style demo anchor. Pure BC drifts: once a stochastic rollout
        # deviates from the demonstration the policy is in states cloning never
        # saw, the route breaks, and on ice a broken route is a fall. Re-fitting
        # the demo actions every few updates holds the prior in place while PPO
        # makes the surrounding states robust.
        if bc_data is not None and args.bc_every > 0 and update % args.bc_every == 0:
            xb, yb = bc_data
            opt = torch.optim.Adam(agent.net.parameters(), lr=args.bc_refresh_lr)
            for _ in range(args.bc_steps):
                logits, _ = agent.net(xb)
                l = torch.nn.functional.cross_entropy(logits, yb)
                opt.zero_grad(); l.backward(); opt.step()

        if args.ent_floor > 0.0:
            if logs["entropy"] < args.ent_floor:
                agent.ent_coef = min(agent.ent_coef * 1.4, args.ent_coef_max)
            elif logs["entropy"] > args.ent_floor * 1.6:
                agent.ent_coef = max(agent.ent_coef / 1.2, args.ent_coef)

        if update % args.eval_every == 0 or update == 1:
            eval_env.max_steps = (len(snaps) - cur.k) + args.step_slack
            g = greedy_rate(agent.net, eval_env, snaps, cur.k, args.eval_episodes,
                            eval_env.max_steps, args.level, device,
                            (args.gate_jitter_dx, args.gate_jitter_dv), rng)
            # KEEP THE BEST WEIGHTS FOR THIS RUNG. The gate measures, the ladder
            # moves on, and nothing preserves the policy that passed -- if it
            # degrades on the next update those weights are gone. Measured on
            # L36: ppo_5299200 scores 0.85 at rung 4 while every checkpoint
            # before and after scores 0.45, so the run spent 750k further steps
            # stuck below a solution it had already found and thrown away.
            # best-EVER per rung. Keying on a single counter meant that coming
            # back to a rung after a demote restarted its record, and the file
            # was then overwritten by a weaker policy -- measured: a verified
            # 1.00 checkpoint for L36 rung 4 was clobbered at 0.58.
            if g > best_by_rung.get(cur.k, -1.0):
                best_by_rung[cur.k] = g
                bp = os.path.join(save_dir, f"best_rung{cur.k}.pt")
                torch.save(cfg_payload(agent, args, demo, cur, len(snaps),
                                       step_count), bp)
                print(f"  best: rung {cur.k} greedy {g:.2f} -> saved "
                      f"{os.path.basename(bp)}", flush=True)
            advanced = cur.advance(g, args.gate_hold)
            if advanced and args.handoff_seed and cur.k >= 0:
                seed_frontier(agent, eval_env, snaps, cur.k, len(snaps),
                              args.level, args.step_slack, len(table), device,
                              args.seed_steps, args.seed_lr,
                              args.seed_check_every, rng,
                              (args.gate_jitter_dx, args.gate_jitter_dv),
                              args.eval_episodes, args.thresh,
                              demo_acts[cur.k] if cur.k < len(demo_acts) else None)
            if advanced and args.reset_head_on_advance:
                # A HEAD PER RUNG, in effect. The suppression that blocks a rung
                # forms while the frontier is still elsewhere: the new rung's
                # state is never visited, so the head's output there is pure
                # generalisation, and it lands the one useful action at the
                # bottom. Measured on L37 rung 5, twice, from two independent
                # initialisations. Re-initialising the head at the moment the
                # rung becomes the frontier gives it a uniform prior exactly
                # where it is needed; the sub-goal keeps that rung a one-action
                # task, and rehearsal restores the rungs already solved.
                torch.nn.init.orthogonal_(agent.net.policy_head.weight, 0.01)
                torch.nn.init.constant_(agent.net.policy_head.bias, 0.0)
                agent.opt = torch.optim.Adam(agent.net.parameters(), lr=args.lr)
                print(f"  head re-initialised for rung {cur.k}", flush=True)
            if advanced:
                last_move[0] = update
            elif (args.temper_after and update - last_move[0] >= args.temper_after
                  and g < args.temper_below):
                with torch.no_grad():
                    agent.net.policy_head.weight.mul_(args.temper_by)
                    agent.net.policy_head.bias.mul_(args.temper_by)
                # stale Adam moments would undo the scaling within a few steps
                for grp in agent.opt.param_groups:
                    for prm in grp["params"]:
                        st = agent.opt.state.get(prm)
                        if st:
                            st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()
                last_move[0] = update
                pmax = float(torch.softmax(
                    agent.net(torch.as_tensor(
                        eval_env.restore(copy.deepcopy(snaps[cur.k]),
                                         route_i=cur.k - 1)[0],
                        device=device).float().unsqueeze(0))[0], 1).max())
                print(f"  ladder: stalled {args.temper_after} updates at rung "
                      f"{cur.k} -> tempered policy head by {args.temper_by} "
                      f"(max action prob now {pmax:.3f}; argmax unchanged)",
                      flush=True)
            if advanced:
                print(f"  ladder: greedy {g:.2f} at rung {cur.k + 1} -> start moved "
                      f"back to snapshot {cur.k}/{len(snaps) - 1} (update {update})",
                      flush=True)
            elif args.demote and cur.k < len(snaps) - 1:
                # the frontier is not clearing; make sure the rung it stands on
                # still works before blaming the frontier for being hard
                prev = cur.k + 1
                eval_env.max_steps = (len(snaps) - prev) + args.step_slack
                gp = greedy_rate(agent.net, eval_env, snaps, prev,
                                 args.eval_episodes, eval_env.max_steps,
                                 args.level, device,
                                 (args.gate_jitter_dx, args.gate_jitter_dv), rng)
                below = gp < args.demote_thresh
                fails[0] = fails[0] + 1 if below else 0
                if below and fails[0] >= args.demote_hold and cur.demote():
                    fails[0] = 0
                    print(f"  ladder: rung {prev} REGRESSED to greedy {gp:.2f} "
                          f"-> stepping back up to it (update {update})", flush=True)

        if update % 5 == 0 or update == 1:
            sps = (step_count - resumed_step) / max(1e-9, time.monotonic() - t0)
            r = np.mean(hist_ret[-100:]) if hist_ret else float("nan")
            sc = np.mean(hist_succ[-100:]) if hist_succ else float("nan")
            print(f"upd {update:4d} | step {step_count:7d} | {sps:5.0f} sps | "
                  f"ret {r:7.2f} | succ {sc:.2f} | k {cur.k:2d}/{len(snaps) - 1} "
                  f"(front {cur.rate:.2f} greedy {cur.last_greedy:.2f}) | ent {logs['entropy']:.3f} | "
                  f"kl {logs['approx_kl']:+.4f} | vl {logs['value_loss']:7.1f} | "
                  f"ec {agent.ent_coef:.3f}", flush=True)

        if update % args.save_every == 0:
            p = os.path.join(save_dir, f"ppo_{step_count}.pt")
            torch.save(cfg_payload(agent, args, demo, cur, len(snaps), step_count), p)
            print("saved", p, flush=True)

        if cur.solved_from_entry:
            p = os.path.join(save_dir, f"ppo_{step_count}.pt")
            torch.save(cfg_payload(agent, args, demo, cur, len(snaps), step_count), p)
            print(f"SOLVED from the real entry at update {update} "
                  f"(succ {cur.rate:.2f}); saved {p}", flush=True)
            break

    eval_env.close()
    for e in envs:
        e.close()


if __name__ == "__main__":
    sys.exit(main())
