#!/usr/bin/env python
"""Turn a solved combo into a GENUINE warm-started PPO setup.

The combo (from checkpoints/LNN/ppo_bc.pt) is an oracle: it proves the level is
solvable and gives the exact route. We use it to bootstrap real RL, NOT to ship
a script:

  1. Replay the combo step-by-step and record (observation, PRIMITIVE action) at
     every grounded decision point -- a demonstration dataset. The combo action
     itself is NOT in the policy's table, so the policy must act step-by-step.
  2. BC-warm-start a fresh ActorCritic on those primitives (+ small position
     perturbations for robustness) -> checkpoints/LNN/ppo_ppobc.pt.
  3. Emit the on-path grounded spots as starts_LNN_demo.json (reverse-curriculum
     start states, scored by depth).
  4. You then PPO fine-tune (Train.py --resume that ckpt --start-states that
     json --curriculum ...). PPO genuinely optimizes from an initialization that
     pure exploration could never reach; the generalization test (play from
     PERTURBED starts / different wind phases) tells us if it truly learned.

Windy levels keep the timed-jump macros (jump_wind) in the action space and use
wind_obs -- a learned macro-action policy that reacts to state, which is standard
RL, not a fixed script. Ice/deterministic use plain jumps.

    python Demos.py --level 32
    python Demos.py --level 30      # windy (auto-detected from the combo)
"""
import os, argparse, json
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
import torch
import torch.nn as nn
from JK_Env import JumpKingEnv, build_action_table
from PPO import ActorCritic, get_device


def load_combo(level):
    # read the proven route from BC.PATHS (robust to checkpoint housekeeping);
    # fall back to the ppo_bc.pt checkpoint if present.
    try:
        import BC
        return BC.PATHS[level]["wind_combo"][0], {}
    except (ImportError, KeyError):
        ck = torch.load(f"checkpoints/L{level}/ppo_bc.pt", map_location="cpu",
                        weights_only=False)
        return ck["action_cfg"]["wind_combo"][0], ck["action_cfg"]


def is_wind_step(s):
    return (s[0] not in ("walk", "settle", "jump")) and len(s) == 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--goal-level", type=int, default=None)
    ap.add_argument("--fine-walk-frames", type=int, default=3)
    ap.add_argument("--extra-charges", type=str, default="22,24,28,30")
    ap.add_argument("--launch-x", type=int, default=None,
                    help="override the launch x (ice: the POST-settle spot the "
                         "route was searched from, e.g. 452 for L36)")
    ap.add_argument("--vel-obs", action="store_true",
                    help="add (vx,vy) to obs so momentum is observable (ICE)")
    ap.add_argument("--perturb", type=str, default="-6,-3,0,3,6")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()
    perturb = [int(p) for p in args.perturb.split(",")]

    combo, cfg = load_combo(args.level)
    goal = args.goal_level if args.goal_level is not None else args.level + 1
    windy = any(is_wind_step(s) for s in combo)
    wind_obs = windy                 # windy policy must SEE the wind to react
    wind_jump = (0, 4) if windy else ()

    # env whose action table is the PRIMITIVES only (NO wind_combo action)
    env = JumpKingEnv(max_steps=200, goal_level=goal,
                      fine_walk_frames=args.fine_walk_frames, extra_charges=extra,
                      wind_jump=wind_jump, wind_obs=wind_obs, vel_obs=args.vel_obs)
    device = get_device()

    def prim_index(step):
        """Map a combo step to a primitive env action index."""
        if step[0] == "jump":
            return env.actions.index(("jump", step[1], int(step[2])))
        # wind step (bucket,dir,charge) -> the timed-jump macro action
        b, d, c = step
        return env.actions.index(("jump_wind", (int(b), d), int(c)))

    # --- 1) replay the combo, collect (obs, primitive) + grounded spots -------
    # The combo's entry (and any leading settle/walk) lives in BC.PATHS.
    import BC
    bc_cfg = BC.PATHS[args.level]
    ey = bc_cfg["y"]
    entry_x = (args.launch_x if args.launch_x is not None
               else bc_cfg["entries"][len(bc_cfg["entries"]) // 2])
    # with an explicit launch-x (post-settle), the leading settle/walk is already
    # accounted for -> replay the plain route jumps directly from there.
    skip_lead = args.launch_x is not None

    demos_obs, demos_act, spots = [], [], []
    # execute leading settle/walk steps, then record each jump/wind step
    prims = [s for s in combo if s[0] not in ("settle", "walk")]
    lead = [] if skip_lead else [s for s in combo if s[0] in ("settle", "walk")]

    def replay_to_launch():
        env.reset(level=args.level, rect_x=entry_x, rect_y=ey)
        for s in lead:
            if s[0] == "walk":
                # walk to target via fine steps
                fw = env.actions.index(("walk", "right", args.fine_walk_frames))
                for _ in range(60):
                    if abs(env.king.rect_x - int(s[1])) <= 2:
                        break
                    env.step(fw if env.king.rect_x < int(s[1]) else fw - 1)
            # settle: passively damp (no action); done inside env on next step

    # record along the canonical trajectory + perturbations of each grounded spot
    replay_to_launch()
    traj = [(int(env.king.rect_x), int(env.king.rect_y))]
    for s in prims:
        env.step(prim_index(s))
        traj.append((int(env.king.rect_x), int(env.king.rect_y)))
    # traj[i] is the grounded spot BEFORE prims[i]; last entry is the exit
    launch = traj[0]

    for i, s in enumerate(prims):
        a_idx = prim_index(s)
        sx, sy = traj[i]
        spots.append((sx, sy))
        for dx in perturb:
            env.reset(level=args.level, rect_x=sx + dx, rect_y=sy)
            if env.levels.current_level != args.level:
                continue
            if abs(env.king.rect_x - sx) > 12 or abs(env.king.rect_y - sy) > 12:
                continue
            for _ in range(args.reps):
                demos_obs.append(env._obs())
                demos_act.append(a_idx)

    X = np.asarray(demos_obs, dtype=np.float32)
    Y = np.asarray(demos_act, dtype=np.int64)
    print(f"L{args.level}: {'WINDY' if windy else 'plain'} | {len(prims)} route "
          f"steps | {len(X)} demo samples | actions={env.num_actions} "
          f"(no combo) | scalars={env.n_scalars}")
    print("action mix:", {env.actions[a]: int((Y == a).sum())
                          for a in sorted(set(Y.tolist()))})

    # --- 2) BC warm-start ----------------------------------------------------
    net = ActorCritic(env.obs_dim, env.num_actions, grid_shape=env.grid_shape,
                      n_scalars=env.n_scalars).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Xt, Yt = torch.as_tensor(X, device=device), torch.as_tensor(Y, device=device)
    w = torch.ones(env.num_actions, device=device)
    for a in set(Y.tolist()):
        w[a] = len(Y) / max(1, (Y == a).sum())
    lossf = nn.CrossEntropyLoss(weight=w)
    net.train()
    for ep in range(args.epochs):
        logits, _ = net(Xt)
        loss = lossf(logits, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            acc = (logits.argmax(1) == Yt).float().mean().item()
            print(f"  BC epoch {ep+1}: loss {loss.item():.3f} acc {acc:.3f}")

    out_ckpt = f"checkpoints/L{args.level}/ppo_ppobc.pt"
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"model": net.state_dict(), "step": 0,
                "action_cfg": {"fine_walk_frames": args.fine_walk_frames,
                               "extra_charges": list(extra), "wait_frames": [],
                               "wind_jump": list(wind_jump), "wind_combo": [],
                               "wind_obs": bool(wind_obs),
                               "vel_obs": bool(args.vel_obs)}}, out_ckpt)

    # --- 3) emit reverse-curriculum start states -----------------------------
    n = len(spots)
    states = [{"level": args.level, "x": sx, "y": sy,
               "score": round((n - i) / n, 3)}       # exit-first = highest score
              for i, (sx, sy) in enumerate(spots)]
    out_starts = f"starts_L{args.level}_demo.json"
    with open(out_starts, "w") as f:
        json.dump(states, f, indent=2)

    print(f"saved warm-start -> {out_ckpt}")
    print(f"saved curriculum ({n} on-path states) -> {out_starts}")
    print(f"\nNext: PPO fine-tune (genuine RL from the warm-start):\n"
          f"  python Train.py --num-envs 8 --rollout 256 --total-steps 2000000 "
          f"--goal-level {goal} --start-states {out_starts} --curriculum "
          f"--max-steps 120 --p-bottom 0.0 --cur-target-p-bottom 0.0 "
          f"--cur-advance-rate 0.5 --cur-window 20 --ent-coef 0.03 "
          f"--fine-walk-frames {args.fine_walk_frames} "
          f"--extra-charges {args.extra_charges} "
          f"{'--wind-obs --wind-jump 0,4 ' if windy else ''}"
          f"{'--vel-obs ' if args.vel_obs else ''}"
          f"--quarantine-after 100000 --stop-at-succ 0.9 "
          f"--resume {out_ckpt} --save-dir checkpoints/L{args.level}")
    env.close()


if __name__ == "__main__":
    main()
