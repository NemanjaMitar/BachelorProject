#!/usr/bin/env python
"""
Procedural generator of REAL-ENGINE Jump King screens.

Why this exists
---------------
`RandomLevel.py` is a *toy* generator: its own numpy projectile physics, its own
13-scalar observation, its own small MLP. It answers "does PPO generalize on an
abstract ledge-stacking MDP?" -- it says NOTHING about the agent that plays the
actual game, because that agent never runs there (no King.py collisions, no
occupancy grid, no conv policy, no macro-action table).

This module is the honest version of the same experiment. A generated screen is
a `jumpking-world` (CustomWorld.py) whose platforms are fed straight into
`Platforms.Platform`, so King.py's real collision / bounce / slip / slope code
acts on them, JK_Env builds its usual occupancy grid from them, and the SAME
conv policy with the SAME macro-action table trains and is evaluated on them.
A success rate here is a statement about the real agent.

What a generated world is
-------------------------
Two screens. Screen 0 is a random ladder of ledges from the floor to near the
top edge; screen 1 is a catch floor. Success = climbing out of screen 0, i.e.
exactly what "crossing a level" means everywhere else in this project.

Ledges are placed within *measured* jump reach (max charge lifts the king ~150
px and carries him ~200 px sideways), and `prove_ladder` then re-proves each
rung IN THE ENGINE: it stands the king on rung i and searches the real action
table for one action that lands him on a higher rung (and, from the top rung,
one that leaves the screen). A world that survives that is solvable by a chain
of single macro-actions -- a constructive proof, not an assumption.

Usage
-----
    python WorldGen.py --preview --seed 3            # describe one world
    python WorldGen.py --selftest --n 10             # engine-prove random worlds
    python WorldGen.py --out levels/gen_eval --n 40 --seed 100000

Training on an endless stream of these (new screen every episode):

    python Train.py --world-gen --total-steps 400000 --save-dir checkpoints/worldgen

Held-out evaluation:

    python GenEval.py --checkpoint checkpoints/worldgen/ppo_final.pt --worlds levels/gen_eval
"""
import os
import json
import argparse
from dataclasses import dataclass, asdict

import numpy as np

import CustomWorld

SCREEN_W, SCREEN_H = 480, 360
FLOOR_Y, FLOOR_H = 344, 16
WALL_W = 8                       # the original screens' side walls (x 0..8, 472..480)

# The screen the king climbs INTO. Its floor sits BELOW the visible screen, and
# that is the whole trick. King._check_level moves a crossing king by
# screen_h + rect_width, so he arrives in the next screen at rect_y ~ 371..379 --
# *under* its bottom edge. A floor drawn anywhere on-screen is therefore a
# CEILING he rises into: he bonks and drops straight back. Measured over 15
# held-out screens, a full-charge up-jump from the top rung crossed only 53% of
# the time with a floor at y=352, and the failures were insensitive to its
# thickness or height (every variant had its underside at y=360, which is what
# he hits). A floor whose TOP is just below the entry point instead catches him
# on the way down and never blocks the way up: y in 366..384 measured 45/45.
# Below y~390 he passes it and falls back to screen 0 -- 0/45.
CATCH_FLOOR = {"x": 0, "y": 376, "w": SCREEN_W, "h": 40, "type": "land"}


def _walls():
    return [{"x": 0, "y": 0, "w": WALL_W, "h": SCREEN_H, "type": "land"},
            {"x": SCREEN_W - WALL_W, "y": 0, "w": WALL_W, "h": SCREEN_H,
             "type": "land"}]


@dataclass
class GenConfig:
    """Ranges the generator samples from. Defaults sit well inside the king's
    measured reach (max-charge up-jump rises ~150px; a max-charge diagonal
    carries ~200px sideways), so a sampled ladder is reachable by construction
    and `prove_ladder` almost always confirms it."""
    n_ledges: tuple = (3, 5)        # ledges above the floor (inclusive range)
    ledge_w: tuple = (60, 140)      # ledge width in px
    ledge_h: int = 14
    dy: tuple = (56, 92)            # vertical gap between consecutive ledges
    stagger: tuple = (-40, 110)     # near-edge offset of the next ledge: negative
                                    # = overlap, positive = a gap to clear. The
                                    # lower bound must stay well short of the
                                    # ledge width so the rung below always keeps
                                    # an EXPOSED launch strip -- a rung roofed
                                    # over by the next one is a dead end (the
                                    # king bonks his head instead of rising).
    min_exposed: int = 34           # px of clear sky the rung below must keep
    top_y: tuple = (48, 88)         # y of the highest ledge; a full-charge
                                    # up-jump lifts the king ~150 px, so the
                                    # last rung must sit inside that of the
                                    # screen top.
    ice_p: float = 0.0              # probability a ledge is ice
    snow_p: float = 0.0             # probability a ledge is snow
    wind: bool = False              # per-screen wind switch

    @staticmethod
    def from_args(args):
        c = GenConfig()
        for f in ("ice_p", "snow_p", "wind"):
            v = getattr(args, f, None)
            if v is not None:
                setattr(c, f, v)
        return c


def _ledge_type(rng, cfg):
    r = rng.random()
    if r < cfg.ice_p:
        return "ice"
    if r < cfg.ice_p + cfg.snow_p:
        return "snow"
    return "land"


def gen_screen(rng, cfg=None):
    """One random climbable screen: floor + a ladder of ledges bottom -> top."""
    cfg = cfg or GenConfig()
    plats = [{"x": 0, "y": FLOOR_Y, "w": SCREEN_W, "h": FLOOR_H, "type": "land"}]
    plats += _walls()

    n = int(rng.integers(cfg.n_ledges[0], cfg.n_ledges[1] + 1))
    # Spread the rungs evenly between the floor and the top rung, then jitter
    # each one -- this keeps the ladder reachable whatever n is (a pure bottom-up
    # dy walk would either overshoot the ceiling or stop far short of it).
    top_y = float(rng.uniform(*cfg.top_y))
    ys = np.linspace(FLOOR_Y - 12, top_y, n + 1)[1:]
    ys = ys + rng.uniform(-6, 6, size=n)
    ys = np.sort(ys)[::-1]                     # bottom (large y) -> top

    # enforce the vertical gap budget after jitter
    prev = float(FLOOR_Y)
    for i in range(n):
        gap = prev - ys[i]
        if gap > cfg.dy[1]:
            ys[i] = prev - cfg.dy[1]
        elif gap < cfg.dy[0]:
            ys[i] = prev - cfg.dy[0]
        prev = float(ys[i])

    # Horizontal placement is STAGGERED, not centre-jittered: each rung is put
    # beside the one below (to its left or right, with a bounded overlap or gap)
    # so the rung below always keeps open sky to launch from.
    cl, cr = float(WALL_W), float(SCREEN_W - WALL_W)   # the floor spans everything
    for y in ys:
        w = float(rng.uniform(*cfg.ledge_w))
        for _ in range(8):                     # retry the side/offset draw
            side = 1.0 if rng.random() < 0.5 else -1.0
            off = float(rng.uniform(*cfg.stagger))
            nl = (cr + off) if side > 0 else (cl - off - w)
            nl = float(np.clip(nl, WALL_W + 2, SCREEN_W - WALL_W - w - 2))
            exposed = (nl - cl) if side > 0 else (cr - (nl + w))
            if exposed >= cfg.min_exposed:
                break
        cl, cr = nl, nl + w
        plats.append({"x": int(round(nl)), "y": int(round(y)),
                      "w": int(round(w)), "h": cfg.ledge_h,
                      "type": _ledge_type(rng, cfg)})
    return plats


def gen_world(rng, cfg=None, name=None):
    """A 2-screen world: random screen 0 + a catch floor to climb into."""
    cfg = cfg or GenConfig()
    return {
        "format": CustomWorld.FORMAT, "version": 1,
        "name": name or "gen",
        "levels": [
            {"wind": bool(cfg.wind), "platforms": gen_screen(rng, cfg)},
            {"wind": False, "platforms": [dict(CATCH_FLOOR)] + _walls()},
        ],
    }


# ------------------------------------------------------------------ proving
def ledge_tops(world, level=0):
    """Rungs of screen `level`, bottom -> top, as (x_left, x_right, y_top)."""
    ps = [p for p in world["levels"][level]["platforms"] if p["h"] < SCREEN_H // 2]
    ps.sort(key=lambda p: -p["y"])            # walls are not rungs
    return [(p["x"], p["x"] + p["w"], p["y"]) for p in ps]


def prove_ladder(env, world, xs_per_rung=5, verbose=False):
    """Prove IN THE ENGINE that the world is solvable rung by rung.

    For each rung, the king is teleported onto it at a few x positions and every
    macro-action in the table is tried; the rung is 'linked' if some action from
    some x lands him on a HIGHER rung (or off the top of the screen). A full
    chain of links is a constructive solution path.

    Returns (ok, links) where links[i] is the winning (x, action_index, result)
    for rung i, or None if that rung is a dead end."""
    CustomWorld.apply_world(env, world)
    rungs = ledge_tops(world, 0)
    kh = env.king.rect_height
    links = []
    for i, (xl, xr, y) in enumerate(rungs):
        # sample stand-on-rung x's: left third, middle, right third
        span = max(1, xr - xl - env.king.rect_width)
        cand_x = [int(xl + span * f) for f in np.linspace(0.04, 0.96, xs_per_rung)]
        found = None
        for x0 in cand_x:
            slvl, sx, sy = env.teleport(0, x0, y - kh - 2)
            if slvl != 0 or abs(sy - (y - kh)) > 6:
                continue                       # did not settle on this rung
            for a in range(env.num_actions):
                env.reset(level=0, rect_x=sx, rect_y=sy)
                _, _, _term, _tr, info = env.step(a)
                if info["level"] >= 1:
                    found = (sx, a, "exit")
                    break
                if info["level"] == 0 and info["y"] + kh <= y - 20:
                    found = (sx, a, (int(info["x"]), int(info["y"])))
                    break
            if found:
                break
        links.append(found)
        if verbose:
            print(f"   rung {i} y={y:3d} x=[{xl},{xr}] -> "
                  f"{'OK a=' + str(found[1]) if found else 'DEAD END'}")
        if not found:
            return False, links
    return True, links


def probe_seam(env, world, k, xs_per_rung=3):
    """How many actions carry the king from screen `k` into screen `k+1`?

    The seam is where hand-built multi-screen worlds break, and the reason is
    geometric: `King._check_level` drops a crossing king into the next screen at
    rect_y ~ 371..379, BELOW its bottom edge. A ledge above him there is a
    ceiling he bonks on; a ledge away from his column he simply rises past and
    falls back through the bottom. He lands only if the exit jump carries him
    SIDEWAYS onto a ledge -- which is why a seam can have exactly one solving
    action out of the whole table while the screen below is easy.

    Returns (crossing_actions, landings): the action indices that cross, and
    where they left him. An empty list means the seam is impassable as drawn.
    """
    rungs = ledge_tops(world, k)
    if len(rungs) < 2:
        return [], []
    xl, xr, y = rungs[-1]                    # the screen's top rung
    kh = env.king.rect_height
    span = max(1, xr - xl - env.king.rect_width)
    hits, land = [], []
    for f in np.linspace(0.1, 0.9, xs_per_rung):
        x0 = int(xl + span * f)
        lvl, sx, sy = env.teleport(k, x0, y - kh - 2)
        if lvl != k:
            continue
        for a in range(env.num_actions):
            env.reset(level=k, rect_x=sx, rect_y=sy)
            _, _, _, _, info = env.step(a)
            if info["level"] > k:
                hits.append(a)
                land.append((int(info["level"]), int(info["x"]), int(info["y"])))
    return hits, land


def _make_prover_env(**kw):
    from JK_Env import JumpKingEnv
    kw.setdefault("max_steps", 10_000)
    kw.setdefault("goal_level", 1)
    kw.setdefault("max_settle_frames", 1000)
    return JumpKingEnv(**kw)


# ------------------------------------------------ endless stream for training
def attach_generator(env, seed=0, cfg=None, tag="", **kw):
    """Make `env` draw a BRAND-NEW random world on every reset().

    The pool variant can only show the policy as many screens as the pool holds,
    and that ceiling is visible: trained on 800 proved screens the policy reached
    0.83 on them and 0.38 on held-out ones, i.e. it had started memorising the
    pool. An endless stream removes the ceiling -- no screen is ever seen twice,
    so there is nothing to memorise. The price is that ~15% of the screens are
    unsolvable (they are not put through the prover, which costs ~6 s each), so
    training success saturates near 0.85 even for a perfect policy. The reported
    number always comes from `GenEval` on a PROVED held-out pool, so that price
    is paid in training noise only."""
    cfg = cfg or GenConfig()
    return _attach(env, lambda rng: gen_world(rng, cfg, name=f"{tag}gen"),
                   seed=seed, **kw)


def load_pool(world_dir):
    """Every proved world in a directory, sorted so pools are reproducible."""
    files = sorted(f for f in os.listdir(world_dir) if f.endswith(".json"))
    if not files:
        raise SystemExit(f"no world files in {world_dir}")
    return [CustomWorld.load_world(os.path.join(world_dir, f)) for f in files]


MAX_STAGE = 8                    # stage k = start k rungs below the top; the
                                 # last stages fall through to a floor start


def _stage_start(rng, world, kh, stage):
    """Where a stage-`stage` episode begins on this screen.

    Stage 0 is the top rung (one proved action from the exit), stage 1 the rung
    below it, and any stage at or past the screen's rung count is the floor --
    the real task. Screens have 3-5 rungs, so the same stage number means
    "equally far from the exit" across screens of different heights."""
    rungs = ledge_tops(world, 0)[1:]           # skip the floor
    if stage >= len(rungs):
        return int(rng.integers(24, SCREEN_W - 24)), FLOOR_Y - 60
    xl, xr, y = rungs[len(rungs) - 1 - int(stage)]
    lo, hi = xl + 6, max(xl + 7, xr - 26)
    return int(rng.integers(lo, hi)), int(y - kh - 2)


def attach_pool(env, world_dir, seed=0, **kw):
    """`_attach` drawing from a directory of PROVED worlds."""
    pool = load_pool(world_dir)
    env.gen_pool = pool
    return _attach(env, lambda rng: pool[int(rng.integers(0, len(pool)))],
                   seed=seed, **kw)


def _attach(env, draw, seed=0, curriculum=True, cur_target=0.65,
            cur_window=150, mix_below=0.3):
    """Make `env` draw a random world from a PROVED pool on every reset().

    `draw(rng)` returns the world for one episode -- from a proved pool, or
    generated fresh. Everything else is the shared training protocol.

    The reverse curriculum is ADAPTIVE, not scheduled. A blind schedule was
    measured to fail here: opening the stages over a fixed 5000 episodes drove
    training success from 0.56 down to 0.25 and left bottom-start success at
    1/120, because the stages widened whether or not the policy had learnt the
    previous one. Instead the wrapper watches the outcome of each episode (the
    level the king ended the previous one on) and only steps to the next stage
    once the last `cur_window` episodes clear `cur_target`. `mix_below` of the
    episodes are drawn from an easier already-cleared stage so the skill is not
    forgotten. Stage advances print one line, so the run log shows the ladder.

    With curriculum=False every episode simply starts on the floor."""
    base_reset = env.reset
    state = {"n": 0, "seed": int(seed), "stage": 0, "hist": [], "advances": 0}
    kh = env.king.rect_height

    def reset(level=None, rect_x=None, rect_y=None, vx=None, vy=None):
        if level is None:
            rng = np.random.default_rng((state["seed"], state["n"]))
            if curriculum and state["n"] > 0:
                # the level the king ENDED the previous episode on, read before
                # the new world overwrites it
                state["hist"].append(1.0 if env.levels.current_level >= 1 else 0.0)
                if len(state["hist"]) >= cur_window:
                    rate = float(np.mean(state["hist"][-cur_window:]))
                    if rate >= cur_target and state["stage"] < MAX_STAGE:
                        state["stage"] += 1
                        state["advances"] += 1
                        state["hist"].clear()
                        print(f"[worldgen] stage -> {state['stage']} "
                              f"(rate {rate:.2f} over {cur_window} eps, "
                              f"seed {state['seed']})", flush=True)
                    elif len(state["hist"]) > 4 * cur_window:
                        del state["hist"][:cur_window]
            state["n"] += 1
            world = draw(rng)
            CustomWorld.apply_world(env, world)
            env.gen_world = world
            st = state["stage"] if curriculum else 99
            if curriculum and st and rng.random() < mix_below:
                st = int(rng.integers(0, st))          # revisit an easier stage
            spot = _stage_start(rng, world, kh, st)
            return base_reset(level=0, rect_x=spot[0], rect_y=spot[1])
        return base_reset(level, rect_x, rect_y, vx, vy)

    env.reset = reset
    env.gen_state = state
    return env


# ------------------------------------------------------------------- dataset
def build_set(out_dir, n, seed, cfg=None, verify=True, verbose=True, tag=None):
    """Write `n` generated worlds to `out_dir` (optionally engine-proved).

    File names carry the seed, so several shards can fill the same directory
    without colliding -- that is how --jobs parallelises the proving, which is
    the expensive part (~6 s per candidate, one full engine per process)."""
    cfg = cfg or GenConfig()
    os.makedirs(out_dir, exist_ok=True)
    tag = f"s{seed}" if tag is None else tag
    env = _make_prover_env() if verify else None
    written, tried = 0, 0
    try:
        while written < n:
            rng = np.random.default_rng((seed, tried))
            world = gen_world(rng, cfg, name=f"{tag}_{written:04d}")
            tried += 1
            if verify:
                ok, _ = prove_ladder(env, world)
                if not ok:
                    if verbose:
                        print(f"  candidate {tried}: unsolvable -> resampled")
                    continue
            world["gen"] = {"seed": int(seed), "index": int(tried - 1),
                            "verified": bool(verify), "cfg": asdict(cfg)}
            path = os.path.join(out_dir, f"{tag}_{written:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(world, f, indent=1)
            written += 1
            if verbose:
                print(f"  wrote {path} "
                      f"({len(ledge_tops(world, 0)) - 1} rungs)")
    finally:
        if env is not None:
            env.close()
    if verbose:
        print(f"{written} worlds -> {out_dir} "
              f"({tried} sampled, {tried - written} rejected)")
    return written, tried


def _shard(job):
    out_dir, n, seed, cfg_kw, verify = job
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    return build_set(out_dir, n, seed, GenConfig(**cfg_kw), verify=verify,
                     verbose=False)


def build_set_parallel(out_dir, n, seed, cfg=None, verify=True, jobs=4):
    """`build_set` split over `jobs` processes, each with its own seed shard."""
    import multiprocessing as mp
    cfg = cfg or GenConfig()
    per = [n // jobs + (1 if i < n % jobs else 0) for i in range(jobs)]
    tasks = [(out_dir, per[i], seed + i, asdict(cfg), verify)
             for i in range(jobs) if per[i]]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        outs = pool.map(_shard, tasks)
    written = sum(w for w, _ in outs)
    tried = sum(t for _, t in outs)
    print(f"{written} worlds -> {out_dir} "
          f"({tried} sampled, {tried - written} rejected, {len(tasks)} jobs)")
    return written, tried


def _describe(world):
    for i, lvl in enumerate(world["levels"]):
        print(f"  screen {i}{'  [wind]' if lvl['wind'] else ''}")
        for p in sorted(lvl["platforms"], key=lambda q: -q["y"]):
            print(f"    x={p['x']:3d}..{p['x']+p['w']:3d} y={p['y']:3d} "
                  f"w={p['w']:3d} {p['type']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="describe one world")
    ap.add_argument("--selftest", action="store_true",
                    help="engine-prove --n random worlds and report the rate")
    ap.add_argument("--out", default=None, help="directory for a world set")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes to prove candidates in parallel")
    ap.add_argument("--ice-p", type=float, default=None)
    ap.add_argument("--snow-p", type=float, default=None)
    ap.add_argument("--wind", action="store_true", default=None)
    args = ap.parse_args()
    cfg = GenConfig.from_args(args)

    if args.preview:
        w = gen_world(np.random.default_rng(args.seed), cfg)
        print(f"world '{w['name']}' seed={args.seed}")
        _describe(w)
        return
    if args.selftest:
        env = _make_prover_env()
        ok_n = 0
        try:
            for i in range(args.n):
                w = gen_world(np.random.default_rng((args.seed, i)), cfg)
                ok, links = prove_ladder(env, w, verbose=True)
                ok_n += ok
                print(f"world {i}: {'SOLVABLE' if ok else 'unsolvable'} "
                      f"({sum(l is not None for l in links)}/{len(links)} rungs linked)")
        finally:
            env.close()
        print(f"\nengine-proved solvable: {ok_n}/{args.n} "
              f"= {100*ok_n/max(1,args.n):.0f}%")
        return
    if args.out:
        if args.jobs > 1:
            build_set_parallel(args.out, args.n, args.seed, cfg,
                               verify=not args.no_verify, jobs=args.jobs)
        else:
            build_set(args.out, args.n, args.seed, cfg,
                      verify=not args.no_verify)
        return
    ap.print_help()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    main()
