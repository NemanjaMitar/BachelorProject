#!/usr/bin/env python
"""
Validate starts/start_states.json by EXHAUSTIVE short-horizon reachability.

For every captured checkpoint this teleports the King there (through the
canonical JK_Env.teleport, so grounding is identical to training), then does a
breadth-first search over macro-actions: depth 1 tries every action, depth 2
tries every action from every resulting state, and so on. A state scores by
how few actions it needs to clear its level:

    score = (depth - d + 1) / depth      d = actions needed, 0 => dead end

Usage:
    python Validate_Start_States.py starts/start_states.json --depth 2
    python Validate_Start_States.py starts/start_states.json --level 5      # one level only
    python Validate_Start_States.py starts/start_states.json --depth 3 --min-score 0.0

Writes a list of {level,x,y,score} dicts (the format JK_Env._load_start_states
already understands) keeping states with score > --min-score. Output file:
<path>_scored.json, or starts/starts_L<n>.json when --level is given -- ready to pass
straight to Train.py --start-states.
"""

import os
import json
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from JK_Env import JumpKingEnv


def _load(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "states" in data:
        data = data["states"]

    pool = []
    if isinstance(data, dict):
        for lvl, pts in data.items():
            try:
                lvl_i = int(lvl)
            except (ValueError, TypeError):
                continue
            for pt in (pts or []):
                if isinstance(pt, dict):
                    pool.append({"level": int(pt.get("level", lvl_i)),
                                 "x": int(pt["x"]), "y": int(pt["y"])})
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pool.append({"level": lvl_i, "x": int(pt[0]), "y": int(pt[1])})
    elif isinstance(data, list):
        for pt in data:
            if isinstance(pt, dict):
                pool.append({"level": int(pt["level"]),
                             "x": int(pt["x"]), "y": int(pt["y"])})
    return pool


def reach_depth(env, st, max_depth, grid=8, max_nodes=120):
    """BFS over macro-actions from a checkpoint.

    Returns (d, settled): d = fewest actions to clear the level (0 = none
    found within max_depth), settled = where the checkpoint actually grounded.
    States are deduplicated on a coarse pixel grid to keep the tree small."""
    settled = env.teleport(st["level"], st["x"], st["y"])
    start_lvl = settled[0]

    frontier = [settled]
    seen = {(settled[0], settled[1] // grid, settled[2] // grid)}

    for depth in range(1, max_depth + 1):
        nxt = []
        for node in frontier:
            for a in range(env.num_actions):
                env.reset(level=node[0], rect_x=node[1], rect_y=node[2])
                _, _, _term, _trunc, info = env.step(a)
                if info["level"] > start_lvl:
                    return depth, settled
                if info["level"] < start_lvl:
                    continue          # fell off the level: dead branch
                key = (info["level"], info["x"] // grid, info["y"] // grid)
                if key not in seen and len(seen) < max_nodes:
                    seen.add(key)
                    nxt.append((info["level"], int(info["x"]), int(info["y"])))
        frontier = nxt
        if not frontier:
            break
    return 0, settled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--level", type=int, default=None,
                    help="only validate states captured on this level and write "
                         "them to starts/starts_L<n>.json (per-level training pool)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--fine-walk-frames", type=int, default=0,
                    help="include the micro-walk action pair in the BFS "
                         "(match what training uses)")
    ap.add_argument("--extra-charges", type=str, default="",
                    help="comma list of appended jump charges (match training)")
    ap.add_argument("--min-score", type=float, default=-1.0,
                    help="drop states with score <= this. Default keeps "
                         "EVERYTHING, including score-0 states: a 0 only means "
                         "'exit is further than --depth actions', which is "
                         "exactly what the harder curriculum stages look like. "
                         "Use 0.0 to keep only provably exit-reaching states.")
    args = ap.parse_args()

    pool = _load(args.path)
    if args.level is not None:
        pool = [st for st in pool if st["level"] == args.level]
        if not pool:
            print(f"no captured states for level {args.level} in {args.path}")
            return
    extra = (tuple(int(c) for c in args.extra_charges.split(","))
             if args.extra_charges else ())
    env = JumpKingEnv(max_steps=10_000,
                      fine_walk_frames=args.fine_walk_frames,
                      extra_charges=extra)

    scored = []
    for i, st in enumerate(pool):
        d, settled = reach_depth(env, st, args.depth)
        score = round((args.depth - d + 1) / args.depth, 4) if d > 0 else 0.0
        # Store the SETTLED position: it is where training will actually
        # start from, and re-teleporting there is stable (already grounded).
        rec = {"level": settled[0], "x": settled[1], "y": settled[2],
               "score": score}
        scored.append(rec)

        if settled[0] != st["level"]:
            tag = (f"SETTLED OFF-LEVEL {st['level']} -> {settled[0]} "
                   f"(bad capture point)")
        elif d > 0:
            tag = f"reaches exit in {d} action(s)"
        else:
            tag = "DEAD END (no exit within depth)"
        print(f"[{i + 1:4d}/{len(pool)}] lvl {st['level']:2d} "
              f"x={st['x']:3d} y={st['y']:3d}  score={score:.2f}  {tag}")
    env.close()

    kept = [s for s in scored if s["score"] > args.min_score]
    if args.out:
        out = args.out
    elif args.level is not None:
        out = f"starts/starts_L{args.level}.json"
    else:
        out = os.path.splitext(args.path)[0] + "_scored.json"
    with open(out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"\nwrote {len(kept)}/{len(scored)} states to {out} "
          f"(dropped {len(scored) - len(kept)} dead ends)")


if __name__ == "__main__":
    main()
