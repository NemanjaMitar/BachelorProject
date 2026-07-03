#!/usr/bin/env python
"""
Exhaustive reachability search over ONE level's macro-action graph.

Because the physics is deterministic and the King only acts when grounded,
each level is a finite graph: nodes are settled (x, y) spots, edges are
macro-actions. This searches that graph breadth-first and answers, with proof:

  * is the next level reachable from the given start states AT ALL
    with a given action set (charges / fine walks)?
  * by which exact action sequence(s)?
  * which states are physics traps (king never settles / level seam flicker)?

Usage:
    python LevelGraph.py --level 9 --starts starts_L9.json --fine-walk-frames 3
    python LevelGraph.py --level 9 --starts starts_L9.json --fine-walk-frames 3 ^
        --charges 4,8,12,16,20,22,24,26,32          # test an EXTENDED pool
    python LevelGraph.py --level 9 --seed 375 305   # single seed instead of a file

    --emit-starts FILE   merge the states along the winning path(s) into FILE
                         as curriculum start states with graded scores
                         (closer to the exit = higher score).

Reading the output:
  * "PATH (n actions): ..." -- a proven route; the level is solvable.
  * "no path found" + a small reachable set -- the action pool cannot cross
    some gap; rerun with --charges/--fine-walk-frames variants to find the
    minimal extension that makes the level solvable.
  * TRAP nodes -- spots where an action left the king never-settled
    (bounce pockets, level-seam flicker). Avoid capturing starts there.
"""

import os
import json
import time
import argparse
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from JK_Env import JumpKingEnv


def load_seeds(args):
    seeds = []
    if args.seed:
        seeds.append({"level": args.level, "x": args.seed[0], "y": args.seed[1]})
    if args.starts:
        with open(args.starts) as f:
            data = json.load(f)
        for s in data:
            if isinstance(s, dict) and int(s.get("level", -1)) == args.level:
                seeds.append({"level": args.level,
                              "x": int(s["x"]), "y": int(s["y"])})
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--starts", default=None,
                    help="start-state json (list format); seeds = this level's states")
    ap.add_argument("--seed", type=int, nargs=2, default=None,
                    metavar=("X", "Y"), help="explicit extra seed")
    ap.add_argument("--charges", default=None,
                    help="comma list overriding the jump charge set, "
                         "e.g. 4,8,12,16,20,22,24,26,32")
    ap.add_argument("--fine-walk-frames", type=int, default=0)
    ap.add_argument("--extra-charges", default=None,
                    help="comma list of jump charges APPENDED to the table "
                         "(matches Train.py --extra-charges)")
    ap.add_argument("--grid", type=int, default=4,
                    help="dedup raster in px (smaller = finer but slower)")
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=1500)
    ap.add_argument("--max-paths", type=int, default=5,
                    help="stop after this many distinct goal paths")
    ap.add_argument("--emit-starts", default=None,
                    help="merge winning-path states into this pool file")
    args = ap.parse_args()

    kw = {"max_steps": 10_000,
          "fine_walk_frames": args.fine_walk_frames,
          "max_settle_frames": 1000}
    if args.charges:
        kw["charges"] = tuple(int(c) for c in args.charges.split(","))
    if args.extra_charges:
        kw["extra_charges"] = tuple(int(c) for c in args.extra_charges.split(","))
    env = JumpKingEnv(**kw)
    lvl, goal = args.level, args.level + 1
    print(f"level {lvl} -> {goal} | {env.num_actions} actions "
          f"charges={env.charges} fine_walk={args.fine_walk_frames} "
          f"grid={args.grid}px")

    seeds = load_seeds(args)
    if not seeds:
        raise SystemExit("no seeds: give --starts and/or --seed")

    def key(x, y):
        return (int(x) // args.grid, int(y) // args.grid)

    nodes = {}       # key -> (x, y, path tuple of action idxs, origin seed xy)
    q = deque()
    for s in seeds:
        slvl, sx, sy = env.teleport(lvl, s["x"], s["y"])
        if slvl != lvl:
            print(f"  seed ({s['x']},{s['y']}) settles OFF-LEVEL -> skipped")
            continue
        k = key(sx, sy)
        if k not in nodes:
            nodes[k] = (sx, sy, (), (sx, sy))
            q.append(k)
    print(f"{len(nodes)} seed nodes")

    paths, traps = [], []
    steps = 0
    t0 = time.time()

    while q and len(nodes) < args.max_nodes and len(paths) < args.max_paths:
        k = q.popleft()
        x, y, path, origin = nodes[k]
        if len(path) >= args.max_depth:
            continue
        for a in range(env.num_actions):
            env.reset(level=lvl, rect_x=x, rect_y=y)
            _, _, term, _tr, info = env.step(a)
            steps += 1
            if steps % 1000 == 0:
                print(f"  ... {steps} steps, {len(nodes)} nodes, "
                      f"{len(paths)} paths, {time.time()-t0:.0f}s")
            flvl, fx, fy = info["level"], int(info["x"]), int(info["y"])
            if flvl >= goal:
                p = path + (a,)
                paths.append((p, origin, (fx, fy)))
                print(f"PATH ({len(p)} actions) from seed {origin} "
                      f"-> lvl {flvl} ({fx},{fy}):")
                print(f"    actions: {[str(env.actions[i]) for i in p]}")
                break
            if flvl < lvl:
                continue                      # fell off: dead branch
            if term and not info.get("success") and not env.move_available():
                traps.append((x, y, a, fx, fy))
                continue                      # never settled: physics trap
            nk = key(fx, fy)
            if nk not in nodes:
                nodes[nk] = (fx, fy, path + (a,), origin)
                q.append(nk)

    dt = time.time() - t0
    print(f"\ndone: {steps} macro-steps in {dt:.0f}s | {len(nodes)} reachable "
          f"spots | {len(paths)} goal paths | {len(traps)} trap edges")

    if traps:
        seen = set()
        print("trap edges (king never settled):")
        for (x, y, a, fx, fy) in traps[:12]:
            t = (key(x, y), a)
            if t in seen:
                continue
            seen.add(t)
            print(f"  from ({x},{y}) act {a:2d} {str(env.actions[a]):22s} "
                  f"-> stuck near ({fx},{fy})")

    if not paths:
        print("NO PATH FOUND with this action set.")
        print("Try: --fine-walk-frames 3, extra --charges, larger --max-depth.")
    elif args.emit_starts:
        # Replay the shortest path FROM ITS ORIGIN SEED and bank each
        # intermediate settled state as a curriculum start, scored by
        # progress along the path.
        best = min(paths, key=lambda t: len(t[0]))
        p, seed = best[0], best[1]
        states, cx, cy = [], seed[0], seed[1]
        for i, a in enumerate(p):
            env.reset(level=lvl, rect_x=cx, rect_y=cy)
            _, _, _t, _tr, info = env.step(a)
            cx, cy = int(info["x"]), int(info["y"])
            if info["level"] == lvl:          # intermediate spot on the level
                states.append({"level": lvl, "x": cx, "y": cy,
                               "score": round((i + 1) / len(p), 3)})
        try:
            with open(args.emit_starts) as f:
                pool = json.load(f)
            assert isinstance(pool, list)
        except (FileNotFoundError, json.JSONDecodeError, AssertionError):
            pool = []
        index = {(s["level"], s["x"] // 8, s["y"] // 8): s
                 for s in pool if isinstance(s, dict)}
        added = 0
        for s in states:
            kk = (s["level"], s["x"] // 8, s["y"] // 8)
            if kk in index:
                # existing entry lies on the proven path: upgrade its score
                index[kk]["score"] = max(float(index[kk].get("score", 0.0)),
                                         s["score"])
            else:
                pool.append(s)
                index[kk] = s
                added += 1

        # Depth-1 scoring pass over the whole pool: any state with a
        # one-action clear is the top of the reverse curriculum (this is what
        # grades the pool when the winning path is too short to emit from).
        rescored = 0
        for s in pool:
            if int(s.get("level", -1)) != lvl:
                continue
            slvl, sx, sy = env.teleport(lvl, s["x"], s["y"])
            if slvl != lvl:
                continue
            for a in range(env.num_actions):
                env.reset(level=lvl, rect_x=sx, rect_y=sy)
                _, _, _t, _tr, info = env.step(a)
                if info["level"] >= goal:
                    if float(s.get("score", 0.0)) < 1.0:
                        s["score"] = 1.0
                        rescored += 1
                    break

        with open(args.emit_starts, "w") as f:
            json.dump(pool, f, indent=2)
        print(f"emitted {added} path states into {args.emit_starts} "
              f"({rescored} pool states re-scored to 1.0 by depth-1 check)")

    env.close()


if __name__ == "__main__":
    main()
