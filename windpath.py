#!/usr/bin/env python
"""Deterministic wind-jump path search from a start state to the next level.

wind_jump actions are atomic (wait-for-bucket-then-jump) so their landing from a
settled spot is deterministic -> a BFS over them is a reliable route finder on
windy levels (unlike a raw-action search, which the stochastic gust breaks).
Prints the shortest action sequence that reaches goal_level.

    python windpath.py --level 26 --x 184 --y 281 --goal-level 27
"""
import os, argparse, collections
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from JK_Env import JumpKingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)
    ap.add_argument("--goal-level", type=int, required=True)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--fine-walk-frames", type=int, default=3)
    ap.add_argument("--extra-charges", type=str, default="22,24,28,30")
    args = ap.parse_args()
    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()

    e = JumpKingEnv(max_steps=300, goal_level=args.goal_level,
                    fine_walk_frames=args.fine_walk_frames, extra_charges=extra,
                    wind_jump=(0, 4), wind_obs=True)
    # expand action set: wind_jumps (deterministic) + fine walks (reposition)
    idxs = [i for i, a in enumerate(e.actions)
            if a[0] == "jump_wind" or (a[0] == "walk" and a[2] == args.fine_walk_frames)]

    start = (args.level, args.x, args.y)
    # BFS; a node is a settled (level,x,y). dedup on coarse cell.
    q = collections.deque([(start, [])])
    seen = {(start[0], start[1] // 6, start[2] // 6)}
    while q:
        (lvl, x, y), path = q.popleft()
        if len(path) >= args.max_depth:
            continue
        for i in idxs:
            e.reset(level=lvl, rect_x=x, rect_y=y)
            if e.levels.current_level != lvl or int(e.king.rect_x) != x:
                # couldn't reproduce this node cleanly; skip expanding from a
                # mismatched settle (only expand faithfully-reachable nodes)
                if e.levels.current_level != lvl:
                    continue
            _, _, term, trunc, info = e.step(i)
            nl, nx, ny = info["level"], int(info["x"]), int(info["y"])
            if nl >= args.goal_level:
                seq = path + [e.actions[i]]
                print(f"FOUND ({len(seq)} actions) {start} -> lvl {nl} ({nx},{ny}):")
                for k, a in enumerate(seq):
                    print(f"   {k+1}. {a}")
                e.close(); return
            if nl < args.level:      # fell backward off this level; prune
                continue
            key = (nl, nx // 6, ny // 6)
            if key in seen:
                continue
            seen.add(key)
            q.append(((nl, nx, ny), path + [e.actions[i]]))
    print(f"NO path found within depth {args.max_depth} "
          f"({len(seen)} states explored)")
    e.close()


if __name__ == "__main__":
    main()
