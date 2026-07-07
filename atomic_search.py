#!/usr/bin/env python
"""Phase-robust ATOMIC route search for windy levels.

windpath.py teleport-resets between steps, so it returns routes that don't
reproduce under atomic wind_combo execution (and can be phase-dependent). This
searches the way a combo actually runs: sequences are executed atomically from
the entry (no reset between steps), and a transition is only kept if it lands in
the SAME cell across several wind phases -- so the resulting route is guaranteed
deterministic and safe to bake into a wind_combo.

Beam search (keeps the most-promising frontier by altitude) with re-execution;
no env snapshot (pygame surfaces aren't deep-copyable). Optional --walk-norm
prepends a walk-to-x so the launch is identical regardless of entry-x drift.

    python atomic_search.py --level 31 --x 9 --y 305 --goal-level 32 \
        --walk-norm 20 --max-depth 7 --beam 10 --trials 4
"""
import os, argparse, heapq
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from JK_Env import JumpKingEnv


def run_path(e, level, x, y, walk_norm, walk_idx, path_idxs):
    """Reset to entry and replay walk-norm + path atomically. Returns the final
    (level, x, y) or None if it left the start level partway (invalid path)."""
    e.reset(level=level, rect_x=x, rect_y=y)
    if walk_norm is not None:
        # walk (many fine steps) to the normalize x before the wind sequence
        for _ in range(40):
            if abs(e.king.rect_x - walk_norm) <= 2 or e.levels.current_level != level:
                break
            e.step(walk_idx if e.king.rect_x < walk_norm else walk_idx - 1)
        if e.levels.current_level != level:
            return None
    for i in path_idxs:
        _, _, _, _, info = e.step(i)
    # include the king's residual SPEED: on ice move_available() fires mid-slide,
    # so the same position with different momentum leads to different next jumps.
    win = bool(e.levels.ending)   # reached the babe -> game won
    if not path_idxs:
        return (e.levels.current_level, int(e.king.rect_x),
                int(e.king.rect_y), float(e.king.speed), win)
    return (info["level"], int(info["x"]), int(info["y"]),
            float(e.king.speed), win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)
    ap.add_argument("--goal-level", type=int, required=True)
    ap.add_argument("--walk-norm", type=int, default=None,
                    help="prepend walk-to-this-x to normalize the launch")
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--trials", type=int, default=4,
                    help="wind phases a transition must agree across to be kept")
    ap.add_argument("--fine-walk-frames", type=int, default=3)
    ap.add_argument("--extra-charges", type=str, default="22,24,28,30")
    ap.add_argument("--regular", action="store_true",
                    help="search plain jumps (windless/ice levels); forces "
                         "trials=1 since physics is deterministic")
    args = ap.parse_args()
    if args.regular:
        args.trials = 1
    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()

    e = JumpKingEnv(max_steps=900, goal_level=args.goal_level,
                    fine_walk_frames=args.fine_walk_frames, extra_charges=extra,
                    wind_jump=(0, 4), wind_obs=True)
    kind = "jump" if args.regular else "jump_wind"
    wj = [i for i, a in enumerate(e.actions) if a[0] == kind]
    walk_idx = None
    if args.walk_norm is not None:
        # a one-off approach_jump-like walk via a temporary env would be cleaner,
        # but we just reuse a fine walk loop: encode walk as repeated fine-walk.
        walk_idx = next(i for i, a in enumerate(e.actions)
                        if a == ("walk", "right", args.fine_walk_frames))

    entry = (args.level, args.x, args.y)

    def robust_step(path_idxs, a):
        """Return the unique (level,x,y) reached by path+a if it's identical
        across `trials` phases AND doesn't fall below the start level; else None.
        Early-prune: if the first phase already fails/falls, skip the rest."""
        first = run_path(e, args.level, args.x, args.y, args.walk_norm,
                         walk_idx, path_idxs + [a])
        if first is None or first[0] < args.level:
            return None
        if first[4]:                     # reached the babe -> WIN
            return ("WIN",)
        # (level, cx, cy, x, y, speed_bucket)
        seen = (first[0], first[1] // 6, first[2] // 6, first[1], first[2],
                round(first[3]))
        for _ in range(args.trials - 1):
            res = run_path(e, args.level, args.x, args.y, args.walk_norm,
                           walk_idx, path_idxs + [a])
            if res is None or res[0] < args.level:
                return None
            if (seen[0], seen[1], seen[2]) != (res[0], res[1] // 6, res[2] // 6):
                return None      # phase-dependent -> reject
        return seen  # (level, cx, cy, x, y, speed_bucket)

    # beam search; frontier entries: (path_idxs, (level,x,y))
    start_lvl = args.level
    frontier = [([], entry)]
    visited = set()
    for depth in range(args.max_depth):
        scored = []
        for path, state in frontier:
            for a in wj:
                r = robust_step(path, a)
                if r is None:
                    continue
                if r == ("WIN",):
                    seq = [e.actions[i] for i in path + [a]]
                    print(f"WIN ROUTE ({len(seq)} steps) {entry} -> reached the "
                          f"babe (game complete):")
                    for k, act in enumerate(seq):
                        print(f"   {k+1}. {act}")
                    e.close(); return
                lvl, cx, cy, xx, yy, spd = r
                if lvl >= args.goal_level:
                    seq = [e.actions[i] for i in path + [a]]
                    print(f"ROBUST ROUTE ({len(seq)} steps) {entry} -> lvl {lvl} "
                          f"({xx},{yy}), agreed across {args.trials} phases:")
                    if args.walk_norm is not None:
                        print(f"   0. ('walk', {args.walk_norm})")
                    for k, act in enumerate(seq):
                        print(f"   {k+1}. {act}")
                    e.close(); return
                key = (lvl, cx, cy, spd)   # momentum-aware: same cell, diff speed
                if key in visited:         # -> different state on ice
                    continue
                visited.add(key)
                # score: prefer higher level, then higher up (smaller y)
                scored.append((-(lvl * 1000 - yy), path + [a], (lvl, xx, yy)))
        if not scored:
            print(f"no robust progress at depth {depth+1}")
            break
        scored.sort(key=lambda t: t[0])
        frontier = [(p, s) for _, p, s in scored[:args.beam]]
        print(f"depth {depth+1}: frontier {len(frontier)} "
              f"best={frontier[0][1]}")
    else:
        print(f"NO robust route within depth {args.max_depth}")
    e.close()


if __name__ == "__main__":
    main()
