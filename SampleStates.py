#!/usr/bin/env python
"""Sample a broad distribution of valid grounded states across a level, so PPO
(warm-started from a demonstration) learns to reach the exit from ANYWHERE on
the level -- funnelling any state back onto the known-good path.

Teleport() drops the King from each grid point and settles him; we keep the
spots where he comes to rest ON this level (didn't fall through). Scored by
altitude for the reverse curriculum (near-exit = easy = high score), so training
starts near the top and works outward. Optionally merges the on-path demo states
(starts/starts_LN_demo.json) so the proven route is always represented.

    python SampleStates.py --level 30 --merge-demo
"""
import os, argparse, json
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from JK_Env import JumpKingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--x-step", type=int, default=12)
    ap.add_argument("--y-step", type=int, default=14)
    ap.add_argument("--dedup", type=int, default=14, help="px cell for dedup")
    ap.add_argument("--max-states", type=int, default=None,
                    help="subsample to ~N states spread across the level height")
    ap.add_argument("--fine-walk-frames", type=int, default=3)
    ap.add_argument("--extra-charges", type=str, default="22,24,28,30")
    ap.add_argument("--windy", action="store_true",
                    help="build with wind_obs/wind_jump (for windy levels)")
    ap.add_argument("--merge-demo", action="store_true",
                    help="also include starts/starts_L<level>_demo.json (the route)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()

    env = JumpKingEnv(max_steps=50, fine_walk_frames=args.fine_walk_frames,
                      extra_charges=extra,
                      wind_jump=(0, 4) if args.windy else (),
                      wind_obs=args.windy)
    W, H = env.screen_w, env.screen_h

    seen = {}
    for gx in range(4, W, args.x_step):
        for gy in range(4, H, args.y_step):
            env.reset(level=args.level, rect_x=gx, rect_y=gy)
            # valid = settled ON this level AND a real grounded decision point
            # (move_available) -> a genuine platform, never inside solid map.
            if env.levels.current_level != args.level or not env.move_available():
                continue
            sx, sy = int(env.king.rect_x), int(env.king.rect_y)
            key = (sx // args.dedup, sy // args.dedup)
            if key not in seen:
                seen[key] = (sx, sy)
    spots = sorted(seen.values(), key=lambda p: p[1])   # top (small y) first
    if args.max_states and len(spots) > args.max_states:
        # keep an even spread across the level height
        idx = [round(i * (len(spots) - 1) / (args.max_states - 1))
               for i in range(args.max_states)]
        spots = [spots[i] for i in sorted(set(idx))]

    # score by altitude: highest spot -> 1.0, lowest -> ~0.1 (reverse curriculum)
    ys = [s[1] for s in spots]
    ymin, ymax = min(ys), max(ys)
    span = max(1, ymax - ymin)
    states = [{"level": args.level, "x": sx, "y": sy,
               "score": round(0.15 + 0.85 * (ymax - sy) / span, 3)}
              for (sx, sy) in spots]

    if args.merge_demo:
        dpath = f"starts/starts_L{args.level}_demo.json"
        if os.path.exists(dpath):
            demo = json.load(open(dpath))
            # boost the proven route slightly so it's always well-represented
            for d in demo:
                d["score"] = min(1.0, d.get("score", 0.5) + 0.05)
            states = demo + states

    out = args.out or f"starts/starts_L{args.level}_broad.json"
    json.dump(states, open(out, "w"), indent=2)
    print(f"L{args.level}: sampled {len(spots)} valid grounded states "
          f"(y {ymin}..{ymax})"
          f"{' + '+str(len(states)-len(spots))+' demo route states' if args.merge_demo else ''}"
          f" -> {out}")
    env.close()


if __name__ == "__main__":
    main()
