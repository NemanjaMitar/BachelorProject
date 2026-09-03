#!/usr/bin/env python
"""How much does the arrival into each level ACTUALLY vary?

`--entry-jitter-dx` decides how wide a band the entry rung is trained (and
gated) on, and picking it by feel is how a level ends up either brittle or
unlearnable. Too narrow and the model only works from one x; too wide and the
gate demands a corridor the level does not have -- measured on L32, where the
route runs along the right edge of a 3-4 px corridor and a 6 px entry gate
capped the crossing rate at 0.20.

So measure it: play the relay under several wind phases and record the x the
king lands on when he first enters each level.

    python tools/arrival_band.py                     # every level, 8 seeds
    python tools/arrival_band.py --levels 32,34,40
    python tools/arrival_band.py --learned           # with the learned models in
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
import FullRelay as FR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=None,
                    help="comma-separated; default every level the relay reaches")
    ap.add_argument("--seeds", type=int, default=8,
                    help="wind phase is seeded per run; each seed is one sample")
    ap.add_argument("--learned", action="store_true",
                    help="put each screen's own training dir in the chain "
                         "(tools/relay_bundle.py selection) instead of checkpoints/L<N>/")
    args = ap.parse_args()

    want = ({int(x) for x in args.levels.split(",") if x.strip()}
            if args.levels else None)

    swaps = {}
    if args.learned:
        sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
        import relay_bundle
        swaps = relay_bundle.learned_models()
        print(f"{len(swaps)} learned models in the chain")

    samples = {}
    for s in range(args.seeds):
        seed = 20260826 + s
        e = FR.build_env(seed=seed)
        models = FR.bind(e, {L: FR.load(L, (e.grid_h, e.grid_w), swaps.get(L))
                             for L in range(0, 43)})
        r = FR.run_trial(e, models, 0)
        for lvl, (x, y) in r["arrival"].items():
            samples.setdefault(int(lvl), []).append((float(x), float(y)))
        e.close()
        print(f"  seed {seed}: reached L{max(r['traj'])}", flush=True)

    print(f"\n{'level':>6} {'n':>3} {'x min':>8} {'x max':>8} {'spread':>7} "
          f"{'y':>6}   suggested --entry-jitter-dx")
    for lvl in sorted(samples):
        if want and lvl not in want:
            continue
        xs = [p[0] for p in samples[lvl]]
        ys = {round(p[1]) for p in samples[lvl]}
        spread = max(xs) - min(xs)
        # half the observed spread, plus a 2 px margin, is the band the entry
        # rung has to cover; below 2 px there is nothing to be robust to
        sug = max(2.0, spread / 2.0 + 2.0)
        print(f"  L{lvl:<4} {len(xs):>3} {min(xs):8.2f} {max(xs):8.2f} "
              f"{spread:7.2f} {sorted(ys)!s:>6}   {sug:.1f}")


if __name__ == "__main__":
    sys.exit(main())
