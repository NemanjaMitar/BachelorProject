#!/usr/bin/env python
"""Pick the best generated-screen policy out of a training run.

Scores the last N checkpoints of a --world-gen / --world-pool run on the
held-out pool, both greedily and by sampling, and reports the ranking. PPO
checkpoints wobble from save to save, so the last one is not reliably the best
-- this is what decides which file to ship.

    python tools/gen_harvest.py --dir checkpoints/experiments/worldgen8 \
        --worlds levels/gen_eval --last 6 --limit 80

Add --copy-to checkpoints/worldgen_best.pt to copy the winner (ranked by the
greedy score, ties broken by the sampled one) to a stable path.
"""
import os
import re
import sys
import glob
import json
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import WorldGen
from GenEval import load_policy, run_world
from PPO import get_device


def step_of(path):
    m = re.search(r"ppo_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def score(ck, pool, starts, max_steps, device, greedy):
    env, net, ndim = load_policy(ck, device)
    ok = tot = worlds = 0
    try:
        for w in pool:
            res = run_world(env, net, ndim, device, w, starts, max_steps,
                            greedy=greedy)
            ok += sum(res)
            tot += len(res)
            worlds += all(res)
    finally:
        env.close()
    return ok / max(1, tot), worlds / max(1, len(pool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--worlds", default="levels/gen_eval")
    ap.add_argument("--last", type=int, default=6)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--copy-to", default=None)
    ap.add_argument("--json", default="logs/gen_harvest.json")
    args = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(args.dir, "ppo_*.pt")), key=step_of)
    if not cks:
        raise SystemExit(f"no checkpoints in {args.dir}")
    cks = cks[-args.last:]
    pool = WorldGen.load_pool(args.worlds)[:args.limit]
    device = get_device()

    rows = []
    for ck in cks:
        g_start, g_world = score(ck, pool, args.starts, args.max_steps, device, True)
        s_start, s_world = score(ck, pool, args.starts, args.max_steps, device, False)
        rows.append({"ckpt": ck, "step": step_of(ck),
                     "greedy_per_start": g_start, "greedy_worlds": g_world,
                     "sampled_per_start": s_start, "sampled_worlds": s_world})
        print(f"{os.path.basename(ck):>18s}  greedy {100*g_start:5.1f}% "
              f"({100*g_world:5.1f}% of screens)   sampled {100*s_start:5.1f}% "
              f"({100*s_world:5.1f}%)", flush=True)

    best = max(rows, key=lambda r: (r["greedy_per_start"], r["sampled_per_start"]))
    print(f"\nBEST: {best['ckpt']}  greedy {100*best['greedy_per_start']:.1f}% | "
          f"sampled {100*best['sampled_per_start']:.1f}%  "
          f"on {len(pool)} held-out screens x {args.starts} starts")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"dir": args.dir, "worlds": args.worlds, "n": len(pool),
                   "starts": args.starts, "rows": rows, "best": best["ckpt"]},
                  open(args.json, "w", encoding="utf-8"), indent=1)
        print(f"wrote {args.json}")
    if args.copy_to:
        os.makedirs(os.path.dirname(args.copy_to) or ".", exist_ok=True)
        shutil.copy2(best["ckpt"], args.copy_to)
        print(f"copied winner -> {args.copy_to}")


if __name__ == "__main__":
    main()
