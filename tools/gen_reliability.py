#!/usr/bin/env python
"""How reliable is the generated-screen policy, per attempt and per difficulty?

`GenEval.py` reports one rate. That is not what "reliable" means for this agent,
for a reason built into the screens: every generated screen has a full-width
floor, so a missed jump drops the king back onto it and the episode continues --
exactly like a human retrying. A single greedy pass therefore understates it and
a single sampled pass overstates the role of luck. This measures the curve:

  * one GREEDY pass (deterministic -- retrying it is pointless)
  * then up to `--attempts` SAMPLED passes, reporting the cumulative rate after
    each one, which is the number a viewer of a GIF is actually estimating
  * both broken down by how many rungs the screen has, so "it fails sometimes"
    can be read as "it fails on the tall ones"

    python tools/gen_reliability.py --limit 40
    python tools/gen_reliability.py --model <ckpt> --limit 80 --attempts 5
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

import CustomWorld
import WorldGen
from GenEval import load_policy
from PPO import get_device


def one_pass(env, net, ndim, device, x0, max_steps, greedy):
    obs, _ = env.reset(level=0, rect_x=int(x0), rect_y=WorldGen.FLOOR_Y - 60)
    for _ in range(max_steps):
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs[:ndim], device=device)
                            .float().unsqueeze(0))
            a = int(logits.argmax(1).item() if greedy else
                    torch.distributions.Categorical(logits=logits).sample())
        obs, _, term, trunc, info = env.step(a)
        if info["level"] >= 1:
            return True
        if term or trunc:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/worldgen_best.pt")
    ap.add_argument("--worlds", default="levels/gen_eval")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--json", default="logs/gen_reliability.json")
    args = ap.parse_args()

    device = get_device()
    env, net, ndim = load_policy(args.model, device)
    pool = WorldGen.load_pool(args.worlds)[:args.limit]
    xs = np.linspace(40, WorldGen.SCREEN_W - 40, args.starts).astype(int)

    runs = 0
    greedy_ok = 0
    # cum[k] = runs that had succeeded by sampled attempt k (1-indexed)
    cum = [0] * (args.attempts + 1)
    by_rungs = defaultdict(lambda: {"runs": 0, "greedy": 0, "any": 0})
    rows = []

    for w, world in enumerate(pool):
        CustomWorld.apply_world(env, world)
        rungs = len(WorldGen.ledge_tops(world, 0)) - 1
        for x0 in xs:
            runs += 1
            b = by_rungs[rungs]
            b["runs"] += 1
            g = one_pass(env, net, ndim, device, x0, args.max_steps, True)
            greedy_ok += g
            b["greedy"] += g
            first = 0                       # first sampled attempt that worked
            for k in range(1, args.attempts + 1):
                if one_pass(env, net, ndim, device, x0, args.max_steps, False):
                    first = k
                    break
            if first:
                for k in range(first, args.attempts + 1):
                    cum[k] += 1
            b["any"] += bool(g or first)
            rows.append({"world": world.get("name", f"w{w}"), "rungs": rungs,
                         "x0": int(x0), "greedy": bool(g), "sampled_first": first})
        print(f"  {world.get('name','w')}: rungs={rungs} "
              f"greedy={sum(r['greedy'] for r in rows[-args.starts:])}/{args.starts}",
              flush=True)
    env.close()

    print(f"\n{os.path.basename(args.model)} on {len(pool)} UNSEEN screens "
          f"x {args.starts} starts = {runs} runs")
    print(f"  greedy (1 deterministic pass) : {greedy_ok}/{runs} "
          f"= {100 * greedy_ok / runs:.1f}%")
    for k in range(1, args.attempts + 1):
        print(f"  sampled, within {k} attempt{'s' if k > 1 else ' '}     : "
              f"{cum[k]}/{runs} = {100 * cum[k] / runs:.1f}%")

    print("\n  by screen height (a rung is a ledge the king must reach):")
    print(f"  {'rungs':>5} {'runs':>5} {'greedy':>8} {'greedy or 5 tries':>19}")
    for r in sorted(by_rungs):
        b = by_rungs[r]
        print(f"  {r:>5} {b['runs']:>5} {100 * b['greedy'] / b['runs']:>7.1f}% "
              f"{100 * b['any'] / b['runs']:>18.1f}%")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "worlds": args.worlds, "runs": runs,
                   "greedy": greedy_ok, "cumulative_sampled": cum,
                   "by_rungs": {str(k): v for k, v in by_rungs.items()},
                   "rows": rows}, f, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
