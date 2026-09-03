#!/usr/bin/env python
"""Stage-by-stage score of a --world-pool checkpoint on HELD-OUT screens.

GenEval answers the thesis question (bottom start, unseen screen). This answers
the training question: how far down the reverse curriculum has the policy
actually got? Stage 0 is the top rung, stage k is k rungs below it, "floor" is
the real task. Watching all of them at once shows whether a stalled run is
stalled everywhere or only on the deepest stage.

    python tools/gen_stages.py --checkpoint checkpoints/experiments/worldgen3/ppo_1075200.pt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

import CustomWorld
import WorldGen
from GenEval import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--worlds", default="levels/gen_eval")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tries", type=int, default=2, help="starts per stage/world")
    ap.add_argument("--max-steps", type=int, default=32)
    args = ap.parse_args()

    device = "cpu"
    env, net, ndim = load_policy(args.checkpoint, device)
    pool = WorldGen.load_pool(args.worlds)[:args.n]
    kh = env.king.rect_height
    rng = np.random.default_rng(0)

    stages = [0, 1, 2, 3, 4, "floor"]
    hits = {s: [0, 0] for s in stages}
    for w in pool:
        n_rungs = len(WorldGen.ledge_tops(w, 0)) - 1
        CustomWorld.apply_world(env, w)
        for s in stages:
            if s != "floor" and s >= n_rungs:
                continue                      # this screen has no such rung
            for _ in range(args.tries):
                st = 99 if s == "floor" else s
                x0, y0 = WorldGen._stage_start(rng, w, kh, st)
                obs, _ = env.reset(level=0, rect_x=x0, rect_y=y0)
                ok = False
                for _ in range(args.max_steps):
                    with torch.no_grad():
                        a = int(net(torch.as_tensor(obs[:ndim]).float()
                                    .unsqueeze(0))[0].argmax(1).item())
                    obs, _, term, trunc, info = env.step(a)
                    if info["level"] >= 1:
                        ok = True
                        break
                    if term or trunc:
                        break
                hits[s][0] += ok
                hits[s][1] += 1
    env.close()
    print(f"{os.path.basename(args.checkpoint)} on {len(pool)} held-out screens")
    for s in stages:
        w, n = hits[s]
        if n:
            print(f"  stage {str(s):>5s}: {w:3d}/{n:3d} = {100*w/n:5.1f}%")


if __name__ == "__main__":
    main()
