#!/usr/bin/env python
"""Run the generated-screen policy on a screen YOU drew.

`GenEval.py` scores a whole pool; this takes one hand-made screen -- anything
`LevelEditor.py` saves, or any `jumpking-world` file -- and answers the only
question that matters in practice: would the model climb out of it, and on which
attempt.

    python tools/gen_try.py --world levels/mine.json --model checkpoints/worldgen_best.pt
    python tools/gen_try.py --world levels/mine.json --screen 2      # a later screen
    python tools/gen_try.py --world levels/mine.json --prove         # is it even solvable?

The file's screen `--screen` is used as the climb; the screen above it is
replaced by the generator's standard catch floor, because the king enters the
next screen BELOW its bottom edge and any on-screen floor there is a ceiling he
bonks on (see WorldGen.CATCH_FLOOR). Nothing else about your screen is touched.

`--attempts` runs the sampled policy repeatedly from the same spot and reports
which attempt first succeeded -- that is the "would it get it first or second
try" number. `--greedy` uses the argmax instead, where retries are pointless
because the policy is deterministic.
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

import CustomWorld
import WorldGen
from GenEval import load_policy
from PPO import get_device


def as_climb_world(path, screen=0):
    """Your screen + the generator's catch screen, as a 2-screen world."""
    src = CustomWorld.load_world(path)
    if screen >= len(src["levels"]):
        raise SystemExit(f"{path} has {len(src['levels'])} screens; "
                         f"--screen {screen} is out of range")
    return {"format": CustomWorld.FORMAT, "version": 1,
            "name": f"{src.get('name', 'custom')}#{screen}",
            "levels": [src["levels"][screen],
                       {"wind": False,
                        "platforms": [dict(WorldGen.CATCH_FLOOR)] + WorldGen._walls()}]}


def newest_model():
    for pat in ("checkpoints/worldgen_best.pt",
                "checkpoints/experiments/worldgen*/ppo_*.pt"):
        hits = sorted(glob.glob(pat), key=os.path.getmtime)
        if hits:
            return hits[-1]
    raise SystemExit("no generated-screen checkpoint found; pass --model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, help="a jumpking-world json")
    ap.add_argument("--screen", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--starts", type=int, default=3,
                    help="bottom-floor x positions to try from")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--emit", default=None,
                    help="write the prepared 2-screen climb world here (your "
                         "screen + the catch screen) so Play.py can render it")
    ap.add_argument("--seams", action="store_true",
                    help="report, for every seam of a MULTI-screen world, how "
                         "many actions actually carry the king into the screen "
                         "above -- the usual reason a hand-built world stalls")
    ap.add_argument("--prove", action="store_true",
                    help="also engine-prove the screen is solvable rung by rung")
    args = ap.parse_args()

    if args.seams:
        src = CustomWorld.load_world(args.world)
        env = WorldGen._make_prover_env(goal_level=len(src["levels"]) - 1)
        CustomWorld.apply_world(env, src)
        try:
            for k in range(len(src["levels"]) - 1):
                hits, land = WorldGen.probe_seam(env, src, k)
                if not hits:
                    print(f"  seam {k} -> {k+1}: IMPASSABLE "
                          f"(0 of {env.num_actions} actions cross)")
                    continue
                uniq = sorted(set(hits))
                print(f"  seam {k} -> {k+1}: {len(hits)} crossings from "
                      f"{len(uniq)} distinct action(s) {uniq} "
                      f"-> lands e.g. {land[0][1:]} on screen {land[0][0]}")
                if len(uniq) <= 2:
                    print(f"      only {len(uniq)} action(s) of {env.num_actions} "
                          f"work here -- a policy trained on single screens has "
                          f"no reason to find it; widen the landing ledge in "
                          f"screen {k+1} or move it under the exit column")
        finally:
            env.close()
        return

    world = as_climb_world(args.world, args.screen)
    rungs = WorldGen.ledge_tops(world, 0)
    print(f"{world['name']}: {len(rungs) - 1} rungs above the floor")
    for (xl, xr, y) in reversed(rungs):
        print(f"    x={xl:3d}..{xr:3d}  y={y:3d}")

    if args.emit:
        CustomWorld.save_world(world, args.emit)
        print(f"wrote {args.emit}  -- watch it with:")
        print(f"    python Play.py --world {args.emit} --goal-level 1 "
              f"--level 0 --stochastic --checkpoint <model.pt>")

    if args.prove:
        env = WorldGen._make_prover_env()
        try:
            ok, links = WorldGen.prove_ladder(env, world, verbose=True)
        finally:
            env.close()
        print("solvable by single actions" if ok else
              f"NOT solvable: rung {len(links) - 1} is a dead end")
        if not ok:
            return

    model = args.model or newest_model()
    device = get_device()
    env, net, ndim = load_policy(model, device)
    print(f"model: {model}  ({env.num_actions} actions)")

    xs = np.linspace(40, WorldGen.SCREEN_W - 40, args.starts).astype(int)
    first_try = solved = 0
    try:
        CustomWorld.apply_world(env, world)
        for x0 in xs:
            got = None
            for attempt in range(1, args.attempts + 1):
                obs, _ = env.reset(level=0, rect_x=int(x0),
                                   rect_y=WorldGen.FLOOR_Y - 60)
                for _ in range(args.max_steps):
                    with torch.no_grad():
                        logits, _ = net(torch.as_tensor(obs[:ndim], device=device)
                                        .float().unsqueeze(0))
                        a = int(logits.argmax(1).item() if args.greedy else
                                torch.distributions.Categorical(
                                    logits=logits).sample())
                    obs, _, term, trunc, info = env.step(a)
                    if info["level"] >= 1:
                        got = attempt
                        break
                    if term or trunc:
                        break
                if got:
                    break
            solved += got is not None
            first_try += got == 1
            print(f"  start x={x0:3d}: "
                  + (f"climbed out on attempt {got}" if got
                     else f"failed all {args.attempts} attempts"))
    finally:
        env.close()

    print(f"\n{solved}/{len(xs)} starting positions climbed out "
          f"({first_try} of them on the first attempt), "
          f"{'greedy' if args.greedy else 'sampled'} policy")


if __name__ == "__main__":
    main()
