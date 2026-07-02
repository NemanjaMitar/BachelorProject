#!/usr/bin/env python
"""
Discover HANDOFF states: where does the level-N model actually deliver the
king on level N+1? Those arrival spots are exactly the start states the
level-N+1 model must master, or the relay chain breaks between the two.

Runs the level-N policy headlessly from its own start-state pool until it
reaches the goal level, records the settled arrival position, dedupes on a
coarse grid, and merges the results into the next level's start-state file
(list-of-dicts format, score 0.0 = hardest curriculum stage, unlocked last).

Usage (after level 6 is trained):
    python Handoff.py --level 6 --checkpoint checkpoints/L6/ppo_XXXX.pt --start-states starts_L6.json
    -> merges arrival states into starts_L7.json

    --episodes 50        more episodes = better coverage of arrival spots
    --greedy             argmax policy instead of sampling (fewer, canonical spots)
    --out FILE           merge somewhere else than starts_L<N+1>.json
"""

import os
import json
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch

from JK_Env import JumpKingEnv
from PPO import ActorCritic, get_device


def merge(path, states, grid=8):
    """Append new {level,x,y,score} dicts to a list-format pool file,
    deduping against what's already there on a `grid`-px raster."""
    try:
        with open(path) as f:
            pool = json.load(f)
        if not isinstance(pool, list):
            print(f"WARNING: {path} is not a list-format pool; leaving it "
                  f"untouched and writing nothing")
            return 0
    except (FileNotFoundError, json.JSONDecodeError):
        pool = []

    seen = {(int(s["level"]), int(s["x"]) // grid, int(s["y"]) // grid)
            for s in pool if isinstance(s, dict)}
    added = 0
    for st in states:
        key = (st["level"], st["x"] // grid, st["y"] // grid)
        if key in seen:
            continue
        pool.append(dict(st, score=0.0))
        seen.add(key)
        added += 1
    if added:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(pool, f, indent=2)
        os.replace(tmp, path)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True,
                    help="the SOLVED level whose model does the delivering")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--start-states", required=True,
                    help="that level's own start pool (starts_L<N>.json)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax policy (default: sample like training)")
    ap.add_argument("--out", default=None,
                    help="pool file to merge into (default starts_L<N+1>.json)")
    args = ap.parse_args()

    goal = args.level + 1
    out = args.out or f"starts_L{goal}.json"
    device = get_device()

    env = JumpKingEnv(max_steps=args.max_steps, goal_level=goal,
                      start_states=args.start_states, p_bottom=0.0)
    net = ActorCritic(env.obs_dim, env.num_actions,
                      grid_shape=env.grid_shape, n_scalars=env.n_scalars).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded {args.checkpoint} | level {args.level} -> {goal} | "
          f"{args.episodes} episodes ({'greedy' if args.greedy else 'stochastic'})")

    arrivals, succ = [], 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        term = trunc = False
        info = {}
        while not (term or trunc):
            ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(ob)
                if args.greedy:
                    a = int(torch.argmax(logits, dim=1).item())
                else:
                    a = int(torch.distributions.Categorical(logits=logits)
                            .sample().item())
            obs, _, term, trunc, info = env.step(a)
        if info.get("success") and info.get("level", -1) >= goal:
            succ += 1
            st = {"level": int(info["level"]),
                  "x": int(info["x"]), "y": int(info["y"])}
            arrivals.append(st)
            print(f"  ep {ep:3d}: arrived lvl {st['level']} "
                  f"x={st['x']} y={st['y']}")
    env.close()

    print(f"\n{succ}/{args.episodes} episodes reached level {goal}, "
          f"{len(arrivals)} arrival states collected")
    added = merge(out, arrivals)
    print(f"merged {added} new handoff states into {out} "
          f"(rest were duplicates of existing entries)")


if __name__ == "__main__":
    main()
