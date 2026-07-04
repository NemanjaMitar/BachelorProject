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
    ap.add_argument("--fine-walk-frames", type=int, default=0,
                    help="pass the same value the model was TRAINED with")
    ap.add_argument("--extra-charges", type=str, default="",
                    help="comma list of appended jump charges the model was "
                         "trained with (e.g. 22,24,28,30)")
    ap.add_argument("--wait-frames", type=str, default="",
                    help="comma list of wait actions the model was trained "
                         "with (wind levels)")
    ap.add_argument("--wind-obs", action="store_true",
                    help="the model was trained with wind observation")
    args = ap.parse_args()

    goal = args.level + 1
    out = args.out or f"starts_L{goal}.json"
    device = get_device()

    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()
    wait = tuple(int(c) for c in args.wait_frames.split(",")) if args.wait_frames else ()
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=goal,
                      start_states=args.start_states, p_bottom=0.0,
                      fine_walk_frames=args.fine_walk_frames,
                      extra_charges=extra,
                      wait_frames=wait,
                      wind_obs=args.wind_obs)
    ckpt = torch.load(args.checkpoint, map_location=device)
    n_act = ckpt["model"]["policy_head.weight"].shape[0]
    if n_act > env.num_actions:
        raise SystemExit(
            f"{args.checkpoint} was trained with {n_act} actions but the env "
            f"has {env.num_actions} -- match --fine-walk-frames / "
            f"--extra-charges to training.")
    # Actions are appended, never reordered, so a smaller model's indices
    # stay valid in a bigger env.
    net = ActorCritic(env.obs_dim, n_act,
                      grid_shape=env.grid_shape, n_scalars=env.n_scalars).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded {args.checkpoint} | level {args.level} -> {goal} | "
          f"{args.episodes} episodes ({'greedy' if args.greedy else 'stochastic'})")

    # Cycle DETERMINISTICALLY through every pool state instead of letting
    # reset() sample. The score-weighted sampler never picks score-0 states
    # (the arrivals and hard spots!), which silently turned every previous
    # evaluation into a test of only the easy states.
    pool = [s for s in env._start_pool
            if int(s.get("level", args.level)) == args.level]
    if not pool:
        raise SystemExit(f"no level-{args.level} states in {args.start_states}")
    print(f"cycling through {len(pool)} pool states, "
          f"{args.episodes} episodes total")

    arrivals, succ = [], 0
    per_state = {}
    for ep in range(args.episodes):
        st0 = pool[ep % len(pool)]
        key0 = (int(st0["x"]), int(st0["y"]))
        obs, _ = env.reset(level=args.level,
                           rect_x=st0["x"], rect_y=st0["y"])
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
        ok = bool(info.get("success")) and info.get("level", -1) >= goal
        s_ok, s_n = per_state.get(key0, (0, 0))
        per_state[key0] = (s_ok + (1 if ok else 0), s_n + 1)
        if ok:
            succ += 1
            st = {"level": int(info["level"]),
                  "x": int(info["x"]), "y": int(info["y"])}
            # sanity: never bank an off-screen/seam-glitch position
            if 0 <= st["x"] <= 460 and 0 <= st["y"] <= 340:
                arrivals.append(st)
            print(f"  ep {ep:3d}: from ({key0[0]},{key0[1]}) arrived "
                  f"lvl {st['level']} x={st['x']} y={st['y']}")
    env.close()

    weak = [(k, v) for k, v in sorted(per_state.items()) if v[0] < v[1]]
    if weak:
        print("\nWEAK/FAILING start states:")
        for (x, y), (s_ok, s_n) in weak:
            print(f"  ({x},{y}): {s_ok}/{s_n} successful")

    print(f"\n{succ}/{args.episodes} episodes reached level {goal}, "
          f"{len(arrivals)} arrival states collected")
    added = merge(out, arrivals)
    print(f"merged {added} new handoff states into {out} "
          f"(rest were duplicates of existing entries)")


if __name__ == "__main__":
    main()
