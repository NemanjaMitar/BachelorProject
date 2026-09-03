#!/usr/bin/env python
"""
Held-out evaluation on procedurally generated REAL-ENGINE screens.

The generalization question the thesis asks -- "did the agent learn to climb, or
did it memorise 43 hand-made screens?" -- only has an answer if the agent is
measured on screens it has never seen, IN THE ENGINE IT WAS TRAINED IN. That is
what this does: it loads a checkpoint, plays it greedily from the bottom of each
world in a directory built by `WorldGen.py --out` (seeds disjoint from the
training pool), and reports the fraction of unseen screens it climbs out of.

    python GenEval.py --checkpoint checkpoints/worldgen/ppo_final.pt \
        --worlds levels/gen_eval --starts 3

`--starts` is how many bottom-floor x positions each world is tried from; a
world counts as solved only if EVERY one of them works (--any relaxes that to
"at least one"), so the number is a robustness figure, not a lucky-spawn one.
"""
import os
import json
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

import CustomWorld
import WorldGen
from JK_Env import JumpKingEnv, build_action_table
from Occupancy import resolve_channels
from PPO import ActorCritic, get_device


def load_policy(path, device):
    """The env + net a checkpoint was trained with (same recipe as GenTest.py)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("action_cfg", {})
    tbl = build_action_table(
        fine_walk_frames=cfg.get("fine_walk_frames", 0),
        extra_charges=tuple(cfg.get("extra_charges", ())),
        wait_frames=tuple(cfg.get("wait_frames", ())),
        wind_jump=tuple(cfg.get("wind_jump", ())),
        settle_action=bool(cfg.get("settle_action", False)))
    channels = resolve_channels(cfg.get("grid_channels"))
    env = JumpKingEnv(max_steps=200, goal_level=1,
                      fine_walk_frames=cfg.get("fine_walk_frames", 0),
                      extra_charges=tuple(cfg.get("extra_charges", ())),
                      wind_jump=tuple(cfg.get("wind_jump", ())) or (0, 4),
                      wind_obs=bool(cfg.get("wind_obs")),
                      vel_obs=bool(cfg.get("vel_obs")),
                      grid_channels=channels)
    env.actions = tbl
    env.num_actions = len(tbl)
    n_act = ck["model"]["policy_head.weight"].shape[0]
    net = ActorCritic(env.obs_dim, n_act, grid_shape=env.grid_shape,
                      n_scalars=env.n_scalars,
                      extra_conv=cfg.get("extra_conv", 0),
                      scalar_embed=cfg.get("scalar_embed", 0)).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return env, net, env._grid_flat + env.n_scalars


def run_world(env, net, ndim, device, world, starts, max_steps, greedy=True):
    """Play `world` from `starts` bottom positions; return a list of bools."""
    CustomWorld.apply_world(env, world)
    out = []
    xs = np.linspace(40, WorldGen.SCREEN_W - 40, starts).astype(int)
    for x0 in xs:
        obs, _ = env.reset(level=0, rect_x=int(x0),
                           rect_y=WorldGen.FLOOR_Y - 60)
        ok = False
        for _ in range(max_steps):
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs[:ndim], device=device)
                                .float().unsqueeze(0))
                a = int(logits.argmax(1).item() if greedy else
                        torch.distributions.Categorical(logits=logits).sample())
            obs, _, term, trunc, info = env.step(a)
            if info["level"] >= 1:
                ok = True
                break
            if term or trunc:
                break
        out.append(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--worlds", required=True,
                    help="directory of generated worlds (WorldGen.py --out)")
    ap.add_argument("--starts", type=int, default=3,
                    help="bottom-floor x positions tried per world")
    ap.add_argument("--any", action="store_true",
                    help="count a world solved if ANY start works "
                         "(default: every start must)")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of taking its argmax. "
                        "Generated screens have a full-width floor, so a missed "
                        "jump only drops the king back onto it -- the README "
                        "notes that this removes the pressure for PPO to make "
                        "the winning action the MOST likely one, and a policy "
                        "can then be far better when sampled than when greedy.")
    ap.add_argument("--json", default=None, help="write per-world results here")
    args = ap.parse_args()

    device = get_device()
    env, net, ndim = load_policy(args.checkpoint, device)
    pool = WorldGen.load_pool(args.worlds)
    if args.limit:
        pool = pool[:args.limit]

    rows, solved, runs_ok, runs = [], 0, 0, 0
    for i, world in enumerate(pool):
        res = run_world(env, net, ndim, device, world, args.starts,
                        args.max_steps, greedy=not args.stochastic)
        good = any(res) if args.any else all(res)
        solved += good
        runs_ok += sum(res)
        runs += len(res)
        rows.append({"world": world.get("name", f"w{i}"),
                     "rungs": len(WorldGen.ledge_tops(world, 0)) - 1,
                     "starts": [bool(r) for r in res], "solved": bool(good)})
        print(f"  {rows[-1]['world']:>16s} rungs={rows[-1]['rungs']} "
              f"{''.join('X' if r else '.' for r in res)} "
              f"{'SOLVED' if good else ''}")
    env.close()

    print(f"\n{os.path.basename(args.checkpoint)} on {len(pool)} UNSEEN "
          f"generated screens: {solved}/{len(pool)} solved "
          f"= {100*solved/max(1,len(pool)):.1f}%  "
          f"({'any-start' if args.any else 'all-starts'} rule) | "
          f"per-start {runs_ok}/{runs} = {100*runs_ok/max(1,runs):.1f}%")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": args.checkpoint, "worlds": args.worlds,
                       "solved": solved, "n": len(pool),
                       "per_start": [runs_ok, runs], "rows": rows}, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
