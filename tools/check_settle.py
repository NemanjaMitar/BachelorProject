#!/usr/bin/env python
"""How much does a trained policy actually USE the standalone settle action?

The training log reports success, not which actions the policy picks -- so a run
with --settle-action can look identical whether the agent learned to damp its
momentum or ignored the action completely. This reports both.

    python tools/check_settle.py checkpoints/experiments/L36_settle
    python tools/check_settle.py checkpoints/experiments/L36_settle --episodes 10

Reports, for at-rest and for momentum-carrying starts:
  * success rate
  * the share of chosen actions that were `settle`
  * how often settle was the FIRST action (the moment it matters most)
"""

import os
import sys
import glob
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from JK_Env import JumpKingEnv
from PPO import ActorCritic


def latest(d):
    fs = glob.glob(os.path.join(d, "*.pt"))
    if not fs:
        raise SystemExit(f"no .pt files in {d}")
    return max(fs, key=lambda p: int("".join(c for c in os.path.basename(p)
                                            if c.isdigit()) or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("save_dir")
    ap.add_argument("--level", type=int, default=36)
    ap.add_argument("--pool", default="starts/starts_L36_momentum.json")
    ap.add_argument("--states", type=int, default=8, help="how many ladder rungs")
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    path = latest(args.save_dir)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("action_cfg") or {}
    n_act = ck["model"]["policy_head.weight"].shape[0]

    env = JumpKingEnv(goal_level=args.level + 1, max_steps=150,
                      fine_walk_frames=cfg.get("fine_walk_frames", 3),
                      extra_charges=tuple(cfg.get("extra_charges", ())),
                      vel_obs=bool(cfg.get("vel_obs", False)),
                      settle_action=bool(cfg.get("settle_action", False)),
                      start_states=args.pool, curriculum=True)
    has_settle = any(a[0] == "settle" for a in env.actions)
    settle_idx = ([i for i, a in enumerate(env.actions) if a[0] == "settle"] or [None])[0]

    net = ActorCritic(env.obs_dim, n_act, grid_shape=env.grid_shape,
                      n_scalars=env.n_scalars)
    net.load_state_dict(ck["model"])
    net.eval()

    states = env._cp_ranked[:args.states]
    print(f"{os.path.basename(path)}  step={ck.get('step')}  {n_act} actions  "
          f"settle={'yes (idx %d)' % settle_idx if has_settle else 'NOT IN TABLE'}")
    print(f"pool={args.pool}  rungs={len(states)}  episodes/rung={args.episodes}\n")

    for label, jit in (("at rest ", 0.0), ("momentum", args.jitter)):
        rng = np.random.default_rng(args.seed)
        wins = eps = n_settle = n_act_total = first_settle = 0
        for s in states:
            for _ in range(args.episodes):
                vx = vy = None
                if jit > 0:
                    vx, vy = rng.uniform(-jit, jit), rng.uniform(-jit, jit)
                obs, _ = env.reset(level=args.level, rect_x=s["x"], rect_y=s["y"],
                                   vx=vx, vy=vy)
                info = {}
                for t in range(30):
                    with torch.no_grad():
                        logits, _ = net(torch.as_tensor(obs).float().unsqueeze(0))
                        a = int(torch.distributions.Categorical(logits=logits).sample())
                    n_act_total += 1
                    if a == settle_idx:
                        n_settle += 1
                        if t == 0:
                            first_settle += 1
                    obs, _, term, trunc, info = env.step(a)
                    if term or trunc:
                        break
                wins += bool(info.get("success"))
                eps += 1
        sr = wins / eps
        ci = 1.96 * (sr * (1 - sr) / eps) ** 0.5
        share = n_settle / max(n_act_total, 1)
        first = first_settle / max(eps, 1)
        print(f"  {label}: success {sr:.3f} +/- {ci:.3f}   "
              f"settle share {share:.3f}   settle-as-first-action {first:.3f}")
    env.close()


if __name__ == "__main__":
    main()
