#!/usr/bin/env python
"""Seed a newly-added approach_jump action with an EXISTING action's learned
policy-head row, so a resumed model actually USES the macro instead of ignoring
it (a fresh near-zero logit never gets sampled against a confident old action).

We copy the row of the crossing action the agent already trusts
(jump_wind(0,left,32)) into the approach_jump row and add a small bias nudge so
greedy prefers the macro. PPO fine-tuning then refines it.

    python seed_approach.py --src checkpoints/L25/ppo_471040.pt \
        --out checkpoints/L25/ppo_seed.pt
"""
import os, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
from JK_Env import build_action_table

FINE, EXTRA, WJUMP = 3, (22, 24, 28, 30), (0, 4)
SRC_ACTION = ("jump_wind", (0, "left"), 32)
APPROACH = (220, 0, "left", 32)                       # target_x,bucket,dir,charge
DST_ACTION = ("approach_jump", (220, 0, "left"), 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nudge", type=float, default=2.0,
                    help="added to the approach_jump bias so greedy prefers it")
    args = ap.parse_args()

    old = build_action_table(fine_walk_frames=FINE, extra_charges=EXTRA,
                             wind_jump=WJUMP)
    new = build_action_table(fine_walk_frames=FINE, extra_charges=EXTRA,
                             wind_jump=WJUMP, approach_jump=(APPROACH,))
    src_idx = old.index(SRC_ACTION)
    dst_idx = new.index(DST_ACTION)
    assert dst_idx == len(new) - 1 == len(old), (dst_idx, len(old), len(new))

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["model"]
    n_old = sd["policy_head.weight"].shape[0]
    if n_old != len(old):
        raise SystemExit(f"{args.src} has {n_old} actions, expected {len(old)}")

    w, b = sd["policy_head.weight"], sd["policy_head.bias"]
    sd["policy_head.weight"] = torch.cat([w, w[src_idx:src_idx + 1].clone()], 0)
    sd["policy_head.bias"] = torch.cat(
        [b, (b[src_idx:src_idx + 1] + args.nudge).clone()], 0)

    cfg = ck.setdefault("action_cfg", {})
    cfg.setdefault("fine_walk_frames", FINE)
    cfg.setdefault("extra_charges", list(EXTRA))
    cfg.setdefault("wind_jump", list(WJUMP))
    cfg.setdefault("wind_obs", True)
    cfg["approach_jump"] = [list(APPROACH)]

    # Drop the old optimizer state: it has 73-action moment tensors that no
    # longer match the 74-action head, so resuming it crashes Adam. Absent, the
    # resume just starts a fresh optimizer (the correct thing after a head edit).
    ck.pop("opt", None)

    torch.save(ck, args.out)
    print(f"seeded approach_jump (idx {dst_idx}) from {SRC_ACTION} (idx "
          f"{src_idx}) +nudge {args.nudge}; {n_old}->{len(new)} actions -> "
          f"{args.out}")


if __name__ == "__main__":
    main()
