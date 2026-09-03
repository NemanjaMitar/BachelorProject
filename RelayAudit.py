#!/usr/bin/env python
"""Cascade audit: for each level N, find where level N-1 actually DELIVERS the
king, then play level N's own model from that real entry and flag a FALL (the
level ever drops below N) or a STALL (never reaches N+1). This catches the
fall-then-reclimb bug (relay hands off to a spot the single-level curriculum
never trained) across the whole chain -- the same defect found on L12/L23.

    python RelayAudit.py --lo 3 --hi 24
"""
import os, argparse, glob, json
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
from JK_Env import JumpKingEnv, build_action_table, obs_selector
from PPO import ActorCritic, get_device

dev = get_device()
SKIP = ("seed", "temp", "backup")


def latest(level):
    fs = [f for f in glob.glob(f"checkpoints/L{level}/*.pt")
          if not any(s in f.lower() for s in SKIP)]
    return max(fs, key=os.path.getmtime) if fs else None


def load(level, goal):
    f = latest(level)
    if not f:
        return None
    ck = torch.load(f, map_location="cpu", weights_only=False)
    cfg = ck.get("action_cfg", {})
    ex = tuple(cfg.get("extra_charges", (22, 24, 28, 30)))
    n = ck["model"]["policy_head.weight"].shape[0]
    # Build the env to the checkpoint's OWN observation spec. The audit used to
    # hardcode wind_obs=False / n_scalars=4, so it could only ever load plain
    # models -- a wind or velocity checkpoint failed to load_state_dict.
    wo, vo = bool(cfg.get("wind_obs")), bool(cfg.get("vel_obs"))
    chans = cfg.get("grid_channels")          # None = the legacy triple
    e = JumpKingEnv(max_steps=300, goal_level=goal, fine_walk_frames=3,
                    extra_charges=ex, wind_obs=wo, vel_obs=vo,
                    vel_encoding=cfg.get("vel_encoding", "xy"),
                    grid_channels=chans)
    e.actions = build_action_table(fine_walk_frames=3, extra_charges=ex)
    e.num_actions = len(e.actions)
    sel = obs_selector(e, chans, wo, vo, os.path.basename(f),
                       cfg.get("vel_encoding", "xy"))
    net = ActorCritic(sel.obs_dim, n, grid_shape=sel.grid_shape,
                      n_scalars=sel.n_scalars,
                      extra_conv=bool(cfg.get("extra_conv", False)),
                      scalar_embed=int(cfg.get("scalar_embed", 0))).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    return e, net, sel, os.path.basename(f)


def greedy(e, net, sel, lvl, x, y, goal, cap=30):
    obs, _ = e.reset(level=lvl, rect_x=x, rect_y=y)
    seq = [lvl]
    for _ in range(cap):
        with torch.no_grad():
            a = int(net(torch.as_tensor(sel(obs), device=dev).float()
                        .unsqueeze(0))[0].argmax(1).item())
        obs, _, t, tr, info = e.step(a)
        if info["level"] != seq[-1]:
            seq.append(info["level"])
        if info["level"] >= goal or t or tr:
            break
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=3)
    ap.add_argument("--hi", type=int, default=24)
    args = ap.parse_args()

    print(f"{'lvl':>4} {'entries (from prev)':<26} {'result':<10} status")
    for N in range(args.lo, args.hi + 1):
        prev = load(N - 1, N)
        cur = load(N, N + 1)
        if not prev or not cur:
            print(f"{N:>4} {'-- missing model --':<26}")
            continue
        ep, npv, selp, _ = prev
        starts = (json.load(open(f"starts/starts_L{N-1}.json"))
                  if os.path.exists(f"starts/starts_L{N-1}.json") else [])
        entries = set()
        for s in (starts[:8] or [{"x": 240, "y": 300}]):
            seq_x = greedy(ep, npv, selp, N - 1, int(s["x"]), int(s["y"]), N)
            if seq_x[-1] >= N:               # actually reached N
                entries.add((int(ep.king.rect_x), int(ep.king.rect_y)))
        ep.close()
        e, net, sel, name = cur
        bad = []
        for (x, y) in sorted(entries):
            seq = greedy(e, net, sel, N, x, y, N + 1)
            if min(seq) < N:
                bad.append((x, y, "FALL", seq))
            elif seq[-1] < N + 1:
                bad.append((x, y, "STALL", seq))
        e.close()
        estr = ",".join(f"{x}" for (x, y) in sorted(entries))[:24]
        if bad:
            b = bad[0]
            print(f"{N:>4} {estr:<26} {b[2]:<10} <-- {len(bad)} bad entry(s), "
                  f"e.g. ({b[0]},{b[1]}) -> {b[3]}")
        else:
            print(f"{N:>4} {estr:<26} {'OK':<10} clean cascade")


if __name__ == "__main__":
    main()
