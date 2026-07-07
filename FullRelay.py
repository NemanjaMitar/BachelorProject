#!/usr/bin/env python
"""Headless full-relay verification: play 0 -> top (the babe) using the actual
per-level model the relay would pick (latest checkpoint by filename number),
logging the level trajectory and flagging any FALL (level ever decreases) or
STALL (can't leave a level). This is the trustworthy "clean 0-42 run" check.

Handles mixed observation families in one relay: the env produces the SUPERSET
scalars [x,y,level,alt, wind_sin,wind_cos, vx,vy] and each model takes only the
columns it was trained on (base + wind if wind_obs + vel if vel_obs). That is
exactly the generalization Play.py needs for wind/ice/plain models to coexist.

    python FullRelay.py                 # from level 0
    python FullRelay.py --from 25       # start mid-chain
"""
import os, re, argparse, glob
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np, torch
from JK_Env import JumpKingEnv, build_action_table
from PPO import ActorCritic, get_device

dev = get_device()
SKIP = ("temp", "backup")


def _step_num(f):
    m = re.search(r"(\d+)", os.path.basename(f))
    return int(m.group(1)) if m else 0


def latest(level):
    if level <= 3:                       # levels 0-3 share one combined model
        return "checkpoints/ppo_1868800.pt"
    fs = [f for f in glob.glob(f"checkpoints/L{level}/*.pt")
          if not any(s in f.lower() for s in SKIP)]
    # match Play.py's latest_ckpt EXACTLY: highest number in the FILENAME
    return max(fs, key=_step_num) if fs else None


def load(level):
    f = latest(level)
    if not f:
        return None
    ck = torch.load(f, map_location="cpu", weights_only=False)
    cfg = ck.get("action_cfg", {})
    tbl = build_action_table(
        fine_walk_frames=cfg.get("fine_walk_frames", 3),
        extra_charges=tuple(cfg.get("extra_charges", (22, 24, 28, 30))),
        wait_frames=tuple(cfg.get("wait_frames", ())),
        wind_jump=tuple(cfg.get("wind_jump", ())),
        approach_jump=tuple(tuple(a) for a in cfg.get("approach_jump", ())),
        wind_combo=tuple(tuple(tuple(s) for s in r)
                         for r in cfg.get("wind_combo", ())))
    n = ck["model"]["policy_head.weight"].shape[0]
    wo, vo = bool(cfg.get("wind_obs")), bool(cfg.get("vel_obs"))
    nsc = 4 + 2 * wo + 2 * vo
    net = ActorCritic(0, n, grid_shape=(3, 45, 60), n_scalars=nsc).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    return dict(net=net, tbl=tbl, wo=wo, vo=vo, name=os.path.basename(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    # superset-obs env; each model selects its own scalar columns
    e = JumpKingEnv(max_steps=400, fine_walk_frames=3, extra_charges=(22, 24, 28, 30),
                    wind_jump=(0, 4), wind_obs=True, vel_obs=True)
    gf = e._grid_flat
    models = {L: load(L) for L in range(args.frm, 43)}

    def model_input(full, m):
        sc = full[gf:]                    # [x,y,level,alt,sin,cos,vx,vy]
        cols = [0, 1, 2, 3] + ([4, 5] if m["wo"] else []) + ([6, 7] if m["vo"] else [])
        return np.concatenate([full[:gf], sc[cols]]).astype(np.float32)

    wins = 0
    stuck_report = {}
    for t in range(args.trials):
        e.reset(level=args.frm)
        traj = [args.frm]; result = "?"
        arrival = {args.frm: (int(e.king.rect_x), int(e.king.rect_y))}
        for _ in range(320):
            if e.levels.ending:
                result = "REACHED BABE"; break
            lvl = e.levels.current_level
            if lvl < min(traj):
                pass
            m = models.get(lvl)
            if m is None:
                result = f"no model for L{lvl}"; break
            e.actions = m["tbl"]; e.num_actions = len(m["tbl"])
            obs = model_input(e._obs(), m)
            with torch.no_grad():
                a = int(m["net"](torch.as_tensor(obs, device=dev).float()
                                 .unsqueeze(0))[0].argmax(1).item())
            _, _, term, trunc, info = e.step(a)
            nl = info["level"]
            if nl not in arrival:
                arrival[nl] = (int(info["x"]), int(info["y"]))
            if nl != traj[-1]:
                traj.append(nl)
            if e.levels.ending:
                result = "REACHED BABE"; break
            if term or trunc:
                result = "terminated"; break
        # detect falls in the trajectory
        falls = [(traj[i], traj[i + 1]) for i in range(len(traj) - 1)
                 if traj[i + 1] < traj[i]]
        ok = result == "REACHED BABE" and not falls
        wins += ok
        if not ok:
            # the blocker is the highest level it reached but couldn't leave;
            # its arrival spot is the real relay entry to fix into that curriculum
            stuck = max(traj)
            stuck_report[stuck] = arrival.get(stuck)
        tag = "CLEAN" if ok else ("FELL@" + str([f for f in falls][:2]) if falls else result)
        print(f"trial {t}: {tag} | max={max(traj)} | traj="
              f"{traj if len(traj)<22 else traj[:22]+['...']}")
    print(f"\n{wins}/{args.trials} clean top-outs from L{args.frm}")
    if stuck_report:
        print("NEXT BLOCKERS (level -> real entry to add to its curriculum):")
        for lvl in sorted(stuck_report):
            print(f"  L{lvl} @ {stuck_report[lvl]}")
    e.close()


if __name__ == "__main__":
    main()
