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
from JK_Env import JumpKingEnv, build_action_table, obs_selector
from PPO import ActorCritic, get_device
from Occupancy import resolve_channels, SUPPORTED_CHANNELS

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


def load(level, hw=(45, 60), path=None, ck=None, name=None):
    """The model that drives `level`, ready to step.

    `ck` takes an ALREADY-LOADED checkpoint dict instead of reading a file, so a
    packed relay (tools/relay_bundle.py) can serve the same models straight out
    of one archive without unpacking them to disk first. The file path is still
    the default; nothing about that path changes."""
    if ck is None:
        f = path or latest(level)
        if not f:
            return None
        ck = torch.load(f, map_location="cpu", weights_only=False)
        name = os.path.basename(f)
    cfg = ck.get("action_cfg", {})
    tbl = build_action_table(
        fine_walk_frames=cfg.get("fine_walk_frames", 3),
        extra_charges=tuple(cfg.get("extra_charges", (22, 24, 28, 30))),
        wait_frames=tuple(cfg.get("wait_frames", ())),
        wind_jump=tuple(cfg.get("wind_jump", ())),
        approach_jump=tuple(tuple(a) for a in cfg.get("approach_jump", ())),
        wind_combo=tuple(tuple(tuple(s) for s in r)
                         for r in cfg.get("wind_combo", ())),
        settle_action=bool(cfg.get("settle_action", False)))
    n = ck["model"]["policy_head.weight"].shape[0]
    wo, vo = bool(cfg.get("wind_obs")), bool(cfg.get("vel_obs"))
    # checkpoints written before grid_channels existed get the legacy triple
    ch = resolve_channels(cfg.get("grid_channels"))
    venc = cfg.get("vel_encoding", "xy")
    nsc = 4 + 2 * wo + (0 if not vo else (3 if venc == "polar" else 2))
    net = ActorCritic(0, n, grid_shape=(len(ch),) + tuple(hw), n_scalars=nsc,
                      extra_conv=bool(cfg.get("extra_conv", False)),
                      scalar_embed=int(cfg.get("scalar_embed", 0))).to(dev)
    net.load_state_dict(ck["model"]); net.eval()
    return dict(net=net, tbl=tbl, wo=wo, vo=vo, ch=ch,
                venc=venc, name=name or f"L{level}")


def bind(e, models):
    """Give every model the selector that narrows THIS env's superset
    observation to the columns that model was trained on."""
    for m in models.values():
        if m is not None:
            m["sel"] = obs_selector(e, m["ch"], m["wo"], m["vo"], m["name"],
                                m.get("venc"))
    return models


def build_env(seed=None):
    """THE relay env: SUPERSET observation, every model selects its own columns.

    Every supported grid channel is produced, not just the legacy triple, so a
    model trained on any subset (the ice family wants solid+slope) can be served
    here. Widening this does not change what any model sees -- obs_selector
    picks the same planes by NAME -- it only widens the intermediate vector."""
    return JumpKingEnv(max_steps=400, fine_walk_frames=3,
                       extra_charges=(22, 24, 28, 30), wind_jump=(0, 4),
                       wind_obs=True, vel_obs=True, vel_encoding="both",
                       grid_channels=SUPPORTED_CHANNELS, seed=seed)


def run_trial(e, models, frm, cap=320):
    """One greedy relay attempt from level `frm`; returns a record dict.

    levels.ending is cleared FIRST. It is set when the king reaches the babe and
    nothing in reset() clears it, so without this every trial after the first
    top-out in a process exits on the very first check and is scored as a win --
    a phantom that inflated the old multi-trial counts."""
    e.levels.ending = False
    e.reset(level=frm)
    traj = [frm]
    result = "?"
    arrival = {frm: (int(e.king.rect_x), int(e.king.rect_y))}
    steps = []                        # (level, action_idx): the run's fingerprint
    spans = []                        # screens driven by a lower level's model
    held, held_lvl = None, -1         # the model in use and the level it is for
    for _ in range(cap):
        if e.levels.ending:
            result = "REACHED BABE"; break
        lvl = e.levels.current_level
        m = models.get(lvl)
        if m is None and held is not None and lvl > held_lvl:
            # A SCREEN WITH NO MODEL IS NOT AUTOMATICALLY THE END OF THE RUN.
            # Two screens (24 and 41) are crossed THROUGH, not stood on: the
            # routes for L23 and L40 climb through them, so the relay never asks
            # for a model there and none was ever trained. A policy spends its
            # actions one at a time, so it does stand on the spanned screen --
            # and reporting "no model" there would fail a policy that is in the
            # middle of a route going exactly where it should. Carry the model
            # that is mid-route instead. Only UPWARD (lvl > held_lvl): after a
            # fall the king is on a screen the held model knows nothing about,
            # and that must still be a break.
            m = held
            if lvl not in spans:
                spans.append(lvl)
        if m is None:
            result = f"no model for L{lvl}"; break
        if models.get(lvl) is not None:
            held, held_lvl = m, lvl
        e.actions = m["tbl"]; e.num_actions = len(m["tbl"])
        obs = m["sel"](e._obs())
        with torch.no_grad():
            a = int(m["net"](torch.as_tensor(obs, device=dev).float()
                             .unsqueeze(0))[0].argmax(1).item())
        steps.append((lvl, a))
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
    return dict(traj=traj, arrival=arrival, result=result, falls=falls,
                steps=steps, spans=spans,
                ok=(result == "REACHED BABE" and not falls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed the env RNG (wind phase) so the run reproduces")
    ap.add_argument("--swap", default=None, metavar="LVL=PATH[,LVL=PATH]",
                    help="use a different checkpoint for these levels without "
                         "touching checkpoints/L<N>/ -- so a candidate model can "
                         "be tried in the real chain while the verified baseline "
                         "stays exactly where it is")
    args = ap.parse_args()

    e = build_env(seed=args.seed)
    swaps = {}
    for item in (args.swap or "").split(","):
        if item.strip():
            lvl, pth = item.split("=", 1)
            swaps[int(lvl)] = pth.strip()
    models = bind(e, {L: load(L, (e.grid_h, e.grid_w), swaps.get(L))
                      for L in range(args.frm, 43)})
    for L, pth in sorted(swaps.items()):
        print(f"SWAP L{L} -> {pth}")

    wins = 0
    stuck_report = {}
    for t in range(args.trials):
        r = run_trial(e, models, args.frm)
        traj, falls = r["traj"], r["falls"]
        wins += r["ok"]
        if not r["ok"]:
            # the blocker is the highest level it reached but couldn't leave;
            # its arrival spot is the real relay entry to fix into that curriculum
            stuck = max(traj)
            stuck_report[stuck] = r["arrival"].get(stuck)
        tag = "CLEAN" if r["ok"] else ("FELL@" + str(falls[:2]) if falls else r["result"])
        span = r.get("spans") or []
        print(f"trial {t}: {tag} | max={max(traj)} | traj="
              f"{traj if len(traj)<22 else traj[:22]+['...']}"
              + (f" | spanned {span} on the model below" if span else ""))
    print(f"\n{wins}/{args.trials} clean top-outs from L{args.frm}")
    if stuck_report:
        print("NEXT BLOCKERS (level -> real entry to add to its curriculum):")
        for lvl in sorted(stuck_report):
            print(f"  L{lvl} @ {stuck_report[lvl]}")
    e.close()


if __name__ == "__main__":
    main()
