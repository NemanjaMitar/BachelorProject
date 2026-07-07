#!/usr/bin/env python
"""
Behavior-cloning warm-start for a level whose proven path RL exploration can't
discover (precise deterministic jumps -- e.g. level 24's middle platform).

We KNOW the winning action sequence (from LevelGraph). This pre-trains the PPO
policy to IMITATE it, so PPO then only has to refine, not discover. Produces a
checkpoint you --resume into a normal Train.py run.

Usage (level 24):
    python BC.py --level 24 --goal-level 25 --out checkpoints/L24/ppo_bc.pt

Then:
    python Train.py ... --resume checkpoints/L24/ppo_bc.pt --reset-opt ...
"""

import os
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
import torch.nn as nn

from JK_Env import JumpKingEnv
from PPO import ActorCritic, get_device


# Per-level demonstration configs. Two kinds:
#   "spots"      -- deterministic level: list of (settled_x, settled_y, action),
#                   the action to take FROM each settled spot (L24 middle-platform
#                   jumps RL couldn't discover).
#   "reposition" -- windy level: the hard part is a DETERMINISTIC walk into a
#                   crossing window (walking on snow ignores wind), then the atomic
#                   wind_jump handles the gust timing. We clone: "left of the
#                   window -> walk; inside it -> wind_jump".
PATHS = {
    24: {
        "kind": "spots", "goal": 25, "wind_obs": False, "wind_jump": (),
        "spots": [
            (103, 280, ("jump", "right", 24)),
            (248, 200, ("jump", "right", 26)),
            (274, 200, ("jump", "right", 12)),
            (377, 209, ("walk", "right", 10)),
            (391, 209, ("jump", "left", 30)),
            (238, 64,  ("jump", "left", 20)),
            (90,  32,  ("jump", "right", 32)),
        ],
    },
    # L26: entry (184,281, where L25 delivers) -> 27. The 4-wind-jump route has
    # two same-row states (184 & 131 @ y281) the conv grid can't tell apart and
    # which need DIFFERENT actions -> unlearnable as separate decisions. So the
    # whole route is ONE atomic wind_combo action, picked at the (distinguishable)
    # entry. Verified 6/6 from x0 184-188.
    26: {
        "kind": "combo", "goal": 27, "wind_obs": False, "wind_jump": (0, 4),
        "wind_combo": (((0, "up", 12), (0, "left", 32),
                        (4, "up", 32), (4, "up", 32)),),
        "y": 281, "entries": [184, 186, 188],
    },
    # L28: entry (189,303) -> 29 in 3 wind-jumps. The mid-ledge landing after the
    # first jump varies (~136-169 @ y149) and the net can't gate that fine-x, so
    # a per-state BC fell left there. Whole level = ONE atomic wind_combo from the
    # entry instead. Verified 6/6 from x0 189-193.
    28: {
        "kind": "combo", "goal": 29, "wind_obs": False, "wind_jump": (0, 4),
        "wind_combo": ((("walk", 192), (0, "up", 32),
                        (4, "right", 26), (0, "left", 32)),),
        "y": 303, "entries": [189, 190, 191, 192],
    },
    # L29: L28 actually delivers to (173-174,306), NOT 143. Route: hop to the
    # mid ledge (~146), then ride the wind up and right to 30. windpath's route
    # was a phantom (teleport-settle != atomic-settle), so this was built by
    # atomic replay. Verified 5/5 across the handoff range 172-179.
    29: {
        "kind": "combo", "goal": 30, "wind_obs": False, "wind_jump": (0, 4),
        "wind_combo": ((("walk", 174), (0, "up", 8), (0, "up", 12),
                        (4, "up", 32), (4, "right", 26), (4, "up", 32)),),
        "y": 306, "entries": [172, 173, 174, 175],
    },
    # L30: L29 delivers to (401,301). Route (atomic-verified 3/3 phases, window
    # 396-406 5/5): up-26, ride-right, up-26, up-32 -> lvl 31 (climbs through the
    # level's own (305,96)/(155,47) waypoints).
    30: {
        "kind": "combo", "goal": 31, "wind_obs": False, "wind_jump": (0, 4),
        "wind_combo": (((0, "up", 26), (0, "right", 32),
                        (0, "up", 26), (0, "up", 32)),),
        "y": 301, "entries": [399, 400, 401, 402, 403],
    },
    # L31: L30 delivers to (8-10,305), far-left edge. Phase-robust route needed a
    # walk off the left wall first (counter-walk drift near the wall made edge
    # jumps phase-dependent). Built by phase-robust atomic search: walk to 32,
    # up the left, ride right to the top ledge, hop left to the exit spot, up->32.
    # Verified 5/5 across handoff x 8-10.
    31: {
        "kind": "combo", "goal": 32, "wind_obs": False, "wind_jump": (0, 4),
        "wind_combo": ((("walk", 32), (4, "up", 26), (4, "right", 32),
                        (0, "left", 16), (0, "left", 8), (0, "up", 32)),),
        "y": 305, "entries": [8, 9, 10],
    },
    # L32: DETERMINISTIC (no wind on 32+). L31 delivers to (56,320). Regular-jump
    # route found by atomic_search --regular, verified 3/3 window 55-58 -> L33.
    32: {
        "kind": "combo", "goal": 33, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "right", 30), ("jump", "left", 32),
                        ("jump", "left", 16), ("jump", "right", 22),
                        ("jump", "left", 32)),),
        "y": 320, "entries": [55, 56, 57],
    },
    # L33: DETERMINISTIC. L32 delivers to (203,264). Route verified 3/3 across
    # 199-207 -> L34 (137,320).
    33: {
        "kind": "combo", "goal": 34, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "right", 24), ("jump", "right", 20),
                        ("jump", "left", 26), ("jump", "left", 32)),),
        "y": 264, "entries": [201, 203, 205],
    },
    # L34: DETERMINISTIC (was untrained). L33 delivers to (137,320). Route
    # verified 3/3 across 133-137 -> L35 (319,320).
    34: {
        "kind": "combo", "goal": 35, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "right", 20), ("jump", "left", 28),
                        ("jump", "right", 24), ("jump", "right", 32),
                        ("jump", "right", 26)),),
        "y": 320, "entries": [134, 136, 137],
    },
    # L35: DETERMINISTIC (was untrained). L34 delivers to (319,320). Route
    # verified 3/3 across 317-323 -> L36 ICE biome (405,304).
    35: {
        "kind": "combo", "goal": 36, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "right", 8), ("jump", "left", 26),
                        ("jump", "left", 32), ("jump", "right", 32)),),
        "y": 320, "entries": [318, 319, 320],
    },
    # L36: ICE. L35 delivers (with momentum) to obs-pos (405,304). Combo LEADS
    # with a ('settle',) step that passively damps the arrival slide to its
    # natural stop (452), then the route searched from 452. Real-chain verified
    # 6/6 (BC greedy-verify falsely fails since a teleport has no momentum).
    36: {
        "kind": "combo", "goal": 37, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("settle",),
                        ("jump", "left", 24), ("jump", "right", 4),
                        ("jump", "left", 16), ("jump", "left", 24),
                        ("jump", "right", 32), ("jump", "right", 12),
                        ("jump", "left", 4), ("jump", "right", 32)),),
        "y": 304, "entries": [404, 405, 406],
    },
    # L37: ICE. L36 delivers (with momentum) to obs-pos (445,296); settles to
    # 452. 11-jump momentum route (found only with momentum-aware search) -> L38
    # (372,304). settle-prefixed; real-relay verified.
    37: {
        "kind": "combo", "goal": 38, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("settle",),
                        ("jump", "left", 32), ("jump", "right", 4),
                        ("jump", "left", 4), ("jump", "up", 4),
                        ("jump", "right", 32), ("jump", "right", 32),
                        ("jump", "up", 12), ("jump", "up", 32),
                        ("jump", "up", 8), ("jump", "right", 4),
                        ("jump", "right", 32)),),
        "y": 296, "entries": [444, 445, 446],
    },
    # L38: ICE (last ice level). L37 delivers (momentum) to obs-pos (372,304);
    # settles to 336. Route -> L39 (442,320), exits the ice biome.
    38: {
        "kind": "combo", "goal": 39, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("settle",),
                        ("jump", "left", 26), ("jump", "left", 20),
                        ("jump", "right", 4), ("jump", "right", 24),
                        ("jump", "right", 32)),),
        "y": 304, "entries": [371, 372, 373],
    },
    # L39: non-ice (deterministic). L38 delivers to (442,320) velocity-free.
    # Route -> L40 (332,320).
    39: {
        "kind": "combo", "goal": 40, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "left", 22), ("jump", "left", 30),
                        ("jump", "right", 26), ("jump", "left", 22),
                        ("jump", "right", 32)),),
        "y": 320, "entries": [441, 442, 443],
    },
    # L40: non-ice. L39 delivers to (332,320). L41's direct exit is a TRAP
    # (116,372, every jump stuck), so this combo spans L40 THROUGH L41 to L42
    # (143,320) -- L41 needs no model of its own. Combined atomic_search --regular.
    40: {
        "kind": "combo", "goal": 42, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "right", 8), ("jump", "left", 26),
                        ("jump", "left", 32), ("jump", "right", 20),
                        ("jump", "left", 32), ("jump", "right", 22),
                        ("jump", "left", 32), ("jump", "left", 24),
                        ("jump", "right", 22), ("jump", "left", 32)),),
        "y": 320, "entries": [331, 332, 333],
    },
    # L42: SUMMIT. L40's combo delivers to (143,320). Route reaches the babe at
    # (376,120) -> levels.ending=True (game won). goal 43 is unreachable so BC's
    # greedy verify won't detect the win (level stays 42); verified via real chain.
    42: {
        "kind": "combo", "goal": 42, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("jump", "left", 4), ("jump", "left", 12),
                        ("jump", "left", 32), ("jump", "right", 26),
                        ("jump", "right", 12)),),
        "y": 320, "entries": [142, 143, 144],
    },
    # L23: WRONG-EXIT fix. L23's own model climbs but exits to the RIGHT platform
    # of L24 (412,320) which is a dead-end. This merged combo (spans L24 like L40
    # spans L41) crosses to L24 and climbs the LEFT platform to L25. From L22
    # delivery band ~(264-266,288); walk-normalized to 266.
    23: {
        "kind": "combo", "goal": 25, "wind_obs": False, "wind_jump": (),
        "wind_combo": ((("settle",),
                        ("jump", "right", 32), ("jump", "left", 20),
                        ("jump", "right", 28), ("jump", "left", 30),
                        ("jump", "right", 24), ("jump", "right", 26),
                        ("jump", "right", 12), ("jump", "left", 30),
                        ("jump", "left", 20), ("jump", "right", 32)),),
        "y": 288, "entries": [266, 267, 268],
    },
    # L25: crossing fires 6/6 from x in [214,228] via ('jump_wind',(0,'left'),32);
    # the agent could do it from ~220 but refused to walk there from 192.
    25: {
        "kind": "reposition", "goal": 26, "wind_obs": True, "wind_jump": (0, 4),
        # walk labels stop strictly below cross_x; window_starts (cross labels)
        # sit strictly above it -> no x gets both labels.
        "y": 69, "walk": ("walk", "right", 3), "cross_x": 214,
        "cross": ("jump_wind", (0, "left"), 32),
        "walk_starts": [178, 182, 186, 190, 192, 196, 200, 204, 208, 212],
        "window_starts": [218, 222, 226],
    },
}


def collect_spots(env, cfg, level, perturb=(-8, -4, -2, 0, 2, 4, 8), reps=6):
    """Teleport to each spot (+ nearby x offsets), label with the proven action."""
    X, Y = [], []
    for (sx, sy, act) in cfg["spots"]:
        a_idx = env.actions.index(act)
        for dx in perturb:
            env.reset(level=level, rect_x=sx + dx, rect_y=sy)
            if env.levels.current_level != level:
                continue
            if abs(env.king.rect_x - sx) > 14 or abs(env.king.rect_y - sy) > 14:
                continue
            for _ in range(reps):
                X.append(env._obs()); Y.append(a_idx)
    return X, Y


def collect_reposition(env, cfg, level, reps=10):
    """From each start left of the window: record obs->walk while x < cross_x, then
    obs->wind_jump once inside. reps give wind-phase diversity (phase randomizes
    per reset), so the policy learns to gate on POSITION, not phase."""
    X, Y = [], []
    y = cfg["y"]
    walk_idx = env.actions.index(cfg["walk"])
    cross_idx = env.actions.index(cfg["cross"])
    cross_x = cfg["cross_x"]
    for x0 in cfg["walk_starts"]:
        for _ in range(reps):
            env.reset(level=level, rect_x=x0, rect_y=y)
            if env.levels.current_level != level:
                continue
            steps = 0
            while (env.king.rect_x < cross_x
                   and env.levels.current_level == level and steps < 25):
                X.append(env._obs()); Y.append(walk_idx)
                env.step(walk_idx); steps += 1
            if env.levels.current_level == level:
                X.append(env._obs()); Y.append(cross_idx)
    for x0 in cfg["window_starts"]:
        for _ in range(reps):
            env.reset(level=level, rect_x=x0, rect_y=y)
            if env.levels.current_level == level:
                X.append(env._obs()); Y.append(cross_idx)
    return X, Y


def collect_combo(env, cfg, level, reps=20):
    """Single decision: at the entry, pick the atomic wind_combo action that runs
    the whole level. Label the entry (+ working perturbations) with the combo
    index. One distinguishable state -> one action, trivially learnable."""
    X, Y = [], []
    combo_idx = next(i for i, a in enumerate(env.actions) if a[0] == "wind_combo")
    for x0 in cfg["entries"]:
        for _ in range(reps):
            env.reset(level=level, rect_x=x0, rect_y=cfg["y"])
            if env.levels.current_level != level:
                continue
            X.append(env._obs()); Y.append(combo_idx)
    return X, Y


def collect(env, cfg, level):
    if cfg["kind"] == "reposition":
        X, Y = collect_reposition(env, cfg, level)
    elif cfg["kind"] == "combo":
        X, Y = collect_combo(env, cfg, level)
    else:
        X, Y = collect_spots(env, cfg, level)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--goal-level", type=int, default=None)
    ap.add_argument("--fine-walk-frames", type=int, default=3)
    ap.add_argument("--extra-charges", type=str, default="22,24,28,30")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra = tuple(int(c) for c in args.extra_charges.split(",")) if args.extra_charges else ()
    device = get_device()

    cfg = PATHS.get(args.level)
    if not cfg:
        raise SystemExit(f"no proven path defined for level {args.level}")
    goal = args.goal_level if args.goal_level is not None else cfg["goal"]
    wind_obs = cfg.get("wind_obs", False)
    wind_jump = cfg.get("wind_jump", ())
    wind_combo = cfg.get("wind_combo", ())

    env = JumpKingEnv(max_steps=400, goal_level=goal,
                      fine_walk_frames=args.fine_walk_frames, extra_charges=extra,
                      wind_jump=wind_jump, wind_combo=wind_combo, wind_obs=wind_obs)

    X, Y = collect(env, cfg, args.level)
    print(f"collected {len(X)} demonstration samples ({cfg['kind']})")
    print("action distribution:", {env.actions[a]: int((Y == a).sum())
                                    for a in sorted(set(Y.tolist()))})

    net = ActorCritic(env.obs_dim, env.num_actions,
                      grid_shape=env.grid_shape, n_scalars=env.n_scalars).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    Xt = torch.as_tensor(X, device=device)
    Yt = torch.as_tensor(Y, device=device)
    # inverse-frequency class weights: reposition levels are dominated by 'walk'
    # samples, which otherwise make the net collapse to always-walk and never fire
    # the crossing.
    w = torch.ones(env.num_actions, device=device)
    for a in set(Y.tolist()):
        w[a] = len(Y) / (Y == a).sum()
    lossf = nn.CrossEntropyLoss(weight=w)

    net.train()
    for ep in range(args.epochs):
        logits, _ = net(Xt)
        loss = lossf(logits, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            acc = (logits.argmax(1) == Yt).float().mean().item()
            print(f"  epoch {ep+1:4d}  loss {loss.item():.4f}  acc {acc:.3f}")

    # verify: greedy-play from the hardest entry (the spot the agent couldn't do)
    net.eval()
    if cfg["kind"] == "reposition":
        starts = [(cfg["walk_starts"][0], cfg["y"])]
        budget = 30
    elif cfg["kind"] == "combo":
        starts = [(cfg["entries"][0], cfg["y"])]
        budget = 2
    else:
        starts = [(cfg["spots"][0][0], cfg["spots"][0][1])]
        budget = len(cfg["spots"]) + 3
    for (sx, sy) in starts:
        obs, _ = env.reset(level=args.level, rect_x=sx, rect_y=sy)
        cleared = False
        for step in range(budget):
            with torch.no_grad():
                a = int(net(torch.as_tensor(obs, device=device)
                            .float().unsqueeze(0))[0].argmax(1).item())
            obs, _, term, trunc, info = env.step(a)
            if info["level"] > args.level:
                print(f"BC policy GREEDILY CLEARS from ({sx},{sy}) in {step+1} "
                      f"steps -> lvl {info['level']} "
                      f"({int(info['x'])},{int(info['y'])})")
                cleared = True; break
            if term or trunc:
                break
        if not cleared:
            print(f"WARNING: greedy BC did not clear from ({sx},{sy}) "
                  "(PPO fine-tune should still fix it, but check the path).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "step": 0,
                "action_cfg": {"fine_walk_frames": args.fine_walk_frames,
                               "extra_charges": list(extra),
                               "wait_frames": [],
                               "wind_jump": list(wind_jump),
                               "wind_combo": [[list(s) for s in r]
                                              for r in wind_combo],
                               "wind_obs": bool(wind_obs)}}, args.out)
    print(f"saved BC warm-start -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
