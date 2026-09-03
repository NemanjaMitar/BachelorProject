#!/usr/bin/env python
"""Generalization test: does a policy GENUINELY solve a level (step-by-step with
primitive actions), and does it generalize beyond the exact demo states?

Plays the policy greedily, one primitive action per grounded decision, from a
band of start x's around the entry (and, for windy levels, several wind phases).
Reports completion rate. Use it on the warm-started net (baseline) and again after
PPO fine-tune to see whether PPO added real robustness.

    python GenTest.py --checkpoint checkpoints/L32_king/best_rung0.pt --level 32 \
        --goal-level 33 --x 56 --y 320
"""
import os, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import torch
from JK_Env import JumpKingEnv, build_action_table
from PPO import ActorCritic, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--goal-level", type=int, required=True)
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--states", type=str, default=None,
                    help="robustness score: fraction of the states in this JSON "
                         "(e.g. starts/starts_LN_broad.json) the policy completes")
    ap.add_argument("--band", type=int, default=6, help="+/- x jitter to test")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--trials", type=int, default=4, help="wind phases per x")
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["action_cfg"]
    tbl = build_action_table(fine_walk_frames=cfg.get("fine_walk_frames", 3),
                             extra_charges=tuple(cfg.get("extra_charges", ())),
                             wait_frames=tuple(cfg.get("wait_frames", ())),
                             wind_jump=tuple(cfg.get("wind_jump", ())),
                             settle_action=bool(cfg.get("settle_action", False)))
    n = ck["model"]["policy_head.weight"].shape[0]
    nsc = 4 + (2 if cfg.get("wind_obs") else 0) + (2 if cfg.get("vel_obs") else 0)
    env = JumpKingEnv(max_steps=300, goal_level=args.goal_level,
                      fine_walk_frames=cfg.get("fine_walk_frames", 3),
                      extra_charges=tuple(cfg.get("extra_charges", ())),
                      wind_jump=tuple(cfg.get("wind_jump", ())) or (0, 4),
                      wind_obs=bool(cfg.get("wind_obs")),
                      vel_obs=bool(cfg.get("vel_obs")))
    env.actions = tbl; env.num_actions = len(tbl)
    net = ActorCritic(env.obs_dim, n, grid_shape=env.grid_shape, n_scalars=nsc).to(device)
    net.load_state_dict(ck["model"]); net.eval()
    ndim = env._grid_flat + nsc

    def solves(x0, y0):
        for _ in range(args.trials):
            obs, _ = env.reset(level=args.level, rect_x=x0, rect_y=y0)
            if env.levels.current_level != args.level:
                return None            # invalid state
            ok = False
            for _ in range(args.max_steps):
                with torch.no_grad():
                    a = int(net(torch.as_tensor(obs[:ndim], device=device)
                                .float().unsqueeze(0))[0].argmax(1).item())
                obs, _, term, trunc, info = env.step(a)
                if info["level"] >= args.goal_level:
                    ok = True; break
                if term or trunc:
                    break
            if not ok:
                return False           # failed at least one phase
        return True

    if args.states:                    # robustness over a whole-level state set
        import json
        states = json.load(open(args.states))
        states = [s for s in states if s.get("level", args.level) == args.level]
        solved = tested = 0
        for s in states:
            r = solves(int(s["x"]), int(s["y"]))
            if r is None:
                continue
            tested += 1; solved += bool(r)
        print(f"{os.path.basename(args.checkpoint)} L{args.level}: ROBUSTNESS "
              f"{solved}/{tested} states solved = {100*solved/max(1,tested):.0f}% "
              f"(all {args.trials} phases each)")
        env.close(); return

    xs = list(range(args.x - args.band, args.x + args.band + 1, args.step))
    total = won = exact_won = exact_tot = 0
    for x0 in xs:
        for _ in range(args.trials):
            obs, _ = env.reset(level=args.level, rect_x=x0, rect_y=args.y)
            if env.levels.current_level != args.level:
                continue
            ok = False
            for _ in range(args.max_steps):
                with torch.no_grad():
                    a = int(net(torch.as_tensor(obs[:ndim], device=device)
                                .float().unsqueeze(0))[0].argmax(1).item())
                obs, _, term, trunc, info = env.step(a)
                if info["level"] >= args.goal_level:
                    ok = True; break
                if term or trunc:
                    break
            total += 1; won += ok
            if x0 == args.x:
                exact_tot += 1; exact_won += ok
    print(f"{os.path.basename(args.checkpoint)} L{args.level}: "
          f"exact-entry {exact_won}/{exact_tot} | "
          f"generalization (x {xs[0]}..{xs[-1]}, {args.trials} phases) "
          f"{won}/{total} = {100*won/max(1,total):.0f}%")
    env.close()


if __name__ == "__main__":
    main()
