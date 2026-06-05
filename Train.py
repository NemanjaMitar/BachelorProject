#!/usr/bin/env python
"""
Train a PPO agent on headless Jump King.

Run from the game root (folder containing King.py and the asset folders).

Validate the environment first (single in-process env, no subprocesses):
    python train.py --smoke

Then train:
    python train.py --num-envs 8 --rollout 256 --goal-level 1

Start small: --goal-level 1 trains the agent just to clear level 0 (reach
level 1). Only widen the goal once that works -- that's your correctness check
that the whole stack (env macro-step, reward sign, PPO math) is sound.

Warm-start onto a harder goal (the "staircase"): once level 0 is solved, load
that checkpoint and bump the goal. Use --reset-opt when switching tasks:
    python train.py --num-envs 8 --rollout 64 --goal-level 2 \
        --start-states start_states.json --p-bottom 0.5 \
        --resume checkpoints/ppo_XXXX.pt --reset-opt
"""

import os
import time
import argparse
import numpy as np
import torch

from PPO import PPO, RolloutBuffer, get_device


def maybe_resume(agent, args, device):
    """Warm-start from a saved checkpoint. Returns the step to continue from.

    Loads the policy/value weights always. By default also restores the Adam
    optimizer state so a continued run resumes cleanly; pass --reset-opt when
    warm-starting onto a NEW task (e.g. bumping --goal-level) where fresh
    optimizer moments usually train better. The step counter is carried forward
    so new checkpoint files don't overwrite the old ones."""
    if not args.resume:
        return 0
    ckpt = torch.load(args.resume, map_location=device)
    agent.net.load_state_dict(ckpt["model"])
    if not args.reset_opt and "opt" in ckpt:
        try:
            agent.opt.load_state_dict(ckpt["opt"])
        except Exception as e:
            print("warning: could not load optimizer state, continuing fresh:", e)
    step = int(ckpt.get("step", 0))
    print(f"resumed from {args.resume} at step {step} "
          f"(opt {'reset' if args.reset_opt else 'restored'})")
    return step


def env_factory(args):
    """Returns a zero-arg callable that builds one env (picklable via cloudpickle)."""
    def _make():
        from JK_Env import JumpKingEnv
        return JumpKingEnv(
            max_steps=args.max_steps,
            goal_level=args.goal_level,
            level_reward=args.level_reward,
            level_penalty=args.level_penalty,
            altitude_breadcrumb=args.altitude_breadcrumb,
            start_states=args.start_states,   # curriculum checkpoint pool
            p_bottom=args.p_bottom,           # fraction of resets that start at the bottom
            curriculum=args.curriculum,       # reverse easy->hard curriculum
            cur_advance_rate=args.cur_advance_rate,
            cur_target_p_bottom=args.cur_target_p_bottom,
            terminate_on_fall_below_start=args.terminate_on_fall,
        )
    return _make


def smoke(args):
    """Single env, in-process. Confirms the env loads assets, the macro-step
    runs, rewards have the right sign (up => positive), and PPO can do one
    update. If this passes, scaling to subprocesses is the easy part."""
    from JK_Env import JumpKingEnv
    device = get_device()
    print("device:", device)

    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      level_reward=args.level_reward, level_penalty=args.level_penalty,
                      altitude_breadcrumb=args.altitude_breadcrumb,
                      start_states=args.start_states, p_bottom=args.p_bottom)
    agent = PPO(env.obs_dim, env.num_actions, device=device)
    maybe_resume(agent, args, device)
    print(f"obs_dim={env.obs_dim} num_actions={env.num_actions} "
          f"params={sum(p.numel() for p in agent.net.parameters())} "
          f"| checkpoints={len(env._start_pool)} p_bottom={env.p_bottom}")

    obs, _ = env.reset()
    T = 64
    buf = RolloutBuffer(T, 1, env.obs_dim, device)
    for t in range(T):
        ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
        a, lp, v = agent.net.act(ob)
        nobs, r, term, trunc, info = env.step(a.item())
        buf.add(ob.squeeze(0), a.squeeze(0), lp.squeeze(0),
                torch.tensor(r, device=device),
                v.squeeze(0),
                torch.tensor(float(term), device=device),
                torch.tensor(float(trunc), device=device))
        print(f"t={t:2d} act={a.item():2d} {str(env.actions[a.item()]):24s} "
              f"r={r:+.3f} lvl={info['level']} alt={info['altitude']:.0f}")
        obs = nobs
        if term or trunc:
            obs, _ = env.reset()
    with torch.no_grad():
        last_v = agent.net.forward(
            torch.as_tensor(obs, device=device).float().unsqueeze(0))[1]
    logs = agent.update(buf, last_v)
    print("one PPO update ok:", {k: round(v, 4) for k, v in logs.items()})
    env.close()
    print("SMOKE TEST PASSED")


def train(args):
    from VecEnv import SubprocVecEnv
    device = get_device()
    print("device:", device, "| num_envs:", args.num_envs)

    envs = SubprocVecEnv([env_factory(args) for _ in range(args.num_envs)])
    obs_dim, num_actions = envs.obs_dim, envs.num_actions
    agent = PPO(obs_dim, num_actions, device=device,
                lr=args.lr, gamma=args.gamma, lam=args.lam,
                clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
                ent_coef=args.ent_coef)

    os.makedirs(args.save_dir, exist_ok=True)
    obs = envs.reset()
    global_step = maybe_resume(agent, args, device)
    recent_returns, recent_levels, recent_success = [], [], []
    cur_stat = None
    start = time.time()

    n_updates = args.total_steps // (args.rollout * args.num_envs)
    for update in range(1, n_updates + 1):
        buf = RolloutBuffer(args.rollout, args.num_envs, obs_dim, device)

        for _ in range(args.rollout):
            ob = torch.as_tensor(obs, device=device).float()
            with torch.no_grad():
                action, logprob, value = agent.net.act(ob)
            a_np = action.cpu().numpy()
            nobs, rew, term, trunc, infos = envs.step(a_np)

            buf.add(ob, action, logprob,
                    torch.as_tensor(rew, device=device),
                    value,
                    torch.as_tensor(term, device=device),
                    torch.as_tensor(trunc, device=device))

            for info in infos:
                if "episode" in info:
                    recent_returns.append(info["episode"]["r"])
                    recent_levels.append(info["episode"]["level"])
                    recent_success.append(1.0 if info.get("success") else 0.0)
                    if info.get("curriculum"):
                        cur_stat = info["curriculum"]

            obs = nobs
            global_step += args.num_envs

        with torch.no_grad():
            last_value = agent.net.forward(
                torch.as_tensor(obs, device=device).float())[1]
        logs = agent.update(buf, last_value)

        if update % args.log_every == 0:
            sps = int(global_step / (time.time() - start))
            mr = np.mean(recent_returns[-100:]) if recent_returns else float("nan")
            ml = np.mean(recent_levels[-100:]) if recent_levels else float("nan")
            mx = max(recent_levels[-100:]) if recent_levels else -1
            sr = np.mean(recent_success[-100:]) if recent_success else float("nan")
            cur = ""
            if cur_stat:
                cur = (f" | cur {cur_stat['unlocked']}/{cur_stat['total']} "
                       f"pb {cur_stat['p_bottom']}")
            print(f"upd {update:4d} | step {global_step:8d} | {sps:5d} sps | "
                  f"ret {mr:8.2f} | succ {sr:4.2f} | mean_lvl {ml:4.1f} | "
                  f"max_lvl {mx:2d}{cur} | "
                  f"kl {logs['approx_kl']:+.4f} | ent {logs['entropy']:.3f}")

        if update % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ppo_{global_step}.pt")
            torch.save({"model": agent.net.state_dict(),
                        "opt": agent.opt.state_dict(),
                        "step": global_step}, path)
            print("saved", path)

    envs.close()


def train_visible(args):
    """Single in-process env in a REAL window, so you can WATCH training.

    This is for diagnosis, not speed: it runs ONE env (N=1), samples actions
    stochastically (so you see exploration), and renders the settled frame
    after every macro-action. If the policy has collapsed you will literally
    see the king repeat the same jump; healthy training shows varied actions
    and the king working its way upward. Subprocess vec-envs can't render (each
    has its own SDL), which is why visible mode is single-env."""
    # Must override the headless driver BEFORE JK_Env imports pygame.
    os.environ["SDL_VIDEODRIVER"] = ""
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    from JK_Env import JumpKingEnv

    device = get_device()
    print("device:", device, "| VISIBLE single-env training (N=1)")
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      level_reward=args.level_reward, level_penalty=args.level_penalty,
                      altitude_breadcrumb=args.altitude_breadcrumb,
                      start_states=args.start_states, p_bottom=args.p_bottom,
                      curriculum=args.curriculum,
                      cur_advance_rate=args.cur_advance_rate,
                      cur_target_p_bottom=args.cur_target_p_bottom,
                      terminate_on_fall_below_start=args.terminate_on_fall)
    agent = PPO(env.obs_dim, env.num_actions, device=device,
                lr=args.lr, gamma=args.gamma, lam=args.lam, clip=args.clip,
                epochs=args.epochs, minibatches=args.minibatches, ent_coef=args.ent_coef)
    global_step = maybe_resume(agent, args, device)
    print(f"checkpoints={len(env._start_pool)} curriculum={env.curriculum}")

    SCALE = 2
    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King - VISIBLE training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    def draw(update, a, r, info):
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1(); env.king.blitme(); env.babe.blitme(); env.levels.blit2()
        except Exception:
            pass
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))
        cs = info.get("curriculum", {})
        l1 = font.render(f"upd {update}  act {a} {env.actions[a]}  r={r:+.2f}",
                         True, (255, 255, 0))
        l2 = font.render(f"lvl={info['level']} y={int(info['y'])} "
                         f"cur={cs.get('unlocked','-')}/{cs.get('total','-')} "
                         f"pb={cs.get('p_bottom','-')} succ={cs.get('succ_rate','-')}",
                         True, (120, 255, 120))
        window.blit(l1, (6, 6)); window.blit(l2, (6, 28))
        pygame.display.flip()
        clock.tick(args.fps)

    os.makedirs(args.save_dir, exist_ok=True)
    obs, _ = env.reset()
    n_updates = args.total_steps // args.rollout
    for update in range(1, n_updates + 1):
        buf = RolloutBuffer(args.rollout, 1, env.obs_dim, device)
        for t in range(args.rollout):
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN
                                             and e.key == pygame.K_ESCAPE):
                    env.close(); return
            ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                a, lp, v = agent.net.act(ob)
            nobs, r, term, trunc, info = env.step(a.item())
            buf.add(ob.squeeze(0), a.squeeze(0), lp.squeeze(0),
                    torch.tensor(r, device=device), v.squeeze(0),
                    torch.tensor(float(term), device=device),
                    torch.tensor(float(trunc), device=device))
            draw(update, a.item(), r, info)
            obs = nobs
            global_step += 1
            if term or trunc:
                obs, _ = env.reset()
        with torch.no_grad():
            last_v = agent.net.forward(
                torch.as_tensor(obs, device=device).float().unsqueeze(0))[1]
        logs = agent.update(buf, last_v)
        print(f"upd {update:4d} | step {global_step:7d} | ent {logs['entropy']:.3f} "
              f"| kl {logs['approx_kl']:+.4f}")
        if update % args.save_every == 0:
            path = os.path.join(args.save_dir, f"ppo_visible_{global_step}.pt")
            torch.save({"model": agent.net.state_dict(),
                        "opt": agent.opt.state_dict(), "step": global_step}, path)
            print("saved", path)
    env.close()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="single-env sanity check")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--max-steps", type=int, default=600, help="env step limit per episode")
    p.add_argument("--goal-level", type=int, default=1,
                   help="terminate+reward on reaching this level (curriculum)")
    p.add_argument("--level-reward", type=float, default=10.0)
    p.add_argument("--level-penalty", type=float, default=10.0)
    p.add_argument("--altitude-breadcrumb", type=float, default=0.01)
    p.add_argument("--start-states", type=str, default=None,
                   help="path to start_states.json; enables the checkpoint curriculum")
    p.add_argument("--p-bottom", type=float, default=0.2,
                   help="fraction of episode resets that start at the bottom "
                        "instead of a checkpoint (only matters with --start-states)")
    p.add_argument("--curriculum", action="store_true",
                   help="reverse curriculum: start only from the easiest (near-exit) "
                        "checkpoint and unlock harder/lower ones as success rises")
    p.add_argument("--cur-advance-rate", type=float, default=0.6,
                   help="recent success rate that unlocks the next curriculum stage")
    p.add_argument("--cur-target-p-bottom", type=float, default=0.6,
                   help="bottom-start fraction reached once all checkpoints are mastered")
    p.add_argument("--terminate-on-fall", action="store_true",
                   help="end the episode (as a failure) if the king falls below the "
                        "level it started this attempt on; with level-N start states "
                        "this re-anchors the next attempt on level N")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint .pt to warm-start from")
    p.add_argument("--reset-opt", action="store_true",
                   help="when resuming, start with a fresh optimizer "
                        "(recommended when switching tasks, e.g. a new --goal-level)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.04)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--render", action="store_true",
                   help="watch training in a real window (single env, slow, for diagnosis)")
    p.add_argument("--fps", type=int, default=60, help="render cap in --render mode")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.smoke:
        smoke(args)
    elif args.render:
        train_visible(args)
    else:
        train(args)