#!/usr/bin/env python
"""
Watch a trained PPO checkpoint play Jump King in a real window.

Run from the game root:
    python play.py --checkpoint checkpoints/ppo_1945600.pt
    python play.py --checkpoint checkpoints/ppo_1945600.pt --stochastic
    python play.py --checkpoint checkpoints/ppo_1945600.pt --fps 30 --episodes 3

Notes:
* jk_env.py sets SDL_VIDEODRIVER=dummy at import time for headless training.
  We override that to a REAL driver BEFORE importing jk_env, so a window opens.
* Greedy by default (argmax) so you see the learned policy, not exploration
  noise. Pass --stochastic to sample like training did.
"""

import os
# IMPORTANT: undo the headless drivers BEFORE jk_env imports pygame.
os.environ["SDL_VIDEODRIVER"] = ""      # let SDL pick the real default
os.environ["SDL_AUDIODRIVER"] = ""

import sys
import time
import argparse
import numpy as np
import torch
import pygame

from JK_Env import JumpKingEnv
from PPO import ActorCritic, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=60, help="render speed cap")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--goal-level", type=int, default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking argmax")
    p.add_argument("--scale", type=int, default=2, help="window upscaling")
    args = p.parse_args()

    device = get_device()
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level)

    # Load policy
    net = ActorCritic(env.obs_dim, env.num_actions,
                      grid_shape=env.grid_shape, n_scalars=env.n_scalars).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded {args.checkpoint} (trained to step {ckpt.get('step', '?')})")
    print(f"device={device} actions={env.num_actions}")

    # A real window to mirror the env's internal game_screen into.
    w, h = env.screen_w * args.scale, env.screen_h * args.scale
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — trained PPO agent")
    clock = pygame.time.Clock()

    def draw():
        # The env renders the world onto env.game_screen if we ask the level to.
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception as e:
            # Cosmetic blit failures shouldn't crash the viewer.
            print("draw warning:", e)
        scaled = pygame.transform.scale(env.game_screen, (w, h))
        window.blit(scaled, (0, 0))
        pygame.display.flip()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        best_alt = -1e9
        steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

            ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            with torch.no_grad():
                logits, _ = net(ob)
                if args.stochastic:
                    a = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    a = torch.argmax(logits, dim=1).item()

            # Draw EVERY physics frame of the macro-action so the jump arc and
            # the king's crouch/jump/fall animation are visible, not just the
            # settled landing frame. The env calls render_cb() after each
            # physics frame; we redraw and pace it there.
            def render_cb():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit()
                draw()
                clock.tick(args.fps)
            obs, r, term, trunc, info = env.step(a, render_cb=render_cb)
            ep_ret += r
            best_alt = max(best_alt, info["altitude"])
            steps += 1
            done = term or trunc

            draw()
            print(f"ep{ep} step{steps:3d} act={a:2d} {str(env.actions[a]):22s} "
                  f"lvl={info['level']} y={info['y']} alt={info['altitude']:.0f} "
                  f"r={r:+.2f}", end="\r")

        print(f"\nepisode {ep}: return={ep_ret:+.2f}  max_level={info['level']}  "
              f"best_altitude={best_alt:.0f}  ({'GOAL' if term else 'timeout'})")

    env.close()


if __name__ == "__main__":
    main()