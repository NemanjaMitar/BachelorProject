#!/usr/bin/env python
"""
Watch trained PPO checkpoints play Jump King in a real window.

Two ways to choose the policy:

* SINGLE MODEL (old behaviour):
    python Play.py --checkpoint checkpoints/L4/ppo_409600.pt --level 4 --goal-level 5

* PER-LEVEL RELAY: whenever the king stands on level N, the newest checkpoint
  in <model-dir>/L<N>/ drives him. This chains your per-level specialists into
  one full climb:
    python Play.py --model-dir checkpoints
    python Play.py --model-dir checkpoints --level 2   # start the relay on level 2

  Levels without a model dir keep the model that carried the king there
  (with a one-time warning); --checkpoint, if also given, is the preferred
  fallback instead.

Notes:
* JK_Env.py sets SDL_VIDEODRIVER=dummy at import time for headless training.
  We override that to a REAL driver BEFORE importing jk_env, so a window opens.
* Greedy by default (argmax) so you see the learned policy, not exploration
  noise. Pass --stochastic to sample like training did.
* A checkpoint trained with a different action set / observation (e.g. the old
  12-dim env, or --fine-walk-frames) will fail to load with a clear message.
"""

import os
# IMPORTANT: undo the headless drivers BEFORE jk_env imports pygame.
os.environ["SDL_VIDEODRIVER"] = ""      # let SDL pick the real default
os.environ["SDL_AUDIODRIVER"] = ""

import re
import sys
import glob
import argparse
import numpy as np
import torch
import pygame

from JK_Env import JumpKingEnv
from PPO import ActorCritic, get_device


def latest_ckpt(d):
    """Newest checkpoint in a directory, by the step number in the filename."""
    best, best_step = None, -1
    for f in glob.glob(os.path.join(d, "*.pt")):
        m = re.search(r"(\d+)", os.path.basename(f))
        step = int(m.group(1)) if m else 0
        if step > best_step:
            best, best_step = f, step
    return best


class ModelBank:
    """Lazily loads and caches one policy per level from <model_dir>/L<n>/.

    for_level(n) returns the net for level n, or None if that level has no
    loadable checkpoint (caller decides the fallback)."""

    def __init__(self, env, device, model_dir):
        self.env, self.device, self.model_dir = env, device, model_dir
        self._cache = {}

    def load(self, path):
        net = ActorCritic(self.env.obs_dim, self.env.num_actions,
                          grid_shape=self.env.grid_shape,
                          n_scalars=self.env.n_scalars).to(self.device)
        ckpt = torch.load(path, map_location=self.device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        print(f"loaded {path} (trained to step {ckpt.get('step', '?')})")
        return net

    def for_level(self, lvl):
        if lvl in self._cache:
            return self._cache[lvl]
        net = None
        if self.model_dir:
            d = os.path.join(self.model_dir, f"L{lvl}")
            path = latest_ckpt(d) if os.path.isdir(d) else None
            if path:
                try:
                    net = self.load(path)
                except Exception as e:
                    print(f"WARNING: {path} does not fit the current env "
                          f"(action/obs mismatch?) -- {e}")
        self._cache[lvl] = net
        return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="single checkpoint used for all levels (and as the "
                        "fallback when --model-dir has a gap)")
    p.add_argument("--model-dir", default=None,
                   help="per-level relay: use the newest checkpoint in "
                        "<model-dir>/L<n>/ while the king is on level n")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=60, help="render speed cap")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--goal-level", type=int, default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking argmax")
    p.add_argument("--scale", type=int, default=2, help="window upscaling")
    p.add_argument("--level", type=int, default=None,
                   help="start every episode on this level (with --x/--y: at "
                        "that exact spot, e.g. a captured start state)")
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--start-states", type=str, default=None,
                   help="start each episode from a random state in this pool "
                        "(same file training used), instead of the bottom")
    args = p.parse_args()

    if not args.checkpoint and not args.model_dir:
        p.error("give --checkpoint, --model-dir, or both")

    device = get_device()
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      start_states=args.start_states)

    bank = ModelBank(env, device, args.model_dir)
    fallback = bank.load(args.checkpoint) if args.checkpoint else None

    # Explicit start state (bypasses the pool): reset() teleports + settles.
    reset_kwargs = {}
    if args.level is not None or args.x is not None or args.y is not None:
        reset_kwargs = {"level": args.level or 0,
                        "rect_x": args.x, "rect_y": args.y}

    # A real window to mirror the env's internal game_screen into.
    w, h = env.screen_w * args.scale, env.screen_h * args.scale
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — trained PPO agent")
    clock = pygame.time.Clock()

    def draw():
        # The env renders the world onto env.game_screen if we ask the level to.
        # Hidden walls (levels 6/7/21) are fake bricks with no collision; this
        # fades them when the king overlaps, like the real game -- otherwise he
        # looks embedded in solid wall while using a secret passage.
        try:
            env.levels.update_hiddenwalls(env.king)
        except Exception:
            pass
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

    warned_levels = set()

    for ep in range(args.episodes):
        obs, _ = env.reset(**reset_kwargs)
        done = False
        ep_ret = 0.0
        best_alt = -1e9
        steps = 0
        net = None
        net_lvl = None

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()

            # ---- pick the policy for the level the king stands on ----------
            lvl = env.levels.current_level
            if lvl != net_lvl:
                cand = bank.for_level(lvl)
                if cand is not None:
                    if net is not None and cand is not net:
                        print(f"\n>>> level {lvl}: switching to model L{lvl}")
                    net = cand
                elif fallback is not None:
                    if net is not None and net is not fallback:
                        print(f"\n>>> level {lvl}: no L{lvl} model, using the "
                              f"--checkpoint fallback")
                    net = fallback
                elif net is not None:
                    if lvl not in warned_levels:
                        print(f"\n>>> level {lvl}: no model found, keeping the "
                              f"previous level's policy")
                        warned_levels.add(lvl)
                else:
                    print(f"no model available for start level {lvl} -- give "
                          f"--checkpoint as a fallback or add "
                          f"{args.model_dir}/L{lvl}/")
                    env.close()
                    sys.exit(1)
                net_lvl = lvl

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

        outcome = ("GOAL" if info.get("success")
                   else "fell below start" if term else "timeout")
        print(f"\nepisode {ep}: return={ep_ret:+.2f}  max_level={info['level']}  "
              f"best_altitude={best_alt:.0f}  ({outcome})")

    env.close()


if __name__ == "__main__":
    main()
