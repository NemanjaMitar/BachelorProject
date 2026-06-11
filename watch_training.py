#!/usr/bin/env python
"""
watch_training.py - watch episodes EXACTLY as training produces them.

The difference from Play.py: this builds the env with the SAME start-state /
curriculum / fall-rule config the trainer uses, and calls env.reset() with no
args, so you see the REAL start distribution (checkpoint sampling, p_bottom,
curriculum, frontier bias) instead of always starting at the bottom. It renders
every physics frame (so you see the arcs), draws the start-altitude anchor line,
and -- the point of the tool -- classifies and tallies WHY each episode ends.

This is the empirical confirmation for the "spends the episode reclimbing" and
"each level should be its own sequence" questions: after a run it prints a
start-level histogram, an end-reason breakdown, and the fraction of episodes
that ended having fallen below where they started.

Run from the game root (folder with King.py):

    # watch the policy under the real training start distribution
    python watch_training.py --checkpoint checkpoints/ppo_XXXX.pt \
        --start-states start_states.json --p-bottom 0.2 --goal-level 5 \
        --terminate-on-fall

    # watch the start distribution itself with a random policy (no checkpoint)
    python watch_training.py --random --start-states start_states.json

Mirror whatever flags you pass to train.py so the picture matches training.

    SPACE  pause / resume        N  skip to next episode
    ESC / close window  quit
"""

import os
# Real window, silent audio -- MUST precede JK_Env's pygame import.
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import pygame

from JK_Env import JumpKingEnv
from PPO import ActorCritic, get_device


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="policy .pt to watch; omit and pass --random to watch starts only")
    p.add_argument("--random", action="store_true",
                   help="act uniformly at random (no checkpoint needed)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of argmax")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--scale", type=int, default=2)
    # --- env config: keep these identical to your train.py invocation --------
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--goal-level", type=int, default=None)
    p.add_argument("--start-states", type=str, default=None)
    p.add_argument("--p-bottom", type=float, default=0.2)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--frontier-bias", type=float, default=0.0)
    p.add_argument("--frontier-min-level", type=int, default=1)
    p.add_argument("--terminate-on-fall", action="store_true")
    p.add_argument("--only-level", type=int, default=None,
                   help="restrict starts to ONE level's checkpoints, so you can watch "
                        "(e.g.) pure level-4 attempts and see whether the agent tries "
                        "the exit jump or refuses it. Forces p_bottom=0.")
    return p


def classify_end(term, trunc, info, start_level, fell_below):
    """Return a short human label for why the episode ended."""
    if info.get("success"):
        if info["level"] >= (info.get("start_level", start_level) + 1) and \
           (args_goal is None or info["level"] < args_goal):
            return "GOAL"
        return "GOAL"
    if trunc and not term:
        if info["level"] < start_level:
            return "TIMEOUT (below start)"
        if fell_below:
            return "TIMEOUT (recovered, no goal)"
        return "TIMEOUT (stuck on start lvl)"
    if term and info["level"] < start_level:
        return "FELL BELOW START"
    return "TERMINATED"


def main():
    global args_goal
    args = build_argparser().parse_args()
    args_goal = args.goal_level

    if not args.checkpoint and not args.random:
        sys.exit("pass --checkpoint PATH, or --random to watch starts with no policy")

    device = get_device()
    env = JumpKingEnv(
        max_steps=args.max_steps,
        goal_level=args.goal_level,
        start_states=args.start_states,
        p_bottom=args.p_bottom,
        curriculum=args.curriculum,
        frontier_bias=args.frontier_bias,
        frontier_min_level=args.frontier_min_level,
        terminate_on_fall_below_start=args.terminate_on_fall,
    )

    if args.only_level is not None:
        env._start_pool = [s for s in env._start_pool
                           if int(env._state_field(s, "level", "current_level",
                                                    default=0)) == args.only_level]
        env.p_bottom = 0.0
        if not env._start_pool:
            sys.exit(f"no checkpoints on level {args.only_level} in the pool")
        print(f"restricted to level {args.only_level}: {len(env._start_pool)} checkpoints")

    net = None
    if args.checkpoint:
        net = ActorCritic(env.obs_dim, env.num_actions,
                          grid_shape=env.grid_shape, n_scalars=env.n_scalars).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        print(f"loaded {args.checkpoint} (step {ckpt.get('step', '?')})")
    print(f"pool size={len(env._start_pool)}  device={device}  "
          f"goal_level={args.goal_level}  terminate_on_fall={args.terminate_on_fall}")

    w, h = env.screen_w * args.scale, env.screen_h * args.scale
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King - watch training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    big = pygame.font.Font(None, 26)

    # running aggregates across episodes
    start_hist = Counter()
    end_reasons = Counter()
    net_progress = []          # max_level - start_level per episode
    ended_below_start = 0

    # mutable HUD context the render callback reads
    hud = {"ep": 0, "a": 0, "r": 0.0, "info": {}, "start_level": 0,
           "start_y": 0, "max_level": 0, "min_level": 0, "from_cp": False,
           "step": 0, "paused": False}

    def draw():
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1(); env.king.blitme(); env.babe.blitme(); env.levels.blit2()
        except Exception:
            pass
        surf = pygame.transform.scale(env.game_screen, (w, h))
        window.blit(surf, (0, 0))

        info = hud["info"]
        cur_level = info.get("level", hud["start_level"])
        # start-altitude anchor: only meaningful while on the start screen
        if cur_level == hud["start_level"]:
            y = hud["start_y"] * args.scale
            for x0 in range(0, w, 16):
                pygame.draw.line(window, (90, 140, 255), (x0, y), (x0 + 8, y), 2)

        agg_ep = max(1, hud["ep"])
        below_frac = ended_below_start / agg_ep
        goal_frac = end_reasons.get("GOAL", 0) / agg_ep
        lines = [
            (f"ep {hud['ep']}/{args.episodes}   start L{hud['start_level']}"
             f"{' (cp)' if hud['from_cp'] else ' (bottom)'}   "
             f"now L{cur_level}   max L{hud['max_level']}", (255, 255, 0)),
            (f"act {hud['a']} {env.actions[hud['a']]}   r={hud['r']:+.2f}   "
             f"step {hud['step']}", (120, 255, 120)),
            (f"so far: goal {goal_frac:.0%}   ended-below-start {below_frac:.0%}   "
             f"mean net +{np.mean(net_progress):.2f}" if net_progress else
             "so far: (first episode)", (255, 160, 160)),
        ]
        if hud["paused"]:
            lines.append(("PAUSED  (SPACE resume,  N next,  ESC quit)", (255, 255, 255)))
        for i, (txt, col) in enumerate(lines):
            window.blit((big if i == 0 else font).render(txt, True, col), (6, 6 + i * 24))
        pygame.display.flip()

    def pump():
        """Handle quit/pause/skip. Returns 'skip' if N pressed, else None."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                env.close(); pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    env.close(); pygame.quit(); sys.exit()
                if e.key == pygame.K_SPACE:
                    hud["paused"] = not hud["paused"]
                if e.key == pygame.K_n:
                    return "skip"
        return None

    skip_flag = {"v": False}

    def render_cb():
        if pump() == "skip":
            skip_flag["v"] = True
        # honor pause inside the physics loop too
        while hud["paused"]:
            if pump() == "skip":
                skip_flag["v"] = True
                break
            draw()
            clock.tick(30)
        draw()
        clock.tick(args.fps)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        start_level = env._episode_start_level
        start_y = int(env.king.rect_y)
        from_cp = bool(env._episode_is_curric)
        start_hist[start_level] += 1

        hud.update(ep=ep, start_level=start_level, start_y=start_y,
                   max_level=start_level, min_level=start_level,
                   from_cp=from_cp, step=0, info={"level": start_level})
        skip_flag["v"] = False

        done = False
        info = {"level": start_level, "start_level": start_level, "success": False}
        fell_below = False
        steps = 0

        while not done:
            if pump() == "skip":
                break
            ob = torch.as_tensor(obs, device=device).float().unsqueeze(0)
            if net is not None:
                with torch.no_grad():
                    logits, _ = net(ob)
                    if args.stochastic:
                        a = torch.distributions.Categorical(logits=logits).sample().item()
                    else:
                        a = int(torch.argmax(logits, dim=1).item())
            else:
                a = int(env.rng.integers(env.num_actions))

            hud.update(a=a, step=steps + 1)
            obs, r, term, trunc, info = env.step(a, render_cb=render_cb)
            steps += 1
            lvl = info["level"]
            hud.update(r=r, info=info, max_level=max(hud["max_level"], lvl),
                       min_level=min(hud["min_level"], lvl))
            if lvl < start_level:
                fell_below = True
            done = term or trunc or skip_flag["v"]
            draw()

        reason = "SKIPPED" if skip_flag["v"] else \
            classify_end(term, trunc, info, start_level, fell_below)
        end_reasons[reason] += 1
        net_progress.append(hud["max_level"] - start_level)
        if info.get("level", start_level) < start_level or fell_below:
            ended_below_start += 1
        print(f"ep {ep:3d}  start L{start_level} {'cp ' if from_cp else 'bot'}  "
              f"-> max L{hud['max_level']}  end={reason:24s} steps={steps}")

    # ---- summary --------------------------------------------------------
    print("\n================ SUMMARY ================")
    print("start-level histogram (where episodes BEGAN):")
    tot = sum(start_hist.values())
    for lvl in sorted(start_hist):
        n = start_hist[lvl]
        print(f"   L{lvl}: {n:3d}  ({n / tot:5.1%})  " + "#" * int(40 * n / tot))
    print("\nend reasons:")
    for reason, n in end_reasons.most_common():
        print(f"   {reason:28s} {n:3d}  ({n / tot:5.1%})")
    print(f"\nepisodes that ended having fallen below their start level: "
          f"{ended_below_start}/{tot}  ({ended_below_start / tot:.1%})")
    print(f"mean net level progress (max - start): +{np.mean(net_progress):.2f}")
    print("=========================================")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    args_goal = None
    main()