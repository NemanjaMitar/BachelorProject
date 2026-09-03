# -*- coding: utf-8 -*-
"""
Play with a live neural-network side panel (NNViz): the game on the left,
action probabilities / observation / V(s) / layer activations on the right.

    python VizPlay.py --model-dir checkpoints --level 29 --goal-level 31   # wind
    python VizPlay.py --checkpoint checkpoints/experiments/L36_veltest/ppo_307200.pt --level 36 --goal-level 37   # ice (vel-obs)
    python VizPlay.py --model-dir checkpoints                              # full relay from 0

Extends Play.py's relay with:
  * velocity-observation models (vel_obs) via a per-model scalar-column selector
  * forward hooks that capture conv/trunk activations for the panel
  * --shots DIR to dump one PNG per decision (for the thesis / headless check)
"""
import os
import sys
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = "dummy" if "--mute" in sys.argv else ""

import argparse
import glob
import numpy as np
import torch
import pygame

from Play import latest_ckpt, scan_action_cfg                   # noqa: E402
from JK_Env import JumpKingEnv, obs_selector                   # noqa: E402
from PPO import get_device                                     # noqa: E402
from Occupancy import SUPPORTED_CHANNELS                       # noqa: E402
from NNViz import NNVisualizer                                 # noqa: E402
import FullRelay as FR                                         # noqa: E402


class VizModelBank:
    """FullRelay's loader plus the forward hooks the side panel reads.

    The relay serves models trained on different observation families, so the
    env is built as the SUPERSET and every model gets the selector that narrows
    it to its own planes and scalar columns -- the same binding FullRelay.bind
    does. Slicing scalar columns by hand (what this used to do) silently feeds a
    model whose grid channels differ from the env's the WRONG planes: an ice
    model wants solid+slope, and it would have been handed solid+king instead."""

    def __init__(self, env, device, model_dir, acts_store):
        self.env, self.device, self.model_dir = env, device, model_dir
        self.acts = acts_store
        self._cache = {}

    def _hook(self, name):
        def fn(_m, _i, out):
            self.acts[name] = out.detach().cpu().numpy().ravel()
        return fn

    def load(self, path, level=0):
        m = FR.load(level, (self.env.grid_h, self.env.grid_w), path=path)
        if m is None:
            raise SystemExit(f"{path}: no such checkpoint")
        m["sel"] = obs_selector(self.env, m["ch"], m["wo"], m["vo"],
                                m["name"], m.get("venc"))
        m["mask"] = scalar_mask(self.env, m)
        net = m["net"]
        net.conv[-1].register_forward_hook(self._hook("conv"))   # Flatten out
        net.trunk[1].register_forward_hook(self._hook("h1"))     # tanh 1
        net.trunk[3].register_forward_hook(self._hook("h2"))     # tanh 2
        print(f"loaded {path} ({len(m['tbl'])} actions, "
              f"channels {','.join(m['ch'])}"
              f"{', wind' if m['wo'] else ''}{', vel' if m['vo'] else ''})")
        return m

    def for_level(self, lvl):
        if lvl in self._cache:
            return self._cache[lvl]
        m = None
        if self.model_dir:
            d = os.path.join(self.model_dir, f"L{lvl}")
            path = latest_ckpt(d) if os.path.isdir(d) else None
            if path:
                try:
                    m = self.load(path, lvl)
                except Exception as e:
                    print(f"WARNING: {path}: {e}")
        self._cache[lvl] = m
        return m


def scalar_mask(env, m):
    """Which of the env's scalar columns this model actually reads.

    Cosmetic -- the panel greys out the rest. The packing order is the one
    obs_selector documents: [x, y, level, altitude] (+ wind) (+ velocity)."""
    n = len(env._obs()) - env._grid_flat
    mask = [False] * n
    for i in range(min(4, n)):
        mask[i] = True
    if m["wo"]:
        for i in (4, 5):
            if i < n:
                mask[i] = True
    if m["vo"]:
        base = 6 if env.wind_obs else 4
        for i in range(base, min(base + 2, n)):
            mask[i] = True
    return mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--goal-level", type=int, default=None)
    p.add_argument("--mute", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--shots", default=None,
                   help="save a PNG per decision into this dir (also works "
                        "headless with SDL_VIDEODRIVER=dummy)")
    args = p.parse_args()
    if not args.checkpoint and not args.model_dir:
        p.error("give --checkpoint, --model-dir, or both")
    # levels 0-3 live in one combined checkpoint at the model-dir root; use it
    # as the fallback automatically when none is given (same as Play usage).
    if args.model_dir and not args.checkpoint:
        cand = os.path.join(args.model_dir, "ppo_1868800.pt")
        if os.path.exists(cand):
            args.checkpoint = cand
            print(f"fallback checkpoint: {cand} (levels without L<n>/ dir)")
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)
        os.environ["SDL_VIDEODRIVER"] = os.environ.get("VIZ_DRIVER", "dummy")

    candidates = [args.checkpoint]
    if args.model_dir:
        candidates += [latest_ckpt(d) for d in glob.glob(
            os.path.join(args.model_dir, "L*")) if os.path.isdir(d)]
    fine, extra = scan_action_cfg(candidates)
    print(f"env action set: fine_walk_frames={fine} extra_charges={extra}")

    device = get_device()
    # THE relay env: every supported grid channel and the widest scalar vector,
    # so a model trained on any subset can be served here (see FullRelay.build_env)
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      fine_walk_frames=fine, extra_charges=extra,
                      wind_jump=(0, 4), wind_obs=True, vel_obs=True,
                      vel_encoding="both", grid_channels=SUPPORTED_CHANNELS)

    acts = {}
    bank = VizModelBank(env, device, args.model_dir, acts)
    fallback = bank.load(args.checkpoint) if args.checkpoint else None

    reset_kwargs = {}
    if args.level is not None or args.x is not None:
        reset_kwargs = {"level": args.level or 0,
                        "rect_x": args.x, "rect_y": args.y}

    gw, gh = env.screen_w * args.scale, env.screen_h * args.scale
    window = pygame.display.set_mode((gw + NNVisualizer.PANEL_WIDTH, gh))
    pygame.display.set_caption("Jump King — PPO + NN monitor")
    clock = pygame.time.Clock()
    hud_font = pygame.font.Font(None, 22)
    viz = NNVisualizer(window, gw, gh)

    def draw():
        try:
            env.levels.update_hiddenwalls(env.king)
        except Exception:
            pass
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1(); env.king.blitme()
            env.babe.blitme(); env.levels.blit2()
        except Exception as e:
            print("draw warning:", e)
        window.blit(pygame.transform.scale(env.game_screen, (gw, gh)), (0, 0))
        try:
            hud = hud_font.render(
                f"lvl {env.levels.current_level}  x={int(env.king.rect_x)} "
                f"y={int(env.king.rect_y)}", True, (255, 255, 0))
            window.blit(hud, (6, 6))
        except Exception:
            pass
        viz.render()
        pygame.display.flip()

    shot_i = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(**reset_kwargs)
        done = False
        ep_ret = 0.0
        steps = 0
        m = None
        net_lvl = None

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); sys.exit()

            lvl = env.levels.current_level
            if lvl != net_lvl:
                cand = bank.for_level(lvl)
                if cand is not None:
                    m = cand
                elif fallback is not None:
                    m = fallback
                elif m is None:
                    print(f"no model for level {lvl}"); env.close(); sys.exit(1)
                # a screen with no model of its own keeps the one that is
                # mid-route: 24 and 41 are climbed THROUGH, never stood on
                env.actions = m["tbl"]
                env.num_actions = len(m["tbl"])
                net_lvl = lvl

            scal = obs[env._grid_flat:]
            obt = torch.as_tensor(m["sel"](obs), device=device).float().unsqueeze(0)
            with torch.no_grad():
                logits, value = m["net"](obt)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            a = (torch.distributions.Categorical(logits=logits).sample().item()
                 if args.stochastic else int(np.argmax(probs)))

            viz.update(action_probs=probs, actions=m["tbl"], selected_action=a,
                       state=list(scal), state_mask=m["mask"], value=float(value),
                       episode=ep, level=lvl, model_name=m["name"],
                       conv_out=acts.get("conv"),
                       actor_h1=acts.get("h1"), actor_h2=acts.get("h2"))

            # For macro routes, show the primitive being executed live instead
            # of the macro's index name: advance one label per jump launch.
            from NNViz import ARROW
            act = m["tbl"][a]
            step_labels, step_state = [], {"i": 0, "air": False}
            if act[0] == "wind_combo":
                for s in act[2]:
                    if s[0] == "jump":
                        step_labels.append(f"J{ARROW.get(s[1], '?')}{int(s[2]):02d}")
                    elif s[0] not in ("walk", "settle"):
                        b, d, c = s
                        step_labels.append(f"V{b}{ARROW.get(d, '?')}{int(c):02d}")
                if step_labels:
                    viz.update(exec_label=step_labels[0])
            else:
                viz.update(exec_label=None)

            def render_cb():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close(); sys.exit()
                if step_labels:
                    airborne = bool(getattr(env.king, "isJump", False))
                    if airborne and not step_state["air"]:
                        step_state["i"] = min(step_state["i"] + 1,
                                              len(step_labels))
                        idx = min(step_state["i"] - 1, len(step_labels) - 1)
                        viz.update(exec_label=step_labels[idx])
                    step_state["air"] = airborne
                draw()
                if args.shots:
                    step_state["f"] = step_state.get("f", 0) + 1
                    if step_state["f"] % 25 == 0:
                        nonlocal shot_i
                        pygame.image.save(window, os.path.join(
                            args.shots, f"shot_{ep:02d}_{shot_i:04d}.png"))
                        shot_i += 1
                clock.tick(args.fps)
            obs, r, term, trunc, info = env.step(a, render_cb=render_cb)
            ep_ret += r
            steps += 1
            done = term or trunc
            viz.update(reward=r, ep_reward=ep_ret, altitude=info["altitude"],
                       level=info["level"], exec_label=None)
            draw()
            if args.shots:
                pygame.image.save(window, os.path.join(
                    args.shots, f"shot_{ep:02d}_{shot_i:04d}.png"))
                shot_i += 1
            print(f"ep{ep} step{steps:3d} act={a:2d} lvl={info['level']} "
                  f"V={float(value):+.2f} r={r:+.2f}", end="\r")

        viz.add_episode_reward(ep_ret)
        print(f"\nepisode {ep}: return={ep_ret:+.2f} max_level={info['level']}")
    env.close()


if __name__ == "__main__":
    main()
