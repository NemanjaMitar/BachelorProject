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
import sys
# IMPORTANT: undo the headless drivers BEFORE jk_env imports pygame.
# (--mute must be handled here, before pygame initialises the mixer.)
os.environ["SDL_VIDEODRIVER"] = ""      # let SDL pick the real default
os.environ["SDL_AUDIODRIVER"] = "dummy" if "--mute" in sys.argv else ""

import re
import glob
import argparse
import numpy as np
import torch
import pygame

from JK_Env import JumpKingEnv, build_action_table, obs_selector
from Occupancy import SUPPORTED_CHANNELS
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


# For checkpoints saved before 'action_cfg' was stamped in, infer the config
# from the policy-head size (this project's historical action sets).
KNOWN_ACTION_CFGS = {23: (0, ()), 25: (3, ()), 37: (3, (22, 24, 28, 30))}


def scan_action_cfg(paths):
    """Read every candidate checkpoint's action config and return the LARGEST
    one (fine_walk_frames, extra_charges). Actions are only ever appended, so
    the biggest table subsumes the smaller models."""
    best, best_n = (0, ()), 0
    for p in paths:
        if not p:
            continue
        try:
            ck = torch.load(p, map_location="cpu")
            n = ck["model"]["policy_head.weight"].shape[0]
        except Exception:
            continue
        cfg = ck.get("action_cfg")
        if cfg is not None:
            cfg = (int(cfg.get("fine_walk_frames", 0)),
                   tuple(cfg.get("extra_charges", ())))
        else:
            cfg = KNOWN_ACTION_CFGS.get(n)
            if cfg is None:
                print(f"WARNING: {p} has {n} actions and no stored action_cfg; "
                      f"cannot infer its action set -- pass the flags manually")
                continue
        if n > best_n:
            best, best_n = cfg, n
    return best


class ModelBank:
    """Lazily loads and caches one policy per level from <model_dir>/L<n>/.

    for_level(n) returns the net for level n, or None if that level has no
    loadable checkpoint (caller decides the fallback)."""

    def __init__(self, env, device, model_dir):
        self.env, self.device, self.model_dir = env, device, model_dir
        self._cache = {}

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        # Build the net with the checkpoint's OWN action count, and pair it
        # with the action TABLE its indices were trained against (from the
        # stored action_cfg, or inferred for pre-stamp checkpoints). The
        # relay swaps env.actions to this table whenever the model is active,
        # so models trained with DIFFERENT action sets coexist safely --
        # index meaning travels with the model, not with the env.
        n_act = ckpt["model"]["policy_head.weight"].shape[0]
        cfg = ckpt.get("action_cfg")
        if cfg is not None:
            fine = int(cfg.get("fine_walk_frames", 0))
            extra = tuple(cfg.get("extra_charges", ()))
            wait = tuple(cfg.get("wait_frames", ()))
            wjump = tuple(cfg.get("wind_jump", ()))
            approach = tuple(tuple(a) for a in cfg.get("approach_jump", ()))
            wcombo = tuple(tuple(tuple(s) for s in r)
                           for r in cfg.get("wind_combo", ()))
            wind = bool(cfg.get("wind_obs", False))
            vel = bool(cfg.get("vel_obs", False))
            chans = cfg.get("grid_channels")      # None = the legacy triple
            xconv = bool(cfg.get("extra_conv", False))
            sembed = int(cfg.get("scalar_embed", 0))
            settle = bool(cfg.get("settle_action", False))
        elif n_act in KNOWN_ACTION_CFGS:
            fine, extra = KNOWN_ACTION_CFGS[n_act]
            wait, wjump, approach, wcombo, wind = (), (), (), (), False
            vel, chans = False, None
            xconv, sembed = False, 0
            settle = False
        else:
            raise SystemExit(
                f"{path}: {n_act} actions and no stored action_cfg -- cannot "
                f"reconstruct its action table.")
        table = build_action_table(fine_walk_frames=fine, extra_charges=extra,
                                   wait_frames=wait, wind_jump=wjump,
                                   approach_jump=approach, wind_combo=wcombo,
                                   settle_action=settle)
        if len(table) != n_act:
            raise SystemExit(
                f"{path}: action_cfg gives {len(table)} actions but the "
                f"policy head has {n_act} -- checkpoint metadata is corrupt.")
        # One selector per model narrows the env's SUPERSET observation to the
        # grid channels and scalar columns this checkpoint was trained on. The
        # old code took the first obs_dim entries instead, which is only correct
        # when a model's scalars are a PREFIX of the env's -- true for wind
        # models, false for velocity models (their vx,vy sit after the wind
        # pair), so ice checkpoints could not be played at all.
        sel = obs_selector(self.env, chans, wind, vel, os.path.basename(path),
                           cfg.get("vel_encoding", "xy") if cfg else "xy")
        net = ActorCritic(sel.obs_dim, n_act,
                          grid_shape=sel.grid_shape,
                          n_scalars=sel.n_scalars,
                          extra_conv=xconv, scalar_embed=sembed).to(self.device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        print(f"loaded {path} (trained to step {ckpt.get('step', '?')}, "
              f"{n_act} actions, extras={extra or 'none'}"
              f"{', wind-aware' if wind else ''}"
              f"{', velocity-aware' if vel else ''})")
        return net, table, sel

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


OVERLAY = {"Land": (108, 122, 96), "Ice": (128, 196, 232), "Snow": (226, 232, 240)}


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
    p.add_argument("--mute", action="store_true",
                   help="disable game audio (handled before pygame init)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking argmax")
    p.add_argument("--scale", type=int, default=2, help="window upscaling")
    p.add_argument("--level", type=int, default=None,
                   help="start every episode on this level (with --x/--y: at "
                        "that exact spot, e.g. a captured start state)")
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--world", type=str, default=None,
                   help="watch a model play a custom world built with "
                        "LevelEditor.py (levels/<name>.json)")
    p.add_argument("--start-states", type=str, default=None,
                   help="start each episode from a random state in this pool "
                        "(same file training used), instead of the bottom")
    p.add_argument("--fine-walk-frames", type=int, default=None,
                   help="override the auto-detected micro-walk setting "
                        "(default: read from the checkpoints themselves)")
    p.add_argument("--extra-charges", type=str, default=None,
                   help="override the auto-detected appended jump charges "
                        "(default: read from the checkpoints themselves)")
    args = p.parse_args()

    if not args.checkpoint and not args.model_dir:
        p.error("give --checkpoint, --model-dir, or both")

    # Auto-size the env action set from the models in use; flags override.
    candidates = [args.checkpoint]
    if args.model_dir:
        candidates += [latest_ckpt(d) for d in glob.glob(
            os.path.join(args.model_dir, "L*")) if os.path.isdir(d)]
    fine, extra = scan_action_cfg(candidates)
    if args.fine_walk_frames is not None:
        fine = args.fine_walk_frames
    if args.extra_charges is not None:
        extra = (tuple(int(c) for c in args.extra_charges.split(","))
                 if args.extra_charges else ())
    print(f"env action set: fine_walk_frames={fine} extra_charges={extra}")

    device = get_device()
    # The env ALWAYS produces the SUPERSET observation (wind AND velocity
    # scalars); each model selects what it was trained on (see ModelBank.load),
    # which is what lets plain, windy and icy models share one relay.
    env = JumpKingEnv(max_steps=args.max_steps, goal_level=args.goal_level,
                      start_states=args.start_states,
                      fine_walk_frames=fine,
                      extra_charges=extra,
                      wind_jump=(0, 4),
                      wind_obs=True, vel_obs=True, vel_encoding="both",
                      grid_channels=SUPPORTED_CHANNELS)
    if args.world:
        import CustomWorld
        CustomWorld.apply_world(env, CustomWorld.load_world(args.world))

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
    hud_font = pygame.font.Font(None, 22)

    # Custom / generated screens carry no art -- the engine still paints the
    # ORIGINAL level's background behind them, so without this overlay the real
    # collision geometry is invisible and the painted ledges you can see are not
    # solid. PlayWorld.py does the same thing for human play; H toggles it.
    show_overlay = [args.world is not None]

    def handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                show_overlay[0] = not show_overlay[0]

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
        if show_overlay[0]:
            for p in (env.levels.levels[env.levels.current_level].platforms or []):
                col = OVERLAY.get(getattr(p, "type", "Land"), OVERLAY["Land"])
                pygame.draw.rect(env.game_screen, col,
                                 pygame.Rect(p.x, p.y, p.width, p.height))
        scaled = pygame.transform.scale(env.game_screen, (w, h))
        window.blit(scaled, (0, 0))
        try:
            hud = hud_font.render(
                f"lvl {env.levels.current_level}  "
                f"x={int(env.king.rect_x)}  y={int(env.king.rect_y)}",
                True, (255, 255, 0))
            window.blit(hud, (6, 6))
        except Exception:
            pass
        pygame.display.flip()

    warned_levels = set()

    for ep in range(args.episodes):
        obs, _ = env.reset(**reset_kwargs)
        done = False
        ep_ret = 0.0
        best_alt = -1e9
        steps = 0
        net = None
        net_table = None
        net_select = None
        net_lvl = None
        held_lvl = -1            # the level whose model is in use
        held_own = False         # ...and whether it is that level's OWN

        while not done:
            handle_events()

            # ---- pick the policy for the level the king stands on ----------
            lvl = env.levels.current_level
            if lvl != net_lvl:
                cand = bank.for_level(lvl)
                if cand is not None:
                    if net is not None and cand[0] is not net:
                        print(f"\n>>> level {lvl}: switching to model L{lvl}")
                    net, net_table, net_select = cand
                    held_lvl, held_own = lvl, True
                elif held_own and lvl > held_lvl:
                    # A SPANNED SCREEN, NOT A GAP. Screens 24 and 41 are climbed
                    # THROUGH and never stood on -- the routes for 23 and 40
                    # cross them mid-route -- so no model was ever trained there.
                    # Keep the policy that is mid-route: reaching for the
                    # --checkpoint fallback here hands the screen to the levels
                    # 0-3 model, which has no idea where it is and falls. Only
                    # UPWARD (lvl > held_lvl): after a fall the held model knows
                    # nothing about the screen the king landed on, and that
                    # really is a break. Same rule as FullRelay.run_trial.
                    if lvl not in warned_levels:
                        print(f"\n>>> level {lvl}: no L{lvl} model -- crossed "
                              f"mid-route by L{held_lvl}'s policy")
                        warned_levels.add(lvl)
                elif fallback is not None:
                    if net is not None and net is not fallback[0]:
                        print(f"\n>>> level {lvl}: no L{lvl} model, using the "
                              f"--checkpoint fallback")
                    net, net_table, net_select = fallback
                    held_lvl, held_own = lvl, False
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
                # The env executes whatever table it holds: install the ACTIVE
                # model's table so its action indices mean what they meant in
                # training, regardless of what other models in the relay use.
                env.actions = net_table
                env.num_actions = len(net_table)
                net_lvl = lvl

            ob = torch.as_tensor(net_select(obs),
                                 device=device).float().unsqueeze(0)
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
                handle_events()
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
