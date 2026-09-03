#!/usr/bin/env python
"""
Play a custom world yourself, before training anything on it.

Same engine the agent trains on -- real King physics, real ice slip, real wind --
just driven by your keyboard instead of a policy. Use it to answer the question
that wastes the most time otherwise: *is this level actually possible?*

    python PlayWorld.py levels/tower.json
    python PlayWorld.py levels/tower.json --screen 1     # start on a later screen
    python PlayWorld.py                                  # the original game's levels

Controls
    LEFT / RIGHT ... walk, and steer the jump you are charging
    SPACE (hold) ... charge a jump; release to launch (longer hold = higher)
    R .............. respawn at the bottom      TAB .. jump to the next screen
    H .............. show/hide the platform overlay (ice and snow are colour-coded)
    ESC ............ quit
"""

import os
import sys
import argparse

# JK_Env sets SDL_VIDEODRIVER=dummy at import time so training can run headless.
# We need a REAL window here, so claim the variable first -- "" lets SDL pick the
# platform default, and JK_Env's setdefault then leaves it alone. (Setting
# SDL_VIDEODRIVER=dummy yourself still works, for headless testing.)
os.environ.setdefault("SDL_VIDEODRIVER", "")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # skip the mixer, keep the window

import pygame

# Grab the REAL keyboard function before JK_Env replaces it with the agent's
# fake key dict (it monkeypatches pygame.key.get_pressed at construction time).
_real_get_pressed = pygame.key.get_pressed

import CustomWorld as CW

HUD_BG = (0, 0, 0)
OVERLAY = {"Land": (108, 122, 96), "Ice": (128, 196, 232), "Snow": (226, 232, 240)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None,
                    help="world file (omit to play the original game's levels)")
    ap.add_argument("--screen", type=int, default=0, help="screen to start on")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    from JK_Env import JumpKingEnv
    if args.path:
        world = CW.load_world(args.path)
        env = CW.make_env(world, max_steps=10 ** 9)
        top = len(world["levels"]) - 1
        title = f"{os.path.basename(args.path)} — screen {{}} / {top}"
    else:
        world, top = None, 42
        env = JumpKingEnv(max_steps=10 ** 9)
        title = "Jump King — screen {} / 42"

    env.reset(level=args.screen)

    w, h = env.screen_w * args.scale, env.screen_h * args.scale
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — playtest")
    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()

    show_overlay = world is not None       # custom screens have no art, so start on
    best = env.levels.current_level

    def draw():
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception:
            pass
        if show_overlay:
            for p in (env.levels.levels[env.levels.current_level].platforms or []):
                col = OVERLAY.get(getattr(p, "type", "Land"), OVERLAY["Land"])
                pygame.draw.rect(env.game_screen, col,
                                 pygame.Rect(p.x, p.y, p.width, p.height))
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))

        lvl = env.levels.current_level
        windy = False
        try:
            windy = bool(env.levels.levels[lvl].weather.hasWind)
        except Exception:
            pass
        bar = pygame.Surface((w, 26)); bar.set_alpha(180); bar.fill(HUD_BG)
        window.blit(bar, (0, 0))
        txt = (f"{title.format(lvl)}   x={int(env.king.rect_x)} y={int(env.king.rect_y)}"
               f"   best={best}" + ("   WIND" if windy else ""))
        window.blit(font.render(txt, True, (255, 255, 0)), (8, 5))
        if lvl >= top:
            t = font.render("TOP SCREEN REACHED — this world is solvable", True, (126, 208, 140))
            window.blit(t, t.get_rect(center=(w // 2, h // 2)))
        hint = small.render("SPACE hold=charge  arrows=steer  R=respawn  TAB=next screen  H=overlay",
                            True, (170, 176, 190))
        window.blit(hint, (8, h - 20))
        pygame.display.flip()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    env.reset(level=args.screen)
                elif e.key == pygame.K_h:
                    show_overlay = not show_overlay
                elif e.key == pygame.K_TAB:
                    env.reset(level=min(top, env.levels.current_level + 1))

        # feed the REAL keyboard into the dict King reads
        keys = _real_get_pressed()
        env._keys.clear()
        for k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_UP):
            env._keys[k] = 1 if keys[k] else 0

        env._physics_frame()
        best = max(best, env.levels.current_level)
        draw()
        clock.tick(args.fps)

    pygame.quit()
    print(f"highest screen reached: {best} / {top}")
    if args.path and best >= top:
        print("the world is human-solvable -- worth training on:")
        print(f"  python CustomWorld.py --seed {args.path}")
        print(f"  python Train.py --world {args.path} "
              f"--start-states {CW.default_pool_path(args.path)} --curriculum")


if __name__ == "__main__":
    main()
