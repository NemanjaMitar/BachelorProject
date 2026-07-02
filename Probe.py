#!/usr/bin/env python
"""
Diagnose why a captured "near-exit" state won't clear its level.

Run on the machine that HAS the game assets (images/, Audio/), from the game root:

    python Probe.py                  # defaults to level 0, x=193, y=16
    python Probe.py 0 193 16
    python Probe.py 4 55 88          # a level-4 left-launch spot, etc.
    python Probe.py 4 55 88 --render # WATCH it in a window instead of reading logs

For ONE strong action it prints (and with --render, shows) the per-frame rect_y
and current_level so you can see exactly what happens at the top edge. Read the
output like this:

  * level flickers 0 -> 1 -> 0 within a frame or two, rect_y stuck near 0:
        the _check_level if/elif bug is undoing the transition.
        Fix: change the second `if` in King._check_level to `elif`.

  * rect_y goes negative, wraps to ~360 (level 1), then keeps INCREASING past
    360 and the level drops back to 0:
        the king popped over the edge but fell back down through an open next
        level without landing -- this state is a knife-edge perch, not a real
        one-jump-from-safety spot. The validator is right to score it 0; capture
        a spot a jump BELOW a solid landing platform in the next level instead.

  * rect_y never goes negative at all:
        the jump isn't launching the king high enough from here (too little
        runway). Same remedy: capture from lower / from the actual launch platform.

Then it sweeps EVERY action and reports the resulting level, so you can see if
ANY single action clears durably from this spot. With --render you watch each
sweep action play out (the HUD top-left shows which action is running).
"""

import os
import sys
import argparse


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int, nargs="?", default=0)
    p.add_argument("x", type=int, nargs="?", default=193)
    p.add_argument("y", type=int, nargs="?", default=16)
    p.add_argument("--render", action="store_true",
                   help="show the probe in a real window (slower, visual)")
    p.add_argument("--fps", type=int, default=60,
                   help="render speed cap in --render mode (try 30 for slow-mo)")
    p.add_argument("--scale", type=int, default=2, help="window upscaling")
    p.add_argument("--action", type=int, default=None,
                   help="action index to trace (default: strongest jump)")
    return p


def make_renderer(env, scale, fps):
    """Real-window renderer. Returns (draw, hud) where draw() blits one frame
    and hud['msg'] is the caption text drawn top-left."""
    import pygame
    w, h = env.screen_w * scale, env.screen_h * scale
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — probe (ESC quits)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    hud = {"msg": ""}

    def draw():
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN
                                         and e.key == pygame.K_ESCAPE):
                env.close()
                sys.exit()
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception:
            pass
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))
        if hud["msg"]:
            window.blit(font.render(hud["msg"], True, (255, 255, 0)), (6, 6))
        k = env.king
        stat = font.render(
            f"lvl {env.levels.current_level}  x={int(k.rect_x)} y={int(k.rect_y)}",
            True, (120, 255, 120))
        window.blit(stat, (6, 28))
        pygame.display.flip()
        clock.tick(fps)

    return draw, hud


def drive_jump(env, action_idx, draw=None):
    """Replicate _apply_action's charge/release but WITHOUT the settle loop, so
    the caller can step frame-by-frame afterwards and watch the trajectory.
    Mirrors the env: direction key held during the WHOLE charge, so charges
    past maxJumpCount auto-fire toward the direction instead of straight up."""
    kind, direction, mag = env.actions[action_idx]
    if kind == "jump":
        hold_left = (direction == "left")
        hold_right = (direction == "right")
        for _ in range(mag):
            env._set_keys(space=True, left=hold_left, right=hold_right)
            env._physics_frame()
            if draw:
                draw()
            if env.levels.ending or env.king.isJump or env.king.isFalling:
                break
        env._set_keys(left=hold_left, right=hold_right)
        env._physics_frame()
        if draw:
            draw()
    else:
        for _ in range(mag):
            env._set_keys(left=(direction == "left"), right=(direction == "right"))
            env._physics_frame()
            if draw:
                draw()
    env._set_keys()


def trace(env, level, x, y, action_idx, max_frames=160, draw=None, hud=None):
    # env.reset teleports through the canonical JK_Env.teleport, which
    # already settles the King onto the ground.
    env.reset(level=level, rect_x=x, rect_y=y)
    gx, gy, gl = env.king.rect_x, env.king.rect_y, env.levels.current_level
    print(f"\n=== TRACE action {action_idx} {env.actions[action_idx]} "
          f"from grounded (x={gx}, y={gy}) on level {gl} ===")
    if hud is not None:
        hud["msg"] = f"TRACE {env.actions[action_idx]}"
    drive_jump(env, action_idx, draw=draw)
    last_lvl = env.levels.current_level
    for f in range(max_frames):
        env._physics_frame()
        if draw:
            draw()
        ry = env.king.rect_y
        lvl = env.levels.current_level
        note = ""
        if lvl != last_lvl:
            note = f"   <<< LEVEL {last_lvl} -> {lvl}"
            last_lvl = lvl
        if f < 70 or note:
            print(f"  f{f:3d}  rect_y={ry:8.1f}  level={lvl}{note}")
        if env.move_available():
            print(f"  --> settled: rect_y={ry:.1f}  level={lvl}  (after {f} frames)")
            break
    else:
        print(f"  --> still not settled after {max_frames} frames "
              f"(rect_y={env.king.rect_y:.1f}, level={env.levels.current_level})")


def sweep(env, level, x, y, draw=None, hud=None):
    print(f"\n=== SWEEP every action from (level={level}, x={x}, y={y}) ===")
    cleared = []
    for a in range(env.num_actions):
        if hud is not None:
            hud["msg"] = f"SWEEP act {a}/{env.num_actions - 1}  {env.actions[a]}"
        env.reset(level=level, rect_x=x, rect_y=y)
        _, _, _, _, info = env.step(a, render_cb=draw)
        mark = ""
        if info["level"] > level:
            mark = "  *** CLEARED ***"
            cleared.append(a)
        print(f"  act {a:2d} {str(env.actions[a]):22s} -> "
              f"level {info['level']}  (x={info['x']}, y={info['y']}){mark}")
    print(f"\n  actions that cleared durably: {cleared if cleared else 'NONE'}")


if __name__ == "__main__":
    args = build_argparser().parse_args()

    # SDL drivers must be chosen BEFORE JK_Env imports pygame.
    if args.render:
        os.environ["SDL_VIDEODRIVER"] = ""      # real window
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from JK_Env import JumpKingEnv

    env = JumpKingEnv(max_steps=10_000)
    draw, hud = make_renderer(env, args.scale, args.fps) if args.render else (None, None)

    if args.action is not None:
        probe_action = args.action
    else:
        # Pick the strongest available jump for the trace (most likely to clear).
        jumps = [(i, a[2]) for i, a in enumerate(env.actions) if a[0] == "jump"]
        probe_action = max(jumps, key=lambda t: t[1])[0]

    trace(env, args.level, args.x, args.y, probe_action, draw=draw, hud=hud)
    sweep(env, args.level, args.x, args.y, draw=draw, hud=hud)
    env.close()
