#!/usr/bin/env python
"""
capture_helper.py - drive the king yourself and capture start states.

The training agent can't reach the level-4 exit on its own, so it never learns
the crossing and the curriculum deadlocks. This tool lets YOU place the king on
the exact ledge you want and bank it as a start state, giving the curriculum a
genuinely-near-the-goal checkpoint to drill from.

Run from the game root (folder with King.py):

    python capture_helper.py
    python capture_helper.py --start-states start_states.json   # load + extend it

Controls:
    LEFT / RIGHT          walk / steer
    SPACE (hold, release) charge and jump, exactly like the real game
    N                     teleport to the next existing checkpoint (sorted
                          highest-level-and-highest-on-screen first), so you can
                          jump straight up near the wall instead of climbing
                          from the bottom every time
    C                     CAPTURE the king's current spot (only works when
                          grounded & settled -- same gate the trainer uses)
    S                     save all captured points to the json on disk
    R                     reset to the bottom of level 0
    ESC / close window    quit (reminds you to save if unsaved)

Captured points are written in capture.py's format: {level_str: [[x, y], ...]}.
Existing points are preserved and de-duplicated on a coarse pixel grid.
"""

import os
os.environ["SDL_VIDEODRIVER"] = ""          # real window
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import json
import argparse
import pygame

# The env monkeypatches pygame.key.get_pressed to return its FAKED key dict, so
# we grab the real one now (this binding survives the env's reassignment) to
# read the actual keyboard for manual driving.
_REAL_GET_PRESSED = pygame.key.get_pressed

from JK_Env import JumpKingEnv

SCALE = 2
FPS = 60
DEDUP_GRID = 12


def load_points(path):
    """Return {level_int: [(x, y), ...]} from an existing json, or {}."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(data, dict) and "states" in data:
        data = data["states"]
    out = {}
    if isinstance(data, dict):
        for lvl, pts in data.items():
            try:
                li = int(lvl)
            except (ValueError, TypeError):
                continue
            for p in pts or []:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    out.setdefault(li, []).append((int(p[0]), int(p[1])))
    return out


def save_points(path, points):
    """Write {level_int: [(x,y)]} back out, merging with whatever is on disk."""
    disk = load_points(path)
    for lvl, pts in points.items():
        disk.setdefault(lvl, [])
        seen = {(x // DEDUP_GRID, y // DEDUP_GRID) for x, y in disk[lvl]}
        for x, y in pts:
            key = (x // DEDUP_GRID, y // DEDUP_GRID)
            if key not in seen:
                disk[lvl].append((x, y))
                seen.add(key)
    out = {str(lvl): [[x, y] for x, y in pts] for lvl, pts in sorted(disk.items())}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)
    total = sum(len(v) for v in out.values())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-states", type=str, default="start_states.json")
    args = ap.parse_args()

    env = JumpKingEnv(goal_level=None, max_steps=10**9)
    existing = load_points(args.start_states)
    # flat list of teleport targets, highest level + highest on screen first
    targets = sorted(
        [(lvl, x, y) for lvl, pts in existing.items() for (x, y) in pts],
        key=lambda t: (-t[0], t[1]))
    captured = {}          # level_int -> [(x, y)] captured THIS session
    t_idx = -1
    unsaved = 0

    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King - capture helper")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    env.reset(level=0)
    msg = "drive with arrows+space; C=capture  N=teleport  S=save  R=reset"

    def draw():
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1(); env.king.blitme(); env.babe.blitme(); env.levels.blit2()
        except Exception:
            pass
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))
        grounded = env.move_available()
        n_cap = sum(len(v) for v in captured.values())
        hud = [
            (f"lvl={env.levels.current_level}  x={int(env.king.rect_x)} "
             f"y={int(env.king.rect_y)}  grounded={'YES' if grounded else 'no'}",
             (255, 255, 0)),
            (f"captured this session: {n_cap}   unsaved: {unsaved}", (120, 255, 120)),
            (msg, (200, 200, 255)),
        ]
        for i, (txt, col) in enumerate(hud):
            window.blit(font.render(txt, True, col), (6, 6 + i * 22))
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
                    env.reset(level=0)
                    msg = "reset to bottom"
                elif e.key == pygame.K_n and targets:
                    t_idx = (t_idx + 1) % len(targets)
                    lvl, x, y = targets[t_idx]
                    env.reset(level=lvl, rect_x=x, rect_y=y)
                    msg = f"teleported to existing checkpoint L{lvl} ({x},{y})"
                elif e.key == pygame.K_c:
                    if env.move_available():
                        lvl = env.levels.current_level
                        x, y = int(env.king.rect_x), int(env.king.rect_y)
                        captured.setdefault(lvl, []).append((x, y))
                        unsaved += 1
                        msg = f"CAPTURED L{lvl} ({x},{y})  -- press S to save"
                    else:
                        msg = "can't capture mid-air -- land first"
                elif e.key == pygame.K_s:
                    if captured:
                        total = save_points(args.start_states, captured)
                        msg = f"saved -> {args.start_states}  (pool now {total} points)"
                        unsaved = 0
                    else:
                        msg = "nothing captured yet"

        # feed REAL keyboard into the env's faked key dict, then step physics
        keys = _REAL_GET_PRESSED()
        env._set_keys(space=keys[pygame.K_SPACE],
                      left=keys[pygame.K_LEFT],
                      right=keys[pygame.K_RIGHT])
        env._physics_frame()

        draw()
        clock.tick(FPS)

    if unsaved:
        print(f"WARNING: {unsaved} captured point(s) were not saved (you didn't press S).")
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()