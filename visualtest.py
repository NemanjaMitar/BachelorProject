#!/usr/bin/env python
"""visualtest.py - WATCH each macro-action play out in a real window.

Run from the game root:  python visualtest.py

A real window opens. The script drives the king through a scripted list of
actions, rendering EVERY physics frame so you see the charge, the launch, the
arc, and the landing for each action. Between actions it pauses:

    SPACE / RIGHT ARROW   -> run the next action
    R                     -> reset to the bottom and restart the script
    T                     -> jump to the checkpoint phase (teleport test)
    ESC / close window    -> quit

The top caption shows the current action and the king's live (x, y, level).
This lets you see directly whether a jump fires, where it lands, and -- in the
checkpoint phase -- whether the teleport actually places a working king.
"""

import os
# Real window, silent audio. MUST be set before jk_env imports pygame.
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import pygame
from JK_Env import JumpKingEnv

SCALE = 2
FPS = 60

# Scripted actions to watch from the BOTTOM. Indices into env.actions:
#   0-6   jump left  (charges 4,8,12,16,20,26,32)
#   7-13  jump up
#   14-20 jump right
#   21 walk left, 22 walk right
# This sequence deliberately tests directional jumps at a charge BELOW
# maxJumpCount (20) and ABOVE it (32) in both directions, so you can confirm
# the king now steers left/right instead of always launching straight up.
BOTTOM_SCRIPT = [22, 21,        # walk right, walk left
                 18,            # jump RIGHT charge 20  (<= maxJumpCount)
                 20,            # jump RIGHT charge 32  (> maxJumpCount, was broken)
                 4,             # jump LEFT  charge 20
                 6,             # jump LEFT  charge 32  (> maxJumpCount, was broken)
                 11]            # jump UP    charge 20
# Checkpoints to watch in the teleport phase: (x, y) on level 0
CHECKPOINTS = [(193, 16), (210, 16), (247, 16)]
CHECKPOINT_ACTIONS = [7, 9, 11, 13]   # up jumps, weak -> strong


def main():
    env = JumpKingEnv(goal_level=1, max_steps=10**9)
    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King - visual action test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    def render(caption):
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception as e:
            # cosmetic draw systems may complain; physics is unaffected
            pass
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))
        line1 = font.render(caption, True, (255, 255, 0))
        line2 = font.render(
            f"x={int(env.king.rect_x)} y={int(env.king.rect_y)} "
            f"lvl={env.levels.current_level}  |  SPACE=next  R=reset  T=teleport phase  ESC=quit",
            True, (120, 255, 120))
        window.blit(line1, (6, 6))
        window.blit(line2, (6, 28))
        pygame.display.flip()
        clock.tick(FPS)

    def poll_quit():
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()

    def wait_for_key(caption):
        """Render a static frame and wait. Returns 'next', 'reset', or 'teleport'."""
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if e.key in (pygame.K_SPACE, pygame.K_RIGHT):
                        return "next"
                    if e.key == pygame.K_r:
                        return "reset"
                    if e.key == pygame.K_t:
                        return "teleport"
            render(caption)

    def run_action(action_idx, label):
        """Mirror env._apply_action (the FIXED version) but render every frame."""
        kind, direction, magnitude = env.actions[action_idx]
        if kind == "jump":
            hold_left = (direction == "left")
            hold_right = (direction == "right")
            for _ in range(magnitude):
                env._set_keys(space=True, left=hold_left, right=hold_right)
                env._physics_frame()
                render(f"{label}  [charging {magnitude}]")
                poll_quit()
                if env.levels.ending or env.king.isJump or env.king.isFalling:
                    break
            env._set_keys(left=hold_left, right=hold_right)
            env._physics_frame()
            render(f"{label}  [released {direction}]")
        else:
            for _ in range(magnitude):
                env._set_keys(left=(direction == "left"), right=(direction == "right"))
                env._physics_frame()
                render(f"{label}  [walking]")
                poll_quit()
        env._set_keys()
        frames = 0
        while (not env.move_available()
               and frames < env.max_settle_frames
               and not env.levels.ending):
            env._physics_frame()
            render(f"{label}  [settling {frames}]")
            poll_quit()
            frames += 1

    # ----- run the script -------------------------------------------------
    phase = "bottom"
    while True:
        if phase == "bottom":
            env.reset(level=0)
            for a in BOTTOM_SCRIPT:
                cmd = wait_for_key(f"NEXT: {env.actions[a]}  (from bottom)")
                if cmd == "reset":
                    break
                if cmd == "teleport":
                    phase = "teleport"; break
                run_action(a, f"{env.actions[a]}")
            else:
                cmd = wait_for_key("bottom script done - SPACE to replay, T for teleport")
                if cmd == "teleport":
                    phase = "teleport"
        else:  # teleport phase
            for (cx, cy) in CHECKPOINTS:
                env.reset(level=0, rect_x=cx, rect_y=cy)
                cmd = wait_for_key(f"TELEPORTED to ({cx},{cy}) - does the king sit on a ledge?")
                if cmd == "reset":
                    phase = "bottom"; break
                for a in CHECKPOINT_ACTIONS:
                    run_action(a, f"{env.actions[a]} @cp({cx},{cy})")
                    cmd = wait_for_key(f"after {env.actions[a]} - SPACE to continue")
                    if cmd == "reset":
                        phase = "bottom"; break
                    if env.levels.current_level > 0:
                        wait_for_key(">>> reached level 1! SPACE to continue")
                        break
                if phase == "bottom":
                    break
            else:
                cmd = wait_for_key("teleport phase done - R to restart from bottom")
                if cmd == "reset":
                    phase = "bottom"


if __name__ == "__main__":
    main()