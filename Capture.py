#!/usr/bin/env python
"""
Capture curriculum start-states by playing Jump King with the keyboard.

Run from the game root:
    python capture.py

Controls (standard Jump King):
    LEFT / RIGHT      walk
    hold SPACE        crouch / charge a jump
    release SPACE     jump (hold a direction while releasing to jump that way)

Capture hotkeys:
    P                 record the king's CURRENT grounded position as a checkpoint
    S                 save all recorded checkpoints to start_states.json
    BACKSPACE         delete the last recorded checkpoint
    ESC / close       save and quit

Strategy: climb level 0, and every time you're standing somewhere useful
(especially near the TOP / the exit to level 1), press P. Capture a few spots
at different heights so the curriculum can start the agent low, middle, and
high. Then do the same for levels 1, 2, ... as far as you can play.

The level each checkpoint belongs to is recorded automatically.
"""

import os
# Real window + silent audio. MUST be set before jk_env imports pygame.
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import json
import pygame

# Save the REAL keyboard function before the env monkeypatches it.
import JK_Env
_real_get_pressed = pygame.key.get_pressed

OUT = "start_states.json"
SCALE = 2


def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            raw = json.load(f)
        return {int(k): [tuple(p) for p in v] for k, v in raw.items()}
    return {}


def save(states):
    serial = {str(k): [list(p) for p in v] for k, v in states.items()}
    with open(OUT, "w") as f:
        json.dump(serial, f, indent=2)
    total = sum(len(v) for v in states.values())
    print(f"\nsaved {total} checkpoints across {len(states)} levels -> {OUT}")


def main():
    env = JK_Env.JumpKingEnv(max_steps=10**9)   # huge limit: we control it manually
    pygame.key.get_pressed = _real_get_pressed  # hand the keyboard back to the human

    env.reset(level=0)                          # start at the very bottom

    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — checkpoint capture (P=record  S=save)")
    clock = pygame.time.Clock()

    states = load_existing()
    order = []   # remember insertion order so BACKSPACE can undo
    if states:
        print(f"loaded {sum(len(v) for v in states.values())} existing checkpoints")

    king = env.king
    last_recorded = ""

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save(states)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    save(states)
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_p:
                    if env.move_available():     # only record grounded, settled spots
                        lvl = env.levels.current_level
                        pt = (int(king.rect_x), int(king.rect_y))
                        states.setdefault(lvl, []).append(pt)
                        order.append(lvl)
                        last_recorded = f"recorded  level {lvl}  x={pt[0]} y={pt[1]}"
                        print(last_recorded)
                    else:
                        last_recorded = "NOT grounded - move to a ledge first"
                        print(last_recorded)
                elif event.key == pygame.K_s:
                    save(states)
                    last_recorded = "saved"
                elif event.key == pygame.K_BACKSPACE:
                    if order:
                        lvl = order.pop()
                        states[lvl].pop()
                        if not states[lvl]:
                            del states[lvl]
                        last_recorded = f"deleted last (level {lvl})"
                        print(last_recorded)

        # Advance one game frame with REAL keyboard driving the king.
        env._physics_frame()

        # Render
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception as e:
            print("draw warning:", e)
        window.blit(pygame.transform.scale(env.game_screen, (w, h)), (0, 0))

        # On-screen HUD
        try:
            font = pygame.font.Font(None, 22 * 1)
            n = sum(len(v) for v in states.values())
            hud = font.render(
                f"lvl {env.levels.current_level}  x={int(king.rect_x)} y={int(king.rect_y)}"
                f"  |  checkpoints: {n}  |  P=record S=save",
                True, (255, 255, 0))
            window.blit(hud, (6, 6))
            if last_recorded:
                msg = font.render(last_recorded, True, (120, 255, 120))
                window.blit(msg, (6, 28))
        except Exception:
            pass

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()