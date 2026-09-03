#!/usr/bin/env python
"""
Capture curriculum start-states by playing Jump King with the keyboard.

Run from the game root:
    python Capture.py

Movement (standard Jump King):
    LEFT / RIGHT      walk
    hold SPACE        crouch / charge a jump
    release SPACE     jump (hold a direction while releasing to jump that way)

Fly mode (no need to actually beat the game):
    F                 toggle fly mode on/off
    ARROW KEYS        fly freely in any direction (hold LSHIFT to fly faster)
                      flying past the top/bottom edge changes level automatically

Capture hotkeys:
    P                 record the king's CURRENT grounded position as a checkpoint
                      (exit fly mode and land on a platform first)
    S                 save all recorded checkpoints to starts/start_states.json
    BACKSPACE         delete the last recorded checkpoint
    ESC / close       save and quit

Workflow: press F, fly up to the level you want, hover just above a platform,
press F again (the king drops and lands), then press P. The level each
checkpoint belongs to is recorded automatically. Capture a few spots per level
at different heights so the curriculum can start the agent low, middle, high.
"""

import os
# Real window + silent audio. MUST be set before JK_Env imports pygame.
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
import json
import pygame

# Save the REAL keyboard function before the env monkeypatches it.
import JK_Env
_real_get_pressed = pygame.key.get_pressed

OUT = "starts/start_states.json"
SCALE = 2
FLY_SPEED = 10        # px per frame in fly mode
FLY_SPEED_FAST = 25   # with LSHIFT held


def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            raw = json.load(f)
        return {int(k): [tuple(p) for p in v] for k, v in raw.items()}
    return {}


def save(states):
    serial = {str(k): [list(p) for p in v] for k, v in sorted(states.items())}
    with open(OUT, "w") as f:
        json.dump(serial, f, indent=2)
    total = sum(len(v) for v in states.values())
    print(f"\nsaved {total} checkpoints across {len(states)} levels -> {OUT}")


def set_fly(king, on):
    """Toggle the engine's built-in creative mode. King.update() switches on
    os.environ['mode']: anything != 'normal' skips all physics and moves the
    king with the arrow keys (King._creative), while _check_level still
    handles level transitions at the top/bottom edges."""
    if on:
        os.environ["mode"] = "creative"
    else:
        os.environ["mode"] = "normal"
        # Hand the king back to physics cleanly: drop whatever momentum /
        # charge state was frozen when fly mode was entered, and mark him
        # airborne so gravity lands him before the keyboard is read again.
        king.speed, king.angle = 0, 0
        king.jumpCount = 0
        king.isCrouch = False
        king.isWalk = False
        king.isSplat = False
        king.isFalling = True


def main():
    env = JK_Env.JumpKingEnv(max_steps=10**9)   # huge limit: we control it manually
    pygame.key.get_pressed = _real_get_pressed  # hand the keyboard back to the human

    env.reset(level=0)                          # start at the very bottom

    w, h = env.screen_w * SCALE, env.screen_h * SCALE
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Jump King — checkpoint capture (F=fly  P=record  S=save)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    states = load_existing()
    order = []   # remember insertion order so BACKSPACE can undo
    if states:
        print(f"loaded {sum(len(v) for v in states.values())} existing checkpoints")

    king = env.king
    flying = False
    last_msg = ""

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
                elif event.key == pygame.K_f:
                    flying = not flying
                    set_fly(king, flying)
                    last_msg = "FLY MODE ON" if flying else "fly mode off - landing"
                    print(last_msg)
                elif event.key == pygame.K_p:
                    if flying:
                        last_msg = "in FLY mode - press F and land first"
                        print(last_msg)
                    elif env.move_available():   # only record grounded, settled spots
                        lvl = env.levels.current_level
                        pt = (int(king.rect_x), int(king.rect_y))
                        states.setdefault(lvl, []).append(pt)
                        order.append(lvl)
                        last_msg = f"recorded  level {lvl}  x={pt[0]} y={pt[1]}"
                        print(last_msg)
                    else:
                        last_msg = "NOT grounded - move to a ledge first"
                        print(last_msg)
                elif event.key == pygame.K_s:
                    save(states)
                    last_msg = "saved"
                elif event.key == pygame.K_BACKSPACE:
                    if order:
                        lvl = order.pop()
                        states[lvl].pop()
                        if not states[lvl]:
                            del states[lvl]
                        last_msg = f"deleted last (level {lvl})"
                        print(last_msg)

        # LSHIFT = fast flight (King._creative reads creative_speed each frame).
        if flying:
            mods = pygame.key.get_mods()
            king.creative_speed = FLY_SPEED_FAST if mods & pygame.KMOD_LSHIFT else FLY_SPEED

        # Advance one game frame with the REAL keyboard driving the king.
        env._physics_frame()

        # Don't let fly mode escape the map at the very bottom.
        if env.levels.current_level < 0:
            env.levels.current_level = 0
            king.rect_y = env.screen_h - king.rect_height

        # Render
        try:
            env.levels.update_hiddenwalls(king)   # fade fake walls like the real game
        except Exception:
            pass
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
            n = sum(len(v) for v in states.values())
            mode_tag = "FLY" if flying else "play"
            hud = font.render(
                f"[{mode_tag}] lvl {env.levels.current_level}"
                f"  x={int(king.rect_x)} y={int(king.rect_y)}"
                f"  |  checkpoints: {n}  |  F=fly P=record S=save",
                True, (255, 255, 0))
            window.blit(hud, (6, 6))
            if last_msg:
                msg = font.render(last_msg, True, (120, 255, 120))
                window.blit(msg, (6, 28))
        except Exception:
            pass

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
