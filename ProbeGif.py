#!/usr/bin/env python
"""
Record an animated GIF of the action sweep: from ONE grounded state the king
tries EVERY action in the table, one after another, and the clip labels each
attempt and whether it cleared the level.

This is Probe.py's `sweep` with a recorder attached -- same env, same macro
actions the policy chooses from -- so the GIF shows exactly the decision set a
model faces at that state.

    python ProbeGif.py                          # level 4, the screenshot spot
    python ProbeGif.py --level 4 --x 119 --y 289
    python ProbeGif.py --level 12 --x 200 --y 250 --scale 2 --out figures/L12.gif
    python ProbeGif.py --actions 0-6,14         # only some actions
    python ProbeGif.py --no-hitboxes --no-trail

Frame budget: only every `--every`-th physics frame is captured, so a 60 fps
jump becomes a few frames. Keep the GIF small by raising --every or narrowing
--actions.
"""

import os
import sys
import argparse


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--x", type=int, default=119)
    p.add_argument("--y", type=int, default=289)
    p.add_argument("--out", default=None,
                   help="output .gif (default figures/probe_L<lvl>_actions.gif)")
    p.add_argument("--actions", default=None,
                   help="which action indices, e.g. '0-6,14,20' (default: all)")
    p.add_argument("--every", type=int, default=3,
                   help="capture every Nth physics frame (default 3)")
    p.add_argument("--fps", type=int, default=20, help="GIF playback fps")
    p.add_argument("--scale", type=int, default=1, help="pixel upscale (1 = 480x360)")
    p.add_argument("--hold", type=int, default=8,
                   help="frames to freeze on each action's result")
    p.add_argument("--max-frames-per-action", type=int, default=40)
    p.add_argument("--trail", dest="trail", action="store_true", default=True,
                   help="draw the king's flight path (default on)")
    p.add_argument("--no-trail", dest="trail", action="store_false")
    p.add_argument("--hitboxes", dest="hitboxes", action="store_true", default=True,
                   help="outline platform hitboxes in red (default on)")
    p.add_argument("--no-hitboxes", dest="hitboxes", action="store_false")
    p.add_argument("--fine-walk-frames", type=int, default=0,
                   help="match a model's action table (see Probe.py)")
    p.add_argument("--extra-charges", default="",
                   help="comma list of appended jump charges (match training)")
    return p


def parse_actions(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [a for a in out if 0 <= a < n]


def label(action):
    kind, direction, mag = action
    if kind == "jump":
        return f"jump {direction} charge {mag}"
    if kind == "walk":
        return f"walk {direction} {mag}f"
    return f"{kind} {direction} {mag}"


def main():
    args = build_argparser().parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from PIL import Image
    from JK_Env import JumpKingEnv

    extra = (tuple(int(c) for c in args.extra_charges.split(","))
             if args.extra_charges else ())
    env = JumpKingEnv(max_steps=10_000, fine_walk_frames=args.fine_walk_frames,
                      extra_charges=extra)
    # Environment() wipes this in its constructor, so set it AFTER the env exists.
    os.environ["hitboxes"] = "1" if args.hitboxes else ""

    todo = parse_actions(args.actions, env.num_actions)
    out = args.out or f"figures/probe_L{args.level}_actions.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    font = pygame.font.Font(None, 20)
    small = pygame.font.Font(None, 17)
    W, H = env.screen_w, env.screen_h
    frames = []          # PIL RGB frames
    trail = []           # king positions inside the current action
    hud = {"top": "", "result": None}

    def draw_scene():
        env.game_screen.fill((0, 0, 0))
        try:
            env.levels.blit1()
            env.king.blitme()
            env.babe.blitme()
            env.levels.blit2()
        except Exception:
            pass
        if args.trail and len(trail) > 1:
            for i, (tx, ty) in enumerate(trail[:-1]):
                # older points fade toward the background
                f = i / max(1, len(trail) - 1)
                col = (int(90 + 165 * f), int(90 + 60 * f), int(230 - 90 * f))
                pygame.draw.circle(env.game_screen, col,
                                   (int(tx) + 8, int(ty) + 12), 1)
        env.game_screen.blit(font.render(hud["top"], True, (255, 255, 60)), (6, 5))
        k = env.king
        env.game_screen.blit(
            small.render(f"level {env.levels.current_level}   "
                         f"x={int(k.rect_x)} y={int(k.rect_y)}",
                         True, (150, 255, 150)), (6, 24))
        if hud["result"]:
            text, col = hud["result"]
            surf = font.render(text, True, col)
            box = surf.get_rect()
            box.center = (W // 2, H - 22)
            pad = box.inflate(14, 8)
            shade = pygame.Surface(pad.size)
            shade.set_alpha(170)
            shade.fill((0, 0, 0))
            env.game_screen.blit(shade, pad.topleft)
            env.game_screen.blit(surf, box.topleft)

    def snap():
        im = Image.frombytes("RGB", (W, H),
                             pygame.image.tostring(env.game_screen, "RGB"))
        if args.scale != 1:
            im = im.resize((W * args.scale, H * args.scale), Image.NEAREST)
        frames.append(im)

    cleared = []
    for a in todo:
        env.reset(level=args.level, rect_x=args.x, rect_y=args.y)
        trail.clear()
        hud["result"] = None
        hud["top"] = f"action {a}/{env.num_actions - 1}   {label(env.actions[a])}"

        counter = {"f": 0, "kept": 0}

        def grab():
            counter["f"] += 1
            trail.append((env.king.rect_x, env.king.rect_y))
            if counter["f"] % args.every:
                return
            if counter["kept"] >= args.max_frames_per_action:
                return
            counter["kept"] += 1
            draw_scene()
            snap()

        draw_scene()          # the grounded starting pose
        snap()
        _, _, _, _, info = env.step(a, render_cb=grab)

        if info["level"] > args.level:
            hud["result"] = (f"CLEARED -> level {info['level']}", (120, 255, 120))
            cleared.append(a)
        elif info["level"] < args.level:
            hud["result"] = (f"FELL -> level {info['level']}", (255, 110, 110))
        else:
            hud["result"] = (f"stays on level {info['level']}", (210, 210, 210))
        draw_scene()
        for _ in range(args.hold):
            snap()

        print(f"  act {a:2d} {label(env.actions[a]):22s} -> level {info['level']}"
              f"  (x={info['x']}, y={info['y']})"
              f"{'  *** CLEARED ***' if a in cleared else ''}")
        sys.stdout.flush()

    env.close()

    # One shared palette for the whole clip: per-frame palettes make the GIF
    # bigger and make untouched background pixels flicker.
    base = frames[0].quantize(colors=255, method=Image.MEDIANCUT)
    pal = [f.quantize(palette=base, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True,
                disposal=1)
    size = os.path.getsize(out) / 1e6
    print(f"\n  actions that cleared: {cleared if cleared else 'NONE'}")
    print(f"  wrote {out}  ({len(frames)} frames, {size:.1f} MB)")


if __name__ == "__main__":
    main()
