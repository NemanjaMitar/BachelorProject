#!/usr/bin/env python
"""Render the generated-screen policy climbing UNSEEN screens, to a GIF.

`GenEval.py` gives the number; this is the picture behind it. It plays the same
checkpoint on screens from a held-out pool and captures every physics frame, so
the jump arcs are visible and it is obvious the screens are not the game's own.

    python tools/gen_render.py --model checkpoints/worldgen_best.pt \
        --worlds levels/gen_eval --n 3
    python tools/gen_render.py --world levels/gen_eval/s900000_0007.json

One GIF per screen by default; `--combine` writes a single GIF that plays them
back to back, which is the one to put in a slide. `--only-solved` skips screens
this policy misses, so the file is a demonstration and not a blooper reel --
the honest success rate is `GenEval.py`'s job, not this one's.
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import torch
from PIL import Image

import CustomWorld
import WorldGen
from GenEval import load_policy
from PPO import get_device

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_try import as_climb_world

OUT_DIR = "figures"

# THE ENGINE'S ART IS NOT THE GEOMETRY. A generated screen reuses whatever
# background the original level index happens to have, and those painted
# ledges are NOT solid -- the generated platforms are. Drawing the real
# collision rects over the art is the only way a viewer can see what the
# king is actually landing on; PlayWorld.py does the same (its H key).
#
# Its muted palette is not reused, though: that one is meant to sit UNDER a
# player's attention on a live screen, and in a still frame of a forest
# background it disappears, so the engine's painted terraces -- which are NOT
# solid -- read as more solid than the platforms that are. The art is dimmed
# instead and the rects drawn in colours the background cannot contain.
OVERLAY = {"Land": (255, 206, 84), "Ice": (110, 205, 255),
           "Snow": (240, 248, 255)}
EDGE = (30, 24, 12)
DIM = 190                      # alpha of the black veil over the engine art
GROUND = (18, 20, 26)          # --no-art background: geometry and nothing else


def play_and_capture(env, net, ndim, device, world, x0, max_steps,
                     greedy, scale, every, font, label, overlay=True, art=True):
    """One attempt on one screen; returns (frames, crossed, actions)."""
    frames, n = [], [0]

    def draw(txt):
        n[0] += 1
        if n[0] % max(1, every):
            return
        env.game_screen.fill(GROUND if not art else (0, 0, 0))
        if art:
            try:
                env.levels.blit1()
                env.levels.blit2()
            except Exception:
                pass
        if overlay:
            if art:
                veil = pygame.Surface((env.screen_w, env.screen_h))
                veil.set_alpha(DIM)
                veil.fill((0, 0, 0))
                env.game_screen.blit(veil, (0, 0))
            lvl = env.levels.levels[env.levels.current_level]
            for pf in (lvl.platforms or []):
                r = pygame.Rect(pf.x, pf.y, pf.width, pf.height)
                col = OVERLAY.get(getattr(pf, "type", "Land"), OVERLAY["Land"])
                pygame.draw.rect(env.game_screen, col, r)
                pygame.draw.rect(env.game_screen, EDGE, r, 1)
        try:
            env.king.blitme()          # over the overlay, never under it
        except Exception:
            pass
        surf = env.game_screen
        if scale != 1:
            surf = pygame.transform.scale(
                surf, (env.screen_w * scale, env.screen_h * scale))
        lines = txt.split("\n")
        bar = pygame.Surface((surf.get_width(), 10 + 14 * len(lines)))
        bar.set_alpha(190)
        bar.fill((0, 0, 0))
        surf.blit(bar, (0, 0))          # the HUD sits on the sky, not in it
        for i, line in enumerate(lines):
            surf.blit(font.render(line, True, (255, 230, 60)), (10, 6 + 14 * i))
        frames.append(Image.frombytes("RGB", surf.get_size(),
                                      pygame.image.tostring(surf, "RGB")))

    obs, _ = env.reset(level=0, rect_x=int(x0), rect_y=WorldGen.FLOOR_Y - 60)
    crossed, taken = False, []
    for t in range(max_steps):
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(obs[:ndim], device=device)
                            .float().unsqueeze(0))
            a = int(logits.argmax(1).item() if greedy else
                    torch.distributions.Categorical(logits=logits).sample())
        act = env.actions[a]
        taken.append(act)
        hud = f"{label}\nstep {t + 1}: {act[0]} {act[1]} {act[2]}"
        obs, _, term, trunc, info = env.step(a, render_cb=lambda: draw(hud))
        if info["level"] >= 1:
            crossed = True
            break
        if term or trunc:
            break
    end = f"{label}\n{'CLIMBED OUT' if crossed else 'failed'}"
    for _ in range(12):                      # hold the last frame
        n[0] = 0
        draw(end)
    return frames, crossed, taken


def save_gif(frames, out, fps, colors):
    """Write the GIF, optionally on a shared quantised palette.

    Pillow re-derives a palette per frame otherwise, which on these
    frames produced a 34 MB file for 12 screens. Quantising the first
    frame and mapping the rest onto its palette keeps the geometry exact
    (it is a handful of flat colours) and cuts that by ~15x."""
    if colors:
        base = frames[0].quantize(colors=colors)
        frames = [base] + [f.quantize(palette=base) for f in frames[1:]]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/worldgen_best.pt")
    ap.add_argument("--worlds", default="levels/gen_eval",
                    help="a pool directory; --world names one file instead")
    ap.add_argument("--world", default=None)
    ap.add_argument("--screen", type=int, default=0,
                    help="which screen of --world to climb; the screen "
                         "above it is replaced by the generator's catch "
                         "floor, because the king enters the next screen "
                         "BELOW its bottom edge and a hand-drawn floor "
                         "there is a ceiling he bonks on")
    ap.add_argument("--n", type=int, default=3, help="screens to render")
    ap.add_argument("--skip", type=int, default=0,
                    help="start this far into the pool")
    ap.add_argument("--x0", type=int, default=None,
                    help="bottom-floor start x (default: the pool's middle one)")
    ap.add_argument("--attempts", type=int, default=3,
                    help="sampled retries per screen before giving up")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax instead of sampling (retries then do nothing)")
    ap.add_argument("--only-solved", action="store_true", default=True)
    ap.add_argument("--keep-failures", dest="only_solved", action="store_false")
    ap.add_argument("--combine", action="store_true",
                    help="one GIF with every screen back to back")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--every", type=int, default=4, help="capture every Nth frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--no-art", action="store_true",
                    help="drop the engine background entirely and draw only "
                         "the collision geometry -- the clearest picture of a "
                         "GENERATED screen, and a far smaller GIF, because a "
                         "dimmed photographic background defeats the palette")
    ap.add_argument("--colors", type=int, default=0,
                    help="quantise frames to this many colours before saving "
                         "(64 is plenty for the overlay view)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="hide the collision rects and show only the "
                         "engine art -- which is misleading on a "
                         "generated screen, so the overlay is on")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    env, net, ndim = load_policy(args.model, device)
    # A pool world already carries the catch screen; a hand-drawn file
    # does not, and playing it raw is why moj.json failed here while
    # gen_try.py solved it 3/3.
    pool = ([as_climb_world(args.world, args.screen)] if args.world
            else WorldGen.load_pool(args.worlds)[args.skip:])

    pygame.font.init()
    font = pygame.font.SysFont("consolas", 11)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.model))[0]

    all_frames, made, won = [], 0, 0
    for world in pool:
        if made >= args.n:
            break
        name = world.get("name", "screen")
        CustomWorld.apply_world(env, world)
        x0 = args.x0 if args.x0 is not None else WorldGen.SCREEN_W // 2
        rungs = len(WorldGen.ledge_tops(world, 0)) - 1
        # HOW MANY TIMES TO TRY and WHETHER TO SHOW A SCREEN IT NEVER GOT are
        # two different questions. Tying the retry to --only-solved meant
        # --keep-failures silently rendered a single sampled attempt and
        # labelled every screen "try 1" -- which is the one reading the GIF
        # must not give, since a missed jump on these screens only drops the
        # king back onto the floor and retrying is what the agent really does.
        shown = None
        for attempt in range(1 if args.greedy else args.attempts):
            label = (f"{name}  {rungs} rungs, UNSEEN in training"
                     f"   [{'greedy' if args.greedy else 'sampled'}]"
                     f"   climbed {won}/{made}"
                     + ("" if args.greedy else f"   try {attempt + 1}"))
            frames, ok, taken = play_and_capture(
                env, net, ndim, device, world, x0, args.max_steps,
                args.greedy, args.scale, args.every, font, label,
                overlay=not args.no_overlay, art=not args.no_art)
            print(f"  {name}: try {attempt + 1} -> "
                  f"{'CLIMBED OUT' if ok else 'failed'} in {len(taken)} actions")
            shown = frames                 # keep the last attempt played
            if ok:
                break
        frames = shown
        if args.only_solved and not ok:
            print(f"  {name}: skipped (this policy does not solve it)")
            continue
        made += 1
        won += bool(ok)
        if args.combine:
            all_frames += frames
        else:
            out = os.path.join(OUT_DIR, f"gen_{name}_{tag}.gif")
            save_gif(frames, out, args.fps, args.colors)
            print(f"    {len(frames)} frames -> {out} "
                  f"({os.path.getsize(out) / 1e6:.1f} MB)")
    env.close()

    if args.combine and all_frames:
        out = args.out or os.path.join(OUT_DIR, f"gen_unseen_{tag}.gif")
        save_gif(all_frames, out, args.fps, args.colors)
        print(f"{made} screens ({won} climbed), {len(all_frames)} frames "
              f"-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    elif args.combine:
        raise SystemExit("no screens rendered")


if __name__ == "__main__":
    sys.exit(main())
