#!/usr/bin/env python
"""Draw a level's START STATES over the level itself, numbered -- the picture
behind the reverse curriculum.

Each circle is one entry of a `starts/` pool: a spot the king is teleported to
at the beginning of an episode. They are numbered in pool order, which for a
demo-derived pool is the order along the proven route (0 = the level entry).

    python tools/thesis/build_start_states.py --level 33
    python tools/thesis/build_start_states.py --level 40 --starts starts/starts_L40_demo.json
    python tools/thesis/build_start_states.py --level 40 --demo starts/demo_L40_swap.json

EVERY STATE IS VERIFIED IN THE ENGINE before it is drawn: the king is teleported
there and has to settle on the level being drawn. This is not a formality --
`starts/starts_L40_demo.json` labels all ten of its states `"level": 40`, but
the L40 route climbs THROUGH screen 41, so five of them are really on 41 and
drawing them on the L40 screenshot would put circles in mid-air. States that
land somewhere else are reported and skipped (or drawn on their own screen with
--split).
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
from PIL import Image

from JK_Env import JumpKingEnv

OUT_DIR = "figures"
RING = (255, 225, 40)
RING_EDGE = (40, 30, 0)
TEXT = (255, 240, 120)


def default_pool(level):
    for pat in (f"starts/starts_L{level}.json",
                f"starts/starts_L{level}_demo.json",
                f"starts/starts_L{level}_broad.json"):
        if os.path.isfile(pat):
            return pat
    hits = sorted(glob.glob(f"starts/starts_L{level}*.json"))
    if not hits:
        raise SystemExit(f"no start-state pool for L{level} in starts/")
    return hits[0]


def states_from_pool(path, level):
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)
    if isinstance(pool, dict):                 # the global start_states.json
        pool = pool.get("states", pool.get(str(level), []))
    return [{"x": float(s["x"]), "y": float(s["y"]),
             "level": int(s.get("level", level)),
             "score": s.get("score")} for s in pool]


def states_from_demo(path):
    """The route's own decision points, which carry their TRUE level."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for s in d["steps"]:
        st = s["state_before"]
        out.append({"x": float(st["x"]), "y": float(st["y"]),
                    "level": int(st["level"]), "score": None})
    return out


def settle(env, st, level):
    """Teleport there and report where the king actually ends up."""
    env.reset(level=int(st.get("level", level)),
              rect_x=int(round(st["x"])), rect_y=int(round(st["y"])))
    return (int(env.levels.current_level),
            float(env.king.rect_x), float(env.king.rect_y))


def render(env, level, pts, scale, out, font, label, platforms=False):
    """The level's art with a numbered ring on every state."""
    env.reset(level=level)
    env.game_screen.fill((0, 0, 0))
    try:
        env.levels.blit1()
        env.levels.blit2()                     # no king: the circles ARE the kings
    except Exception:
        pass
    surf = env.game_screen
    if scale != 1:
        surf = pygame.transform.scale(
            surf, (env.screen_w * scale, env.screen_h * scale))

    if platforms:
        # THE ART AND THE COLLISION DISAGREE, and not rarely -- L15 state 7
        # stands on a 24x56 pillar the level simply does not draw, L33's whole
        # pool sits on 8 px pillars, L21 has four invisible ledges. So a start
        # state drawn over bare art looks like it is hovering even when the
        # engine has it firmly on the ground. Drawing the real rects is the only
        # honest picture; the TOP edge is drawn brightest because that edge is
        # the surface the king actually rests on.
        lay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        for pf in (env.levels.levels[level].platforms or []):
            r = pygame.Rect(pf.x * scale, pf.y * scale,
                            max(1, pf.width * scale), max(1, pf.height * scale))
            pygame.draw.rect(lay, (150, 220, 255, 45), r)
            pygame.draw.rect(lay, (150, 220, 255, 110), r, 1)
            pygame.draw.line(lay, (170, 235, 255, 235),
                             (r.left, r.top), (r.right - 1, r.top),
                             max(2, scale - 1))
        surf.blit(lay, (0, 0))

    r = max(6, 5 * scale)
    for i, (x, y) in pts:
        cx, cy = int(x * scale), int(y * scale)
        pygame.draw.circle(surf, RING_EDGE, (cx, cy), r + 2, 0)
        pygame.draw.circle(surf, RING, (cx, cy), r, max(2, scale))
        t = font.render(str(i), True, TEXT)
        # the number sits up-right of the ring, nudged back inside at the edges
        tx = min(cx + r + 2, surf.get_width() - t.get_width() - 2)
        ty = max(2, cy - r - t.get_height() + 2)
        shadow = font.render(str(i), True, (0, 0, 0))
        surf.blit(shadow, (tx + 1, ty + 1))
        surf.blit(t, (tx, ty))

    if label:
        bar = pygame.Surface((surf.get_width(), 6 + font.get_height()))
        bar.set_alpha(190)
        bar.fill((0, 0, 0))
        surf.blit(bar, (0, 0))
        surf.blit(font.render(label, True, TEXT), (6, 3))

    Image.frombytes("RGB", surf.get_size(),
                    pygame.image.tostring(surf, "RGB")).save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--starts", default=None, help="a starts/ pool json")
    ap.add_argument("--demo", default=None,
                    help="a demo json instead; its steps carry their own level")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--split", action="store_true",
                    help="also write a figure for every OTHER level the states "
                         "settle on (the spanning routes put states on two)")
    ap.add_argument("--no-platforms", dest="platforms",
                    action="store_false", default=True,
                    help="draw over the bare art. Off by default ON PURPOSE: "
                         "many levels do not paint their own ledges, so without "
                         "the collision rects the states look like they hover")
    ap.add_argument("--no-label", dest="label", action="store_false", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.demo:
        src, states = args.demo, states_from_demo(args.demo)
    else:
        src = args.starts or default_pool(args.level)
        states = states_from_pool(src, args.level)
    print(f"L{args.level}: {len(states)} states from {src}")

    env = JumpKingEnv(max_steps=10 ** 6)
    pygame.font.init()
    font = pygame.font.SysFont("consolas", max(11, 6 * args.scale), bold=True)

    by_level, dropped = {}, []
    for i, st in enumerate(states):
        lvl, x, y = settle(env, st, args.level)
        # THE RING GOES ON HIS FEET, NOT HIS BODY. A start state is a contact
        # point with a platform; the king's rect is 20x24, so centring the ring
        # puts it 12 px above the ledge -- 36 px at scale 3, which reads as a
        # circle hovering in mid-air over a level whose art does not draw its
        # own ledges clearly. Verified against the geometry: every L33 state
        # has a platform whose top equals rect_y + rect.height.
        cx = x + env.king.rect.width / 2.0
        cy = y + env.king.rect.height
        by_level.setdefault(lvl, []).append((i, (cx, cy)))
        if lvl != int(st.get("level", args.level)):
            dropped.append((i, st.get("level"), lvl))
        print(f"  {i:>2}: asked L{st.get('level', args.level)} "
              f"({st['x']:.0f},{st['y']:.0f}) -> settled L{lvl} "
              f"({x:.0f},{y:.0f})")

    if dropped:
        print("\nSTATES THAT ARE NOT ON THE LEVEL THEIR POOL CLAIMS:")
        for i, claimed, real in dropped:
            print(f"  state {i}: pool says L{claimed}, engine settles it on L{real}")

    os.makedirs(OUT_DIR, exist_ok=True)
    wrote = []
    targets = sorted(by_level) if args.split else [args.level]
    for lvl in targets:
        pts = by_level.get(lvl, [])
        if not pts:
            print(f"  (no state settles on L{lvl} -- nothing to draw)")
            continue
        out = (args.out if (args.out and lvl == args.level)
               else os.path.join(OUT_DIR, f"start_states_L{lvl}.png"))
        label = (f"L{lvl}  start states ({len(pts)})  "
                 f"{os.path.basename(src)}") if args.label else None
        wrote.append(render(env, lvl, pts, args.scale, out, font, label,
                            platforms=args.platforms))
        print(f"  {len(pts)} states -> {out}")
    env.close()
    for w in wrote:
        print(f"wrote {w}")


if __name__ == "__main__":
    sys.exit(main())
