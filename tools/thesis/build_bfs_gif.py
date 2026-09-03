#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Animated BFS over ONE level's macro-action graph, drawn on the REAL screen.

This is LevelGraph.py's search with a recorder attached: same JumpKingEnv, same
macro-action table, same (x//grid, y//grid) node key, same trap rule, same
stop-on-first-goal-path. Nothing is staged -- every circle is a state the engine
actually settled the king into, and every line is one macro-action it executed.

Output (both from a single pass):
    figures/bfs_L<lvl>.gif          the clip for the slide
    figures/bfs_L<lvl>/NN_*.png     key frames, as a projector-proof backup

The clip goes: empty screen -> entry state -> one frame per BFS DEPTH (the
frontier grows as a wave) -> the exit crossing -> everything dims and the proven
path lights up. A HUD counts depth / nodes / paths so "exhaustive" is visible
rather than asserted.

    python tools/thesis/build_bfs_gif.py --level 14
    python tools/thesis/build_bfs_gif.py --level 9 --max-depth 4 --grid 6
    python tools/thesis/build_bfs_gif.py --level 14 --all-seeds --lang en

The defaults are the project's frozen action set (fine walks 3, extra charges
22,24,28,30 -> 37 actions), i.e. exactly what AutoPilot.py proves levels with.
By default the search is seeded ONLY from the entry state (the lowest pooled
state on the level), because that is the honest question -- "is the level
provable from where the king arrives?" -- and because a search seeded from a
near-exit state finishes at depth 1 and makes a boring picture.
"""

import os
import sys
import json
import math
import time
import argparse
from collections import deque

# repo root (this file lives in tools/thesis/), so JK_Env imports and the
# relative starts/ + figures/ defaults both work when run from the root.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


# --------------------------------------------------------------------------
# strings (the thesis figures are Serbian Cyrillic; --lang en for a talk in EN)
# --------------------------------------------------------------------------
TXT = {
    "sr": {
        "level":    "ниво",
        "depth":    "дубина",
        "nodes":    "чворова",
        "paths":    "путања",
        "exit":     "излаз → следећи ниво",
        "entry":    "улазно стање",
        "explored": "истражено стање",
        "trap":     "замка (краљ се не смири)",
        "path":     "доказана путања",
        "proved":   "доказана путања: {n} акција",
        "nopath":   "нема путање са овим скупом акција",
    },
    "en": {
        "level":    "level",
        "depth":    "depth",
        "nodes":    "nodes",
        "paths":    "paths",
        "exit":     "exit → next level",
        "entry":    "entry state",
        "explored": "explored state",
        "trap":     "trap (never settles)",
        "path":     "proven path",
        "proved":   "proven path: {n} actions",
        "nopath":   "no path with this action set",
    },
}

# palette: yellow frontier reads on the dark game art, green = proof, red = trap
COL = {
    "edge":   (150, 168, 186),
    "node":   (255, 210,  63),
    "nodeed": ( 60,  48,  10),
    "pop":    (255, 255, 225),
    "entry":  (255, 140,  60),
    "trap":   (235,  70,  60),
    "path":   ( 52, 214, 122),
    "exit":   ( 90, 235, 140),
    "hud":    (238, 238, 244),
    "mut":    (168, 176, 190),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=14)
    p.add_argument("--starts", default=None,
                   help="start pool (default starts/starts_L<lvl>.json)")
    p.add_argument("--seed", type=int, nargs=2, default=None, metavar=("X", "Y"),
                   help="explicit seed instead of the pool's entry state")
    p.add_argument("--all-seeds", action="store_true",
                   help="seed from EVERY pooled state (LevelGraph's default) "
                        "instead of the entry state only")
    # --- action table: the project's frozen set, same as AutoPilot.py ---
    p.add_argument("--charges", default=None)
    p.add_argument("--fine-walk-frames", type=int, default=3)
    p.add_argument("--extra-charges", default="22,24,28,30")
    # --- search budget ---
    p.add_argument("--grid", type=int, default=4, help="dedup raster in px")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-nodes", type=int, default=260)
    p.add_argument("--max-paths", type=int, default=1)
    # --- rendering ---
    p.add_argument("--out", default=None)
    p.add_argument("--png-dir", default=None)
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--scale", type=int, default=2, help="pixel upscale")
    p.add_argument("--layer-steps", type=int, default=3,
                   help="sub-frames per BFS depth (the frontier grows)")
    p.add_argument("--lang", choices=("sr", "en"), default="sr")
    p.add_argument("--no-legend", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------
# 1. the search (LevelGraph.py's, with every event recorded)
# --------------------------------------------------------------------------
def load_seeds(args, lvl):
    if args.seed:
        return [(args.seed[0], args.seed[1])]
    path = args.starts or f"starts/starts_L{lvl}.json"
    if not os.path.exists(path):
        raise SystemExit(f"no start pool at {path} -- pass --starts or --seed X Y")
    with open(path) as f:
        pool = json.load(f)
    pts = [(int(s["x"]), int(s["y"])) for s in pool
           if isinstance(s, dict) and int(s.get("level", -1)) == lvl]
    if not pts:
        raise SystemExit(f"{path} holds no states for level {lvl}")
    if args.all_seeds:
        return pts
    # the ENTRY state: largest y = lowest on the screen = where he arrives
    return [max(pts, key=lambda p: p[1])]


def search(env, args):
    """BFS identical to LevelGraph.py, returning the recording it walked."""
    lvl, goal = args.level, args.level + 1

    def key(x, y):
        return (int(x) // args.grid, int(y) // args.grid)

    nodes, order, q = {}, [], deque()
    edges, traps, exits = [], [], []

    for (sx, sy) in load_seeds(args, lvl):
        slvl, ax, ay = env.teleport(lvl, sx, sy)
        if slvl != lvl:
            print(f"  seed ({sx},{sy}) settles OFF-LEVEL -> skipped")
            continue
        k = key(ax, ay)
        if k not in nodes:
            nodes[k] = {"x": ax, "y": ay, "depth": 0, "parent": None, "act": None}
            order.append(k)
            q.append(k)
    if not nodes:
        raise SystemExit("no usable seed")
    entry_key = order[0]

    steps, goal_depth, t0 = 0, None, time.time()
    while q and len(nodes) < args.max_nodes and len(exits) < args.max_paths:
        k = q.popleft()
        nd = nodes[k]
        if goal_depth is not None and nd["depth"] > goal_depth:
            break                      # finish the layer the goal was found in
        if nd["depth"] >= args.max_depth:
            continue
        for a in range(env.num_actions):
            env.reset(level=lvl, rect_x=nd["x"], rect_y=nd["y"])
            _, _, term, _tr, info = env.step(a)
            steps += 1
            flvl, fx, fy = info["level"], int(info["x"]), int(info["y"])
            if flvl >= goal:
                exits.append({"u": k, "act": a, "depth": nd["depth"] + 1,
                              "fx": fx, "fy": fy})
                goal_depth = nd["depth"]
                print(f"  PATH found at depth {nd['depth'] + 1} "
                      f"(action {a}: {env.actions[a]}) -> level {flvl}")
                break                  # LevelGraph stops this node here too
            if flvl < lvl:
                continue               # fell off the bottom: dead branch
            if term and not info.get("success") and not env.move_available():
                traps.append({"u": k, "act": a, "depth": nd["depth"] + 1,
                              "fx": fx, "fy": fy})
                continue               # never settled: physics trap
            nk = key(fx, fy)
            if nk not in nodes:
                nodes[nk] = {"x": fx, "y": fy, "depth": nd["depth"] + 1,
                             "parent": k, "act": a}
                order.append(nk)
                edges.append({"u": k, "v": nk})
                q.append(nk)
        if steps % 200 < env.num_actions:
            print(f"  ... {steps} macro-steps, {len(nodes)} nodes, "
                  f"{time.time() - t0:.0f}s")
            sys.stdout.flush()

    # walk the parent chain back from the exit node
    path_keys, path_acts = [], []
    if exits:
        k = exits[0]["u"]
        while k is not None:
            path_keys.append(k)
            path_acts.append(nodes[k]["act"])
            k = nodes[k]["parent"]
        path_keys.reverse()
        path_acts.reverse()
        path_acts = path_acts[1:] + [exits[0]["act"]]   # drop the seed's None

    print(f"  done: {steps} macro-steps in {time.time() - t0:.0f}s | "
          f"{len(nodes)} nodes | {len(traps)} trap edges | {len(exits)} paths")
    return {"nodes": nodes, "order": order, "edges": edges, "traps": traps,
            "exits": exits, "entry": entry_key,
            "path_keys": path_keys, "path_acts": path_acts,
            "max_depth": max(n["depth"] for n in nodes.values())}


# --------------------------------------------------------------------------
# 2. the drawing
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    import pygame
    import pygame.gfxdraw as gfx
    from PIL import Image
    from JK_Env import JumpKingEnv

    T = TXT[args.lang]
    kw = {"max_steps": 10_000, "fine_walk_frames": args.fine_walk_frames,
          "max_settle_frames": 1000}
    if args.charges:
        kw["charges"] = tuple(int(c) for c in args.charges.split(","))
    if args.extra_charges:
        kw["extra_charges"] = tuple(int(c) for c in args.extra_charges.split(","))
    env = JumpKingEnv(**kw)
    lvl = args.level
    print(f"level {lvl} -> {lvl + 1} | {env.num_actions} actions "
          f"fine_walk={args.fine_walk_frames} grid={args.grid}px")

    G = search(env, args)
    nodes, order = G["nodes"], G["order"]

    # ---- background: the real screen, king parked at the entry state -------
    S = args.scale
    W, H = env.screen_w, env.screen_h
    en = nodes[G["entry"]]
    env.teleport(lvl, en["x"], en["y"])
    env.game_screen.fill((0, 0, 0))
    try:
        env.levels.blit1()
        env.king.blitme()
        env.babe.blitme()
        env.levels.blit2()
    except Exception:
        pass
    bg = pygame.transform.scale(env.game_screen.copy(), (W * S, H * S))
    SW, SH = bg.get_size()

    def fnt(sz, bold=True):
        return pygame.font.SysFont(
            "consolas,couriernew,segoeui,arial,dejavusans", int(sz), bold=bold)

    F_HUD, F_SM = fnt(11 * S), fnt(9 * S)
    NR = max(3, int(round(3.4 * S)))         # node radius

    def C(k):
        """node key -> centre of the king's sprite, in output pixels"""
        n = nodes[k]
        return (int((n["x"] + 8) * S), int((n["y"] + 12) * S))

    def rgba(c, a):
        return (c[0], c[1], c[2], max(0, min(255, int(a))))

    def disc(surf, p, r, col, a, ring=None, rw=2):
        if ring is None:
            gfx.filled_circle(surf, p[0], p[1], r, rgba(col, a))
            gfx.aacircle(surf, p[0], p[1], r, rgba(COL["nodeed"], a))
        else:
            for i in range(rw):
                gfx.aacircle(surf, p[0], p[1], r - i, rgba(col, a))

    def text(surf, s, xy, f, col, a=255, anchor="tl"):
        img = f.render(s, True, col)
        sh = f.render(s, True, (0, 0, 0))
        r = img.get_rect()
        setattr(r, {"tl": "topleft", "tr": "topright",
                    "mc": "center"}[anchor], xy)
        if a < 255:
            img = img.copy(); img.set_alpha(a)
            sh = sh.copy();   sh.set_alpha(a)
        surf.blit(sh, (r.x + 1, r.y + 1))
        surf.blit(img, r.topleft)
        return r

    def shade(surf, rect, a=150):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, a))
        surf.blit(s, rect.topleft)

    def arrow(surf, p, q, col, a, w, head=7):
        pygame.draw.line(surf, rgba(col, a), p, q, w)
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        h = head * S / 2.0
        pts = [q,
               (q[0] - h * math.cos(ang - 0.42), q[1] - h * math.sin(ang - 0.42)),
               (q[0] - h * math.cos(ang + 0.42), q[1] - h * math.sin(ang + 0.42))]
        gfx.filled_polygon(surf, [(int(x), int(y)) for x, y in pts], rgba(col, a))

    def compose(reveal=0, partial=1.0, pop=False, dim=False, path_upto=0,
                flash=False, banner=None):
        surf = bg.copy()
        ov = pygame.Surface((SW, SH), pygame.SRCALPHA)
        A = 0.30 if dim else 1.0                     # graph-layer alpha scale

        vis = [k for k in order if nodes[k]["depth"] <= reveal]
        if partial < 1.0:
            top = [k for k in vis if nodes[k]["depth"] == reveal]
            keep = max(1, int(round(partial * len(top))))
            drop = set(top[keep:])
            vis = [k for k in vis if k not in drop]
        visset, fresh = set(vis), set()
        if pop and reveal > 0:
            # only the layer being revealed right now flashes bright
            fresh = {k for k in vis if nodes[k]["depth"] == reveal}

        # edges
        for e in G["edges"]:
            if e["v"] in visset:
                pygame.draw.line(ov, rgba(COL["edge"], 150 * A),
                                 C(e["u"]), C(e["v"]), max(1, S // 2 + 1))
        # traps (only once their whole layer is settled)
        if partial >= 1.0:
            for t in G["traps"]:
                if t["depth"] > reveal or t["u"] not in visset:
                    continue
                x, y = C(t["u"])
                d = int(NR * 1.15)
                for dx, dy in ((1, 1), (1, -1)):
                    pygame.draw.line(ov, rgba(COL["trap"], 235 * A),
                                     (x - d * dx, y - d * dy),
                                     (x + d * dx, y + d * dy), max(2, S))
        # nodes
        for k in vis:
            if k == G["entry"]:
                continue
            p, isnew = C(k), k in fresh
            disc(ov, p, NR + (2 if isnew else 0),
                 COL["pop"] if isnew else COL["node"], 245 * A)
        # entry: a ring, so the king's sprite stays readable inside it
        disc(ov, C(G["entry"]), NR + 3 * S // 2, COL["entry"], 255 * A,
             ring=True, rw=max(2, S))

        # exit crossings
        if G["exits"] and (flash or path_upto or reveal >= G["exits"][0]["depth"]):
            e = G["exits"][0]
            a = 255 if (flash or path_upto) else 200
            arrow(ov, C(e["u"]), (int((e["fx"] + 8) * S), 2 * S),
                  COL["exit"], a * (1.0 if (flash or path_upto) else A),
                  max(2, S + 1))

        # the proven path, revealed node by node
        if path_upto:
            pk = G["path_keys"]
            for i in range(1, min(path_upto, len(pk))):
                arrow(ov, C(pk[i - 1]), C(pk[i]), COL["path"], 255, max(2, S + 1))
            for i in range(min(path_upto, len(pk))):
                disc(ov, C(pk[i]), NR + 1, COL["path"], 255)
            if path_upto > len(pk) and G["exits"]:
                e = G["exits"][0]
                arrow(ov, C(e["u"]), (int((e["fx"] + 8) * S), 2 * S),
                      COL["exit"], 255, max(3, S + 2))
        surf.blit(ov, (0, 0))

        # goal line across the top
        for x in range(0, SW, 8 * S):
            pygame.draw.line(surf, COL["exit"], (x, 3 * S),
                             (x + 4 * S, 3 * S), max(1, S))
        text(surf, T["exit"], (SW - 6 * S, 6 * S), F_SM, COL["exit"], anchor="tr")

        # HUD
        npaths = 1 if path_upto or (G["exits"] and reveal >= G["exits"][0]["depth"]) else 0
        l1 = f"{T['level']} {lvl} → {lvl + 1}"
        l2 = (f"{T['depth']} {reveal} · {len(vis)} {T['nodes']} "
              f"· {npaths} {T['paths']}")
        box = pygame.Rect(4 * S, 4 * S, max(F_HUD.size(l1)[0], F_HUD.size(l2)[0])
                          + 10 * S, 2 * F_HUD.get_height() + 6 * S)
        shade(surf, box, 140)
        text(surf, l1, (box.x + 5 * S, box.y + 3 * S), F_HUD, COL["hud"])
        text(surf, l2, (box.x + 5 * S, box.y + 3 * S + F_HUD.get_height()),
             F_HUD, COL["node"])

        # legend: a full-width strip along the bottom, so it never sits on top
        # of a node the way a floating box does
        floor = SH
        if not args.no_legend:
            rows = [(COL["entry"], T["entry"]), (COL["node"], T["explored"])]
            if G["traps"]:          # don't advertise a mark the screen lacks
                rows.append((COL["trap"], T["trap"]))
            if G["exits"]:
                rows.append((COL["path"], T["path"]))
            lh = F_SM.get_height() + 7 * S
            strip = pygame.Rect(0, SH - lh, SW, lh)
            shade(surf, strip, 170)
            widths = [F_SM.size(s)[0] + 14 * S for _, s in rows]
            gap = max(2 * S, (SW - sum(widths)) // (len(rows) + 1))
            x, cy = gap, strip.y + lh // 2
            for (c, s), w in zip(rows, widths):
                disc(surf, (x + 4 * S, cy), max(3, 2 * S), c, 255)
                text(surf, s, (x + 11 * S, cy - F_SM.get_height() // 2),
                     F_SM, COL["mut"])
                x += w + gap
            floor = strip.y

        if banner:
            img = F_HUD.render(banner, True, COL["path"])
            r = img.get_rect(center=(SW // 2, floor - 11 * S))
            shade(surf, r.inflate(14 * S, 8 * S), 185)
            text(surf, banner, r.center, F_HUD, COL["path"], anchor="mc")
        return surf

    # ---- the storyboard ----------------------------------------------------
    frames, durs, keyframes = [], [], []

    def snap(surf, ms, keep=None):
        frames.append(Image.frombytes(
            "RGB", (SW, SH), pygame.image.tostring(surf, "RGB")))
        durs.append(ms)
        if keep:
            keyframes.append((keep, frames[-1]))

    snap(compose(reveal=-1), 700, "01_level")
    snap(compose(reveal=0), 1000, "02_entry")
    D = min(G["max_depth"], args.max_depth)
    for d in range(1, D + 1):
        for s in range(1, args.layer_steps + 1):
            snap(compose(reveal=d, partial=s / args.layer_steps, pop=True), 190)
        snap(compose(reveal=d), 620, f"{d + 2:02d}_depth{d}")
    if G["exits"]:
        snap(compose(reveal=D, flash=True), 800, f"{D + 3:02d}_exit")
        for i in range(1, len(G["path_keys"]) + 2):
            snap(compose(reveal=D, dim=True, path_upto=i), 220)
        snap(compose(reveal=D, dim=True, path_upto=len(G["path_keys"]) + 1,
                     banner=T["proved"].format(n=len(G["path_acts"]))),
             2600, f"{D + 4:02d}_path")
    else:
        snap(compose(reveal=D, banner=T["nopath"]), 2600, f"{D + 3:02d}_nopath")

    # ---- write -------------------------------------------------------------
    out = args.out or f"figures/bfs_L{lvl}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # One palette for the whole clip, sampled across it -- a palette taken from
    # frame 0 alone would have none of the overlay colours in it.
    idx = sorted({int(i * (len(frames) - 1) / 6) for i in range(7)})
    strip = Image.new("RGB", (SW, SH * len(idx)))
    for i, fi in enumerate(idx):
        strip.paste(frames[fi], (0, i * SH))
    base = strip.quantize(colors=255, method=Image.MEDIANCUT)
    pal = [f.quantize(palette=base, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs,
                loop=0, optimize=True, disposal=1)
    print(f"  wrote {out} ({len(frames)} frames, "
          f"{os.path.getsize(out) / 1e6:.1f} MB, {sum(durs) / 1000:.1f}s)")

    if not args.no_png:
        pdir = args.png_dir or f"figures/bfs_L{lvl}"
        os.makedirs(pdir, exist_ok=True)
        for name, im in keyframes:
            im.save(os.path.join(pdir, f"{name}.png"))
        print(f"  wrote {len(keyframes)} key frames to {pdir}/")

    if G["exits"]:
        print("  proven route: " +
              " -> ".join(str(env.actions[a]) for a in G["path_acts"]))
    env.close()


if __name__ == "__main__":
    main()
