#!/usr/bin/env python
"""
Automatic start-state capture -- replaces manual fly-mode capturing.

Enumerates candidate standing spots from the level's platform geometry
(top of every platform, stepped), teleport-validates each one (the King must
settle exactly there, grounded, on the right level -- same check the manual
pipeline relied on), then picks a spatially spread subset via farthest-point
sampling, anchored at the LOWEST spot (the likeliest entry region).

The result merges into the level's pool file. Combine with:
  * Handoff.py  -- adds the REAL arrival states from the previous level
  * LevelGraph.py --emit-starts -- adds proven route states with graded scores

Usage:
    python AutoSeed.py --level 12                      # -> starts/starts_L12.json
    python AutoSeed.py --level 12 --n 8 --out my.json
"""

import os
import json
import argparse

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from JK_Env import JumpKingEnv


EDGE = 12          # keep seeds this far from the side walls (x=0/8 spots are
                   # pinned against the wall and rarely useful launch spots)


def candidate_spots(env, lvl, step=24):
    """All teleport-validated standing spots on the level's platform tops."""
    kh, kw = env.king.rect_height, env.king.rect_width
    out, seen = [], set()
    for p in env.levels.levels[lvl].platforms:
        top = p.rect.y - kh
        # skip spots hugging the level seam (top) -- they flicker levels
        if top < 8:
            continue
        xs = sorted({p.rect.x,
                     p.rect.x + p.rect.width // 2 - kw // 2,
                     p.rect.x + p.rect.width - kw,
                     *range(p.rect.x, p.rect.x + p.rect.width - kw + 1, step)})
        for x in xs:
            if x < EDGE or x + kw > env.screen_w - EDGE:
                continue
            slvl, gx, gy = env.teleport(lvl, x, top)
            if slvl != lvl:
                continue
            if abs(gy - top) > 6 or abs(gx - x) > 6:
                continue                      # slid/fell away: not a stable spot
            key = (gx // 12, gy // 12)
            if key in seen:
                continue
            seen.add(key)
            out.append((gx, gy))
    return out


def escapable(env, lvl, x, y, need=2):
    """A seed is useful only if the king can actually GO somewhere from it:
    at least `need` jump actions must settle in distinct new cells ON the
    level (a pocket-trapped spot fails -- every action bounces back into the
    same cell; a ledge whose every action falls off the level fails too:
    with the fall rule those episodes end instantly and can never succeed)."""
    start_cell = (int(x) // 16, int(y) // 16)
    outcomes = set()
    for a in range(env.num_actions):
        if env.actions[a][0] != "jump":
            continue
        env.reset(level=lvl, rect_x=x, rect_y=y)
        _, _, _term, _tr, info = env.step(a)
        if info["level"] != lvl:
            continue                          # fell off (or cleared) -- no credit
        cell = (int(info["x"]) // 16, int(info["y"]) // 16)
        if cell != start_cell:
            outcomes.add(cell)
        if len(outcomes) >= need:
            return True
    return False


def spread(cands, n):
    """Greedy farthest-point sampling; first pick = lowest spot on the level."""
    if not cands:
        return []
    picked = [max(cands, key=lambda c: c[1])]
    while len(picked) < min(n, len(cands)):
        best, bd = None, -1.0
        for c in cands:
            if c in picked:
                continue
            d = min((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 for p in picked)
            if d > bd:
                bd, best = d, c
        if best is None:
            break
        picked.append(best)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--n", type=int, default=8,
                    help="number of spread spots to keep")
    ap.add_argument("--out", default=None,
                    help="pool file to merge into (default starts/starts_L<n>.json)")
    args = ap.parse_args()

    out = args.out or f"starts/starts_L{args.level}.json"
    env = JumpKingEnv(max_steps=100)

    cands = candidate_spots(env, args.level)
    # Full farthest-point ordering, then walk it and keep only spots the
    # escapability test accepts -- rejects are replaced by the next most
    # spread-out candidate automatically.
    order = spread(cands, len(cands))
    picked = []
    for c in order:
        if len(picked) >= args.n:
            break
        if escapable(env, args.level, c[0], c[1]):
            picked.append(c)
        else:
            print(f"  rejected ({c[0]},{c[1]}): trapped/no useful moves")
    env.close()
    print(f"level {args.level}: {len(cands)} stable spots found, "
          f"keeping {len(picked)} escapable spread seeds")

    try:
        with open(out) as f:
            pool = json.load(f)
        assert isinstance(pool, list)
    except (FileNotFoundError, json.JSONDecodeError, AssertionError):
        pool = []
    seen = {(s["level"], s["x"] // 8, s["y"] // 8)
            for s in pool if isinstance(s, dict)}
    added = 0
    for (x, y) in picked:
        k = (args.level, x // 8, y // 8)
        if k in seen:
            continue
        pool.append({"level": args.level, "x": int(x), "y": int(y),
                     "score": 0.0})
        seen.add(k)
        added += 1
    with open(out, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"merged {added} new seeds into {out} ({len(pool)} total)")


if __name__ == "__main__":
    main()
