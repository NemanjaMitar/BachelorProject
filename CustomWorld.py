#!/usr/bin/env python
"""
Custom worlds: hand-built level sequences that run on the REAL game engine.

A "world" is an ordered list of screens (level 0 = bottom, the king climbs to
higher indices, exactly like the original game). Each screen holds axis-aligned
platforms of three kinds -- land, ice (slippery) and snow (no walking) -- and a
per-screen wind switch.

The point is that a world is NOT a second physics model: the platforms are fed
straight into `Platforms.Platform`, so King.py's real collision, slopes, ice
slip and wind act on them, JK_Env builds its usual occupancy grid from them, and
the SAME conv policy trains on them. That makes a custom world a genuine
generalisation test of the agent, not a toy re-implementation.

Build worlds with the editor:

    python LevelEditor.py                       # new world
    python LevelEditor.py levels/tower.json     # edit an existing one

Then seed start states and train:

    python CustomWorld.py --seed levels/tower.json
    python Train.py --world levels/tower.json --start-states starts/world_tower.json \
        --curriculum --total-steps 300000 --save-dir checkpoints/world_tower

File format (levels/<name>.json):

    {
      "format": "jumpking-world", "version": 1, "name": "tower",
      "levels": [
        {"wind": false,
         "platforms": [{"x": 0, "y": 344, "w": 480, "h": 16, "type": "land"}]},
        {"wind": true, "platforms": [...]}
      ]
    }
"""

import os
import json
import argparse

SCREEN_W, SCREEN_H = 480, 360
FORMAT = "jumpking-world"
TYPES = ("land", "ice", "snow")

# what each block type means to Platforms.Platform(x, y, w, h, slope, slip, support, snow)
_FLAGS = {"land": (False, False), "ice": (True, False), "snow": (False, True)}


# ----------------------------------------------------------------- the format
def new_world(name="untitled"):
    """A minimal playable world: a floor to stand on and a screen to reach."""
    return {
        "format": FORMAT, "version": 1, "name": name,
        "levels": [
            {"wind": False, "platforms": [
                {"x": 0, "y": 344, "w": 480, "h": 16, "type": "land"}]},
            {"wind": False, "platforms": [
                {"x": 0, "y": 344, "w": 480, "h": 16, "type": "land"}]},
        ],
    }


def new_level(wind=False):
    return {"wind": bool(wind), "platforms": []}


def _clean_platform(p):
    x, y = int(p["x"]), int(p["y"])
    w, h = max(1, int(p["w"])), max(1, int(p["h"]))
    t = str(p.get("type", "land")).lower()
    if t not in TYPES:
        t = "land"
    return {"x": x, "y": y, "w": w, "h": h, "type": t}


def validate(world):
    """Normalise in place and return a list of human-readable warnings."""
    if world.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file (got format={world.get('format')!r})")
    world.setdefault("version", 1)
    world.setdefault("name", "untitled")
    levels = world.get("levels") or []
    if not levels:
        raise ValueError("world has no levels")

    warn = []
    for i, lvl in enumerate(levels):
        lvl["wind"] = bool(lvl.get("wind", False))
        lvl["platforms"] = [_clean_platform(p) for p in (lvl.get("platforms") or [])]
    if len(levels) < 2:
        warn.append("only 1 screen: the king can never climb out of it "
                    "(a world needs >= 2 screens -- the top one is the goal)")
    if not levels[0]["platforms"]:
        warn.append("screen 0 has no platforms: the king falls immediately")
    else:
        floor = max(p["y"] for p in levels[0]["platforms"])
        if floor < SCREEN_H - 60:
            warn.append("screen 0 has no floor near the bottom: the king spawns "
                        "at the bottom and will fall out")
    for i, lvl in enumerate(levels[:-1]):
        if not lvl["platforms"]:
            warn.append(f"screen {i} is empty -- nothing to stand on")
    return warn


def load_world(path):
    with open(path, encoding="utf-8") as f:
        world = json.load(f)
    validate(world)
    return world


def save_world(world, path):
    validate(world)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(world, f, indent=1)
    return path


def platform_tuples(level):
    """The screen's platforms as Platforms.Platform constructor tuples."""
    out = []
    for p in level["platforms"]:
        slip, snow = _FLAGS[p["type"]]
        out.append((p["x"], p["y"], p["w"], p["h"], False, slip, False, snow))
    return out


# ------------------------------------------------------- injection into JK_Env
class _Weather:
    """Stand-in for the game's per-screen weather object.

    Level.update_wind only asks for `.hasWind`; Level.blit1 calls `.blitme`.
    A custom screen has no weather art, so blitting is a no-op."""

    def __init__(self, has_wind):
        self.hasWind = bool(has_wind)

    def blitme(self, screen, rect):
        pass


def apply_world(env, world):
    """Replace the running env's level data with the world's.

    Levels.reset() does not reload level data, so this survives every reset --
    but it must be applied AFTER the env is constructed."""
    from Platforms import Platform

    levels = world["levels"]
    for i, lvl in enumerate(levels):
        game_level = env.levels.levels.get(i)
        if game_level is None:
            raise ValueError(f"the engine has no screen {i} to overwrite "
                             f"(worlds are limited to {env.levels.max_level + 1} screens)")
        game_level.platforms = [Platform(*t) for t in platform_tuples(lvl)]
        game_level.weather = _Weather(lvl["wind"])
        game_level.props = None            # no decoration on custom screens
        game_level.npc = None
        game_level.hiddenwalls = None
        game_level.flyer = None

    # cap the climb at the world's top screen
    env.levels.max_level = len(levels) - 1
    env.max_level = env.levels.max_level
    env.world = world
    return env


def make_env(world, goal_level=None, **kwargs):
    """A JumpKingEnv running `world` (a dict or a path to a world file).

    goal_level defaults to the top screen, so 'success' means climbing the
    whole world."""
    from JK_Env import JumpKingEnv

    if isinstance(world, str):
        world = load_world(world)
    n = len(world["levels"])
    kwargs.setdefault("goal_level", n - 1 if goal_level is None else goal_level)
    env = JumpKingEnv(**kwargs)
    return apply_world(env, world)


def default_pool_path(world_or_path):
    name = (os.path.splitext(os.path.basename(world_or_path))[0]
            if isinstance(world_or_path, str) else world_or_path["name"])
    return os.path.join("starts", f"world_{name}.json")


# ------------------------------------------------------------------- seeding
def seed_world(path, out=None, per_level=6, step=24, verbose=True, include_goal=False):
    """Derive start states from the world's geometry (the AutoSeed recipe).

    Every platform top is teleport-tested; spots that are not stable or that
    lead nowhere are dropped; the survivors are spread out per screen. Writes
    one pool covering ALL screens, which is what the reverse curriculum wants
    (it ranks by level first, so the top screen unlocks first)."""
    import AutoSeed

    world = load_world(path)
    out = out or default_pool_path(path)
    env = make_env(world, max_steps=100)

    pool = []
    n = len(world["levels"])
    # The top screen IS the goal: an episode starting there would succeed on its
    # first step and teach nothing, so it gets no seeds.
    last = n if include_goal else n - 1
    try:
        for lvl in range(last):
            cands = AutoSeed.candidate_spots(env, lvl, step=step)
            picked = []
            for c in AutoSeed.spread(cands, len(cands)):
                if len(picked) >= per_level:
                    break
                if AutoSeed.escapable(env, lvl, c[0], c[1]):
                    picked.append(c)
            for (x, y) in picked:
                pool.append({"level": lvl, "x": int(x), "y": int(y), "score": 0.0})
            if verbose:
                print(f"  screen {lvl}: {len(cands):3d} candidates -> {len(picked)} seeds")
    finally:
        env.close()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=1)
    if verbose:
        print(f"wrote {len(pool)} start states -> {out}")
    return out, pool


# ----------------------------------------------------------------------- CLI
def _cmd_info(path):
    world = load_world(path)
    warn = validate(world)
    print(f"world '{world['name']}'  ({len(world['levels'])} screens)")
    for i, lvl in enumerate(world["levels"]):
        kinds = {}
        for p in lvl["platforms"]:
            kinds[p["type"]] = kinds.get(p["type"], 0) + 1
        kind_s = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())) or "empty"
        print(f"  screen {i}: {kind_s}{'   [wind]' if lvl['wind'] else ''}")
    print(f"  goal = reach screen {len(world['levels']) - 1}")
    for w in warn:
        print(f"  WARNING: {w}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", metavar="PATH", help="write a starter world file")
    ap.add_argument("--info", metavar="PATH", help="describe a world file")
    ap.add_argument("--seed", metavar="PATH", help="generate start states for a world")
    ap.add_argument("--out", default=None, help="pool file for --seed")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--include-goal", action="store_true",
                    help="also seed the top screen (normally pointless: those "
                         "episodes start already finished)")
    args = ap.parse_args()

    if args.new:
        name = os.path.splitext(os.path.basename(args.new))[0]
        print("wrote", save_world(new_world(name), args.new))
    elif args.info:
        _cmd_info(args.info)
    elif args.seed:
        seed_world(args.seed, args.out, per_level=args.per_level,
                   include_goal=args.include_goal)
    else:
        ap.print_help()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    main()
