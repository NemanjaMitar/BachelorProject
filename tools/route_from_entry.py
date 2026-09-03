#!/usr/bin/env python
"""Search a level for a crossing route starting from the arrival the CHAIN
really makes, and write it as a `train_ice.py` demonstration.

A route is only valid for the arrival it was searched from. The moment a lower
screen gets a different model the arrival moves, and a route found for the old
one has nothing left to replay -- so there is nothing for a backward curriculum
to be built from and the route has to be found again, from the new arrival.

This does that. It restores the real arrival snapshot and beam-searches the
action table for a sequence that reaches the goal level, then writes the states
along it as `starts/demo_L<N>_swap.json` in ice_demo.py's schema, so

    python tools/train_ice.py --level 39 --demo starts/demo_L39_swap.json ...

trains on it exactly like every other screen.

    python tools/route_from_entry.py --level 39                    # auto --swap
    python tools/route_from_entry.py --level 39 --beam 40 --max-depth 9

Measured on L39: the previous route was searched for an arrival at x=443.0, the
learned ice models deliver x=449.7, and the same six actions then fall two
biomes. The route found from 449.7 is five actions.

Why snapshot/restore and not `reset(level, x, y)`: the arrival carries flags a
teleport cannot express. Measured here, the L39 arrival is at rest but its
`angle` is 3.66 rad, not the 1.571 a settle produces, and `x` is 449.68, which a
teleport rounds. A route proved from a rounded, re-settled state is not proof
about the state the chain actually hands over.
"""
import os
import sys
import copy
import json
import heapq
import argparse

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import torch
import FullRelay as FR
from JK_Env import JumpKingEnv, build_action_table
import relay_bundle as RB

OUT = "starts"
# levels whose route climbs THROUGH an intermediate screen without landing on it
SPAN_GOAL = {23: 25, 40: 42, 42: 43}


def level_table(cfg):
    return build_action_table(
        fine_walk_frames=int(cfg["fine_walk_frames"]),
        extra_charges=tuple(cfg["extra_charges"]),
        wind_jump=tuple(cfg["wind_jump"]),
        settle_action=bool(cfg["settle_action"]))


def readable(env):
    vx, vy = env.velocity()
    return {"level": int(env.levels.current_level),
            "x": float(env.king.rect_x), "y": float(env.king.rect_y),
            "vx": float(vx), "vy": float(vy),
            "speed": float(env.king.speed), "angle": float(env.king.angle),
            "grounded": bool(env.move_available())}


def jsonable(act):
    """An action tuple as JSON. ('jump_wind', (0,'left'), 32) keeps its pair as a
    LIST, not a string -- so the training side can rebuild the tuple and look it
    up in the action table."""
    kind, d, mag = act
    return [str(kind), (list(d) if isinstance(d, tuple) else str(d)), int(mag)]


def collect_entry(level, seed, trials, swaps):
    """Play the real relay and snapshot the king the moment he ENTERS `level`."""
    e = FR.build_env(seed=seed)
    models = FR.bind(e, {L: FR.load(L, (e.grid_h, e.grid_w), swaps.get(L))
                         for L in range(0, 43)})
    entry = None
    for t in range(trials):
        e.levels.ending = False
        e.reset(level=0)
        for _ in range(320):
            m = models.get(e.levels.current_level)
            if m is None:
                break
            e.actions = m["tbl"]
            e.num_actions = len(m["tbl"])
            with torch.no_grad():
                a = int(m["net"](torch.as_tensor(m["sel"](e._obs()), device=FR.dev)
                                 .float().unsqueeze(0))[0].argmax(1))
            _, _, te, tr, info = e.step(a)
            if int(info["level"]) == level:
                entry = (e.snapshot(), readable(e))
                print(f"  entry to L{level} on trial {t}: "
                      f"x={entry[1]['x']:.2f} y={entry[1]['y']:.2f} "
                      f"speed={entry[1]['speed']:.2f}", flush=True)
                break
            if te or tr or e.levels.ending:
                break
        if entry:
            break
    e.close()
    return entry


def state_key(env, grid=4):
    """Two states are the same node when they settle in the same small cell.

    The king only acts when grounded and the level is windless, so a settled
    (level, x, y) is the whole state -- `speed` is included anyway because a
    seam crossing can leave residual momentum, and two arrivals a pixel apart
    with different momentum are genuinely different nodes."""
    return (int(env.levels.current_level),
            int(float(env.king.rect_x) // grid),
            int(float(env.king.rect_y) // grid),
            round(float(env.king.speed), 1))


def altitude(env, level):
    """Higher is better: a level gained outranks any height inside a screen."""
    return (int(env.levels.current_level) - level) * 1000 - float(env.king.rect_y)


def search(env, entry_snap, level, goal, table, max_depth, beam, verbose=True):
    """Beam search over the macro-action graph from one exact snapshot.

    Returns the winning list of action indices, or None. The frontier is kept
    by altitude because that is the only ordering the game itself rewards; a
    plain BFS over ~44 actions cannot reach the depth these routes need
    (the L39 route is five actions = 44^5 sequences)."""
    seen = {state_key(env)}
    frontier = [([], copy.deepcopy(entry_snap))]
    for depth in range(max_depth):
        scored, crossed = [], None
        for path, snap in frontier:
            for i in range(len(table)):
                env.restore(copy.deepcopy(snap))
                try:
                    _, _, _, _, info = env.step(i)
                except Exception:
                    continue
                lvl = int(info["level"])
                if lvl < level or (lvl > level and lvl < goal and level not in SPAN_GOAL):
                    continue                      # fell, or left the wrong way
                if env.levels.ending or lvl >= goal:
                    return path + [i]
                k = state_key(env)
                if k in seen:
                    continue
                seen.add(k)
                scored.append((altitude(env, level), path + [i], env.snapshot()))
        if not scored:
            if verbose:
                print(f"  depth {depth + 1}: no new states -- search exhausted")
            return None
        scored.sort(key=lambda t: -t[0])
        frontier = [(p, s) for _, p, s in scored[:beam]]
        if verbose:
            print(f"  depth {depth + 1}: {len(scored):5d} new states, "
                  f"best altitude {scored[0][0]:8.1f}, keeping {len(frontier)}",
                  flush=True)
    return None


def write_demo(env, entry_snap, route, level, goal, table, cfg, swaps, out):
    """Replay the winning route once more, recording every state along it."""
    env.restore(copy.deepcopy(entry_snap))
    rec = {"level": level, "goal": goal, "source": "route_from_entry.py",
           "table_cfg": cfg,
           "swaps": {str(k): v for k, v in sorted(swaps.items())},
           "route_actions": [list(table[i]) for i in route], "steps": []}
    rec["entry"] = {"snapshot": env.snapshot(), "state": readable(env)}
    for i in route:
        act = table[i]
        before, st = env.snapshot(), readable(env)
        _, _, _, _, info = env.step(i)
        rec["steps"].append({"i": len(rec["steps"]), "route_action": list(act),
                             "action": jsonable(act), "action_index": int(i),
                             "snapshot": before, "state_before": st,
                             "state_after": readable(env),
                             "level_after": int(info["level"])})
    rec["exit"] = {"snapshot": env.snapshot(), "state": readable(env)}
    rec["reached"] = int(env.levels.current_level)
    rec["ending"] = bool(env.levels.ending)
    rec["crossed"] = bool(env.levels.ending or env.levels.current_level >= goal)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--goal-level", type=int, default=None)
    ap.add_argument("--swap", action="append", default=[], metavar="LVL=PATH",
                    help="replacement model for a lower level; with none given "
                         "the set the packed relay uses is taken")
    ap.add_argument("--entry", default=None,
                    help="a json written by an earlier run, to skip replaying "
                         "the relay (default starts/L<N>_swap_entry.json)")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--beam", type=int, default=30)
    ap.add_argument("--fine-walk-frames", type=int, default=6,
                    help="6 on L39: at 3 the walk and the jump it precedes land "
                         "in the SAME 8px occupancy cell and the two rungs "
                         "cannot be told apart")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    level = args.level
    goal = args.goal_level or SPAN_GOAL.get(level, level + 1)

    if args.swap:
        swaps = {int(s.split("=")[0]): s.split("=", 1)[1] for s in args.swap}
    else:
        swaps = {int(k): v for k, v in RB.learned_models().items()}
        swaps.pop(level, None)          # this level is what we are replacing
    print(f"L{level} -> goal L{goal} | {len(swaps)} swapped lower models")

    entry_file = args.entry or os.path.join(OUT, f"L{level}_swap_entry.json")
    if os.path.exists(entry_file):
        with open(entry_file, encoding="utf-8") as f:
            d = json.load(f)
        entry_snap, st = d["snapshot"], d["state"]
        print(f"entry from {entry_file}: x={st['x']:.2f} y={st['y']:.2f} "
              f"speed={st['speed']:.2f}")
    else:
        ent = collect_entry(level, args.seed, args.trials, swaps)
        if ent is None:
            raise SystemExit(f"the chain never reached L{level}")
        entry_snap, st = ent
        with open(entry_file, "w", encoding="utf-8") as f:
            json.dump({"level": level, "snapshot": entry_snap, "state": st,
                       "swaps": {str(k): v for k, v in sorted(swaps.items())}}, f)
        print(f"wrote {entry_file}")

    cfg = {"fine_walk_frames": int(args.fine_walk_frames),
           "extra_charges": [22, 24, 28, 30],
           "wind_jump": [], "settle_action": True}
    table = level_table(cfg)
    print(f"{len(table)} actions, depth <= {args.max_depth}, beam {args.beam}")

    env = JumpKingEnv(max_steps=10 ** 6, vel_obs=True, wind_obs=True,
                      grid_channels=FR.SUPPORTED_CHANNELS)
    env.actions = list(table)
    env.num_actions = len(table)

    route = search(env, entry_snap, level, goal, table,
                   args.max_depth, args.beam)
    if route is None:
        env.close()
        raise SystemExit(f"no crossing route found from this arrival within "
                         f"depth {args.max_depth} / beam {args.beam}")

    print(f"\nROUTE ({len(route)} actions):")
    for i in route:
        print(f"  {table[i]}")

    out = args.out or os.path.join(OUT, f"demo_L{level}_swap.json")
    rec = write_demo(env, entry_snap, route, level, goal, table, cfg, swaps, out)
    env.close()
    print(f"\ncrossed={rec['crossed']} reached=L{rec['reached']} "
          f"steps={len(rec['steps'])} -> {out}")
    if not rec["crossed"]:
        raise SystemExit("the recorded replay did NOT cross -- refusing to ship "
                         "a demonstration the curriculum cannot be built from")


if __name__ == "__main__":
    sys.exit(main())
