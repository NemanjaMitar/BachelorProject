#!/usr/bin/env python
"""movetest.py - is the king actually moving, and can it cross to level 1?

Run from the game root:  python movetest.py

The smoke test only prints altitude, which hides horizontal movement and
hides whether your near-exit checkpoints can actually reach level 1. This
prints x AND y so we can tell three cases apart:
  1) king frozen (x and y never change)          -> physics/keyboard bug
  2) king moves horizontally but never climbs     -> expected on the flat floor
  3) from a top checkpoint an up-jump reaches lvl1 -> curriculum is doing its job
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from JK_Env import JumpKingEnv

env = JumpKingEnv(goal_level=1)


def show(tag, info, r):
    print(f"  {tag:24s} -> x={info['x']:3f} y={info['y']:3f} "
          f"lvl={info['level']} alt={info['altitude']:.0f} r={r:+.2f}")


print("=== Test A: from the BOTTOM, does the king move at all? ===")
env.reset(level=0)
print(f"  start: x={int(env.king.rect_x)} y={int(env.king.rect_y)} "
      f"lvl={env.levels.current_level}")
# big right jump, big left jump, walks, big up jump
for a in (20, 6, 22, 21, 13):
    obs, r, term, trunc, info = env.step(a)
    show(str(env.actions[a]), info, r)
print("  -> if x changes, the king is NOT frozen (good). y staying ~306 on the")
print("     flat bottom floor is expected.\n")

print("=== Test B: from your TOP checkpoints, can an up-jump reach level 1? ===")
for cx in (193, 210, 247):                       # your captured y=16 spots
    env.reset(level=0, rect_x=cx, rect_y=16)
    print(f"  checkpoint start: x={int(env.king.rect_x)} y={int(env.king.rect_y)} "
          f"lvl={env.levels.current_level}")
    reached = False
    for a in (7, 8, 9, 10, 11, 12, 13):          # up jumps, weak -> strong
        obs, r, term, trunc, info = env.step(a)
        show(str(env.actions[a]), info, r)
        if info["level"] > 0:
            print("  >>> CROSSED to level", info["level"], "- curriculum works!\n")
            reached = True
            break
    if not reached:
        print("  (no up-jump crossed from here; try a different checkpoint/charge)\n")

env.close()