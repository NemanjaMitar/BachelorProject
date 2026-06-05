# tracejump.py
import os
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = ""
import pygame
from JK_Env import JumpKingEnv

env = JumpKingEnv()
env.reset()
k = env.levels  # levels
king = env.king
print(f"start: x={king.rect_x} y={king.rect_y} crouch={king.isCrouch} falling={king.isFalling}")

# Manually do a charged RIGHT jump: hold SPACE only for 20 frames, then release with RIGHT.
for i in range(20):
    env._set_keys(space=True)
    env._physics_frame()
    print(f"charge {i:2d}: y={king.rect_y:.1f} jc={king.jumpCount} crouch={king.isCrouch} fall={king.isFalling} spd={king.speed:.2f}")

env._set_keys(right=True)   # release toward right
env._physics_frame()
print(f"RELEASE: x={king.rect_x:.1f} y={king.rect_y:.1f} crouch={king.isCrouch} fall={king.isFalling} spd={king.speed:.2f} ang={king.angle:.2f}")

env._set_keys()  # all up
for i in range(40):
    env._physics_frame()
    print(f"fly {i:2d}: x={king.rect_x:.1f} y={king.rect_y:.1f} fall={king.isFalling} spd={king.speed:.2f} lvl={env.levels.current_level}")