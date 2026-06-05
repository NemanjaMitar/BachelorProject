# probe.py
import os
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = ""
from JK_Env import JumpKingEnv
env = JumpKingEnv()
env.reset()
king = env.king
# Do ONE real macro action via the env, then inspect state right after.
obs, r, term, trunc, info = env.step(20)   # some right-jump action
print(f"after 1 step: x={king.rect_x:.1f} y={king.rect_y:.1f} crouch={king.isCrouch} fall={king.isFalling} jc={king.jumpCount}")
# Now do a SECOND action and watch whether charge starts clean
print("second action, charge frames:")
env._set_keys(space=True)
for i in range(5):
    env._physics_frame()
    print(f"  f{i}: jc={king.jumpCount} crouch={king.isCrouch} fall={king.isFalling}")