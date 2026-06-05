import os
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
from JK_Env import JumpKingEnv
env = JumpKingEnv(goal_level=1, start_states="start_states.json", p_bottom=0.2)
print("checkpoints loaded into pool:", len(env._start_pool))
for i in range(8):
    obs, _ = env.reset()
    print(f"reset {i}: level={env.levels.current_level} x={env.king.rect_x} y={env.king.rect_y}")