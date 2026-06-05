# randtest.py
import os
os.environ["SDL_VIDEODRIVER"] = ""
os.environ["SDL_AUDIODRIVER"] = ""
import numpy as np
from JK_Env import JumpKingEnv

env = JumpKingEnv(max_steps=2000)
obs, _ = env.reset()
best = 0
rng = np.random.default_rng()
for t in range(2000):
    a = int(rng.integers(env.num_actions))
    obs, r, term, trunc, info = env.step(a)
    if info["level"] > best:
        best = info["level"]
        print(f"t={t} REACHED LEVEL {best}  (action {a} = {env.actions[a]})")
    if term or trunc:
        print(f"reset at t={t}, best level so far = {best}")
        obs, _ = env.reset()
print("FINAL best level reached under random actions:", best)