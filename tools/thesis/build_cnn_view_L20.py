# -*- coding: utf-8 -*-
"""What the agent actually sees on a REAL game level (default: level 20), taken
straight from JK_Env's occupancy-grid observation. Shows the level geometry, the
3-channel 8px grid fed to the CNN, and the 8x-downsampled feature resolution."""
import os, sys, json, numpy as np, matplotlib
sys.path.insert(0, os.getcwd())
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from JK_Env import JumpKingEnv

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
st = next((s for s in json.load(open(f"starts_L{LEVEL}.json")) if s.get("level", LEVEL) == LEVEL), None)
env = JumpKingEnv(max_steps=300, goal_level=LEVEL + 1)
obs, _ = env.reset(level=LEVEL, rect_x=st["x"], rect_y=st["y"])
grid = obs[:env._grid_flat].reshape(env.grid_shape)            # (3, 45, 60): solid, king, hazard
solid, king, hazard = grid[0], grid[1], grid[2]
plats = env.levels.levels[LEVEL].platforms or []
kx, ky = env.king.rect_x, env.king.rect_y
SW, SH = env.screen_w, env.screen_h

def to_rgb(sol, kng, haz):
    img = np.empty((*sol.shape, 3)); img[:] = (0.10, 0.11, 0.14)
    img[sol > 0] = (0.74, 0.71, 0.66)
    img[haz > 0] = (0.42, 0.68, 0.86)
    img[kng > 0] = (0.86, 0.30, 0.26)
    return img

def pool(a):
    r, c = a.shape; r2, c2 = r // 2, c // 2
    return a[:r2*2, :c2*2].reshape(r2, 2, c2, 2).max(axis=(1, 3))
ds = [solid.copy(), king.copy(), hazard.copy()]
for _ in range(3): ds = [pool(a) for a in ds]

plt.rcParams.update({"figure.facecolor": "#181a20", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.titlecolor": "#e4e7ec", "axes.titlesize": 13,
    "axes.titleweight": "normal", "font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

a = ax[0]; a.set_facecolor("#20232b")
for p in plats:
    col = "#c8b18f" if getattr(p, "type", "Land") == "Land" else ("#8fd0e6" if p.type == "Ice" else "#e6ecf2")
    a.add_patch(Rectangle((p.x, p.y), p.width, p.height, color=col))
a.add_patch(Rectangle((kx - 6, ky - 18), 12, 18, color="#d85a30"))
a.set_xlim(0, SW); a.set_ylim(SH, 0); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
a.set_title(f"а) Ниво {LEVEL} (људски поглед)")

a = ax[1]; a.imshow(to_rgb(solid, king, hazard), interpolation="nearest", aspect="equal")
a.set_xticks(np.arange(-.5, grid.shape[2], 5)); a.set_yticks(np.arange(-.5, grid.shape[1], 5))
a.grid(color="#ffffff", alpha=0.10, lw=0.6); a.set_xticklabels([]); a.set_yticklabels([])
a.set_title(f"б) Мрежа заузетости, улаз ({grid.shape[2]}×{grid.shape[1]}, 8 px)")

a = ax[2]; a.imshow(to_rgb(ds[0], ds[1], ds[2]), interpolation="nearest", aspect="equal")
a.set_xticks(np.arange(-.5, ds[0].shape[1], 1)); a.set_yticks(np.arange(-.5, ds[0].shape[0], 1))
a.grid(color="#ffffff", alpha=0.16, lw=0.6); a.set_xticklabels([]); a.set_yticklabels([])
a.set_title(f"в) После 8× сажимања ({ds[0].shape[1]}×{ds[0].shape[0]})")

fig.suptitle(f"Шта агент види — ниво {LEVEL} (мрежа заузетости из JK_Env)", fontsize=12, color="#c9cdd6")
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"figures/cnn_view_L{LEVEL}.png", dpi=130)
print(f"L{LEVEL} king=({kx},{ky}) grid={grid.shape} solid_cells={int(solid.sum())} "
      f"-> ds {ds[0].shape} | saved figures/cnn_view_L{LEVEL}.png")
