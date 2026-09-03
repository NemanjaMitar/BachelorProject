# -*- coding: utf-8 -*-
"""What the agent actually sees: the 8px occupancy grid fed to the CNN, and how
3 stride-2 convolutions (8x downsample) collapse the king marker -> fine position
is lost. Built from a hand-designed level."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LVL = "levels/simple.json"
d = json.load(open(LVL, encoding="utf-8"))
W, H = d["size"]; CS = 8
CX, CY = W // CS, H // CS
solid = np.zeros((CY, CX)); king = np.zeros((CY, CX))
for (x, y, w, h) in d["platforms"]:
    c0, c1 = int(x // CS), int(np.ceil((x + w) / CS)); r0, r1 = int(y // CS), int(np.ceil((y + h) / CS))
    solid[max(0, r0):r1, max(0, c0):c1] = 1
kx, ky = d["king"]
kr = int((ky - 16) // CS); kc = int(kx // CS)
king[max(0, kr):kr + 2, max(0, kc):min(CX, kc + 2)] = 1

def to_rgb(sol, kng):
    img = np.empty((*sol.shape, 3)); img[:] = (0.10, 0.11, 0.14)
    img[sol > 0] = (0.74, 0.71, 0.66); img[kng > 0] = (0.86, 0.30, 0.26)
    return img

def pool(a):
    r, c = a.shape; r2, c2 = r // 2, c // 2
    return a[:r2 * 2, :c2 * 2].reshape(r2, 2, c2, 2).max(axis=(1, 3))
ds_s, ds_k = solid.copy(), king.copy()
for _ in range(3): ds_s, ds_k = pool(ds_s), pool(ds_k)

plt.rcParams.update({"figure.facecolor": "#181a20", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.titlecolor": "#e4e7ec", "axes.titlesize": 13,
    "axes.titleweight": "normal", "font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 6.2))

# (a) human view
a = ax[0]; a.set_facecolor("#20232b")
for i, (x, y, w, h) in enumerate(d["platforms"]):
    a.add_patch(Rectangle((x, y), w, h, color="#c8b18f"))
a.add_patch(Rectangle((kx - 7, ky - 20), 14, 20, color="#d85a30"))
px, py = d["princess"]; a.add_patch(Rectangle((px - 7, py - 20), 14, 20, color="#3caa5a"))
a.set_xlim(0, W); a.set_ylim(H, 0); a.set_aspect("equal")
a.set_title("а) Ниво (људски поглед)"); a.set_xticks([]); a.set_yticks([])

# (b) occupancy grid = CNN input
a = ax[1]; a.imshow(to_rgb(solid, king), interpolation="nearest", aspect="equal")
a.set_xticks(np.arange(-.5, CX, 5), minor=False); a.set_yticks(np.arange(-.5, CY, 5))
a.grid(color="#ffffff", alpha=0.10, lw=0.6); a.set_xticklabels([]); a.set_yticklabels([])
a.set_title(f"б) Мрежа заузетости, улаз ({CX}×{CY}, ћелија 8 px)")

# (c) after 8x downsampling
a = ax[2]; a.imshow(to_rgb(ds_s, ds_k), interpolation="nearest", aspect="equal")
a.set_xticks(np.arange(-.5, ds_s.shape[1], 1)); a.set_yticks(np.arange(-.5, ds_s.shape[0], 1))
a.grid(color="#ffffff", alpha=0.14, lw=0.6); a.set_xticklabels([]); a.set_yticklabels([])
a.set_title(f"в) После 8× сажимања ({ds_s.shape[1]}×{ds_s.shape[0]})")

fig.suptitle("Шта агент види: мрежа заузетости и губитак фине позиције", fontsize=12, color="#c9cdd6")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/cnn_view.png", dpi=130)
print(f"grid {CX}x{CY} -> downsampled {ds_s.shape[1]}x{ds_s.shape[0]} | saved charts/cnn_view.png")
