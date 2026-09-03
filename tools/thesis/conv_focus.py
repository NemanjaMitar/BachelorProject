# -*- coding: utf-8 -*-
"""Focused convolutional-part diagram: input map + 3 conv volumes (funnel) + flatten.
Latin, minimal text, few numbers, 3D feature-map volumes. Dark, elegant."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["Corbel", "Segoe UI", "Trebuchet MS", "DejaVu Sans"]
BG = "#0e131c"
fig, ax = plt.subplots(figsize=(13.5, 6.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")
TXT = "#e9edf4"; DIM = "#8591a3"

def shade(hexc, f):
    c = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)]) * f
    return "#%02x%02x%02x" % tuple(np.clip(c, 0, 255).astype(int))

def volume(x, y, w, h, depth, color, slices=0):
    dx, dy = depth * 0.7, depth * 0.5
    ax.add_patch(Polygon([(x, y+h), (x+dx, y+h+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                         facecolor=shade(color, 1.28), edgecolor=shade(color, 0.5), lw=1.1))
    ax.add_patch(Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                         facecolor=shade(color, 0.62), edgecolor=shade(color, 0.5), lw=1.1))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=shade(color, 0.5), lw=1.3))
    for s in range(1, slices+1):
        fx = x + w*s/(slices+1)
        ax.plot([fx, fx], [y, y+h], color=shade(color, 0.55), lw=0.7, alpha=0.65)
    return x+w+dx, y+h+dy  # far-top corner

def arrow(x1, y1, x2, y2, color, lw=1.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color, ls=ls, zorder=6))

WARM = "#c9a97e"; T1 = "#2bb596"; T2 = "#33a0c4"; T3 = "#1f9e8a"; FL = "#6fae7a"

# ---- input map (3 channels) ----
volume(6, 13, 6, 24, 4, WARM, slices=2)
ax.text(9.5, 8.8, "Ulaz", ha="center", fontsize=13.5, color=TXT, weight="bold")
# small 3x3 kernel window on the input face + projection to one conv cell
kx, ky, ks = 8.3, 27, 2.2
ax.add_patch(Rectangle((kx, ky), ks, ks, facecolor="none", edgecolor="#f2c14e", lw=1.8))
ax.text(kx+ks/2, ky+ks+1.4, "filter", ha="center", fontsize=9, color="#f2c14e")

# ---- three conv volumes: shrinking spatial, growing depth ----
volume(30, 15, 5.2, 20, 8, T1, slices=3)
volume(47, 18, 4.6, 15, 12, T2, slices=3)
volume(63, 21, 4.0, 11, 16, T3, slices=3)
ax.text(48, 7.5, "Konvolucija", ha="center", fontsize=14, color=TXT, weight="bold")
ax.text(48, 4.2, "svaki sloj: prostor manji, dubina veca", ha="center", fontsize=10, color=DIM)

# feeding arrows
arrow(12.6, 25, 29.4, 25, DIM, 1.5)                     # input -> conv1
for x1, x2 in [(41.5, 46.4), (57.5, 62.4)]:
    arrow(x1, 26, x2, 26, DIM, 1.5)
# kernel projection lines to one conv1 cell
for corner in [(kx, ky), (kx+ks, ky), (kx, ky+ks), (kx+ks, ky+ks)]:
    ax.plot([corner[0], 32.1], [corner[1], 31], color="#f2c14e", lw=0.7, alpha=0.55, zorder=1)
ax.add_patch(Rectangle((31.2, 30.2), 1.6, 1.6, facecolor="#f2c14e", alpha=0.9, edgecolor="none"))

# ---- flatten into a vector, then out ----
fx0 = 80
for i in range(16):
    yy = 12 + i*1.55
    ax.add_patch(Rectangle((fx0, yy), 2.2, 1.25, facecolor=shade(FL, 1.0 - (i % 3)*0.12),
                           edgecolor=shade(FL, 0.5), lw=0.5))
ax.text(fx0+1.1, 9.0, "ravnanje", ha="center", fontsize=11, color=TXT)
arrow(70, 26, 79.4, 25, T3, 1.7)                        # conv3 -> flatten
arrow(83, 25, 90, 25, FL, 1.8)                          # flatten -> out (user continues)
ax.text(91, 25, "…", va="center", fontsize=16, color=DIM)

fig.savefig("figures/conv_focus.png", dpi=150, facecolor=BG, bbox_inches="tight")
print("saved figures/conv_focus.png")
