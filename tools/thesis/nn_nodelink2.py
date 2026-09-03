# -*- coding: utf-8 -*-
"""Polished actor-critic diagram. Latin labels, minimal text, few numbers.
Input and conv drawn as 3D feature-map volumes (classic CNN look), trunk and heads
as a node-link graph."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["Corbel", "Segoe UI", "Trebuchet MS", "DejaVu Sans"]

BG = "#0e131c"
fig, ax = plt.subplots(figsize=(14, 7.2))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")

TXT = "#e9edf4"; DIM = "#7f8a9c"
def shade(hexc, f):
    c = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)]) * f
    return "#%02x%02x%02x" % tuple(np.clip(c, 0, 255).astype(int))

def volume(x, y, w, h, depth, color, slices=0):
    """3D feature-map box: front + top + right faces, optional slice lines."""
    dx, dy = depth * 0.75, depth * 0.5
    ax.add_patch(Polygon([(x, y+h), (x+dx, y+h+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                         facecolor=shade(color, 1.25), edgecolor=shade(color, 0.5), lw=1))
    ax.add_patch(Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)],
                         facecolor=shade(color, 0.65), edgecolor=shade(color, 0.5), lw=1))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=shade(color, 0.5), lw=1.2))
    for s in range(1, slices+1):
        fx = x + w*s/(slices+1)
        ax.plot([fx, fx], [y, y+h], color=shade(color, 0.55), lw=0.7, alpha=0.6)

def node_col(x, ys, ring, r=1.3):
    pts = []
    for yy in ys:
        ax.add_patch(Circle((x, yy), r, facecolor="#1c2431", edgecolor=ring, lw=1.7, zorder=5))
        pts.append((x, yy))
    return pts

def fully(A, B, color, alpha, lw=0.7):
    for (x1, y1) in A:
        for (x2, y2) in B:
            ax.plot([x1+1.25, x2-1.25], [y1, y2], color=color, alpha=alpha, lw=lw, zorder=2)

def arrow(x1, y1, x2, y2, color, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=color, zorder=6))

# palette
WARM = "#c9a97e"; TEAL = "#2bb596"; GRN = "#5fd39a"; PUR = "#a982e8"; AMB = "#eaa64d"
TEAL2 = "#33a0c4"

# ---- input volume (screen map) ----
volume(6, 18, 4.5, 20, 4, WARM, slices=2)
ax.text(9.3, 14.2, "Ulaz", ha="center", fontsize=13, color=TXT, weight="bold")

# ---- three conv volumes, shrinking height, growing depth ----
volume(24, 20, 4, 16, 7, TEAL, slices=3)
volume(35, 23, 3.6, 11.5, 10, TEAL2, slices=3)
volume(45.5, 25, 3.4, 8.5, 13, shade(TEAL, 0.85), slices=3)
ax.text(38, 14.2, "Konvolucija", ha="center", fontsize=13, color=TXT, weight="bold")

# feeding arrows between volumes
for (x1, x2, yy) in [(10.5, 23.4, 28), (28.5, 34.4, 29), (39.2, 44.9, 30)]:
    arrow(x1, yy, x2, yy, DIM, 1.4)

# ---- trunk (two columns of neurons) ----
ys = np.linspace(15, 41, 7)
h1 = node_col(64, ys, GRN); h2 = node_col(72, ys, GRN)
for x in (64, 72):
    for dy in (-1.5, 0, 1.5): ax.plot(x, 43.5+dy, ".", ms=3.5, color=DIM)
ax.text(68, 10.5, "Stablo", ha="center", fontsize=13, color=TXT, weight="bold")

# arrow from conv into trunk
arrow(50, 29, 62.4, 28, TEAL, 1.6)

# ---- heads ----
py = np.linspace(30, 46, 5); pol = node_col(84, py, PUR)
val = node_col(84, [18], AMB)
fully(h1, h2, "#4a5568", 0.30)
fully(h2, pol, PUR, 0.22)
fully(h2, val, AMB, 0.22)
ax.text(84, 50.5, "Politika", ha="center", fontsize=13, color=PUR, weight="bold")
ax.text(84, 12.5, "Vrednost", ha="center", fontsize=13, color=AMB, weight="bold")

# policy -> soft bars (no numbers)
bx = 90
for yy, v in zip(py[::-1], [0.66, 0.2, 0.09, 0.03, 0.02]):
    ax.plot([85.3, bx], [yy, yy], color=PUR, alpha=0.35, lw=0.8)
    ax.add_patch(Rectangle((bx, yy-0.75), v*9, 1.5, facecolor=PUR, edgecolor="none"))
# value -> V(s)
arrow(85.3, 18, 90, 18, AMB, 1.5)
ax.text(90.8, 18, "V(s)", va="center", fontsize=12, color=AMB, weight="bold")

ax.text(50, 52.3, "Actor-Critic mreža", ha="center", fontsize=15, color=TXT, weight="bold")
fig.savefig("figures/nn_nodelink.png", dpi=150, facecolor=BG, bbox_inches="tight")
print("saved figures/nn_nodelink.png")
