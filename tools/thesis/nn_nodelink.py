# -*- coding: utf-8 -*-
"""Node-link visualization of the actor-critic network (whiteboard style, clean).
Conv + input shown as a feeding block, the fully-connected trunk and the two heads
drawn as circles with connections. Dark theme to match the rest of the thesis figures."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"figure.facecolor": "#181a20", "savefig.facecolor": "#181a20",
                     "text.color": "#c9cdd6", "font.size": 11})
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

NODE = "#2a3340"; EDGE_C = "#6f7a8a"; RING = "#3a4656"
GRN = "#4fd07a"; ORG = "#f0a94e"; BLU = "#54a8e6"; PUR = "#b184ea"

def block(x, y, w, h, title, sub, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                facecolor="#232a36", edgecolor=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), title, ha="center",
            va="center", fontsize=11, color="#e4e7ec")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center", va="center",
                fontsize=8.6, color="#9aa2af")

def column(x, ys, color, r=1.35):
    pts = []
    for y in ys:
        ax.add_patch(Circle((x, y), r, facecolor=NODE, edgecolor=color, lw=1.6, zorder=4))
        pts.append((x, y))
    return pts

def connect(A, B, color=EDGE_C, alpha=0.35, lw=0.8):
    for (x1, y1) in A:
        for (x2, y2) in B:
            ax.plot([x1 + 1.3, x2 - 1.3], [y1, y2], color=color, alpha=alpha, lw=lw, zorder=2)

def dots(x, y):
    for dy in (-1.4, 0, 1.4):
        ax.plot(x, y + dy, marker=".", ms=4, color="#7a8494")

# ---- input + conv as a feeding block ----
block(3, 30, 15, 14, "Улаз", "мапа заузетости\n3×45×60 + скалари", BLU)
block(21, 30, 15, 14, "3 конв. слоја", "16, 32, 32 филтера\n→ 1536 обележја", BLU)
ax.text(28.5, 27.5, "+ скалари → 1540", ha="center", fontsize=8.6, color="#9aa2af")

# ---- trunk: two hidden columns (representative of 128) ----
H1y = np.linspace(14, 42, 7); H2y = np.linspace(14, 42, 7)
h1 = column(45, H1y, GRN); dots(45, 44.5)
h2 = column(58, H2y, GRN); dots(58, 44.5)
ax.text(45, 9.5, "стабло  128\n(tanh)", ha="center", fontsize=9, color="#9aa2af")
ax.text(58, 9.5, "128\n(tanh)", ha="center", fontsize=9, color="#9aa2af")

# ---- heads ----
Py = np.linspace(30, 46, 5); py = column(76, Py, PUR); dots(76, 48)
vy = column(76, [17], ORG)
ax.text(76, 50, "политика", ha="center", fontsize=9.5, color=PUR)
ax.text(76, 12.5, "вредност", ha="center", fontsize=9.5, color=ORG)

# ---- connections ----
feed = [(36.5, 37)]                       # conv block right edge midpoint
connect(feed, h1, color=BLU, alpha=0.5, lw=1.0)
connect(h1, h2)
connect(h2, py, color=PUR, alpha=0.3)
connect(h2, vy, color=ORG, alpha=0.3)

# ---- policy output: softmax bars ----
bx = 84
ax.text(bx + 6, 49, "softmax, 37 акција", ha="center", fontsize=8.6, color="#9aa2af")
vals = [0.62, 0.18, 0.10, 0.06, 0.04]
for k, (yy, v) in enumerate(zip(Py[::-1], vals)):
    ax.add_patch(FancyArrowPatch((77.4, yy), (bx, yy), arrowstyle="-", color=PUR, alpha=0.4, lw=0.8))
    ax.add_patch(plt.Rectangle((bx, yy - 0.7), v * 12, 1.4, facecolor=PUR, edgecolor="none"))
    ax.text(bx + v * 12 + 0.6, yy, f"{v:.2f}", va="center", fontsize=7.6, color="#c9cdd6")

# value output
ax.add_patch(FancyArrowPatch((77.4, 17), (bx, 17), arrowstyle="-|>", mutation_scale=12, color=ORG, alpha=0.6, lw=1.2))
ax.text(bx + 1, 17, "V(s)  =  +18.3", va="center", fontsize=10, color=ORG)

ax.text(50, 53.5, "Актор-критичар мрежа: заједничко стабло, две главе",
        ha="center", fontsize=13, color="#e4e7ec")
ax.text(50, 4.5, "кружићи представљају неуроне (приказан узорак од 128), линије су потпуне везе међу слојевима",
        ha="center", fontsize=8.6, color="#7a8494")
fig.tight_layout()
fig.savefig("figures/nn_nodelink.png", dpi=140)
print("saved figures/nn_nodelink.png")
