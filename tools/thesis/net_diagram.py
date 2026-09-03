# -*- coding: utf-8 -*-
"""Actor-critic network diagram (dark, matching the chart style)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"figure.facecolor": "#181a20", "savefig.facecolor": "#181a20",
                     "text.color": "#c9cdd6", "font.size": 11})
fig, ax = plt.subplots(figsize=(13.5, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, title, sub, fc, ec, tc="#e4e7ec", sc="#9aa2af", fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                facecolor=fc, edgecolor=ec, lw=1.3))
    ax.text(x + w/2, y + h/2 + (1.6 if sub else 0), title, ha="center", va="center",
            fontsize=fs, color=tc)
    if sub:
        ax.text(x + w/2, y + h/2 - 2.1, sub, ha="center", va="center", fontsize=8.6, color=sc)

def arrow(x1, y1, x2, y2, col="#7c8494"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color=col))

B  = ("#232a36", "#54a8e6")   # conv path
S  = ("#232a36", "#8e97a8")   # scalars
T  = ("#23262e", "#b184ea")   # trunk
PH = ("#1f2d24", "#4fd07a")   # policy
VH = ("#2d2620", "#f0a94e")   # value

box(2, 30, 13, 12, "мапа заузетости", "3×45×60 (8 px ћелија)", *B)
box(20, 30, 13, 12, "конв. 1", "16 ф., 3×3, корак 2\n→ 16×23×30", *B)
box(38, 30, 13, 12, "конв. 2", "32 ф., 3×3, корак 2\n→ 32×12×15", *B)
box(56, 30, 13, 12, "конв. 3", "32 ф., 3×3, корак 2\n→ 32×6×8 = 1536", *B)
box(2, 10, 13, 10, "скаларна обележја", "x, y, ниво, висина\n(+ ветар, + брзина)", *S)
box(56, 10, 13, 10, "конкатенација", "1536 + 4 = 1540", *S)
box(74, 20, 11, 12, "FC стабло", "128 → 128\n(tanh)", *T)
box(89, 32, 10, 10, "политика", "softmax, 37", *PH)
box(89, 10, 10, 10, "вредност", "V(s), 1", *VH)

arrow(15.6, 36, 19.4, 36); arrow(33.6, 36, 37.4, 36); arrow(51.6, 36, 55.4, 36)
arrow(62.5, 29.2, 62.5, 21)            # conv flatten -> concat
arrow(15.6, 15, 55.4, 15)              # scalars -> concat
arrow(69.6, 17, 73.4, 23)              # concat -> trunk
arrow(85.6, 28, 88.4, 35)              # trunk -> policy
arrow(85.6, 24, 88.4, 17)              # trunk -> value
ax.text(50, 51.5, "Актор-критичар мрежа · заједничко стабло, две главе (≈233.000 параметара)",
        ha="center", fontsize=12, color="#c9cdd6")
ax.text(28, 6, "ReLU после сваког конволуционог слоја · ортогонална иницијализација (√2, глава политике 0,01)",
        ha="center", fontsize=9, color="#8e97a8")
fig.tight_layout()
fig.savefig("figures/net_architecture.png", dpi=140)
print("saved figures/net_architecture.png")
