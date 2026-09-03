# -*- coding: utf-8 -*-
import re, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def parse(fn):
    st, sc = [], []
    if not os.path.exists(fn): return st, sc
    for ln in open(fn, encoding="utf-8", errors="ignore"):
        m = re.search(r"step\s+(\d+).*?succ\s+([\d.]+)", ln)
        if m: st.append(int(m[1])); sc.append(float(m[2]))
    return st, sc
def smooth(y, w=7):
    y = np.array(y, float)
    if len(y) < w: return y
    return np.convolve(y, np.ones(w) / w, mode="same")

# level -> (log, label, regime)  (L10 removed)
LV = [("L20_train", "L20", "det"), ("L22_train", "L22", "det"), ("L24_train", "L24", "det"),
      ("L29_broad", "L29 (ветар)", "wind"), ("L36_veltest", "L36 (лед)", "ice")]
COL = {"det": "#54a8e6", "wind": "#4fd07a", "ice": "#2fae94"}
rows = []
for log, lab, reg in LV:
    st, sc = parse(f"logs/{log}.log")
    if not st: continue
    s = smooth(sc)
    reach = next((st[i] for i in range(len(s)) if s[i] >= 0.8), st[-1])
    rows.append((lab, reach / 1e3, COL[reg]))
rows.sort(key=lambda r: r[1])                       # ascending: fastest on top

plt.rcParams.update({
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd", "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#c9cdd6", "axes.edgecolor": "#3a3f4a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlesize": 13, "font.size": 12})
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(rows))[::-1]                       # smallest at top
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
ax.barh(y, vals, color=cols, edgecolor="#181a20", height=0.66)
for yi, v in zip(y, vals):
    ax.text(v + max(vals) * 0.012, yi, f"{v:.0f}k", va="center", fontsize=11, color="#c9cdd6")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, max(vals) * 1.12)
ax.set_xlabel("кораци обучавања до успешности 0,8 (хиљаде)")
ax.set_title("Ефикасност обучавања по нивоу — мања вредност је брже")
ax.grid(axis="y", alpha=0)
fig.tight_layout()
fig.savefig("figures/fig10_efficiency.png", dpi=130)
print("rows:", [(l, round(v)) for l, v, _ in rows], "-> charts/fig10_efficiency.png")
