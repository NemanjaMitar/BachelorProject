# -*- coding: utf-8 -*-
import re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
st, S, R, E, C = [], [], [], [], []
for ln in open("logs/L29_broad.log", encoding="utf-8", errors="ignore"):
    m = re.search(r"step\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+).*?cur\s+(\d+)/25.*?ent\s+([\d.]+)", ln)
    if m:
        st.append(int(m[1]) / 1e6); R.append(float(m[2])); S.append(float(m[3])); C.append(int(m[4])); E.append(float(m[5]))
def sm(y, w=5):
    y = np.array(y, float)
    if len(y) < w: return y
    k = np.ones(w) / w; z = np.convolve(y, k, mode="same")
    for i in range(w // 2): z[i] = y[:i + 1].mean(); z[-(i + 1)] = y[-(i + 1):].mean()
    return z
plt.rcParams.update({
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd", "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#868c98", "axes.edgecolor": "#3a3f4a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlesize": 13, "font.size": 11})
COL = {"s": "#4fd07a", "r": "#54a8e6", "e": "#f0a94e", "c": "#b184ea"}
fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8))
a = ax[0, 0]; a.plot(st, sm(S), color=COL["s"], lw=2.1); a.set_ylim(0, 1.02); a.set_title("Успешност")
a = ax[0, 1]; a.plot(st, sm(R), color=COL["r"], lw=2.1); a.axhline(0, color="#454b57", lw=0.9); a.set_title("Добит")
a = ax[1, 0]; a.plot(st, sm(E), color=COL["e"], lw=2.1); a.set_title("Ентропија")
a = ax[1, 1]; a.step(st, C, color=COL["c"], lw=2.1, where="post"); a.set_ylim(0, 26); a.set_title("Курикулум")
for a in ax.flat: a.set_xlabel("кораци окружења (милиони)"); a.margins(x=0.01)
fig.suptitle("Ниво 29 · ветар", fontweight="normal", fontsize=12, color="#c9cdd6")
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("charts/L29_broad.png", dpi=130)
print("ok")
