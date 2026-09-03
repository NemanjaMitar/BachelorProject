# -*- coding: utf-8 -*-
"""Regenerate the whole chart set in one consistent dark style."""
import re, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd", "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#868c98", "axes.edgecolor": "#3a3f4a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlesize": 13, "font.size": 11})
CO = {"s": "#4fd07a", "r": "#54a8e6", "e": "#f0a94e", "c": "#b184ea"}

def parse(fn):
    st, S, R, E, C, N = [], [], [], [], [], None
    for ln in open(fn, encoding="utf-8", errors="ignore"):
        m = re.search(r"step\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+)", ln)
        if not m: continue
        st.append(int(m[1])); R.append(float(m[2])); S.append(float(m[3]))
        e = re.search(r"ent\s+([\d.]+)", ln); E.append(float(e[1]) if e else np.nan)
        cu = re.search(r"cur\s+(\d+)/(\d+)", ln)
        if cu: C.append(int(cu[1])); N = int(cu[2])
        else: C.append(None)
    return st, S, R, E, C, N

def sm(y, w=5):
    y = np.array(y, float)
    if len(y) < w: return y
    z = np.convolve(y, np.ones(w) / w, mode="same")
    for i in range(w // 2): z[i] = y[:i + 1].mean(); z[-(i + 1)] = y[-(i + 1):].mean()
    return z

def xaxis(st):
    st = np.array(st, float); mx = st.max() if len(st) else 0
    if mx > 5e5: return (st - st.min()) / 1e6, "кораци окружења (милиони)"
    if mx > 5e4: return st / 1e3, "кораци окружења (хиљаде)"
    return np.arange(1, len(st) + 1), "ажурирање"

FIGURES = {"randlevel"}   # thesis figures live in figures/, run curves in charts/


def curve(log, title, save):
    st, S, R, E, C, N = parse(f"logs/{log}.log")
    if not st: print("skip", log); return
    x, xl = xaxis(st)
    has_cur = N and len(set(c for c in C if c is not None)) > 1
    if has_cur:
        fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8)); ax = ax.flat
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
    ax[0].plot(x, sm(S), color=CO["s"], lw=2.1); ax[0].set_ylim(-0.02, 1.02); ax[0].set_title("Успешност")
    ax[1].plot(x, sm(R), color=CO["r"], lw=2.1); ax[1].axhline(0, color="#454b57", lw=0.9); ax[1].set_title("Добит")
    ax[2].plot(x, sm(E), color=CO["e"], lw=2.1); ax[2].set_title("Ентропија")
    if has_cur:
        ax[3].step(x, [c if c is not None else np.nan for c in C], color=CO["c"], lw=2.1, where="post")
        ax[3].set_ylim(0, N + 1); ax[3].set_title("Курикулум")
    for a in ax: a.set_xlabel(xl); a.margins(x=0.01)
    fig.suptitle(title, fontweight="normal", fontsize=12, color="#c9cdd6")
    fig.tight_layout(rect=[0, 0, 1, 0.96 if has_cur else 0.93]); fig.savefig(f"{'figures' if save in FIGURES else 'charts'}/{save}.png", dpi=130)
    print("ok", save, "panels", 4 if has_cur else 3)

curve("L10to11_multi", "Нивои 10–11 · вишенивовски", "L10to11_multi")
curve("randlevel",     "Случајни нивои · генератор", "randlevel")
curve("L36_broad",     "Л36 · без опсервације брзине", "L36_broad")
curve("L22_train",     "Ниво 22 · конвергенција", "L22_train")
curve("L20_train",     "Ниво 20", "L20_train")

# regime comparison bar (dark)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
labs = ["детермини-\nстички", "ветровити", "ледени\n(са брзином)", "ледени\n(без брзине)", "генератор"]
vals = [0.92, 1.00, 0.87, 0.00, 1.00]; cols = ["#54a8e6", "#4fd07a", "#2fae94", "#e0574f", "#b184ea"]
b = ax.bar(labs, vals, color=cols, edgecolor="#181a20")
for bx, v in zip(b, vals): ax.text(bx.get_x() + bx.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", color="#c9cdd6")
ax.set_ylim(0, 1.12); ax.set_ylabel("финална успешност"); ax.set_title("Успешност по режимима нивоа")
ax.grid(axis="x", alpha=0)
fig.tight_layout(); fig.savefig("figures/fig09_regime_bar.png", dpi=130); print("ok fig09_regime_bar")
print("ALL DONE")
