# -*- coding: utf-8 -*-
"""The basic per-update figure (as in `plot_l02.py`) for the two short runs.

`tools/quick_two.sh` trains one icy level (36) and one windy level (29) with
`tools/train_ice.py --no-bc-init` -- no behaviour cloning, the policy starts
random. Panels are the ones of the "Нивои 0–2" figure, against the update
environment-step axis. The curriculum panel plots SOLVED RUNGS (`kmax - k`): the trainer
counts `k` down from the snapshot next to the exit towards the level's real
entry, so `kmax - k` rises the way an unlocking curriculum does, and touching
the top line means the level is solved from the state the relay delivers.

    python tools/thesis/quick_dark.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Train.py rows ("cur unlocked/total") and train_ice.py rows ("k rung/kmax")
ROW = re.compile(r"upd\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+).*?"
                 r"cur\s+(\d+)/(\d+).*?ent\s+([\d.]+)")
ROW_ICE = re.compile(r"upd\s+(\d+)\s*\|.*?ret\s+(-?[\d.]+)\s*\|\s*succ\s+([\d.]+)\s*\|"
                     r"\s*k\s+(\d+)/(\d+).*?ent\s+([\d.]+)")

plt.rcParams.update({
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd", "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#868c98", "axes.edgecolor": "#3a3f4a",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.titleweight": "normal",
    "axes.titlesize": 13, "font.size": 11})
COL = {"s": "#4fd07a", "r": "#54a8e6", "e": "#f0a94e", "c": "#b184ea"}


def cumulative(steps):
    """Sum only forward progress, so a resumed continuation of the same run
    does not fold the axis back on itself."""
    out, tot, prev = [], 0.0, None
    for v in steps:
        if prev is not None and v > prev:
            tot += v - prev
        prev = v
        out.append(tot)
    return out


def plot(log, title, out):
    U, R, S, C, E, N = [], [], [], [], [], 1
    for ln in open(log, encoding="utf-8", errors="ignore"):
        m = ROW.search(ln)
        rungs = False
        if not m:
            m = ROW_ICE.search(ln)
            rungs = True
        if not m:
            continue
        st = re.search(r"step\s+(\d+)", ln)
        U.append(int(st[1]) if st else int(m[1]))
        R.append(float(m[2])); S.append(float(m[3]))
        N = int(m[5]); E.append(float(m[6]))
        # train_ice counts the rung DOWN towards the entry; plot rungs solved
        C.append(N - int(m[4]) if rungs else int(m[4]))
    if len(U) < 5:
        print("skip", log, len(U)); return
    n = len(U)
    U = [v / 1e3 for v in cumulative(U)]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8))
    a = ax[0, 0]; a.plot(U, S, color=COL["s"], lw=2.2); a.set_ylim(0, 1.02); a.set_title(u"Успешност")
    a = ax[0, 1]; a.plot(U, R, color=COL["r"], lw=2.2); a.axhline(0, color="#454b57", lw=0.9); a.set_title(u"Добит")
    a = ax[1, 0]; a.plot(U, E, color=COL["e"], lw=2.2); a.set_title(u"Ентропија")
    a = ax[1, 1]; a.step(U, C, color=COL["c"], lw=2.2, where="post"); a.set_ylim(0, N + 1); a.set_title(u"Курикулум")
    for a in ax.flat:
        a.set_xlabel(u"кораци окружења (хиљаде)"); a.margins(x=0.01)
    fig.suptitle(title, fontweight="normal", fontsize=12, color="#c9cdd6")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=130)
    plt.close(fig)
    print("ok %s | %d rows | %.0fk steps | succ %.2f->%.2f | cur %d->%d/%d | ent %.2f->%.2f"
          % (out, n, U[-1], S[0], S[-1], C[0], C[-1], N, E[0], E[-1]))


RUNS = [
    ("logs/quick2_L36_ice.log", u"Ниво 36 · лед", "figures/quick_L36_ice.png"),
    ("logs/quick2_L29_wind.log", u"Ниво 29 · ветар", "figures/quick_L29_wind.png"),
]

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    for log, title, out in RUNS:
        if os.path.exists(log):
            plot(log, title, out)
