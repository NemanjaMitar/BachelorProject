# -*- coding: utf-8 -*-
"""Thesis-style (dark, Serbian) learning curves for the icy levels.

Same 2x2 layout and palette as `wind_dark.py` / `dark_charts.py`, but fed from
the ice campaign logs, which use the backward-curriculum row format:

    upd 12 | step 24576 | 91 sps | ret 3.40 | succ 0.42 | k 3/5 (front 0.42 greedy 1.00) | ent 1.31

Two differences from the plain PPO logs:

* the runs were resumed many times, so `step` restarts -- the x axis sums only
  forward progress (`cumulative`), exactly like `tools/chart_ice_level.py`;
* the curriculum counter runs *down* (rung k = how many demo snapshots ahead of
  the level entry the episode starts), so the panel plots `kmax - k`, i.e. how
  many rungs of the backward curriculum are already solved. It rises, like the
  curriculum panel of every other figure, and touching the top line means the
  level is solved from the state the relay really hands over.
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW = re.compile(
    r"upd\s+(\d+)\s*\|\s*step\s+(\d+)\s*\|\s*(-?[\d.]+) sps\s*\|\s*"
    r"ret\s+(-?[\d.nan]+)\s*\|\s*succ\s+([\d.nan]+)\s*\|\s*k\s+(\d+)/(\d+)\s*"
    r"\(front\s+([\d.nan]+)(?:\s+greedy\s+([\d.nan]+))?\)\s*\|\s*"
    r"ent\s+([\d.]+)")

plt.rcParams.update({
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b", "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd", "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#868c98", "axes.edgecolor": "#3a3f4a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlesize": 13, "font.size": 11})
CO = {"s": "#4fd07a", "r": "#54a8e6", "e": "#f0a94e", "c": "#b184ea"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse(path):
    st, R, S, K, F, E, N = [], [], [], [], [], [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = ROW.search(line)
        if not m:
            continue
        st.append(int(m[2])); R.append(_f(m[4])); S.append(_f(m[5]))
        K.append(int(m[6])); N = int(m[7]); F.append(_f(m[8])); E.append(_f(m[10]))
    return st, np.array(R), np.array(S), np.array(K), np.array(F), np.array(E), N


def cumulative(steps):
    """Sum only forward progress: a resume reports a smaller `step`, and using
    it directly would fold the axis back on itself."""
    out, tot, prev = [], 0.0, None
    for s in steps:
        if prev is not None and s > prev:
            tot += s - prev
        prev = s
        out.append(tot)
    return np.array(out)


def sm(y, w):
    """Centred rolling mean, NaN-safe, with shrinking window at both ends."""
    y = np.asarray(y, float)
    if len(y) < 3:
        return y
    w = max(3, min(int(w) | 1, len(y) // 2 * 2 - 1))
    ok = np.isfinite(y).astype(float)
    z = np.convolve(np.nan_to_num(y), np.ones(w), "same")
    n = np.convolve(ok, np.ones(w), "same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = z / n
    return out


def segments(K):
    """Index blocks over which the curriculum rung is constant."""
    K = np.asarray(K)
    cuts = np.flatnonzero(np.diff(K)) + 1
    return np.split(np.arange(len(K)), cuts)


def curve(log, title, save, window=None, frontier=False):
    st, R, S, K, F, E, N = parse(log)
    if len(st) < 30:
        print("skip", log, len(st)); return
    x = cumulative(st) / 1e6
    w = window or max(9, len(st) // 25)
    raw_a = 0.18 if len(st) < 600 else 0.07

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8))
    a = ax[0, 0]
    if frontier:
        # Success of the episodes that actually STARTED at the frontier rung,
        # drawn per rung: the counter is cleared every time the ladder steps
        # back, so the curve rises inside a rung and restarts at the next one.
        for seg in segments(K):
            if len(seg) < 2:
                continue
            a.plot(x[seg], F[seg], color=CO["s"], lw=0.8, alpha=0.20)
            a.plot(x[seg], sm(F[seg], max(3, len(seg) // 5)), color=CO["s"], lw=2.1)
        for seg in segments(K)[1:]:
            a.axvline(x[seg[0]], color="#868c98", lw=0.6, ls=":", alpha=0.35)
        a.set_title(u"Успешност на текућој пречки")
    else:
        a.plot(x, S, color=CO["s"], lw=0.8, alpha=raw_a)
        a.plot(x, sm(S, w), color=CO["s"], lw=2.1)
        a.set_title(u"Успешност")
    a.set_ylim(-0.02, 1.02)

    a = ax[0, 1]
    a.plot(x, R, color=CO["r"], lw=0.8, alpha=raw_a)
    a.plot(x, sm(R, w), color=CO["r"], lw=2.1)
    a.axhline(0, color="#454b57", lw=0.9); a.set_title(u"Добит")

    a = ax[1, 0]
    a.plot(x, sm(E, w), color=CO["e"], lw=2.1); a.set_title(u"Ентропија")

    a = ax[1, 1]
    a.step(x, N - K, color=CO["c"], lw=2.1, where="post")
    a.fill_between(x, 0, N - K, step="post", color=CO["c"], alpha=0.10)
    a.axhline(N, color="#4fd07a", lw=0.9, ls="--", alpha=0.7)
    a.text(x[-1], N + 0.12, u"прави улаз нивоа", ha="right", va="bottom",
           fontsize=9, color="#4fd07a", alpha=0.9)
    a.set_ylim(0, N + 0.9); a.set_title(u"Курикулум")

    for a in ax.flat:
        a.set_xlabel(u"кораци окружења (милиони)"); a.margins(x=0.01)
    fig.suptitle(title, fontweight="normal", fontsize=12, color="#c9cdd6")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, dpi=130)
    plt.close(fig)
    print("ok", save, "n=%d" % len(st), "steps=%.2fM" % x[-1], "k %d->%d" % (K[0], K[-1]), "w=%d" % w)


# ---------------------------------------------------------------------------
# The earlier L36 ice campaign was run by `Train.py`, whose rows look like
#   upd 42 | step 43008 | ... | succ 0.79 | ... | cur 1/14 pb 0.0 | ... | ent 1.21
# There `succ` is measured on the ONE curriculum stage that is currently
# active, so the panel shows the familiar saw tooth: the rate climbs while the
# stage is being learned, the stage is promoted, and the rate drops on the
# harder one. `train_ice.py` cannot draw that curve -- its episodes start at a
# uniformly chosen unlocked rung (rehearsal against forgetting), so its `succ`
# is a mixture over the solved tail and the frontier and stays near one half by
# construction; there the progress signal is the curriculum panel.
STAGE_ROW = re.compile(
    r"step\s+(\d+).*?ret\s+(-?[\d.]+).*?succ\s+([\d.]+).*?cur\s+(\d+)/(\d+).*?ent\s+([\d.]+)")


def parse_stage(path):
    st, R, S, C, E, N = [], [], [], [], [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = STAGE_ROW.search(line)
        if not m:
            continue
        st.append(int(m[1])); R.append(_f(m[2])); S.append(_f(m[3]))
        C.append(int(m[4])); N = int(m[5]); E.append(_f(m[6]))
    return st, np.array(R), np.array(S), np.array(C), np.array(E), N


def curve_stage(log, title, save, window=7, xmax=None, cur_mean=True):
    """2x2 panel for a `Train.py`-format run (single active curriculum stage).

    `xmax` (in thousands of steps) crops the run; `cur_mean` draws the
    curriculum as a rolling mean of the logged counter. That counter is read
    off whichever of the eight parallel envs happened to finish an episode last
    (Train.py:342), and each env unlocks its own start states, so the raw
    series is a jittery sample of a per-env counter that only ever grows. The
    faint line is the raw sample, the solid one the fleet average.
    """
    st, R, S, C, E, N = parse_stage(log)
    if len(st) < 30:
        print("skip", log, len(st)); return
    x = (cumulative(st)) / 1e3
    if xmax:
        keep = x <= xmax
        x, R, S, C, E = x[keep], R[keep], S[keep], C[keep], E[keep]
    w = window

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8))
    a = ax[0, 0]
    a.plot(x, S, color=CO["s"], lw=0.8, alpha=0.20)
    a.plot(x, sm(S, w), color=CO["s"], lw=2.1)
    a.set_ylim(-0.02, 1.02); a.set_title(u"Успешност")

    a = ax[0, 1]
    a.plot(x, R, color=CO["r"], lw=0.8, alpha=0.20)
    a.plot(x, sm(R, w), color=CO["r"], lw=2.1)
    a.axhline(0, color="#454b57", lw=0.9); a.set_title(u"Добит")

    a = ax[1, 0]
    a.plot(x, sm(E, w), color=CO["e"], lw=2.1); a.set_title(u"Ентропија")

    a = ax[1, 1]
    if cur_mean:
        a.plot(x, C, color=CO["c"], lw=0.7, alpha=0.25)
        y = sm(C, 15)
        a.plot(x, y, color=CO["c"], lw=2.1)
        a.fill_between(x, 0, y, color=CO["c"], alpha=0.10)
    else:
        a.step(x, C, color=CO["c"], lw=2.1, where="post")
        a.fill_between(x, 0, C, step="post", color=CO["c"], alpha=0.10)
    a.set_ylim(0, N + 1); a.set_title(u"Курикулум")

    for a in ax.flat:
        a.set_xlabel(u"кораци окружења (хиљаде)"); a.margins(x=0.01)
    fig.suptitle(title, fontweight="normal", fontsize=12, color="#c9cdd6")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save, dpi=130)
    plt.close(fig)
    print("ok", save, "n=%d" % len(st), "steps=%.0fk" % x[-1], "cur %d/%d->%d" % (C[0], N, C[-1]))


STAGE_RUNS = [
    # (log, title, out, smoothing window, x crop in thousands of steps, curriculum as fleet mean)
    ("logs/L36_iceimproved.log", u"Ниво 36 · лед",
     "figures/L36_ice_curriculum_dark.png", 5, None, True),
    ("logs/L36_iceimproved.log", u"Ниво 36 · лед · откључавање курикулума",
     "figures/L36_ice_unlock_dark.png", 3, 200, True),
    ("logs/L36_train.log", u"Ниво 36 · лед · прелазак степена",
     "figures/L36_ice_early_dark.png", 3, None, False),
]

RUNS = [
    ("logs/ice_L38_pure.log", u"Ниво 38 · лед", "figures/L38_ice_dark.png", None, False),
    ("logs/ice_L36_pure.log", u"Ниво 36 · лед", "figures/L36_ice_dark.png", None, False),
    ("logs/ice_L37_pure.log", u"Ниво 37 · лед", "figures/L37_ice_dark.png", None, False),
    ("logs/ice_L38_pure.log", u"Ниво 38 · лед", "figures/L38_ice_dark_frontier.png", None, True),
    ("logs/ice_L36_pure.log", u"Ниво 36 · лед", "figures/L36_ice_dark_frontier.png", None, True),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log"); p.add_argument("--title"); p.add_argument("--out")
    p.add_argument("--window", type=int)
    p.add_argument("--frontier", action="store_true",
                   help=u"success panel = frontier rung rate, drawn per rung")
    a = p.parse_args()
    os.makedirs("figures", exist_ok=True)
    if a.log:
        curve(a.log, a.title or os.path.basename(a.log), a.out or "figures/ice_dark.png",
              a.window, a.frontier)
    else:
        for log, title, out, w, fr in RUNS:
            if os.path.exists(log):
                curve(log, title, out, w, fr)
        for log, title, out, w, xm, cm in STAGE_RUNS:
            if os.path.exists(log):
                curve_stage(log, title, out, w, xm, cm)
