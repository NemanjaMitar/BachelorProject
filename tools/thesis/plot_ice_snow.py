# -*- coding: utf-8 -*-
"""Thesis figure for an ICE or SNOW level, in the same visual language as
`plot_l02.py` (Slika 7.1) so the two can sit next to each other in the text.

One panel differs, and it is the point of the chapter. On levels 0-2 the fourth
panel is the FORWARD curriculum: a stage counter that climbs as the agent is
allowed to start further from the goal. The ice and snow levels are trained by a
BACKWARD curriculum over a demonstration (Salimans & Chen 2018), so the same
panel is a ladder that walks DOWN: it starts one action from the exit and steps
back through the demonstration as each rung is cleared, reaching 0 -- the state
the relay really delivers the king into -- only when the level is solved.

    python tools/thesis/plot_ice_snow.py                     # both defaults
    python tools/thesis/plot_ice_snow.py ice_L36_pure night_L31
"""
import os, sys

# the labels are Cyrillic and the Windows console is cp1252, which cannot encode
# them -- without this the figure is written and the summary line then crashes
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chart_ice as C

# level -> (biome word, Serbian title)
TITLES = {
    "ice_L36_pure": "Ниво 36 — ледени биом",
    "ice_L37_walk": "Ниво 37 — ледени биом",
    "ice_L38_relay": "Ниво 38 — ледени биом",
    "ice_L36_nosettle": "Ниво 36 — ледени биом (без settle акције)",
    "night_L26": "Ниво 26 — снежни биом (бочни ветар)",
    "night_L29": "Ниво 29 — снежни биом (бочни ветар)",
    "night_L30": "Ниво 30 — снежни биом (бочни ветар)",
    "night_L31": "Ниво 31 — снежни биом (бочни ветар)",
}

STYLE = {
    "figure.facecolor": "#181a20", "axes.facecolor": "#20232b",
    "savefig.facecolor": "#181a20",
    "text.color": "#c9cdd6", "axes.labelcolor": "#aeb3bd",
    "axes.titlecolor": "#e4e7ec",
    "xtick.color": "#868c98", "ytick.color": "#868c98",
    "axes.edgecolor": "#3a3f4a", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#ffffff", "grid.alpha": 0.05,
    "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.titleweight": "normal", "axes.titlesize": 13, "font.size": 11,
}
COL = {"s": "#4fd07a", "r": "#54a8e6", "e": "#f0a94e", "c": "#b184ea"}


def smooth(y, w):
    """Rolling mean, same length. A 2700-update ice run is far denser than the
    79 updates of the 0-2 figure, so the raw trace is drawn faintly underneath
    and the mean carries the shape."""
    if w <= 1:
        return y
    out, acc = [], []
    for v in y:
        acc.append(v)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def runs(rows):
    """Split a log into separate TRAINING RUNS.

    A log is appended to across restarts, and `upd` restarts at 1 each time, so
    plotting the file as one series draws every run on top of the others against
    a non-monotonic x -- which is exactly the tangle the first version of this
    figure produced from ice_L36_pure.log (five runs in one file)."""
    out, cur = [], []
    for r in rows:
        if cur and r[0] <= cur[-1][0]:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def panel(a, x, y, colour, title, win, ylim=None, zero=False):
    if win > 1:
        a.plot(x, y, color=colour, lw=0.8, alpha=0.22)
    a.plot(x, smooth(y, win), color=colour, lw=2.2)
    if zero:
        a.axhline(0, color="#454b57", lw=0.9)
    if ylim:
        a.set_ylim(*ylim)
    a.set_title(title)


def figure(name):
    path = os.path.join("logs", name + ".log")
    rows = C.parse(path)
    if not rows:
        print(f"  {name}: no parsable rows, skipped")
        return None
    segs = runs(rows)
    if len(segs) > 1:
        rows = max(segs, key=len)
        print(f"  {name}: {len(segs)} runs in the log, plotting the longest "
              f"({len(rows)} updates)")
    U = [r[0] for r in rows]
    R = [r[3] for r in rows]
    S = [r[4] for r in rows]
    K = [r[5] for r in rows]
    KMAX = rows[0][6]
    E = [r[9] for r in rows]
    win = max(1, len(U) // 110)

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.8))
    panel(ax[0, 0], U, S, COL["s"], "Успешност", win, ylim=(0, 1.02))
    panel(ax[0, 1], U, R, COL["r"], "Добит", win, zero=True)
    panel(ax[1, 0], U, E, COL["e"], "Ентропија", win)

    a = ax[1, 1]
    a.step(U, K, color=COL["c"], lw=2.2, where="post")
    a.set_ylim(-0.4, KMAX + 0.4)
    a.set_yticks(range(0, KMAX + 1, max(1, KMAX // 8)))
    a.axhline(0, color="#4fd07a", lw=1.0, ls="--", alpha=0.55)
    a.text(U[-1], 0.28, "прави улаз у ниво  ", color="#4fd07a", fontsize=9,
           ha="right", va="bottom", alpha=0.9)
    a.set_title("Курикулум уназад (полазни снимак)")

    for a in ax.flat:
        a.set_xlabel("ажурирање")
        a.margins(x=0.01)
    fig.suptitle(TITLES.get(name, name), fontweight="normal", fontsize=12,
                 color="#c9cdd6")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("charts", exist_ok=True)
    out = os.path.join("charts", f"{name}_train.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {name}: {len(U)} ажурирања -> {out} | лествица {K[0]}->{K[-1]}"
          f"/{KMAX} | успешност {S[0]:.2f}->{S[-1]:.2f} | "
          f"ентропија {E[0]:.2f}->{E[-1]:.2f}")
    return out


def main():
    names = sys.argv[1:] or ["ice_L36_pure", "night_L31"]
    made = [figure(n) for n in names]
    return 0 if any(made) else 1


if __name__ == "__main__":
    sys.exit(main())
