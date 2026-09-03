#!/usr/bin/env python
"""One thesis-ready learning curve for a single icy level.

`chart_ice.py` overlays every ice log against the PPO update counter, which
restarts at 1 on every resume -- fine for watching a run, wrong for a figure of
a campaign that was resumed a dozen times. This plots CUMULATIVE ENVIRONMENT
STEPS instead (positive deltas summed, so a resume from an earlier checkpoint
does not fold the axis back on itself) and reports the one number a reader of
the thesis wants: how the backward curriculum walked from the rung next to the
exit down to the level's real entry, and what the deterministic policy scored
along the way.

    python tools/chart_ice_level.py --log logs/ice_L36_pure.log --level 36
    python tools/chart_ice_level.py --log logs/ice_L36_pure.log --level 36 --dark
"""
import os, re, sys, csv, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROW = re.compile(
    r"upd\s+(\d+)\s*\|\s*step\s+(\d+)\s*\|\s*(-?[\d.]+) sps\s*\|\s*"
    r"ret\s+(-?[\d.nan]+)\s*\|\s*succ\s+([\d.nan]+)\s*\|\s*k\s+(\d+)/(\d+)\s*"
    r"\(front\s+([\d.nan]+)(?:\s+greedy\s+([\d.nan]+))?\)\s*\|\s*"
    r"ent\s+([\d.]+)")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse(path):
    out = []
    for line in open(path, errors="ignore"):
        m = ROW.search(line)
        if m:
            out.append(dict(upd=int(m[1]), step=int(m[2]), ret=_f(m[4]),
                            succ=_f(m[5]), k=int(m[6]), kmax=int(m[7]),
                            front=_f(m[8]), greedy=_f(m[9]) if m[9] else np.nan,
                            ent=_f(m[10])))
    return out


def cumulative(steps):
    """Sum only forward progress: a resume from an older checkpoint reports a
    smaller `step`, and treating that as the x position would draw the run
    going backwards in time."""
    out, tot, prev = [], 0.0, None
    for s in steps:
        if prev is not None and s > prev:
            tot += s - prev
        prev = s
        out.append(tot)
    return np.array(out)


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y
    kern = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, kern, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--smooth", type=int, default=25)
    ap.add_argument("--dark", action="store_true",
                    help="dark palette; the default is the light one a printed "
                         "thesis wants")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit(f"no parsable training lines in {args.log}")
    x = cumulative([r["step"] for r in rows]) / 1e6
    k = np.array([r["k"] for r in rows], float)
    kmax = rows[0]["kmax"]
    greedy = np.array([r["greedy"] for r in rows], float)
    succ = np.array([r["succ"] for r in rows], float)
    ent = np.array([r["ent"] for r in rows], float)
    ret = np.array([r["ret"] for r in rows], float)

    if args.dark:
        bg, fg, dim, grid = "#11151c", "#e8ecf2", "#98a0ad", "#2a3038"
        c_g, c_s, c_k, c_e, c_r = "#2fd07a", "#5fb7d8", "#ffd166", "#c58af0", "#ff8f6b"
    else:
        bg, fg, dim, grid = "white", "#1b1f27", "#5b6270", "#d7dbe2"
        c_g, c_s, c_k, c_e, c_r = "#1a7f4b", "#1f6f9c", "#b8860b", "#6b3fa0", "#b5442a"

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(alpha=0.35, color=grid, linewidth=0.7)
        ax.tick_params(colors=dim, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(grid)

    # ---- 1: does the DETERMINISTIC policy cross? ---------------------------
    ax = axes[0]
    ax.plot(x, smooth(succ, args.smooth), color=c_s, lw=1.4,
            label="stochastic rollout success")
    # The gate re-measures the greedy policy every `eval_every` updates and the
    # log repeats the last value in between, so plotting every row draws a
    # sawtooth that is an artefact of the logging, not of the policy. Show the
    # measurements themselves faintly and the trend over them in full.
    ok = ~np.isnan(greedy)
    ax.plot(x[ok], greedy[ok], color=c_g, lw=0.7, alpha=0.22)
    ax.plot(x[ok], smooth(greedy[ok], args.smooth), color=c_g, lw=1.8,
            label="greedy (argmax) crossing rate at the frontier rung")
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("crossing rate", color=fg, fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right", facecolor=bg, edgecolor=grid,
              labelcolor=fg)
    ax.set_title(f"L{args.level}: backward curriculum over demonstration "
                 f"snapshots", color=fg, fontsize=12, pad=10)

    # ---- 2: how far down the demonstration the start has walked ------------
    ax = axes[1]
    ax.step(x, k, where="post", color=c_k, lw=1.6)
    ax.fill_between(x, k, kmax, step="post", color=c_k, alpha=0.12)
    ax.set_ylim(kmax + 0.4, -0.4)                      # rung 0 at the bottom
    ax.set_yticks(range(0, kmax + 1))
    ax.set_ylabel("curriculum rung $k$", color=fg, fontsize=10)
    ax.axhline(0, color=c_g, lw=1.1, ls="--", alpha=0.9)
    ax.annotate("rung 0 = the state the relay really delivers",
                xy=(x[-1], 0), xytext=(-8, 10), textcoords="offset points",
                ha="right", color=c_g, fontsize=8.5)
    ax.annotate(f"rung {kmax} = one action from the exit",
                xy=(x[0], kmax), xytext=(8, 12), textcoords="offset points",
                color=dim, fontsize=8.5)
    # the plateau is the result worth reading off the figure
    plateau = np.argmax(np.bincount(k.astype(int)))
    span = x[k == plateau]
    if len(span) > 20:
        # y is inverted, so a POSITIVE offset lifts the label off the staircase
        ax.annotate(f"{span[-1] - span[0]:.1f}M steps stuck on rung {plateau}",
                    xy=((span[0] + span[-1]) / 2, plateau), xytext=(0, 24),
                    textcoords="offset points", ha="center", va="center",
                    color=dim, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec=grid, lw=0.6))

    # ---- 3: entropy and return --------------------------------------------
    ax = axes[2]
    ax.plot(x, smooth(ent, args.smooth), color=c_e, lw=1.4)
    ax.set_ylabel("policy entropy (nats)", color=c_e, fontsize=10)
    ax.tick_params(axis="y", colors=c_e)
    ax2 = ax.twinx()
    ax2.plot(x, smooth(ret, args.smooth), color=c_r, lw=1.2, alpha=0.85)
    ax2.set_ylabel("mean episode return", color=c_r, fontsize=10)
    ax2.tick_params(axis="y", colors=c_r, labelsize=9)
    ax2.set_facecolor("none")
    for sp in ax2.spines.values():
        sp.set_color(grid)
    ax.set_xlabel("cumulative environment steps (millions)", color=fg,
                  fontsize=10)

    fig.tight_layout()
    out = args.out or os.path.join("figures", f"L{args.level}_learning_curve.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=bg)

    csv_out = os.path.splitext(out)[0] + ".csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cum_steps", "rung_k", "greedy_at_frontier",
                    "rollout_success", "entropy", "mean_return"])
        for i in range(len(rows)):
            w.writerow([int(x[i] * 1e6), int(k[i]), greedy[i], succ[i],
                        ent[i], ret[i]])

    print(f"{len(rows)} points | {x[-1]:.2f}M cumulative steps | "
          f"ladder {kmax} -> {int(k[-1])}")
    print("wrote", out)
    print("wrote", csv_out)


if __name__ == "__main__":
    sys.exit(main())
