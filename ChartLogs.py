#!/usr/bin/env python
"""Parse Train.py logs into thesis-ready data + charts.

Reads logs/*.log, extracts the per-update metrics (return, success rate, policy
entropy, KL, mean level, curriculum stage), writes a tidy CSV per run and a
multi-panel learning-curve PNG. Also a combined success-rate figure across runs.

    python ChartLogs.py                 # all logs/*.log
    python ChartLogs.py L30_broad L32   # specific runs
"""
import os, re, sys, csv, glob

ROW = re.compile(
    r"upd\s+(\d+)\s*\|\s*step\s+(\d+).*?ret\s+(-?[\d.]+)\s*\|\s*succ\s+([\d.]+)"
    r"\s*\|\s*mean_lvl\s+([\d.]+).*?cur\s+(\d+)/(\d+).*?kl\s+([-+]?[\d.]+)"
    r"\s*\|\s*ent\s+([\d.]+)")
COLS = ["upd", "step", "return", "success", "mean_level", "cur_stage",
        "cur_total", "kl", "entropy"]


def parse(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = ROW.search(line)
        if m:
            rows.append([int(m[1]), int(m[2]), float(m[3]), float(m[4]),
                         float(m[5]), int(m[6]), int(m[7]), float(m[8]),
                         float(m[9])])
    return rows


def main():
    os.makedirs("charts", exist_ok=True)
    names = sys.argv[1:]
    files = ([f"logs/{n}.log" for n in names] if names
             else sorted(glob.glob("logs/*.log")))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except Exception:
        have_plt = False
        print("matplotlib not available -> CSV only")

    summary = []
    for path in files:
        rows = parse(path)
        if not rows:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        with open(f"charts/{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
        steps = [r[1] for r in rows]
        succ = [r[3] for r in rows]
        summary.append((name, steps, succ))
        final = rows[-1]
        print(f"{name:<16} updates={len(rows):>4} final succ={final[3]:.2f} "
              f"mean_lvl={final[4]:.1f} ent={final[8]:.2f} -> charts/{name}.csv")
        if have_plt:
            fig, ax = plt.subplots(2, 2, figsize=(11, 7))
            fig.suptitle(f"{name} — PPO learning curves", fontweight="bold")
            panels = [("success", 3, "Success rate", "tab:green"),
                      ("return", 2, "Episode return", "tab:blue"),
                      ("entropy", 8, "Policy entropy (exploration)", "tab:red"),
                      ("mean_level", 4, "Mean level reached", "tab:purple")]
            for a, (key, idx, title, c) in zip(ax.ravel(), panels):
                a.plot(steps, [r[idx] for r in rows], color=c, lw=1.3)
                a.set_title(title, fontsize=10); a.set_xlabel("env steps")
                a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"charts/{name}.png", dpi=130); plt.close(fig)

    if have_plt and len(summary) > 1:
        plt.figure(figsize=(9, 5))
        for name, steps, succ in summary:
            plt.plot(steps, succ, lw=1.2, label=name)
        plt.title("Success rate across runs", fontweight="bold")
        plt.xlabel("env steps"); plt.ylabel("success rate")
        plt.grid(alpha=0.3); plt.legend(fontsize=7, ncol=2)
        plt.tight_layout(); plt.savefig("charts/_all_success.png", dpi=130)
        print("wrote charts/_all_success.png (combined)")


if __name__ == "__main__":
    main()
