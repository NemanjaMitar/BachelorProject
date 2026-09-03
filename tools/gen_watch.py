#!/usr/bin/env python
"""Periodically score the newest --world-pool checkpoint on the HELD-OUT pool.

Training success (Train.py's `succ`) is measured with the reverse curriculum
mixed in, so it overstates the real thing. This runs GenEval on the newest
checkpoint every --every seconds and appends one line per evaluation, so the
generalization curve is recorded while the run is still going.

    python tools/gen_watch.py --dir checkpoints/experiments/worldgen \
        --worlds levels/gen_eval --every 1800 --limit 40
"""
import os
import re
import sys
import time
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "ppo_*.pt"))
    if not files:
        return None
    def step(p):
        m = re.search(r"ppo_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(files, key=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints/experiments/worldgen")
    ap.add_argument("--worlds", default="levels/gen_eval")
    ap.add_argument("--every", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--out", default="logs/worldgen_eval.log")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen = set()
    while True:
        ck = newest(args.dir)
        if ck and ck not in seen:
            seen.add(ck)
            cmd = [sys.executable, "-u", os.path.join(HERE, "GenEval.py"),
                   "--checkpoint", ck, "--worlds", args.worlds,
                   "--limit", str(args.limit), "--starts", str(args.starts)]
            try:
                out = subprocess.run(cmd, cwd=HERE, capture_output=True,
                                     text=True, timeout=3600).stdout
            except subprocess.TimeoutExpired:
                out = "TIMEOUT"
            line = next((l for l in out.splitlines() if "UNSEEN" in l), out[-200:])
            stamp = time.strftime("%H:%M:%S")
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(f"{stamp} {line.strip()}\n")
            print(f"{stamp} {line.strip()}", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
