#!/usr/bin/env python
"""
Level production line: runs the whole per-level pipeline with NO human,
level after level, and stops loudly when a level genuinely needs attention.

Per level N it chains (each stage a subprocess -- pygame state stays isolated):

  1. HANDOFF   run level N-1's model, bank its real arrival states into
               starts/starts_LN.json. Doubles as the GATE: if the previous model's
               success rate is below --gate, the chain stops (previous level
               was not actually solid).
  2. AUTOSEED  add ~--seeds spread standing spots from the level geometry.
  3. LEVELGRAPH prove a route to level N+1 exists and emit the path states
               as graded curriculum. "NO PATH" stops the chain.
  4. TRAIN     fresh model, standard recipe, early-stops at --stop-succ.



  5. loop      -> level N+1 (its handoff evaluates + feeds from this model).

All NEW levels train with the project's frozen action set:
fine walks 3 + extra charges 22,24,28,30 (37 actions).

Usage:
    python AutoPilot.py --levels 10-14
    python AutoPilot.py --levels 25-30 --no-handoff     # band without a
        predecessor model: geometric seeds only (reconcile with Handoff later)

The chain resumes cleanly: already-trained levels (checkpoints/LN/ exists with
a .pt) are skipped unless --retrain.
"""

import os
import re
import sys
import glob
import json
import argparse
import subprocess

PY = sys.executable
FWF = "3"                      # frozen project action set for all new levels
EXTRA = "22,24,28,30"
WIND_LEVELS = set(range(25, 32))   # wind resets to 0 on entry, so 24 is deterministic
WAIT = "30,60"                     # legacy fixed waits (unused now)
WIND_JUMP = "0,4"                  # atomic wait-for-wind-then-jump buckets
KNOWN_ACTION_CFGS = {23: ("0", ""), 25: ("3", ""), 37: ("3", EXTRA)}


def latest_ckpt(d):
    best, best_step = None, -1
    for f in glob.glob(os.path.join(d, "*.pt")):
        m = re.search(r"(\d+)", os.path.basename(f))
        step = int(m.group(1)) if m else 0
        if step > best_step:
            best, best_step = f, step
    return best


def ckpt_action_flags(path):
    """(fine_walk_frames, extra_charges, wait_frames, wind_obs) CLI values
    for a checkpoint."""
    import torch
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("action_cfg")
    if cfg is not None:
        return (str(int(cfg.get("fine_walk_frames", 0))),
                ",".join(str(c) for c in cfg.get("extra_charges", [])),
                ",".join(str(c) for c in cfg.get("wait_frames", [])),
                bool(cfg.get("wind_obs", False)),
                ",".join(str(c) for c in cfg.get("wind_jump", [])))
    n = ck["model"]["policy_head.weight"].shape[0]
    if n in KNOWN_ACTION_CFGS:
        fine, extra = KNOWN_ACTION_CFGS[n]
        return (fine, extra, "", False, "")
    raise SystemExit(f"cannot infer action set of {path} ({n} actions)")


def run(cmd, log_path, live=False):
    # PYTHONUNBUFFERED: child prints line-by-line into our pipe instead of
    # holding output in 8 KB blocks (which looks like a hung training run).
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(">>", " ".join(cmd), flush=True)
    if live:                      # stream (training) output to the console
        with open(log_path, "w", encoding="utf8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    env=env)
            out = []
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                out.append(line)
            proc.wait()
            return proc.returncode, "".join(out)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (res.stdout or "") + (res.stderr or "")
    with open(log_path, "w", encoding="utf8") as f:
        f.write(out)
    return res.returncode, out


def run_handoff(from_lvl, ckpt, episodes, log_path, out=None):
    """Run Handoff for `from_lvl`'s model; returns (success_rate, new_states)."""
    pf, pe, pw, pwind, pwj = ckpt_action_flags(ckpt)
    cmd = [PY, "Handoff.py", "--level", str(from_lvl),
           "--checkpoint", ckpt,
           "--start-states", f"starts/starts_L{from_lvl}.json",
           "--episodes", str(episodes),
           "--fine-walk-frames", pf]
    if pe:
        cmd += ["--extra-charges", pe]
    if pw:
        cmd += ["--wait-frames", pw]
    if pwind:
        cmd += ["--wind-obs"]
    if pwj:
        cmd += ["--wind-jump", pwj]
    if out:
        cmd += ["--out", out]
    rc, txt = run(cmd, log_path)
    mm = re.search(r"(\d+)/(\d+) episodes reached", txt)
    aa = re.search(r"merged (\d+) new handoff states", txt)
    if rc != 0 or not mm:
        raise SystemExit(f"handoff from level {from_lvl} failed; see {log_path}")
    return (int(mm.group(1)) / max(1, int(mm.group(2))),
            int(aa.group(1)) if aa else 0)


def reconcile(lo, hi, args):
    """Seal the relay across an already-trained band: for each level N in
    [lo, hi], bank level N-1's real arrivals into starts/starts_LN.json, evaluate
    model N from its (now enriched) pool, and briefly resume its training
    when new arrivals appeared or its success rate is below the gate."""
    for lvl in range(lo, hi + 1):
        print(f"\n{'='*66}\n=== RECONCILE LEVEL {lvl}\n{'='*66}", flush=True)
        prev_ck = latest_ckpt(f"checkpoints/L{lvl - 1}")
        my_ck = latest_ckpt(f"checkpoints/L{lvl}")
        if not my_ck:
            print(f"level {lvl}: no model -- skip (train it first)")
            continue
        added = 0
        if prev_ck:
            rate_prev, added = run_handoff(lvl - 1, prev_ck,
                                           args.handoff_episodes,
                                           f"logs/L{lvl}_reconcile_in.log")
            print(f"level {lvl-1} delivers with success {rate_prev:.2f}; "
                  f"{added} new arrival states banked into starts/starts_L{lvl}.json")
        else:
            print(f"level {lvl}: no predecessor model, skipping arrival bank")
        # evaluate THIS level's model from its pool (also feeds N+1's pool)
        rate, _ = run_handoff(lvl, my_ck, args.handoff_episodes,
                              f"logs/L{lvl}_reconcile_eval.log")
        print(f"level {lvl} model success from pool: {rate:.2f}")
        if added == 0 and rate >= args.gate:
            print(f"level {lvl}: sealed, nothing to do")
            continue
        print(f"level {lvl}: resuming training "
              f"({'new arrivals' if added else ''}"
              f"{' + ' if added and rate < args.gate else ''}"
              f"{'low success' if rate < args.gate else ''})")
        mf, me, mw, mwind, mwj = ckpt_action_flags(my_ck)
        cmd = [PY, "Train.py",
               "--num-envs", str(args.num_envs), "--rollout", "256",
               "--total-steps", str(args.reconcile_budget),
               "--goal-level", str(lvl + 1),
               "--start-states", f"starts/starts_L{lvl}.json", "--curriculum",
               "--max-steps", "80",
               "--p-bottom", "0.0", "--cur-target-p-bottom", "0.0",
               "--cur-advance-rate", "0.45", "--cur-window", "20",
               "--ent-coef", "0.06",
               "--fine-walk-frames", mf,
               "--stop-at-succ", str(args.stop_succ),
               "--save-dir", f"checkpoints/L{lvl}",
               "--resume", my_ck]
        if me:
            cmd += ["--extra-charges", me]
        if mw:
            cmd += ["--wait-frames", mw]
        if mwind:
            cmd += ["--wind-obs"]
        if mwj:
            cmd += ["--wind-jump", mwj]
        rc, _txt = run(cmd, f"logs/L{lvl}_reconcile_train.log", live=True)
        if rc != 0:
            raise SystemExit(f"reconcile training failed on level {lvl}")
        rate, _ = run_handoff(lvl, latest_ckpt(f"checkpoints/L{lvl}"),
                              args.handoff_episodes,
                              f"logs/L{lvl}_reconcile_eval2.log")
        print(f"level {lvl} after resume: success {rate:.2f}"
              + ("" if rate >= args.gate else "  (STILL below gate -- "
                 "needs a human look)"))
    print("\nreconcile pass finished.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True, help="e.g. 10-14 or 12")
    ap.add_argument("--no-handoff", action="store_true",
                    help="skip stage 1 (band with no predecessor model); "
                        "geometric seeds only")
    ap.add_argument("--gate", type=float, default=0.7,
                    help="min handoff success rate of the PREVIOUS model")
    ap.add_argument("--stop-succ", type=float, default=0.85)
    ap.add_argument("--budget", type=int, default=3_000_000,
                    help="max training steps per level")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--handoff-episodes", type=int, default=40)
    ap.add_argument("--retrain", action="store_true",
                    help="train even when a checkpoint already exists")
    ap.add_argument("--reconcile", action="store_true",
                    help="RECONCILE mode for already-trained bands: for each "
                        "level, bank the predecessor's real arrival states, "
                        "evaluate the level's model from its pool, and give "
                        "it a short warm-started resume if new arrivals were "
                        "added or the success rate is below the gate")
    ap.add_argument("--reconcile-budget", type=int, default=800_000,
                    help="max extra training steps per level in reconcile mode")
    args = ap.parse_args()
    m = re.match(r"(\d+)(?:-(\d+))?$", args.levels)
    if not m:
        raise SystemExit("--levels must look like 12 or 10-14")
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    os.makedirs("logs", exist_ok=True)

    if args.reconcile:
        reconcile(lo, hi, args)
        return

    for lvl in range(lo, hi + 1):
        print(f"\n{'='*66}\n=== LEVEL {lvl}\n{'='*66}", flush=True)
        starts = f"starts/starts_L{lvl}.json"
        save_dir = f"checkpoints/L{lvl}"

        if not args.retrain and latest_ckpt(save_dir):
            print(f"level {lvl}: checkpoint exists in {save_dir}, skipping "
                  f"(use --retrain to redo)")
            continue

        # ---- 1. handoff from the previous level's model (also the gate) ----
        if not args.no_handoff:
            prev_dir = f"checkpoints/L{lvl - 1}"
            prev_ck = latest_ckpt(prev_dir)
            if not prev_ck:
                raise SystemExit(f"no model for level {lvl-1} in {prev_dir}; "
                                 f"use --no-handoff or train it first")
            rate, _added = run_handoff(lvl - 1, prev_ck,
                                       args.handoff_episodes,
                                       f"logs/L{lvl}_handoff.log")
            print(f"gate: level {lvl-1} model success {rate:.2f}")
            if rate < args.gate:
                raise SystemExit(
                    f"STOP: level {lvl-1} success {rate:.2f} < gate "
                    f"{args.gate}. Its model is not solid enough to hand off "
                    f"-- resume its training before continuing.")

        # ---- 2. geometric seeds --------------------------------------------
        rc, out = run([PY, "AutoSeed.py", "--level", str(lvl),
                       "--n", str(args.seeds), "--out", starts],
                      f"logs/L{lvl}_seed.log")
        if rc != 0:
            raise SystemExit(f"AutoSeed failed; see logs/L{lvl}_seed.log")

        # ---- 3. route proof + emitted curriculum ---------------------------
        windy = lvl in WIND_LEVELS
        graph_cmd = [PY, "LevelGraph.py", "--level", str(lvl),
                     "--starts", starts,
                     "--fine-walk-frames", FWF, "--extra-charges", EXTRA,
                     "--max-depth", "14", "--max-nodes", "2500",
                     "--max-paths", "6", "--emit-starts", starts]
        if windy:
            graph_cmd += ["--wind-jump", WIND_JUMP]
        rc, out = run(graph_cmd, f"logs/L{lvl}_graph.log")
        if rc != 0 or "NO PATH FOUND" in out:
            raise SystemExit(
                f"STOP: no proven route on level {lvl} with the standard "
                f"action set -- needs a human look "
                f"(logs/L{lvl}_graph.log; try Probe --render).")

        # ---- 4. train -------------------------------------------------------
        train_cmd = [PY, "Train.py",
                     "--num-envs", str(args.num_envs), "--rollout", "256",
                     "--total-steps", str(args.budget),
                     "--goal-level", str(lvl + 1),
                     "--start-states", starts, "--curriculum",
                     "--max-steps", "80",
                     "--p-bottom", "0.0", "--cur-target-p-bottom", "0.0",
                     "--cur-advance-rate", "0.45", "--cur-window", "20",
                     "--ent-coef", "0.06",
                     "--fine-walk-frames", FWF, "--extra-charges", EXTRA,
                     "--stop-at-succ", str(args.stop_succ),
                     "--save-dir", save_dir]
        if windy:
            # wind levels: agent sees the wind phase and has atomic
            # wait-for-wind-then-jump actions; trains against randomized phases
            train_cmd += ["--wind-obs", "--wind-jump", WIND_JUMP]
        rc, out = run(train_cmd, f"logs/L{lvl}_train.log", live=True)
        if rc != 0 or not latest_ckpt(save_dir):
            raise SystemExit(f"training failed on level {lvl}; "
                             f"see logs/L{lvl}_train.log")
        print(f"level {lvl} trained -> {latest_ckpt(save_dir)}")

    print("\nAutoPilot finished the requested band.")


if __name__ == "__main__":
    main()
