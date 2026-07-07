#!/usr/bin/env python
"""Batch the combo->PPO conversion across all combo levels, for the thesis:
produce a warm-start + demo curriculum for each, and emit ready-to-run PPO
commands (with per-level log files) so the work can be split across machines.

  python Batch.py demos          # BC warm-start + curriculum for every level
  python Batch.py sample 29,30,34,36,37   # broad state sets for the 'big' runs
  python Batch.py plan           # write train_fast.txt / train_broad.txt cmds

Fast run  = quick PPO from the demo curriculum (policy follows the known path,
            robust to small jitter) -> checkpoints/LNN_ppo, log logs/LNN.log
Broad run = PPO over SampleStates' whole-level distribution -> robust from
            anywhere -> checkpoints/LNN_broad, log logs/LNN_broad.log
"""
import os, sys, subprocess

# combo levels that need conversion. launch = post-settle x for ICE levels;
# goal override for L40 (its combo spans the L41 trap to level 42).
LEVELS = {
    26: {}, 28: {}, 29: {}, 31: {},
    32: {}, 33: {}, 34: {}, 35: {},
    36: {"launch": 452}, 37: {"launch": 452}, 38: {"launch": 336},
    39: {}, 40: {"goal": 42},
    # L42 (summit/win) is trained separately -- success = reaching the babe,
    # not level>=goal, so it needs the win-reward variant.
}
PY = sys.executable


def is_windy(level):
    import torch
    ck = torch.load(f"checkpoints/L{level}/ppo_bc.pt", map_location="cpu",
                    weights_only=False)
    combo = ck["action_cfg"]["wind_combo"][0]
    return any((s[0] not in ("walk", "settle", "jump")) and len(s) == 3
               for s in combo)


def demos():
    for lvl, c in LEVELS.items():
        cmd = [PY, "Demos.py", "--level", str(lvl)]
        if "launch" in c:
            cmd += ["--launch-x", str(c["launch"])]
        if "goal" in c:
            cmd += ["--goal-level", str(c["goal"])]
        print(f"\n=== Demos L{lvl} ===")
        subprocess.run(cmd)


def sample(levels):
    for lvl in levels:
        cmd = [PY, "SampleStates.py", "--level", str(lvl), "--merge-demo"]
        if is_windy(lvl):
            cmd += ["--windy"]
        subprocess.run(cmd)


def plan():
    os.makedirs("logs", exist_ok=True)
    fast, broad = [], []
    for lvl, c in LEVELS.items():
        goal = c.get("goal", lvl + 1)
        windy = " --wind-obs --wind-jump 0,4" if is_windy(lvl) else ""
        base = (f"--num-envs 8 --rollout 256 --goal-level {goal} --curriculum "
                f"--max-steps 150 --p-bottom 0.0 --cur-target-p-bottom 0.0 "
                f"--cur-advance-rate 0.5 --cur-window 20 --ent-coef 0.04 "
                f"--fine-walk-frames 3 --extra-charges 22,24,28,30{windy} "
                f"--quarantine-after 400 --resume checkpoints/L{lvl}/ppo_ppobc.pt")
        fast.append(
            f"{PY} Train.py --total-steps 600000 --start-states "
            f"starts_L{lvl}_demo.json {base} --stop-at-succ 0.9 "
            f"--save-dir checkpoints/L{lvl}_ppo > logs/L{lvl}.log 2>&1")
        broad.append(
            f"{PY} Train.py --total-steps 3000000 --start-states "
            f"starts_L{lvl}_broad.json {base} "
            f"--save-dir checkpoints/L{lvl}_broad > logs/L{lvl}_broad.log 2>&1")
    open("train_fast.txt", "w").write("\n".join(fast) + "\n")
    open("train_broad.txt", "w").write("\n".join(broad) + "\n")
    print(f"wrote train_fast.txt ({len(fast)} cmds) and train_broad.txt")
    print("logs go to logs/L<n>.log -- that's your per-level training data.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if mode == "demos":
        demos()
    elif mode == "sample":
        sample([int(x) for x in sys.argv[2].split(",")])
    else:
        plan()
