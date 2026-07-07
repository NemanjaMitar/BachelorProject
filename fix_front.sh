#!/bin/bash
# Fix the front-half relay-entry falls, then CONSOLIDATE each fix as the highest-
# numbered file in checkpoints/LXY/ so Play.py --model-dir picks it up. Sequential
# to avoid CPU oversubscription; logs to logs/.
cd "/c/Users/Nemanja/Desktop/LastButNotLeast/BachelorProject" || exit 1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
M=logs/fix_front.log; log(){ echo "[$(date '+%H:%M:%S')] $1" | tee -a "$M"; }

# ---- L12: heal the 444 delivery spot (resume the partial fix) ----------------
log "L12 fix start"
python Train.py --num-envs 8 --rollout 256 --total-steps 500000 --goal-level 13 \
  --start-states starts_L12_fix3.json --curriculum --max-steps 200 --p-bottom 0.0 \
  --cur-target-p-bottom 0.0 --cur-advance-rate 0.5 --cur-window 30 --ent-coef 0.02 \
  --level-penalty 25 --fine-walk-frames 3 --extra-charges 22,24,25,28,30 \
  --stop-at-succ 0.97 --resume checkpoints/L12_fix2/ppo_1103872.pt \
  --save-dir checkpoints/L12_fix3 > logs/L12_fix3.log 2>&1
cp "$(ls -t checkpoints/L12_fix3/*.pt | head -1)" checkpoints/L12/ppo_9999999.pt && log "L12 consolidated"

# ---- L17: heal the (45,272) delivery spot ------------------------------------
L17R=$(ls -t checkpoints/L17/*.pt | grep -viE "temp|backup|9999999" | head -1)
log "L17 fix start (resume $L17R)"
python Train.py --num-envs 8 --rollout 256 --total-steps 600000 --goal-level 18 \
  --start-states starts_L17_fix.json --curriculum --max-steps 200 --p-bottom 0.0 \
  --cur-target-p-bottom 0.0 --cur-advance-rate 0.5 --cur-window 30 --ent-coef 0.02 \
  --level-penalty 20 --fine-walk-frames 3 --extra-charges 22,24,28,30 \
  --stop-at-succ 0.95 --resume "$L17R" \
  --save-dir checkpoints/L17_fix > logs/L17_fix.log 2>&1
cp "$(ls -t checkpoints/L17_fix/*.pt | head -1)" checkpoints/L17/ppo_9999999.pt && log "L17 consolidated"
log "FRONT FIX DONE"
