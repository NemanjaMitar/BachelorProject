#!/bin/bash
# Overnight autonomous run. Sequential so it never oversubscribes the CPU; each
# job logs to logs/ and failures don't stop the queue.
cd "/c/Users/Nemanja/Desktop/LastButNotLeast/BachelorProject" || exit 1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
mkdir -p logs
M=logs/night_master.log
log(){ echo "[$(date '+%H:%M:%S')] $1" | tee -a "$M"; }
log "NIGHT RUN START"

# ============ Task 1: convert every combo level to a genuine PPO model ========
# (warm-start -> PPO; --stop-at-succ 0.9 so it stops before over-fitting a small
#  curriculum. Windy levels carry wind_obs + jump_wind macros.)
for L in 26 28 31 33 35 37 38 39 40; do
  WIND=""; case $L in 26|28|31) WIND="--wind-obs --wind-jump 0,4";; esac
  GOAL=$((L+1)); [ "$L" = "40" ] && GOAL=42
  log "FAST L$L start"
  python Train.py --num-envs 8 --rollout 256 --total-steps 500000 --goal-level $GOAL \
    --start-states starts_L${L}_demo.json --curriculum --max-steps 150 --p-bottom 0.0 \
    --cur-target-p-bottom 0.0 --cur-advance-rate 0.5 --cur-window 20 --ent-coef 0.04 \
    --fine-walk-frames 3 --extra-charges 22,24,28,30 $WIND --quarantine-after 100000 \
    --stop-at-succ 0.9 --resume checkpoints/L$L/ppo_ppobc.pt \
    --save-dir checkpoints/L${L}_ppo > logs/L${L}_fast.log 2>&1 || log "FAST L$L FAILED"
  log "FAST L$L done"
done

# ============ Task 3: L12 / L23 -- stop the fall-then-reclimb ==================
# strong --level-penalty punishes dropping a level; p-bottom exposes the full
# traversal (incl. the relay-entry region) so the bad first move gets corrected.
log "FIX L12 start"
python Train.py --num-envs 8 --rollout 256 --total-steps 800000 --goal-level 13 \
  --start-states starts_L12.json --curriculum --max-steps 200 --p-bottom 0.35 \
  --cur-target-p-bottom 0.35 --cur-advance-rate 0.5 --cur-window 30 --ent-coef 0.02 \
  --level-penalty 15 --fine-walk-frames 3 --extra-charges 22,24,25,28,30 \
  --stop-at-succ 0.93 --resume checkpoints/L12/ppo_1011712.pt \
  --save-dir checkpoints/L12_fix > logs/L12_fix.log 2>&1 || log "FIX L12 FAILED"
log "FIX L12 done"

log "FIX L23 start"
python Train.py --num-envs 8 --rollout 256 --total-steps 800000 --goal-level 24 \
  --start-states starts_L23.json --curriculum --max-steps 200 --p-bottom 0.35 \
  --cur-target-p-bottom 0.35 --cur-advance-rate 0.5 --cur-window 30 --ent-coef 0.02 \
  --level-penalty 15 --fine-walk-frames 3 --extra-charges 22,24,28,30 \
  --stop-at-succ 0.93 --resume checkpoints/L23/ppo_190464.pt \
  --save-dir checkpoints/L23_fix > logs/L23_fix.log 2>&1 || log "FIX L23 FAILED"
log "FIX L23 done"

# ============ Task 2: broad training (1 ice=36, 1 windy=29, 2 normal=32,34) ====
# ~20 whole-level valid-platform states each, warm-started to follow the known
# trajectory -> robust from anywhere. Longest jobs, run last so the quick stuff
# is guaranteed done; they checkpoint continuously if the night runs out.
for spec in "36:1000000:" "32:1000000:" "34:1000000:" "29:1200000:--wind-obs --wind-jump 0,4"; do
  L="${spec%%:*}"; rest="${spec#*:}"; STEPS="${rest%%:*}"; WIND="${rest#*:}"
  log "BROAD L$L start ($STEPS steps)"
  python Train.py --num-envs 8 --rollout 256 --total-steps $STEPS --goal-level $((L+1)) \
    --start-states starts_L${L}_broad20.json --curriculum --max-steps 200 --p-bottom 0.0 \
    --cur-target-p-bottom 0.0 --cur-advance-rate 0.45 --cur-window 25 --ent-coef 0.05 \
    --fine-walk-frames 3 --extra-charges 22,24,28,30 $WIND --quarantine-after 500 \
    --resume checkpoints/L$L/ppo_ppobc.pt \
    --save-dir checkpoints/L${L}_broad > logs/L${L}_broad.log 2>&1 || log "BROAD L$L FAILED"
  log "BROAD L$L done"
done

log "NIGHT RUN COMPLETE"
