#!/bin/bash
# Regenerate the deterministic combo models (verified 0->babe chain) as the
# relay's pick for 25-42. Saved as ppo_9999999.pt so latest_ckpt selects it over
# any conversion checkpoint in the same folder. L25(approach seed) / L27(user
# PPO) / L41(spanned by L40) are left alone.
cd "/c/Users/Nemanja/Desktop/LastButNotLeast/BachelorProject" || exit 1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
mkdir -p logs
for L in 26 28 29 30 31 32 33 34 35 36 37 38 39 40 42; do
  args="--level $L --out checkpoints/L$L/ppo_9999999.pt"
  case $L in
    36|37) args="$args --launch-x 452";;
    38)    args="$args --launch-x 336";;
  esac
  case $L in
    40) args="$args --goal-level 42";;
  esac
  if python BC.py $args > logs/regen_L$L.log 2>&1; then
    echo "L$L regenerated"
  else
    echo "L$L FAILED (see logs/regen_L$L.log)"
  fi
done
echo "COMBO REGEN COMPLETE"
