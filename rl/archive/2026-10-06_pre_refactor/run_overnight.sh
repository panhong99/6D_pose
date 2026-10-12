#!/usr/bin/env bash
# Long headless PPO run for Pick-Place-M1013-2F85-v0, then record a play video of the final policy.
# Usage: bash rl/run_overnight.sh [ITERATIONS] [NUM_ENVS]      (4096 envs ~1.0 s/iteration, ~4 GB GPU on RTX 5070)
set -uo pipefail
ITERS="${1:-8000}"; ENVS="${2:-4096}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB="${ISAACLAB_PATH:-$HOME/IsaacLab}"
OUT="$HERE/logs/overnight_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$OUT"
unset CONDA_PREFIX CONDA_DEFAULT_ENV; export TERM=xterm
cd "$LAB"
echo "train start $(date)  iters=$ITERS envs=$ENVS" | tee "$OUT/status.txt"
./isaaclab.sh -p "$HERE/train_pick_place.py" --num_envs "$ENVS" --headless --max_iterations "$ITERS" > "$OUT/train.log" 2>&1
echo "train exit $? $(date)" | tee -a "$OUT/status.txt"
./isaaclab.sh -p "$HERE/train_pick_place.py" --play --num_envs 16 --headless --enable_cameras --video --video_length 900 > "$OUT/play.log" 2>&1
echo "play exit $? $(date)" | tee -a "$OUT/status.txt"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/train.log" | grep -E "Learning iteration|Mean reward|Mean episode length|Episode_Termination|lifting_plate|placed" | tail -n 40 > "$OUT/summary_tail.txt"
echo "done $(date)" | tee -a "$OUT/status.txt"
