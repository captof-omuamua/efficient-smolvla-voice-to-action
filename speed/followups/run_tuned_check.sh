#!/usr/bin/env bash
# Order: P M L, then P M L again (each config has its own compile cache: first run cold, second warm).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this script's folder
REPO=$(cd "$HERE/../.." && pwd)                      # the repository root
OUT_DIR=${OUT_DIR:-$REPO/runs}                       # new outputs (gitignored); never the committed results
LOG_DIR=${LOG_DIR:-$REPO/logs}                       # logs (gitignored)
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/efficient-smolvla-voice-to-action}   # temporary compile caches and worktrees
GPU_LOCK=${GPU_LOCK:-/tmp/efficient-smolvla-voice-to-action.gpu.lock}  # one GPU job at a time
export OUT_DIR LOG_DIR SCRATCH GPU_LOCK
mkdir -p "$OUT_DIR/speed" "$LOG_DIR" "$SCRATCH"
LEROBOT_DIR=${LEROBOT_DIR:?set LEROBOT_DIR to the patched LeRobot checkout (see README, Reproduce)}
PYTHON=${PYTHON:-$LEROBOT_DIR/.venv/bin/python}
S=$OUT_DIR/speed/tuned_without_pr4634   # outputs
mkdir -p "$S"
CACHE=$SCRATCH/tuned_check_cache   # one compile cache per configuration
PY=$PYTHON
MAIN=${PLAIN_MAIN_SRC:?set PLAIN_MAIN_SRC to the src folder of a plain LeRobot main clone}
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_LOGS=recompiles,graph_breaks
cd "$HERE"
run() { # label, cache, cameras, pythonpath
  echo "=== $1 $(date '+%F %T') cache=$2 cameras=$3 PYTHONPATH=$4 ==="
  PYTHONPATH=$4 TORCHINDUCTOR_CACHE_DIR=$CACHE/$2 flock "$GPU_LOCK" $PY "$HERE/tuned_check.py" --label $1 --cameras $3 --out-dir $S > $S/$1.log 2>&1
  echo "exit $?"; grep -a "^\[" $S/$1.log
}
for r in 1 2; do
  run P_withPR_batched_r$r P native ""
  run M_plainmain_r$r M native "$MAIN"
  run L_withPR_looped_r$r L looped ""
done
echo "=== all done $(date '+%F %T') ==="
