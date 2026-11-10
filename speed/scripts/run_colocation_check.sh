#!/usr/bin/env bash
# Alternating fresh processes: A0 alone vs A0 next to a compiled B (warm compile cache). About 9 min.
# Log: $LOG_DIR/final_2026-11-07_colocation_check.log
set -euo pipefail
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
PY=$PYTHON
RES=$HERE
LOG=$LOG_DIR/final_2026-11-07_colocation_check.log
mkdir -p "$OUT_DIR/speed/colocation_check"
CACHE=${CACHE:?set CACHE to the warm inductor cache dir from run_startup.sh}
unset PYTHONPATH
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR="$CACHE"
cd "$RES"
exec 9> "$GPU_LOCK"; flock 9
i=0
for mode in alone with_b with_b alone alone with_b; do
  i=$((i + 1))
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy, waiting ($(date +%T))" >> "$LOG"; sleep 30
  done
  echo "=== run $i $mode ($(date '+%F %T')) ===" >> "$LOG"
  "$PY" colocation_check.py --mode "$mode" --out "$OUT_DIR/speed/colocation_check/run${i}_${mode}.json" >> "$LOG" 2>&1
done
echo "=== done $(date '+%F %T') ===" >> "$LOG"
