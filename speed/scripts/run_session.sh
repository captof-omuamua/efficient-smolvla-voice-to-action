#!/usr/bin/env bash
# Final speed + energy session (energy_session_final.py), then the TF32 side check (precision_check.py).
# Code: the worktree's own venv (editable install of $LEROBOT_DIR). B's first call is a
# cold compile: compiler caches off, in a fresh scratch cache dir. About 15 min in all.
# Extra arguments go to energy_session_final.py (e.g. --rounds 2 --calls-a0 10 --out-dir <dir> for a smoke test).
# Log: LOG (default $LOG_DIR/final_2026-11-07_session.log)
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
WT=$LEROBOT_DIR
PY=$PYTHON
RES=$HERE
OUT=$OUT_DIR/speed
LOG=${LOG:-$LOG_DIR/final_2026-11-07_session.log}
SKIP_PRECISION=${SKIP_PRECISION:-0}
unset PYTHONPATH  # the venv's editable install: the final worktree
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1  # model and processor files are cached
cd "$RES"
until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
  echo "GPU busy, waiting ($(date +%T))" >> "$LOG"; sleep 30
done
COLD=$(mktemp -d "$SCRATCH/inductor_cold_session_XXXX")
{
  echo "=== session $(date '+%F %T') WT HEAD=$(git -C "$WT" rev-parse HEAD) MERGE_HEAD=$(cat "$(git -C "$WT" rev-parse --git-dir)/MERGE_HEAD") ==="
  echo "changes: $(git -C "$WT" status --short | tr '\n' ' ')"
  echo "inductor cache dir (fresh, caches off): $COLD"
} >> "$LOG"
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_CACHE_DIR="$COLD" "$PY" energy_session_final.py "$@" >> "$LOG" 2>&1
echo "=== session done $(date '+%F %T') ===" >> "$LOG"
if [[ $SKIP_PRECISION == 0 ]]; then
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy, waiting ($(date +%T))" >> "$LOG"; sleep 30
  done
  echo "=== precision check $(date '+%F %T') ===" >> "$LOG"
  "$PY" precision_check.py --out-dir "$OUT/precision_check" >> "$LOG" 2>&1
  echo "=== precision check done $(date '+%F %T') ===" >> "$LOG"
fi
