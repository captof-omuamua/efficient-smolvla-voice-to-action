#!/usr/bin/env bash
# Startup and peak memory, one fresh process per run (startup_final.py). About 10 min.
#   A0, A1, C: eager, one run each.
#   B cold:  compiler caches off (TORCHINDUCTOR_FORCE_DISABLE_CACHES=1), in a fresh scratch cache dir.
#   B fill:  caches on, a fresh empty cache dir: the first start on a new machine fills it.
#   B warm, warm2: the same cache dir again, as a robot's later starts would reuse it.
#   B diag:  warm cache again, with TORCH_LOGS=graph_breaks,recompiles (its log lines show every graph break).
# Every run waits until no other compute process is on the GPU, and is redone if one appears while it runs.
# Log: $LOG_DIR/final_2026-11-07_startup.log
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
OUT=$OUT_DIR/speed/startup
LOG=${LOG:-$LOG_DIR/final_2026-11-07_startup.log}
unset PYTHONPATH  # the venv's editable install: the final worktree
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$RES"
exec >> "$LOG" 2>&1

is_ours() {  # is PID $1 the job $2 or one of its children?
  local p=$1
  while [[ -n $p && $p -gt 1 ]]; do
    [[ $p == "$2" ]] && return 0
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
  return 1
}

run_checked() {  # run "$@"; redo it if another compute process was on the GPU at any point
  while true; do
    while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
      echo "GPU busy, waiting ($(date +%T))"; sleep 30
    done
    "$@" & local job=$! foreign=""
    while kill -0 "$job" 2>/dev/null; do
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
        is_ours "$p" "$job" || foreign="$foreign $p"
      done
      sleep 1
    done
    wait "$job"
    [[ -z $foreign ]] && return 0
    echo "another compute process ($foreign) was on the GPU during the run; redoing it ($(date +%T))"
  done
}

echo "=== startup runs $(date '+%F %T') WT HEAD=$(git -C "$WT" rev-parse HEAD) ==="
for setup in A0 A1 C; do
  echo "=== $setup ($(date +%T)) ==="
  run_checked "$PY" startup_final.py --setup "$setup" --label eager --out-dir "$OUT"
done
COLD=$(mktemp -d "$SCRATCH/inductor_cold_startup_XXXX")
echo "=== B cold, caches off, $COLD ($(date +%T)) ==="
run_checked env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_CACHE_DIR="$COLD" \
  "$PY" startup_final.py --setup B --label cold --out-dir "$OUT"
CACHE=$(mktemp -d "$SCRATCH/inductor_cache_XXXX")
for label in fill warm warm2; do
  echo "=== B $label, cache $CACHE ($(date +%T)) ==="
  run_checked env TORCHINDUCTOR_CACHE_DIR="$CACHE" "$PY" startup_final.py --setup B --label "$label" --out-dir "$OUT"
done
echo "=== B diag, TORCH_LOGS=graph_breaks,recompiles ($(date +%T)) ==="
run_checked env TORCHINDUCTOR_CACHE_DIR="$CACHE" TORCH_LOGS=graph_breaks,recompiles \
  "$PY" startup_final.py --setup B --label diag --out-dir "$OUT"
echo "=== startup runs done $(date '+%F %T') ==="
