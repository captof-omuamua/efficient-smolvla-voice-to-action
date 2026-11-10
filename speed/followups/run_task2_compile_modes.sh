#!/usr/bin/env bash
# Follow-up 2: B with compile_mode max-autotune (today's B), default and reduce-overhead, one fresh process per run,
# on today's stack. Order: two cold runs per mode (caches off, each in a fresh empty cache dir) in mirrored order,
# then per mode a fill run (caches on, fresh empty dir) and a warm run (that dir again), warm in mirrored order.
# 60 s of GPU cool-down before every run. About 35 min. Outputs: $OUT_DIR/speed/followups/compile_modes/.
# Log: $LOG_DIR/final_followup_2_compile_modes.log
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
OUT=$OUT_DIR/speed/followups/compile_modes
SCR=$SCRATCH/followups
mkdir -p "$OUT" "$SCR"
LOG=$LOG_DIR/final_followup_2_compile_modes.log
unset PYTHONPATH  # the venv's editable install: the kept final worktree (not edited)
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
exec 9> "$GPU_LOCK"; flock 9
exec >> "$LOG" 2>&1

run() {  # run <mode> <label> [VAR=value ...]: one fresh process after a 60 s cool-down
  local mode=$1 label=$2; shift 2
  sleep 60
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy, waiting ($(date +%T))"; sleep 30
  done
  echo "=== $mode $label ($(date '+%F %T')) GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C $* ==="
  env "$@" "$PY" compile_modes.py --mode "$mode" --label "$label" --out-dir "$OUT" 2>&1 | grep -v -E "AUTOTUNE|triton_|strides:|dtypes:|Autotune Choices|best_kernel|SingleProcess AUTOTUNE|^  (mm|bmm|addmm|convolution|extern)|^\s*$" || true
}

echo "=== follow-up 2 start $(date '+%F %T') ==="
"$PY" -c "import lerobot, torch, transformers; print('lerobot code:', lerobot.__file__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
for spec in max-autotune:cold1 default:cold1 reduce-overhead:cold1 reduce-overhead:cold2 default:cold2 max-autotune:cold2; do
  mode=${spec%%:*}; label=${spec##*:}
  run "$mode" "$label" TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_CACHE_DIR="$(mktemp -d "$SCR/cache_cold_XXXX")"
done
declare -A CACHE
for mode in max-autotune default reduce-overhead; do
  CACHE[$mode]=$(mktemp -d "$SCR/cache_${mode}_XXXX")
  run "$mode" fill TORCHINDUCTOR_CACHE_DIR="${CACHE[$mode]}"
done
for mode in reduce-overhead default max-autotune; do
  run "$mode" warm TORCHINDUCTOR_CACHE_DIR="${CACHE[$mode]}"
done
echo "=== follow-up 2 done $(date '+%F %T') ==="
