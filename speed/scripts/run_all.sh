#!/usr/bin/env bash
# Every final GPU job, one after another, under the shared GPU lock ($GPU_LOCK). About 2 h.
#   1. run_session.sh: speed + energy session (A0, A1, C, B), then the TF32 side check   ~17 min
#   2. run_startup.sh: startup and peak memory, one fresh process per run                ~12 min
#   3. LIBERO A1: eager, 10 steps, no blank slot, cameras one call each                   ~35 min
#   4. LIBERO B: whole-model compile, 1 step, no blank slot, cameras batched              ~35 min
#      (compiler caches off in a fresh dir; TORCH_LOGS=recompiles,graph_breaks to catch any recompile)
# Progress: $LOG_DIR/final_2026-11-07_all.log (each job also writes its own final_2026-11-07_*.log)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this script's folder
REPO=$(cd "$HERE/../.." && pwd)                      # the repository root
OUT_DIR=${OUT_DIR:-$REPO/runs}                       # new outputs (gitignored); never the committed results
LOG_DIR=${LOG_DIR:-$REPO/logs}                       # logs (gitignored)
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/efficient-smolvla-voice-to-action}   # temporary compile caches and worktrees
GPU_LOCK=${GPU_LOCK:-/tmp/efficient-smolvla-voice-to-action.gpu.lock}  # one GPU job at a time
export OUT_DIR LOG_DIR SCRATCH GPU_LOCK
mkdir -p "$OUT_DIR/speed" "$LOG_DIR" "$SCRATCH"
RES=$HERE
LOG=$LOG_DIR/final_2026-11-07_all.log
step() { echo "=== $(date '+%F %T') $* ===" >> "$LOG"; }
exec 9> "$GPU_LOCK"
step "waiting for the GPU lock"; flock 9; step "got the GPU lock"

step "1. session + precision check"
"$RES/run_session.sh" || step "session FAILED (exit $?)"
step "2. startup runs"
"$RES/run_startup.sh" || step "startup FAILED (exit $?)"
step "3. LIBERO A1"
"$RES/run_libero_final.sh" final_2026-11-07_A1_eager_steps10 --cameras looped \
  --policy.num_steps=10 --policy.empty_cameras=0 || step "LIBERO A1 FAILED (exit $?)"
step "4. LIBERO B"
COLD=$(mktemp -d "$SCRATCH/inductor_cold_libero_XXXX")
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_CACHE_DIR="$COLD" TORCH_LOGS=recompiles,graph_breaks \
  "$RES/run_libero_final.sh" final_2026-11-07_B_tuned --cameras batched \
  --policy.compile_model=true --policy.num_steps=1 --policy.empty_cameras=0 || step "LIBERO B FAILED (exit $?)"
step "all done; GPU free"
