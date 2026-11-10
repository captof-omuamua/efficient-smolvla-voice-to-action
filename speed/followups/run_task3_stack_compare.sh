#!/usr/bin/env bash
# Follow-up 3: the talk's stack (OLD) vs today's (NEW), alternating fresh processes, same TF32 setting.
#   OLD: lerobot-pr4634 venv (torch 2.11, transformers 5.5.4) + PYTHONPATH to a scratch worktree at 8c4f33e5 with
#        the talk's batching patch (smolvla-fast-libero/patches/smolvla_batch_cameras.patch).
#   NEW: the kept final worktree's venv (torch 2.15, transformers 5.22.1); nothing in that worktree is edited.
# Order: B fill runs on each stack (warm compile caches, one dir per stack), then A0 in mirrored order
# OLD NEW NEW OLD OLD NEW, then B (warm caches) in the same order. Before every process: at least 45 s idle,
# then wait until the CPU package is at most 62 C (up to 180 s in all). About 30 min.
# Outputs: $OUT_DIR/speed/followups/stack_compare/. Log: $LOG_DIR/final_followup_3_stack_compare.log
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
OUT=$OUT_DIR/speed/followups/stack_compare
SCR=$SCRATCH/followups
mkdir -p "$OUT" "$SCR"
OLD_SRC=${OLD_SRC:-$SCR/wt-talk-8c4f33e5/src}   # the talk's 31 Oct code: PR 4634 head 8c4f33e5 + its batching patch
OLD_PY=${OLD_PYTHON:?set OLD_PYTHON to the python of the PR 4634 pinned venv (torch 2.11, transformers 5.5.4)}
NEW_PY=$PYTHON
LOG=$LOG_DIR/final_followup_3_stack_compare.log
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$HERE"
exec 9> "$GPU_LOCK"; flock 9
exec >> "$LOG" 2>&1
OLD_CACHE=$(mktemp -d "$SCR/cache_stack_old_XXXX")
NEW_CACHE=$(mktemp -d "$SCR/cache_stack_new_XXXX")

pkg_temp() { for z in /sys/class/thermal/thermal_zone*; do [[ $(cat "$z/type") == x86_pkg_temp ]] && echo $(( $(cat "$z/temp") / 1000 )); done; }

cool_down() {  # at least 45 s idle, then until the CPU package is at most 62 C, 180 s at most in all
  local waited=45
  sleep 45
  while (( $(pkg_temp) > 62 && waited < 180 )); do sleep 5; waited=$((waited + 5)); done
  echo "cool-down ${waited} s: CPU package $(pkg_temp) C, GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C"
}

run() {  # run <OLD|NEW> <setup> <label>
  local stack=$1 setup=$2 label=$3
  cool_down
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy, waiting ($(date +%T))"; sleep 30
  done
  echo "=== $stack $setup $label ($(date '+%F %T')) ==="
  if [[ $stack == OLD ]]; then
    env PYTHONPATH="$OLD_SRC" TORCHINDUCTOR_CACHE_DIR="$OLD_CACHE" "$OLD_PY" stack_compare.py --stack OLD \
      --setup "$setup" --label "$label" --out-dir "$OUT" 2>&1 | grep -v -E "AUTOTUNE|triton_|strides:|dtypes:|Autotune Choices|best_kernel|^  (mm|bmm|addmm|convolution|extern)|^\s*$" || true
  else
    env -u PYTHONPATH TORCHINDUCTOR_CACHE_DIR="$NEW_CACHE" "$NEW_PY" stack_compare.py --stack NEW \
      --setup "$setup" --label "$label" --out-dir "$OUT" 2>&1 | grep -v -E "AUTOTUNE|triton_|strides:|dtypes:|Autotune Choices|best_kernel|^  (mm|bmm|addmm|convolution|extern)|^\s*$" || true
  fi
}

echo "=== follow-up 3 start $(date '+%F %T'); OLD worktree $(git -C "$OLD_SRC/.." rev-parse HEAD), changes: $(git -C "$OLD_SRC/.." status --short | tr '\n' ' ') ==="
run OLD B fill
run NEW B fill
i=0
for stack in OLD NEW NEW OLD OLD NEW; do i=$((i + 1)); run "$stack" A0 "run$i"; done
i=0
for stack in OLD NEW NEW OLD OLD NEW; do i=$((i + 1)); run "$stack" B "run$i"; done
echo "=== follow-up 3 done $(date '+%F %T') ==="
