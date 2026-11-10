#!/usr/bin/env bash
# Follow-up 1: PR #4634's test_smolvla_tuple_kv_parity on today's main, with the one-line crop fix
# (pr4634_test_fix.diff, the same per-layer form main's #5220 uses in denoise_step), and unchanged on the PR's own
# pinned stack (torch 2.11, transformers 5.5.4). Nothing is edited in the kept worktree or in lerobot-pr4634;
# the fix lives in scratch worktrees. Each pytest run waits until no other compute process is on the GPU.
# Outputs: $OUT_DIR/speed/followups/pytest_*.txt (task1_test_fix.json: summarize_task1.py). Log: $LOG_DIR/final_followup_1_test_fix.log
set -uo pipefail
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
OUT=$OUT_DIR/speed/followups
mkdir -p "$OUT"
SCR=$SCRATCH/followups
FIX=${FIX_DIR:-$SCR/wt-main-pr4634-testfix}   # main d490e472 + PR head 2d200334 merged + batch_cameras.patch + the test fix
HEAD_WT=${PR_HEAD_DIR:-$SCR/wt-pr4634-head}   # PR head 2d200334 as it is (its own pins: transformers 5.5.4)
NEWPY=$PYTHON  # today's env: torch 2.15, transformers 5.22.1
OLD_LEROBOT_DIR=${OLD_LEROBOT_DIR:?set OLD_LEROBOT_DIR to a LeRobot checkout of PR 4634 at a932c505 with its own .venv}
OLDPY=${OLD_PYTHON:-$OLD_LEROBOT_DIR/.venv/bin/python}  # the PR's pinned env: torch 2.11, transformers 5.5.4
FIX_DIFF=$REPO/speed/patches/pr4634_test_fix.diff
LOG=$LOG_DIR/final_followup_1_test_fix.log
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
exec 9> "$GPU_LOCK"; flock 9
exec >> "$LOG" 2>&1

run() {  # run <name> <dir> <python> <PYTHONPATH or ""> <pytest args...>: one pytest run, output to $OUT/pytest_<name>.txt
  local name=$1 dir=$2 py=$3 pp=$4; shift 4
  until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "GPU busy, waiting ($(date +%T))"; sleep 30
  done
  echo "=== $name ($(date '+%F %T')) ==="
  (
    cd "$dir" || exit 1
    if [[ -n $pp ]]; then export PYTHONPATH=$pp; else unset PYTHONPATH; fi
    {
      echo "dir: $dir | git HEAD $(git rev-parse HEAD) | changes: $(git status --short | tr '\n' ' ')"
      "$py" -c "import lerobot, torch, transformers; print('lerobot code:', lerobot.__file__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
      echo "command: $py -m pytest $*"
      "$py" -m pytest "$@" -p no:cacheprovider -rA
      echo "pytest exit code: $?"
    } > "$OUT/pytest_$name.txt" 2>&1
  )
  tail -3 "$OUT/pytest_$name.txt"
}

echo "=== follow-up 1 start $(date '+%F %T') ==="
run new_fixed_compile_parity "$FIX" "$NEWPY" "$FIX/src" tests/policies/smolvla/test_smolvla_compile_parity.py -v
run new_fixed_smolvla_all "$FIX" "$NEWPY" "$FIX/src" tests/policies/smolvla -v
run old_a932c505_unchanged_compile_parity "$OLD_LEROBOT_DIR" "$OLDPY" "" tests/policies/smolvla/test_smolvla_compile_parity.py -v
run old_head_unchanged_compile_parity "$HEAD_WT" "$OLDPY" "$HEAD_WT/src" tests/policies/smolvla/test_smolvla_compile_parity.py -v
git -C "$HEAD_WT" apply "$FIX_DIFF" && echo "applied the fix to the PR head copy"
run old_head_fixed_tuple_kv "$HEAD_WT" "$OLDPY" "$HEAD_WT/src" tests/policies/smolvla/test_smolvla_compile_parity.py -v -k tuple_kv
git -C "$HEAD_WT" apply -R "$FIX_DIFF" && echo "reverted the fix in the PR head copy"
echo "=== follow-up 1 done $(date '+%F %T') ==="
