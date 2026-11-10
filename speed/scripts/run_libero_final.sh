#!/usr/bin/env bash
# Closed-loop LIBERO on the final worktree, with the talk's protocol on today's lerobot-eval:
# 4 suites x 10 tasks x 10 episodes = 400, seed 1000, 10 environments at once (async), n_action_steps=10,
# cameras agentview_image -> camera1, robot0_eye_in_hand_image -> camera2, checkpoint revision pinned (the
# cached one, which LeRobot CI also pins). The checkpoint's own config sets no blank slot; we pass it explicitly.
# NVML's GPU energy counter is read at the start and end of the whole run (the GPU also renders the simulator),
# after a 20 s idle reading. A background loop logs any other compute process seen on the GPU during the run.
# Usage: run_libero_final.sh <run name> <libero_eval_final.py arguments, e.g. --cameras looped --policy.num_steps=10>
# Log: $LOG_DIR/final_2026-11-07_libero_<run name>.log
set -euo pipefail
NAME=$1; shift
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
OUT=${RUNS:-$OUT_DIR/speed/libero_runs}/$NAME
LOG=${LOG:-$LOG_DIR/final_2026-11-07_libero_$NAME.log}
ENERGY=$OUT_DIR/speed/libero_energy_$NAME.jsonl
EPISODES=${EPISODES:-10}
TASKS=${TASKS:-libero_spatial,libero_object,libero_goal,libero_10}
unset PYTHONPATH  # the venv's editable install: the final worktree
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1  # model and LIBERO assets are cached
cd "$RES"
until [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
  echo "GPU busy, waiting ($(date +%T))" >> "$LOG"; sleep 30
done
cmd=("$PY" libero_eval_final.py "$@"
  --policy.path=lerobot/smolvla_libero --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d
  --policy.n_action_steps=10 --policy.device=cuda
  --env.type=libero --env.task="$TASKS" --env.max_parallel_tasks=1
  --eval.batch_size=10 --eval.n_episodes="$EPISODES" --eval.use_async_envs=true
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
  --seed=1000 --output_dir="$OUT")
{
  echo "=== LIBERO $NAME, $(date '+%F %T') ==="
  echo "WT HEAD $(git -C "$WT" rev-parse HEAD), MERGE_HEAD $(cat "$(git -C "$WT" rev-parse --git-dir)/MERGE_HEAD"), changes: $(git -C "$WT" status --short | tr '\n' ' ')"
  "$PY" -c "import lerobot, torch, transformers, mujoco; print('lerobot code:', lerobot.__file__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| mujoco', mujoco.__version__)"
  echo "env: MUJOCO_GL=$MUJOCO_GL HF_HUB_OFFLINE=$HF_HUB_OFFLINE TORCHINDUCTOR_FORCE_DISABLE_CACHES=${TORCHINDUCTOR_FORCE_DISABLE_CACHES:-} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-} TORCH_LOGS=${TORCH_LOGS:-}"
  printf 'command:'; printf ' %q' "${cmd[@]}"; echo
} >> "$LOG"
"$PY" nvml_energy.py --out "$ENERGY" --mark idle --idle-seconds 20 >> "$LOG"
"$PY" nvml_energy.py --out "$ENERGY" --mark start >> "$LOG"
"${cmd[@]}" >> "$LOG" 2>&1 &
job=$!
(  # watch for other compute processes while the eval runs
  while kill -0 "$job" 2>/dev/null; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      q=$p; ours=0
      while [[ -n $q && $q -gt 1 ]]; do [[ $q == "$job" ]] && { ours=1; break; }; q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' '); done
      [[ $ours == 0 ]] && echo "FOREIGN compute process $p on the GPU at $(date '+%F %T')" >> "$LOG"
    done
    sleep 5
  done
) &
status=0; wait "$job" || status=$?
"$PY" nvml_energy.py --out "$ENERGY" --mark end >> "$LOG"
grep "Success rate" "$LOG" | tail -1 || true
echo "=== done $(date '+%F %T'), exit $status ===" >> "$LOG"
exit $status
