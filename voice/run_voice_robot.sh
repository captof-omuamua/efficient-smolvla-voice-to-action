#!/usr/bin/env bash
# Run the voice robot loop with the GPU lock held; output is added to $LOG_DIR/voice_robot.log.
# Usage: ./run_voice_robot.sh --stage stage1 --name stage1     (watch with: tail -f $LOG_DIR/voice_robot.log)
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)                                 # the repository root
LEROBOT_DIR=${LEROBOT_DIR:?set LEROBOT_DIR to the patched LeRobot checkout (see README, Reproduce)}
PYTHON=${PYTHON:-$LEROBOT_DIR/.venv/bin/python}
OUT_DIR=${OUT_DIR:-$REPO/runs}                       # results go to $OUT_DIR/voice_robot/<name> (gitignored)
LOG_DIR=${LOG_DIR:-$REPO/logs}
GPU_LOCK=${GPU_LOCK:-/tmp/efficient-smolvla-voice-to-action.gpu.lock}
export OUT_DIR
mkdir -p "$LOG_DIR" .cache/inductor

unset PYTHONPATH                                   # use only the venv's own packages
export HF_HUB_OFFLINE=1                            # models come from the local cache, no network
export MUJOCO_GL=egl                               # the simulator draws on the GPU without a window
export PYTHONDONTWRITEBYTECODE=1                   # no .pyc files
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one CPU thread each, as on 8 Nov
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"               # keep the compiled policy between starts

echo "=== $(date '+%F %T') start: voice_robot.py $* (waiting for the GPU lock)" >> "$LOG_DIR/voice_robot.log"
# flock waits for the GPU lock, then runs the loop; all output goes to the log.
flock "$GPU_LOCK" "$PYTHON" voice_robot.py "$@" >> "$LOG_DIR/voice_robot.log" 2>&1
echo "=== $(date '+%F %T') end: voice_robot.py $*" >> "$LOG_DIR/voice_robot.log"
