#!/usr/bin/env bash
# Run the voice robot loop with the GPU lock held; output is added to logs/voice_robot.log.
# Usage: ./run_voice_robot.sh --stage stage1 --name stage1     (watch with: tail -f logs/voice_robot.log)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs .cache/inductor

unset PYTHONPATH                                   # use only the venv's own packages
export HF_HUB_OFFLINE=1                            # models come from the local cache, no network
export MUJOCO_GL=egl                               # the simulator draws on the GPU without a window
export PYTHONDONTWRITEBYTECODE=1                   # no .pyc files
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one CPU thread each, as on 8 Nov
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"               # keep the compiled policy between starts

PYTHON=<project>/lerobot-final-2026-11-07/.venv/bin/python
echo "=== $(date '+%F %T') start: voice_robot.py $* (waiting for the GPU lock)" >> logs/voice_robot.log
# flock waits for the GPU lock, then runs the loop; all output goes to the log.
flock <project>/.gpu.lock "$PYTHON" voice_robot.py "$@" >> logs/voice_robot.log 2>&1
echo "=== $(date '+%F %T') end: voice_robot.py $*" >> logs/voice_robot.log
