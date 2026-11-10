#!/usr/bin/env bash
# One GPU, three models (PREREG.md, second half): run the robot loop and the voice load together, then analyse.
# Usage: run_share.sh NAME MODE      MODE = tuned | original | calibrate
# Options (environment): ROUNDS=5 BLOCK_S=120 ALONE_S=180 MIN_ROUNDS=3 CAL_DECISIONS=600 REGIMES= (empty = all 8)
# Holds the GPU lock for the whole run. Progress: tail -f $LOG_DIR/share_timing.log
set -u
NAME=${1:?usage: run_share.sh NAME tuned|original|calibrate}
MODE=${2:?usage: run_share.sh NAME tuned|original|calibrate}
ROUNDS=${ROUNDS:-5}
BLOCK_S=${BLOCK_S:-120}
ALONE_S=${ALONE_S:-180}
MIN_ROUNDS=${MIN_ROUNDS:-3}
CAL_DECISIONS=${CAL_DECISIONS:-600}
REGIMES=${REGIMES:-}
WHISPER_BUDGET_MS=${WHISPER_BUDGET_MS:-100}   # turn-taking time budgets (PREREG.md); changed only by its smoke-test rule
LLM_BUDGET_MS=${LLM_BUDGET_MS:-140}
ROBOT_CORES=${ROBOT_CORES:-0-3}   # each program keeps to its own fast CPU cores, so they share only the GPU
VOICE_CORES=${VOICE_CORES:-4-7}   # (PREREG.md, second amendment; cores 0 to 7 are the fast ones on this laptop)

HERE=$(cd "$(dirname "$0")" && pwd)   # this folder: voice/
REPO=$(cd "$HERE/.." && pwd)           # the repository root
LEROBOT_DIR=${LEROBOT_DIR:?set LEROBOT_DIR to the patched LeRobot checkout (see README, Reproduce)}
PY=${PYTHON:-$LEROBOT_DIR/.venv/bin/python}   # voice load and analysis
ROBOT_PY=${ROBOT_PYTHON:-$PY}                 # robot loop only, with PYTHONPATH below (registered run: see docs/GPU_SHARING.md)
ROBOT_PATH=${ROBOT_PYTHONPATH:-}
export ROBOT_LEROBOT_DIR=${ROBOT_LEROBOT_DIR:-$LEROBOT_DIR}   # the robot loop records this checkout's commit
OUT_DIR=${OUT_DIR:-$REPO/runs}
LOG_DIR=${LOG_DIR:-$REPO/logs}
GPU_LOCK=${GPU_LOCK:-/tmp/efficient-smolvla-voice-to-action.gpu.lock}
OUT=$OUT_DIR/share/$NAME
LOG=$LOG_DIR/share_timing.log
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1

case "$MODE" in
  tuned) POLICY_ARGS=(--compile --num-steps 1); GUARD_MS=25 ;;
  original) POLICY_ARGS=(--num-steps 10 --warmup 50); GUARD_MS=170 ;;
  calibrate) POLICY_ARGS=(--compile --num-steps 1 --max-decisions "$CAL_DECISIONS"); GUARD_MS=0 ;;
  *) echo "MODE must be tuned, original or calibrate" >&2; exit 2 ;;
esac

cd "$HERE" || exit 1
mkdir -p "$LOG_DIR" "$OUT_DIR/share"
step() { echo "=== $(date '+%F %T') [$NAME] $* ===" >> "$LOG"; }
now_ns() { "$PY" -c 'import time; print(time.monotonic_ns())'; }   # the same clock both programs use

if [ -e "$OUT" ]; then echo "refusing: $OUT already exists" >&2; exit 1; fi

exec 9> "$GPU_LOCK"
step "waiting for the GPU lock"; flock 9; step "got the GPU lock"
mkdir "$OUT" || { step "refusing: $OUT appeared while waiting"; exit 1; }   # never write into an old run

POLICY_PID=""; VOICE_PID=""; SLEEP_PID=""
stop_child() {  # stop_child PID SECONDS: wait up to SECONDS for it to exit, then TERM, then KILL
  local pid=$1 grace=$2
  [ -n "$pid" ] || return 0
  for _ in $(seq "$grace"); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 10); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL "$pid" 2>/dev/null
  step "had to KILL pid $pid"
}
cleanup() {  # runs on every exit: ask the robot loop to stop, then make sure no child is left
  touch "$OUT/STOP"
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null
  stop_child "$VOICE_PID" 0   # the voice load has no stop file: stop it at once
  stop_child "$POLICY_PID" 30   # the robot loop sees STOP within one beat once it is looping
  step "exit"
}
trap cleanup EXIT
trap 'step "interrupted"; exit 130' INT
trap 'step "terminated"; exit 143' TERM HUP

pause() {  # sleep in the background so a signal is handled at once
  sleep "$1" & SLEEP_PID=$!
  wait "$SLEEP_PID"; SLEEP_PID=""
}

# Fingerprints, GPU sharing mode and mains power, taken before anything runs.
sha256sum PREREG.md SHARE_PLAN.md router.py speech.py share_*.py analyse_share.py run_share.sh make_share_audio.sh \
  > "$OUT/fingerprints.txt" 2>&1
nvidia-smi compute-policy -l > "$OUT/compute_policy.txt" 2>&1
cat /sys/class/power_supply/ADP0/online > "$OUT/mains_online.txt" 2>&1
step "start $MODE: mains online = $(cat "$OUT/mains_online.txt")"

PYTHONPATH=$ROBOT_PATH taskset -c "$ROBOT_CORES" "$ROBOT_PY" share_policy_loop.py --out "$OUT" "${POLICY_ARGS[@]}" \
  > "$OUT/policy.log" 2>&1 &
POLICY_PID=$!
step "robot loop started (pid $POLICY_PID, CPU cores $ROBOT_CORES): ${POLICY_ARGS[*]}"

if [ "$MODE" = calibrate ]; then
  wait "$POLICY_PID"; status=$?; POLICY_PID=""
  step "robot loop exited ($status)"
  "$PY" analyse_share.py "$OUT" --calibration | tee "$OUT/analysis.log"
  step "calibration analysed"
  exit 0
fi

# Wait up to 15 minutes for the robot loop to load and warm up.
for _ in $(seq 450); do
  [ -e "$OUT/policy_ready.json" ] && break
  kill -0 "$POLICY_PID" 2>/dev/null || { step "FAILED: robot loop exited before it was ready"; exit 1; }
  pause 2
done
[ -e "$OUT/policy_ready.json" ] || { step "FAILED: robot loop not ready after 15 minutes"; exit 1; }
step "robot loop ready"

echo "phase,start_ns,end_ns" > "$OUT/phases.csv"
start=$(now_ns); pause "$ALONE_S"; echo "alone_pre,$start,$(now_ns)" >> "$OUT/phases.csv"
step "alone_pre done"
kill -0 "$POLICY_PID" 2>/dev/null || { step "FAILED: robot loop died during alone_pre"; exit 1; }

VOICE_ARGS=(--out "$OUT" --rounds "$ROUNDS" --block-s "$BLOCK_S" --guard-after-ms "$GUARD_MS"
            --whisper-budget-ms "$WHISPER_BUDGET_MS" --llm-budget-ms "$LLM_BUDGET_MS")
[ -n "$REGIMES" ] && VOICE_ARGS+=(--regimes "$REGIMES")
taskset -c "$VOICE_CORES" "$PY" share_voice_load.py "${VOICE_ARGS[@]}" > "$OUT/voice.log" 2>&1 &
VOICE_PID=$!
step "voice load started (pid $VOICE_PID, CPU cores $VOICE_CORES): ${VOICE_ARGS[*]}"
# Watch both programs while the blocks run: without the robot loop, the rest of the run measures nothing.
N_REGIMES=8; [ -n "$REGIMES" ] && N_REGIMES=$(echo "$REGIMES" | tr ',' '\n' | wc -l)
LIMIT_S=$(awk "BEGIN {print int($ROUNDS * $N_REGIMES * $BLOCK_S + 900)}")   # all blocks, plus 15 minutes to load
waited=0; problem=""
while kill -0 "$VOICE_PID" 2>/dev/null; do
  kill -0 "$POLICY_PID" 2>/dev/null || { problem="robot loop died during the blocks"; break; }
  [ "$waited" -ge "$LIMIT_S" ] && { problem="voice load still running after $LIMIT_S s"; break; }
  pause 5; waited=$((waited + 5))
done
if [ -n "$problem" ]; then
  step "FAILED: $problem"
  stop_child "$VOICE_PID" 0; VOICE_PID=""
else
  wait "$VOICE_PID"; status=$?; VOICE_PID=""
  step "voice load exited ($status)"
  start=$(now_ns); pause "$ALONE_S"; echo "alone_post,$start,$(now_ns)" >> "$OUT/phases.csv"
  step "alone_post done"
fi

touch "$OUT/STOP"
stop_child "$POLICY_PID" 60; wait "$POLICY_PID" 2>/dev/null; POLICY_PID=""
step "robot loop stopped"
"$PY" analyse_share.py "$OUT" --min-rounds "$MIN_ROUNDS" | tee "$OUT/analysis.log"   # complete rounds only
step "analysed"
[ -z "$problem" ] || exit 1
