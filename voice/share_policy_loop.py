"""Robot loop for the shared-GPU test: one SmolVLA decision on every 500 ms beat of the shared clock.

Loads and warms up the policy exactly as fast-and-on-time/scripts/smolvla_tails.py does (the 8 Nov baseline),
then decides once at every whole multiple of 500 ms on time.monotonic_ns(), which the voice program reads too.
Writes into --out:
  policy_ready.json    written once warm-up is done (the orchestrator waits for it)
  decisions.csv        one row per decision, written and flushed as it happens
  policy_summary.json  written on stop: decisions, skipped beats, p50 / p99 / max of the call time
Stops when the stop file appears, after --max-decisions, or on SIGTERM / Ctrl-C.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed

MODEL_ID = "lerobot/smolvla_libero"
LEROBOT_FAST = Path(os.environ.get("ROBOT_LEROBOT_DIR", "."))  # the checkout the robot loop imports
PERIOD_NS = 500_000_000  # beat k sits at k * PERIOD_NS on CLOCK_MONOTONIC
PIDS_EVERY = 20  # look for other GPU programs on every 20th decision (the call is slower than the rest)
COLUMNS = ["i", "beat_ns", "start_ns", "end_ns", "ms", "sm_mhz", "mem_mhz", "temp_c", "power_mw", "throttle",
           "cpu", "cpu_khz", "other_pids"]


# ---- helpers copied from fast-and-on-time (ontime/telemetry.py, ontime/provenance.py, smolvla_tails.py) ----

def make_observation(config, device):
    """One synthetic observation with the inputs the model expects (shapes fixed, so cost is too)."""
    obs = {"task": ["put both the alphabet soup and the tomato sauce in the basket"]}
    for name, feature in config.input_features.items():
        if feature.type == FeatureType.VISUAL:
            obs[name] = torch.rand(1, *feature.shape, device=device)
        elif feature.type == FeatureType.STATE:
            obs[name] = torch.randn(1, *feature.shape, device=device)
    return obs


def timed_call(fn, sync):
    """Run fn once; return (start_ns, end_ns). sync() waits for the GPU, so the time covers all GPU work."""
    sync()
    start_ns = time.monotonic_ns()
    fn()
    sync()
    return start_ns, time.monotonic_ns()


def current_cpu():
    """The CPU core this process last ran on (field 39 of /proc/self/stat; Linux only)."""
    with open("/proc/self/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()  # skip "pid (name)": the name may contain spaces
    return int(fields[36])  # field 39 overall = index 36 after the first two fields


def cpu_khz(cpu):
    """Current frequency of one CPU core in kHz, as the kernel reports it."""
    with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
        return int(f.read())


def gpu_sample(handle):
    """One snapshot of the GPU's clocks, temperature, power and throttle bits (read after the timer stops)."""
    return {
        "sm_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "mem_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
        "temp_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "power_mw": pynvml.nvmlDeviceGetPowerUsage(handle),
        "throttle": pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle),
    }


def other_gpu_pids(handle):
    """Process ids of any other compute program on the GPU (a sign the run was disturbed)."""
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return sorted(p.pid for p in procs if p.pid != os.getpid())


def git_commit(path):
    """Commit hash of the git checkout at path, plus whether it has uncommitted changes (read-only queries)."""
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        # --no-optional-locks: status must not rewrite the index while we only look.
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except OSError:
        return {"commit": None, "dirty": None}


# ---- this experiment ----

def parse_args():
    """Command-line options fixed by SHARE_PLAN.md, Part 1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-steps", type=int, default=1, help="denoising steps (checkpoint default 10)")
    parser.add_argument("--compile", action="store_true", help="whole-model torch.compile")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max-decisions", type=int, default=0, help="0 = run until the stop file appears")
    parser.add_argument("--stop-file", default=None, help="default: OUT/STOP")
    return parser.parse_args()


def load_policy(args, device):
    """Load SmolVLA and build predict() exactly as smolvla_tails.py does (same config, seed, input, noise)."""
    set_seed(0)
    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.num_steps = args.num_steps
    config.empty_cameras = 0  # no blank camera slot (the earlier speed-up)
    config.compile_model = args.compile
    config.compile_mode = "default"
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID, config=config).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL_ID)
    batch = preprocessor(make_observation(policy.config, device))

    # The same starting noise every call, so every call does exactly the same work.
    generator = torch.Generator(device=device).manual_seed(1234)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device=device)

    def predict():
        with torch.inference_mode():
            return policy.predict_action_chunk(dict(batch), noise=noise.clone())

    return predict


def write_json(path, data):
    """Write a JSON file in one step (temp file, then rename), so a reader never sees half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def next_beat_ns(now_ns):
    """The first beat strictly after now: beats sit at whole multiples of PERIOD_NS."""
    return (now_ns // PERIOD_NS + 1) * PERIOD_NS


def sleep_until(target_ns):
    """Sleep until the shared clock reaches target_ns (time.sleep, as the baseline did; loop in case it wakes early)."""
    while True:
        left_ns = target_ns - time.monotonic_ns()
        if left_ns <= 0:
            return
        time.sleep(left_ns / 1e9)


def stop_on_sigterm():
    """Turn SIGTERM into a normal exit, so the summary still gets written."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))


def run_loop(predict, handle, out, args, stop_file):
    """Decide once per beat until stopped; write each row as it happens. Returns (call ms list, skipped beats)."""
    times_ms, skipped, last_beat = [], 0, None
    with open(out / "decisions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        f.flush()
        try:
            i = 0
            while True:
                beat_ns = next_beat_ns(time.monotonic_ns())  # strictly after now, so never twice on one beat
                if last_beat is not None:
                    skipped += (beat_ns - last_beat) // PERIOD_NS - 1  # a slow call makes us miss beats
                sleep_until(beat_ns)
                start_ns, end_ns = timed_call(predict, torch.cuda.synchronize)
                ms = (end_ns - start_ns) / 1e6
                # Everything below runs after the timer stopped, so it never adds to the measured time.
                snap = gpu_sample(handle)
                cpu = current_cpu()
                pids = ";".join(map(str, other_gpu_pids(handle))) if i % PIDS_EVERY == 0 else ""
                writer.writerow([i, beat_ns, start_ns, end_ns, round(ms, 4), snap["sm_mhz"], snap["mem_mhz"],
                                 snap["temp_c"], snap["power_mw"], snap["throttle"], cpu, cpu_khz(cpu), pids])
                f.flush()
                times_ms.append(ms)
                last_beat = beat_ns
                i += 1
                if stop_file.exists() or (args.max_decisions and i >= args.max_decisions):
                    break
        except (KeyboardInterrupt, SystemExit):
            pass  # stopped from outside: fall through and write the summary
    return times_ms, skipped


def write_summary(out, times_ms, skipped):
    """Write policy_summary.json: how many decisions, beats skipped, and p50 / p99 / max call time."""
    ms = np.array(times_ms) if times_ms else np.array([np.nan])
    summary = {
        "decisions": len(times_ms),
        "skipped_beats": int(skipped),
        "ms": {"p50": round(float(np.percentile(ms, 50)), 3), "p99": round(float(np.percentile(ms, 99)), 3),
               "max": round(float(ms.max()), 3)},
        "stopped_ns": time.monotonic_ns(),
    }
    write_json(out / "policy_summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    """Load, warm up, announce ready, then run the beat loop until stopped."""
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file) if args.stop_file else out / "STOP"
    stop_on_sigterm()

    device = "cuda"
    predict = load_policy(args, device)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    start_ns, end_ns = timed_call(predict, torch.cuda.synchronize)  # includes compiling, if on
    first_call_ms = (end_ns - start_ns) / 1e6
    for _ in range(args.warmup):  # back to back, as the baseline warmed up
        timed_call(predict, torch.cuda.synchronize)

    write_json(out / "policy_ready.json", {
        "pid": os.getpid(),
        "first_call_ms": round(first_call_ms, 2),
        "warmup": args.warmup,
        "settings": vars(args) | {"model": MODEL_ID, "empty_cameras": 0, "compile_mode": "default",
                                  "period_ns": PERIOD_NS, "observation": "synthetic, fixed shapes"},
        "ready_ns": time.monotonic_ns(),
        "cpu_cores": sorted(os.sched_getaffinity(0)),  # the CPU cores this program may run on
        "torch": torch.__version__,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "lerobot_fast": git_commit(LEROBOT_FAST),
    })
    print(f"ready: first call {first_call_ms:.0f} ms, warm-up {args.warmup}", flush=True)

    times_ms, skipped = run_loop(predict, handle, out, args, stop_file)
    write_summary(out, times_ms, skipped)


if __name__ == "__main__":
    main()
