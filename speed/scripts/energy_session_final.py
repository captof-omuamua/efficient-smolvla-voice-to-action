"""Final speed + energy for the 10 Nov talk: four SmolVLA setups in one session, on today's LeRobot.

Adapted from bench/results_energy_same_session_2026-11-07/energy_same_session.py (two setups, ABBA blocks) to
four setups in mirrored rounds. Inputs as in bench/feasibility_smolvla.py.

Code: the worktree $LEROBOT_DIR = LeRobot main d490e472 + PR #4634 head 2d200334 merged
(--no-commit) + the camera-batching patch, run through that worktree's own venv (uv sync --locked).

Setups (lerobot/smolvla_libero, batch size 1):
  A0  talk baseline definition: eager, 10 denoising steps, 1 blank camera slot (empty_cameras=1),
      cameras one vision-encoder call each (the per-camera loop is put back on this instance).
  A1  today's default baseline: eager, 10 steps, no blank slot (empty_cameras=0, the checkpoint's default and
      what LeRobot CI uses since 5 Nov), per-camera loop.
  C   no compiler: eager, 1 step, no blank slot, cameras batched (the patch's default).
  B   tuned: whole-model compile (compile_model=True, default compile_mode max-autotune), 1 step, no blank slot,
      cameras batched.
  All four run with TF32 on for float32 matmuls, as today's lerobot-eval and lerobot-rollout set by default
  (--tf32=true). Today's compile_model no longer sets the matmul precision itself (LeRobot #5096).

Method (as on 7 Nov):
  - Blocks of back-to-back calls take turns in mirrored rounds (A0 A1 C B, then B C A1 A0, three times each:
    6 blocks per setup), so slow drift in clocks and temperature hits every setup about equally.
  - Blocks last about the same time: 100 calls for A0, and as many calls of the others as fit in the same time
    (sized from warm-up timings), so the energy counter's ~0.1 s steps are small against every block.
  - An idle window of the same length comes right before every block, plus one after the last block.
    Idle = process alive, all four models loaded, GPU doing nothing.
  - NVML's cumulative energy counter (millijoules) updates about every 100 ms. Every window starts right after
    a counter update, and ends at the first update at least 100 ms after the work ends.
  - Idle power for a block = energy / time over the idle windows right before and right after it.
    J/decision above idle = (window energy - idle power x window time) / calls.
    J/decision total = above idle + idle power x wall time per call.
  - Before every idle window + block: `nvidia-smi --query-compute-apps=pid` must list only this process.
    During them a background thread also checks every 0.5 s. If another compute process shows up, wait for it
    to leave and redo that idle window and block.
  - A background thread logs SM and memory clocks, temperature, power, P-state and clock-event reasons every
    100 ms, and how many CPU cores other programs keep busy (the eager setups are sensitive to CPU contention).
  - torch._dynamo counters are read before and after the measured blocks: new frames during the blocks would
    mean B recompiled while being measured.
Outputs (in --out-dir): session.json, block_latencies_ms.json, gpu_samples.csv, first-call actions per setup.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pynvml
import torch
import transformers
from huggingface_hub import try_to_load_from_cache

import lerobot
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed

MODEL = "lerobot/smolvla_libero"
REPO = Path(__file__).resolve().parents[2]  # the repository root
PATCH = REPO / "speed/patches/batch_cameras.patch"
TALK = REPO / "speed/results/reference_2026-10-31/results_v2"  # the 31 Oct runs this session is compared with
SETUPS = {
    "A0": {"name": "talk baseline definition: eager, 10 steps, 1 blank camera slot, cameras one call each",
           "compile_model": False, "num_steps": 10, "empty_cameras": 1, "batched_cameras": False,
           "talk_stats": TALK / "pr_eager_steps10_stats.json",
           "talk_actions": TALK / "with_blank_slot/pr_eager_steps10_actions.pt"},
    "A1": {"name": "today's default baseline: eager, 10 steps, no blank slot, cameras one call each",
           "compile_model": False, "num_steps": 10, "empty_cameras": 0, "batched_cameras": False,
           "talk_stats": TALK / "no_blank_slot/pr_eager_steps10_stats.json",
           "talk_actions": TALK / "no_blank_slot/pr_eager_steps10_actions.pt"},
    "C": {"name": "no compiler: eager, 1 step, no blank slot, cameras batched",
          "compile_model": False, "num_steps": 1, "empty_cameras": 0, "batched_cameras": True,
          "talk_stats": TALK / "no_blank_slot/batched_eager_steps1_stats.json",
          "talk_actions": TALK / "no_blank_slot/batched_eager_steps1_actions.pt"},
    "B": {"name": "tuned: whole-model compile (default mode), 1 step, no blank slot, cameras batched",
          "compile_model": True, "num_steps": 1, "empty_cameras": 0, "batched_cameras": True,
          "talk_stats": TALK / "no_blank_slot/batched_compile_model_steps1_stats.json",
          "talk_actions": TALK / "no_blank_slot/batched_compile_model_steps1_actions.pt"},
}
KEYS = list(SETUPS)
ROUND = ["A0", "A1", "C", "B"]
CLOCK_REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
                 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


def make_observation(config, device):
    """Copied from bench/bench_smolvla.py, so the inputs match feasibility_smolvla.py exactly."""
    obs = {"task": ["Pick up the red cube and place it in the box"]}
    for name, feature in config.input_features.items():
        if feature.type == FeatureType.VISUAL:
            obs[name] = torch.rand(1, *feature.shape, device=device)  # pixel values in [0, 1]
        elif feature.type == FeatureType.STATE:
            obs[name] = torch.randn(1, *feature.shape, device=device)
    return obs


def timed(fn):
    """Copied from bench/bench_smolvla.py: run fn once, return (result, ms); synchronize() waits for the GPU."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def load_policy(setup):
    """As feasibility_smolvla.load_policy, plus the per-camera loop for the unbatched setups."""
    config = PreTrainedConfig.from_pretrained(MODEL)
    config.compile_model = setup["compile_model"]  # must be set before the model is built
    config.num_steps = setup["num_steps"]  # denoising steps per action chunk
    config.empty_cameras = setup["empty_cameras"]  # blank camera slots
    policy = SmolVLAPolicy.from_pretrained(MODEL, config=config).to("cuda").eval()
    if not setup["batched_cameras"]:
        model = policy.model  # LeRobot before the patch: one vision-encoder call per camera
        model._embed_images = lambda images: [model.vlm_with_expert.embed_image(img) for img in images]
    return policy


def make_inputs(policy):
    """The exact steps of feasibility_smolvla.main(), right after the policy is loaded."""
    set_seed(0)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL)
    obs = make_observation(policy.config, "cuda")
    obs = {k: v for k, v in obs.items() if "empty_camera" not in k}  # blank slots stay missing, as in LIBERO eval
    batch = preprocessor(obs)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator,
                        device="cuda")
    return batch, noise


def same_inputs(x, y):
    (batch_x, noise_x), (batch_y, noise_y) = x, y
    tensors_equal = all(torch.equal(batch_x[k], batch_y[k]) for k in batch_x if torch.is_tensor(batch_x[k]))
    return batch_x.keys() == batch_y.keys() and tensors_equal and torch.equal(noise_x, noise_y)


class Gpu:
    """NVML reads: the energy counter, and which compute processes are on the GPU."""

    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def energy_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)  # millijoules since the driver loaded

    def next_update(self, not_before=None):
        """Wait for the energy counter's next update; return (mJ, time). It updates about every 100 ms."""
        if not_before is not None:
            time.sleep(max(0.0, not_before - time.perf_counter()))
        start = self.energy_mj()
        while (now := self.energy_mj()) == start:
            time.sleep(0.0005)
        return now, time.perf_counter()

    def foreign_pids(self):
        """Compute processes on the GPU other than this one and its children (NVML)."""
        me = psutil.Process()
        ours = {me.pid} | {child.pid for child in me.children(recursive=True)}
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.h)
        return sorted({p.pid for p in procs} - ours)


def nvidia_smi_foreign_pids():
    """The check asked for before every timing block: nvidia-smi's compute apps, minus this process."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return sorted({int(p) for p in out.split()} - {os.getpid()})


class Sampler(threading.Thread):
    """Logs clocks, temperature, power and P-state every 100 ms, and foreign compute processes every 0.5 s."""

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.phase, self.rows, self.foreign = gpu, "start", [], []
        self.me, self.cores = psutil.Process(), psutil.cpu_count()
        psutil.cpu_percent(interval=None)  # start both CPU counters; each later call covers the time since the last
        self.me.cpu_percent(interval=None)
        self._halt = threading.Event()

    def run(self):
        h, i = self.gpu.h, 0
        while not self._halt.is_set():
            try:
                self.rows.append({
                    "t": time.perf_counter(), "phase": self.phase,
                    "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                    "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                    "power_w_avg": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                    "pstate": pynvml.nvmlDeviceGetPerformanceState(h),
                    "clock_reasons": pynvml.nvmlDeviceGetCurrentClocksEventReasons(h),
                    "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
                })
                if i % 5 == 0:
                    busy_cores = psutil.cpu_percent(interval=None) * self.cores / 100  # whole machine
                    self.rows[-1]["other_cpu_cores"] = max(0.0, busy_cores - self.me.cpu_percent(interval=None) / 100)
                    if pids := self.gpu.foreign_pids():
                        self.foreign.append({"t": time.perf_counter(), "phase": self.phase, "pids": pids})
            except pynvml.NVMLError as e:
                self.rows.append({"t": time.perf_counter(), "phase": self.phase, "error": str(e)})
            i += 1
            self._halt.wait(0.1)

    def stop(self):
        self._halt.set()
        self.join()

    def summary(self, phase):
        rows = [r for r in self.rows if r["phase"] == phase and "error" not in r]
        if not rows:
            return {}
        reasons = 0
        for r in rows:
            reasons |= r["clock_reasons"]
        cpu = [r["other_cpu_cores"] for r in rows if "other_cpu_cores" in r]
        return {
            "samples": len(rows),
            "sm_mhz_mean": round(statistics.mean(r["sm_mhz"] for r in rows)),
            "sm_mhz_min": min(r["sm_mhz"] for r in rows),
            "sm_mhz_max": max(r["sm_mhz"] for r in rows),
            "mem_mhz_mean": round(statistics.mean(r["mem_mhz"] for r in rows)),
            "temp_c_start": rows[0]["temp_c"],
            "temp_c_end": rows[-1]["temp_c"],
            "temp_c_max": max(r["temp_c"] for r in rows),
            "pstates": sorted({r["pstate"] for r in rows}),
            "power_limit_w": sorted({r["power_limit_w"] for r in rows}),
            "clock_reasons": [name for bit, name in CLOCK_REASONS.items() if reasons & bit],
            "other_cpu_cores_mean": round(statistics.mean(cpu), 2) if cpu else None,
            "other_cpu_cores_max": round(max(cpu), 2) if cpu else None,
        }


def latency_stats(ms):
    ordered = sorted(ms)
    n = len(ordered)
    return {  # percentiles as in feasibility_smolvla.py: ordered[int(q * n)]
        "calls": n,
        "median_ms": round(statistics.median(ordered), 2),
        "p90_ms": round(ordered[int(0.90 * n)], 2),
        "p99_ms": round(ordered[int(0.99 * n)], 2),
        "mean_ms": round(statistics.mean(ordered), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def spread(values, digits=3):
    return {
        "mean": round(statistics.mean(values), digits),
        "sd": round(statistics.stdev(values), digits) if len(values) > 1 else 0.0,
        "min": round(min(values), digits),
        "max": round(max(values), digits),
        "per_block": [round(v, digits) for v in values],
    }


def dynamo_counters():
    """A plain copy of torch._dynamo's counters (frames compiled, graph breaks, ...)."""
    return {group: dict(values) for group, values in torch._dynamo.utils.counters.items()}


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6, help="mirrored rounds of all four setups (6 = 6 blocks each)")
    parser.add_argument("--calls-a0", type=int, default=100, help="calls per A0 block; the others match its time")
    parser.add_argument("--warmup", type=int, default=10, help="untimed calls per setup after the first call")
    parser.add_argument("--settle-seconds", type=float, default=8, help="unmeasured calls per setup before block 1")
    parser.add_argument("--out-dir", default=str(Path(os.environ.get("OUT_DIR", REPO / "runs")) / "speed" / "session"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpu = Gpu()
    log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)  # noqa: E731

    if pids := gpu.foreign_pids():
        raise SystemExit(f"Another compute process is on the GPU: {pids}")
    root = Path(lerobot.__file__).resolve().parents[2]
    log(f"lerobot code: {lerobot.__file__} (HEAD {git('rev-parse', 'HEAD', cwd=root)})")

    # lerobot-eval and lerobot-rollout default to --tf32=true; set it the same way for every setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log(f"float32 matmul precision: {torch.get_float32_matmul_precision()}")

    t0 = time.perf_counter()
    policies, load_s = {}, {}
    for k in KEYS:
        start = time.perf_counter()
        policies[k] = load_policy(SETUPS[k])
        load_s[k] = time.perf_counter() - start
    log(f"loaded {KEYS} in {time.perf_counter() - t0:.1f} s; float32 matmul precision now "
        f"{torch.get_float32_matmul_precision()}")
    inputs = {k: make_inputs(policies[k]) for k in KEYS}
    identical_inputs = {k: same_inputs(inputs["A1"], inputs[k]) for k in KEYS}
    log(f"inputs identical to A1's: {identical_inputs}")

    def predictor(k):
        policy, (batch, noise) = policies[k], inputs[k]
        return lambda: policy.predict_action_chunk(dict(batch), noise=noise.clone())

    predict = {k: predictor(k) for k in KEYS}

    # First call (B compiles here; caches are off when TORCHINDUCTOR_FORCE_DISABLE_CACHES=1), then warm-up.
    # A0 goes first, so its first call also pays the process's one-time CUDA library start-up.
    first_call_ms, first_actions = {}, {}
    frames_before_first = dynamo_counters()
    for k in KEYS:
        actions, first_call_ms[k] = timed(predict[k])
        first_actions[k] = actions.clone().cpu()  # B replays CUDA graphs, so later calls overwrite its output
        log(f"{k} first call {first_call_ms[k] / 1000:.1f} s")
        for _ in range(args.warmup):
            timed(predict[k])
    counters_after_first = dynamo_counters()

    # Same actions as the talk's runs (torch 2.11, PR head 8c4f33e5)? Expect rounding-level differences only.
    parity = {}
    for k in KEYS:
        talk_actions = torch.load(SETUPS[k]["talk_actions"])
        diff = (first_actions[k] - talk_actions).abs()
        parity[k] = {"vs_talk_run": str(SETUPS[k]["talk_actions"]),
                     "bit_identical": bool(torch.equal(first_actions[k], talk_actions)),
                     "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                     "max_abs_diff_first_10_steps": float(diff[:, :10].max()),
                     "action_scale_max_abs": float(talk_actions.abs().max())}
        torch.save(first_actions[k], out / f"{k}_first_call_actions.pt")
    pairs = {"A0_vs_A1": ("A0", "A1"), "C_vs_B": ("C", "B")}  # same model, only the blank slot / the compiler
    for name, (x, y) in pairs.items():
        diff = (first_actions[x] - first_actions[y]).abs()
        parity[name] = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                        "max_abs_diff_first_10_steps": float(diff[:, :10].max())}
    log(f"parity: {json.dumps(parity)}")

    # Settle: unmeasured back-to-back calls, which also time each setup for sizing the blocks
    settle_ms = {}
    for k in KEYS:
        end, samples = time.perf_counter() + args.settle_seconds, []
        while time.perf_counter() < end:
            samples.append(timed(predict[k])[1])
        settle_ms[k] = statistics.median(samples)
    calls = {k: round(args.calls_a0 * settle_ms["A0"] / settle_ms[k]) for k in KEYS}
    window_s = args.calls_a0 * settle_ms["A0"] / 1000  # every idle window and every block lasts about this long
    log(f"settle medians {settle_ms}; calls per block {calls}; window {window_s:.1f} s")

    counters_before = dynamo_counters()
    sampler = Sampler(gpu)
    sampler.start()

    def idle_window(name):
        sampler.phase = name
        e0, t0 = gpu.next_update()
        e_mid, t_mid = gpu.next_update(not_before=t0 + window_s / 2)
        e1, t1 = gpu.next_update(not_before=t0 + window_s)
        return {"name": name, "seconds": t1 - t0, "energy_j": (e1 - e0) / 1000,
                "watts": (e1 - e0) / 1000 / (t1 - t0),
                "watts_first_half": (e_mid - e0) / 1000 / (t_mid - t0),
                "watts_second_half": (e1 - e_mid) / 1000 / (t1 - t_mid)}

    def block(name, k):
        sampler.phase = name
        e0, t0 = gpu.next_update()
        ms = [timed(predict[k])[1] for _ in range(calls[k])]
        t_work = time.perf_counter()
        e1, t1 = gpu.next_update(not_before=t_work + 0.1)
        return {"name": name, "setup": k, "calls": calls[k], "work_s": t_work - t0, "window_s": t1 - t0,
                "energy_j": (e1 - e0) / 1000, "latency_ms": ms}

    def wait_until_gpu_is_ours():
        """nvidia-smi (as asked) and NVML must both list no other compute process."""
        while (pids := sorted(set(nvidia_smi_foreign_pids()) | set(gpu.foreign_pids()))):
            log(f"another compute process {pids} is on the GPU; waiting")
            sampler.phase = "waiting"
            time.sleep(10)

    def intruders(*phases):  # foreign processes seen during these phases, or right now
        seen = [f for f in sampler.foreign if f["phase"] in phases]
        return seen + ([{"phase": "after", "pids": pids}] if (pids := gpu.foreign_pids()) else [])

    order = []
    for r in range(args.rounds):
        order += ROUND if r % 2 == 0 else ROUND[::-1]
    idles, blocks, redone = [], [], []
    for i, k in enumerate(order + ["-"]):  # "-": the idle window after the last block
        for attempt in range(100):
            wait_until_gpu_is_ours()
            tag = f"{i}" if attempt == 0 else f"{i}_redo{attempt}"
            idle = idle_window(f"idle_{tag}")
            run = block(f"block_{tag}_{k}", k) if k != "-" else None
            seen = intruders(idle["name"], run["name"] if run else idle["name"])
            if not seen:
                break
            redone.append({"position": i, "setup": k, "seen": seen})
            log(f"position {i} ({k}) saw another compute process {seen}; redoing it")
        idles.append(idle)
        if run:
            run["round"] = i // len(ROUND)
            blocks.append(run)
            log(f"block {i} {k}: {run['calls']} calls in {run['work_s']:.1f} s, {run['energy_j']:.1f} J, "
                f"median {statistics.median(run['latency_ms']):.2f} ms; idle before {idle['watts']:.1f} W")
    sampler.stop()
    counters_after = dynamo_counters()

    # Per block: idle power from the windows right before and after it
    for i, b in enumerate(blocks):
        before, after = idles[i], idles[i + 1]
        idle_w = (before["energy_j"] + after["energy_j"]) / (before["seconds"] + after["seconds"])
        above = b["energy_j"] - idle_w * b["window_s"]
        b.update({
            "idle_w": idle_w,
            "busy_w": (b["energy_j"] - idle_w * (b["window_s"] - b["work_s"])) / b["work_s"],
            "j_per_call_above_idle": above / b["calls"],
            "j_per_call_total": (above + idle_w * b["work_s"]) / b["calls"],
            "wall_ms_per_call": 1000 * b["work_s"] / b["calls"],
            "gpu": sampler.summary(b["name"]),
        })
    for idle in idles:
        idle["gpu"] = sampler.summary(idle["name"])

    summary = {}
    for k in KEYS:
        mine = [b for b in blocks if b["setup"] == k]
        talk = json.loads(SETUPS[k]["talk_stats"].read_text())
        summary[k] = {
            "setup": SETUPS[k]["name"],
            "latency": latency_stats([ms for b in mine for ms in b["latency_ms"]]),
            "block_median_ms": spread([statistics.median(b["latency_ms"]) for b in mine], 2),
            "j_per_call_total": spread([b["j_per_call_total"] for b in mine]),
            "j_per_call_above_idle": spread([b["j_per_call_above_idle"] for b in mine]),
            "busy_w": spread([b["busy_w"] for b in mine], 1),
            "idle_w": spread([b["idle_w"] for b in mine], 1),
            "first_call_s_in_session": round(first_call_ms[k] / 1000, 2),
            "load_s_in_session": round(load_s[k], 2),
            "talk_run": {"file": str(SETUPS[k]["talk_stats"])}
                        | {key: talk[key] for key in ("median_ms", "p99_ms", "energy_j_per_call", "first_call_ms",
                                                      "peak_gpu_mb", "torch")},
        }

    # Ratios X/Y: overall (mean over blocks), and per round (the four blocks of one round, next to each other)
    def ratio(x, y):
        res = {}
        for metric in ("j_per_call_total", "j_per_call_above_idle"):
            per_round = []
            for r in range(args.rounds):
                bx = [b for b in blocks if b["round"] == r and b["setup"] == x]
                by = [b for b in blocks if b["round"] == r and b["setup"] == y]
                per_round.append(bx[0][metric] / by[0][metric])
            res[metric] = {"overall": round(summary[x][metric]["mean"] / summary[y][metric]["mean"], 2),
                           "per_round_min": round(min(per_round), 2), "per_round_max": round(max(per_round), 2),
                           "per_round": [round(v, 2) for v in per_round]}
        for stat in ("median_ms", "p99_ms"):
            res[stat] = round(summary[x]["latency"][stat] / summary[y]["latency"][stat], 2)
        return res

    ratios = {f"{x}_over_{y}": ratio(x, y) for x in ("A0", "A1") for y in KEYS if y != x}

    h = gpu.h
    result = {
        "what": "Final speed + energy per decision for four SmolVLA setups, one session, NVML energy counter",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {
            "lerobot_file": lerobot.__file__,
            "worktree": str(root),
            "head": git("rev-parse", "HEAD", cwd=root),
            "merge_head": read_text(Path(git("rev-parse", "--git-dir", cwd=root)) / "MERGE_HEAD"),
            "status": git("status", "--short", cwd=root),
            "patch": str(PATCH),
            "patch_sha256": sha256(PATCH),
            "modeling_smolvla_sha256": sha256(root / "src/lerobot/policies/smolvla/modeling_smolvla.py"),
            "smolvlm_with_expert_sha256": sha256(root / "src/lerobot/policies/smolvla/smolvlm_with_expert.py"),
            "script_sha256": sha256(__file__),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "triton": importlib.metadata.version("triton"),
            "nvidia_ml_py": importlib.metadata.version("nvidia-ml-py"),
            "gpu": pynvml.nvmlDeviceGetName(h),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_w_enforced_at_end": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
            "power_limit_w_default": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
            "platform_profile": read_text("/sys/firmware/acpi/platform_profile"),
            "on_ac_power": read_text("/sys/class/power_supply/ADP0/online"),
            "os": platform.platform(),
            "model_snapshot": try_to_load_from_cache(MODEL, "config.json"),
            "python_executable": sys.executable,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "env": {v: os.environ.get(v) for v in ("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "TORCHINDUCTOR_CACHE_DIR",
                                                    "TRITON_CACHE_DIR", "HF_HUB_OFFLINE", "PYTHONPATH",
                                                    "CUDA_VISIBLE_DEVICES", "TORCH_LOGS")},
        },
        "settings": {
            "model": MODEL,
            "setups": {k: {key: (str(v) if isinstance(v, Path) else v) for key, v in SETUPS[k].items()}
                       | {"compile_mode": policies[k].config.compile_mode} for k in KEYS},
            "batch_size": 1,
            "order": order,
            "calls_per_block": calls,
            "window_s_target": round(window_s, 2),
            "warmup_calls": args.warmup,
            "settle_seconds": args.settle_seconds,
            "settle_median_ms": {k: round(v, 2) for k, v in settle_ms.items()},
            "inputs_identical_to_A1": identical_inputs,
        },
        "first_call_ms": {k: round(v, 1) for k, v in first_call_ms.items()},
        "load_s": {k: round(v, 2) for k, v in load_s.items()},
        "parity": parity,
        "summary": summary,
        "ratios": ratios,
        "dynamo_counters": {"before_first_calls": frames_before_first, "after_first_calls": counters_after_first,
                            "before_blocks": counters_before, "after_blocks": counters_after,
                            "recompiled_during_blocks": counters_before.get("frames") != counters_after.get("frames")},
        "redone_blocks": redone,
        "idle_windows": idles,
        "blocks": [{k: v for k, v in b.items() if k != "latency_ms"} | {"latency": latency_stats(b["latency_ms"])}
                   for b in blocks],
    }
    (out / "session.json").write_text(json.dumps(result, indent=2))
    latencies = {b["name"]: [round(ms, 3) for ms in b["latency_ms"]] for b in blocks}
    (out / "block_latencies_ms.json").write_text(json.dumps(latencies))
    with open(out / "gpu_samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "phase", "sm_mhz", "mem_mhz", "temp_c", "power_w_avg", "pstate",
                                               "clock_reasons", "power_limit_w", "other_cpu_cores", "error"])
        writer.writeheader()
        writer.writerows(sampler.rows)
    log(json.dumps({"summary": {k: {"latency": v["latency"], "j_total": v["j_per_call_total"]["mean"],
                                    "j_above_idle": v["j_per_call_above_idle"]["mean"]} for k, v in summary.items()},
                    "ratios": ratios}, indent=1))


if __name__ == "__main__":
    main()
