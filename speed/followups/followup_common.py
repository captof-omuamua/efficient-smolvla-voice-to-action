"""Helpers shared by compile_modes.py and stack_compare.py. Runs on both stacks: the talk's (torch 2.11,
transformers 5.5.4, LeRobot 8c4f33e5) and today's (torch 2.15, transformers 5.22.1, the kept final worktree).

Inputs and loading are copied from ../energy_session_final.py (itself from bench/feasibility_smolvla.py): a random
observation made right after set_seed(0), starting noise from a generator seeded with 1234, batch size 1.
"""

import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil
import pynvml
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed

MODEL = "lerobot/smolvla_libero"
SETUPS = {  # as in ../energy_session_final.py
    "A0": {"compile_model": False, "num_steps": 10, "empty_cameras": 1, "batched_cameras": False},
    "B": {"compile_model": True, "num_steps": 1, "empty_cameras": 0, "batched_cameras": True},
}


def make_observation(config, device):
    obs = {"task": ["Pick up the red cube and place it in the box"]}
    for name, feature in config.input_features.items():
        if feature.type == FeatureType.VISUAL:
            obs[name] = torch.rand(1, *feature.shape, device=device)  # pixel values in [0, 1]
        elif feature.type == FeatureType.STATE:
            obs[name] = torch.randn(1, *feature.shape, device=device)
    return obs


def timed(fn):
    """Run fn once; return (result, ms). synchronize() waits for the GPU before and after."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def load_policy(setup, compile_mode=None):
    """Load one setup; compile_mode=None keeps the checkpoint's default (max-autotune)."""
    config = PreTrainedConfig.from_pretrained(MODEL)
    config.compile_model = setup["compile_model"]  # must be set before the model is built
    if compile_mode is not None:
        config.compile_mode = compile_mode
    config.num_steps = setup["num_steps"]
    config.empty_cameras = setup["empty_cameras"]
    policy = SmolVLAPolicy.from_pretrained(MODEL, config=config).to("cuda").eval()
    if not setup["batched_cameras"]:
        model = policy.model  # LeRobot without the batching patch: one vision-encoder call per camera
        model._embed_images = lambda images: [model.vlm_with_expert.embed_image(img) for img in images]
    return policy


def make_inputs(policy):
    set_seed(0)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL)
    obs = make_observation(policy.config, "cuda")
    obs = {k: v for k, v in obs.items() if "empty_camera" not in k}  # blank slots stay missing, as in LIBERO eval
    batch = preprocessor(obs)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator,
                        device="cuda")
    return batch, noise


def latency_stats(ms):
    ordered = sorted(ms)
    n = len(ordered)
    return {"calls": n, "median_ms": round(statistics.median(ordered), 2), "p90_ms": round(ordered[int(0.90 * n)], 2),
            "p99_ms": round(ordered[int(0.99 * n)], 2), "mean_ms": round(statistics.mean(ordered), 2),
            "min_ms": round(ordered[0], 2), "max_ms": round(ordered[-1], 2)}


def cpu_pkg_temp():
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                return int((zone / "temp").read_text()) / 1000
        except OSError:
            pass
    return None


def cpu_mhz_mean():
    khz = []
    for f in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            khz.append(int(f.read_text()))
        except OSError:
            pass
    return round(statistics.mean(khz) / 1000) if khz else None


class Gpu:
    """NVML reads: the energy counter, temperature, and which compute processes are on the GPU."""

    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def energy_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)

    def next_update(self, not_before=None):
        """Wait for the energy counter's next update (about every 100 ms); return (mJ, time)."""
        if not_before is not None:
            time.sleep(max(0.0, not_before - time.perf_counter()))
        start = self.energy_mj()
        while (now := self.energy_mj()) == start:
            time.sleep(0.0005)
        return now, time.perf_counter()

    def temp(self):
        return pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)

    def foreign_pids(self):
        me = psutil.Process()
        ours = {me.pid} | {child.pid for child in me.children(recursive=True)}
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.h)
        return sorted({p.pid for p in procs} - ours)

    def wait_until_ours(self, log=print):
        """Before a timed block: nvidia-smi and NVML must list no other compute process."""
        while True:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout
            pids = sorted(({int(p) for p in out.split()} - {psutil.Process().pid}) | set(self.foreign_pids()))
            if not pids:
                return
            log(f"another compute process {pids} is on the GPU; waiting")
            time.sleep(10)


class Sampler(threading.Thread):
    """Every 100 ms: GPU SM clock, temperature and power, CPU package temperature and mean core clock."""

    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.gpu, self.phase, self.rows, self.foreign = gpu, "start", [], []
        self._halt = threading.Event()

    def run(self):
        i = 0
        while not self._halt.is_set():
            try:
                h = self.gpu.h
                self.rows.append({"t": time.perf_counter(), "phase": self.phase,
                                  "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                  "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                                  "power_w_avg": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                                  "cpu_pkg_temp_c": cpu_pkg_temp(), "cpu_mhz_mean": cpu_mhz_mean()})
                if i % 5 == 0 and (pids := self.gpu.foreign_pids()):
                    self.foreign.append({"phase": self.phase, "pids": pids})
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
        mean = lambda key: round(statistics.mean(r[key] for r in rows if r[key] is not None), 1)  # noqa: E731
        return {"samples": len(rows), "sm_mhz_mean": mean("sm_mhz"), "gpu_temp_c_mean": mean("gpu_temp_c"),
                "gpu_temp_c_max": max(r["gpu_temp_c"] for r in rows), "power_w_avg_mean": mean("power_w_avg"),
                "cpu_pkg_temp_c_mean": mean("cpu_pkg_temp_c"),
                "cpu_pkg_temp_c_max": max(r["cpu_pkg_temp_c"] or 0 for r in rows), "cpu_mhz_mean": mean("cpu_mhz_mean")}


def timed_block(gpu, sampler, fn, calls, phase, log=print, max_tries=10):
    """`calls` back-to-back timed calls; redone if another compute process shows up meanwhile."""
    for _ in range(max_tries):
        gpu.wait_until_ours(log)
        sampler.phase, seen_before = phase, len(sampler.foreign)
        ms = [timed(fn)[1] for _ in range(calls)]
        if len(sampler.foreign) == seen_before and not gpu.foreign_pids():
            return ms
        log(f"{phase}: another compute process appeared; redoing the block")
    raise RuntimeError(f"{phase}: GPU never free")


def energy_block(gpu, sampler, fn, seconds, median_ms, phase, log=print, max_tries=10):
    """J per call from NVML's energy counter: idle window, block of calls, idle window (each about `seconds`)."""
    calls = max(50, round(seconds * 1000 / median_ms))

    def idle(name):
        sampler.phase = name
        e0, t0 = gpu.next_update()
        e1, t1 = gpu.next_update(not_before=t0 + seconds)
        return (e1 - e0) / 1000, t1 - t0

    for _ in range(max_tries):
        gpu.wait_until_ours(log)
        seen_before = len(sampler.foreign)
        idle_j0, idle_s0 = idle(f"{phase}_idle_before")
        sampler.phase = phase
        e0, t0 = gpu.next_update()
        for _ in range(calls):
            timed(fn)
        t_work = time.perf_counter()
        e1, t1 = gpu.next_update(not_before=t_work + 0.1)
        idle_j1, idle_s1 = idle(f"{phase}_idle_after")
        if len(sampler.foreign) == seen_before and not gpu.foreign_pids():
            break
        log(f"{phase}: another compute process appeared; redoing the energy block")
    idle_w = (idle_j0 + idle_j1) / (idle_s0 + idle_s1)
    energy_j, work_s, window_s = (e1 - e0) / 1000, t_work - t0, t1 - t0
    above = energy_j - idle_w * window_s
    return {"calls": calls, "work_s": round(work_s, 2), "idle_w": round(idle_w, 1),
            "busy_w": round((energy_j - idle_w * (window_s - work_s)) / work_s, 1),
            "j_per_call_total": round((above + idle_w * work_s) / calls, 4),
            "j_per_call_above_idle": round(above / calls, 4)}


def dynamo_summary():
    """Frames compiled, graphs, graph breaks (with their first-line reasons) from torch._dynamo's counters."""
    counters = {group: dict(values) for group, values in torch._dynamo.utils.counters.items()}
    reasons = {}
    for reason, n in counters.get("unimplemented", {}).items():
        reason_lines = [line.strip() for line in reason.splitlines() if line.strip()]
        reasons[" | ".join(reason_lines[:2])] = n
    return {"frames": counters.get("frames", {}), "unique_graphs": counters.get("stats", {}).get("unique_graphs"),
            "graph_break_count": sum(counters.get("graph_break", {}).values()), "unimplemented_reasons": reasons}


def gpu_kernel_profile(fn, calls=10):
    """GPU kernel time and kernel launches per call, from torch.profiler (CUPTI), over `calls` calls."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                fn()
            torch.cuda.synchronize()
        kernel_us, kernels = 0.0, 0
        for evt in prof.key_averages():
            if getattr(evt, "device_type", None) == DeviceType.CUDA:
                us = getattr(evt, "self_device_time_total", None)
                if us is None:
                    us = getattr(evt, "self_cuda_time_total", 0)
                kernel_us += us
                kernels += evt.count
        return {"gpu_kernel_ms_per_call": round(kernel_us / 1000 / calls, 2), "kernels_per_call": round(kernels / calls),
                "profiled_calls": calls}
    except Exception as e:  # the profiler is a diagnostic only
        return {"error": repr(e)}
