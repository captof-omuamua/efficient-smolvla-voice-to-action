"""Root-cause check: why does one Whisper call take about 60 ms or about 140 ms? (voice program alone)

Three phases on the same clips: A = free to run on any CPU core, B = kept on the fast cores,
C = kept on the slow cores. Each call records the time of its two parts and the core it ran on.
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the voice/ folder
from speech import RATE, Transcriber, load_raw  # noqa: E402

FAST = set(map(int, sys.argv[2].split(",")))  # fast core ids
SLOW = set(range(os.cpu_count())) - FAST
OUT = Path(sys.argv[1])


def current_cpu():
    """The CPU core this thread last ran on (field 39 of /proc/self/stat)."""
    with open("/proc/self/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[36])


def one_call(t, audio):
    """Whisper once, timing the CPU part (sound to features) and the model part separately."""
    t0 = time.monotonic_ns()
    features = t.processor(audio, sampling_rate=RATE, return_tensors="pt").input_features
    t1 = time.monotonic_ns()
    with torch.inference_mode():
        t.model.generate(features.to("cuda", dtype=torch.float16), language="en", task="transcribe", max_length=64)
    torch.cuda.synchronize()
    t2 = time.monotonic_ns()
    return (t1 - t0) / 1e6, (t2 - t1) / 1e6


def main():
    clips = [load_raw(p) for p in sorted((Path(__file__).resolve().parents[1] / "data/share_audio").glob("c*_*.raw"))]
    t = Transcriber()
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    for i in range(5):
        one_call(t, clips[i])
    everything = os.sched_getaffinity(0)
    phases = [("A_any", everything, 150), ("B_fast", FAST, 150), ("C_slow", SLOW, 60), ("D_fast_gaps", FAST, 25)]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "i", "cpu", "features_ms", "model_ms", "total_ms", "gpu_mhz", "throttle", "threads"])
        for name, cores, n in phases:
            for tid in os.listdir("/proc/self/task"):  # every thread of this program
                os.sched_setaffinity(int(tid), cores)
            time.sleep(0.2)
            for i in range(n):
                if name.endswith("gaps"):
                    time.sleep(1.5)  # a pause, as between spoken commands
                a, b = one_call(t, clips[i % len(clips)])
                w.writerow([name, i, current_cpu(), round(a, 2), round(b, 2), round(a + b, 2),
                            pynvml.nvmlDeviceGetClockInfo(gpu, pynvml.NVML_CLOCK_SM),
                            pynvml.nvmlDeviceGetCurrentClocksEventReasons(gpu), torch.get_num_threads()])
    rows = list(csv.DictReader(open(OUT)))
    for name, _, _ in phases:
        x = np.array([float(r["total_ms"]) for r in rows if r["phase"] == name])
        fe = np.array([float(r["features_ms"]) for r in rows if r["phase"] == name])
        mo = np.array([float(r["model_ms"]) for r in rows if r["phase"] == name])
        print(f"{name:12s} n={len(x):3d} total p50={np.percentile(x, 50):6.1f} p99={np.percentile(x, 99):6.1f} "
              f"max={x.max():6.1f} over100={int((x > 100).sum()):3d} | features p50={np.percentile(fe, 50):5.1f} "
              f"max={fe.max():6.1f} | model p50={np.percentile(mo, 50):5.1f} max={mo.max():6.1f}")
    a = [r for r in rows if r["phase"] == "A_any"]
    for label, group in (("fast cores", [r for r in a if int(r["cpu"]) in FAST]),
                         ("slow cores", [r for r in a if int(r["cpu"]) not in FAST])):
        x = np.array([float(r["total_ms"]) for r in group]) if group else np.array([np.nan])
        print(f"A_any on {label}: n={len(group)} total p50={np.percentile(x, 50):.1f} over100={int((x > 100).sum())}")


if __name__ == "__main__":
    main()
