"""Follow-up check: is Whisper's time steady when the CPU maths libraries use few threads? (voice program alone)

Run with OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS set in the environment.
Phase A = 150 calls back to back. Phase D = 15 calls with a 1 s pause before each (like spoken commands).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the voice/ folder
from speech import RATE, Transcriber, load_raw  # noqa: E402

FAST = set(range(8))  # the 8 fast cores on this laptop


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
    return (t1 - t0) / 1e6, (t2 - t1) / 1e6, current_cpu()


def show(name, rows):
    fe, mo = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    total = fe + mo
    on_slow = sum(r[2] not in FAST for r in rows)
    print(f"{name:10s} n={len(rows):3d} total p50={np.percentile(total, 50):6.1f} p99={np.percentile(total, 99):6.1f} "
          f"max={total.max():6.1f} over100={int((total > 100).sum()):3d} | features p50={np.percentile(fe, 50):5.1f} "
          f"p99={np.percentile(fe, 99):5.1f} max={fe.max():5.1f} | model p50={np.percentile(mo, 50):5.1f} "
          f"max={mo.max():5.1f} | calls on slow cores={on_slow}")


def main():
    clips = [load_raw(p) for p in sorted((Path(__file__).resolve().parents[1] / "data/share_audio").glob("c*_*.raw"))]
    t = Transcriber()
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS')} torch={torch.get_num_threads()} "
          f"extractor={type(t.processor.feature_extractor).__name__}")
    for i in range(5):
        one_call(t, clips[i])
    show("A_flat", [one_call(t, clips[i % len(clips)]) for i in range(150)])
    rows = []
    for i in range(15):
        time.sleep(1.0)
        rows.append(one_call(t, clips[i % len(clips)]))
    show("D_gaps", rows)


if __name__ == "__main__":
    main()
