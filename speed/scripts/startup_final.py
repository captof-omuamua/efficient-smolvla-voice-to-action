"""Startup and peak GPU memory for one SmolVLA setup, in a fresh process (one setup per process, as in the talk).

Startup = the first predict_action_chunk call after loading, as in feasibility_smolvla.py; for B it includes
torch.compile. Whether that compile is cold or reuses an earlier run's cache is set by the caller through
TORCHINDUCTOR_FORCE_DISABLE_CACHES / TORCHINDUCTOR_CACHE_DIR (see run_startup.sh). Loading time is recorded
separately. Peak GPU memory = torch.cuda.max_memory_allocated over 50 warm calls (as the talk's peak_gpu_mb).
Dynamo's counters after the first call and after the 50 calls show graph breaks and whether anything recompiled.

Usage: python startup_final.py --setup B --label cold --out-dir <dir>
"""

import time

T_START = time.perf_counter()  # before the heavy imports

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from energy_session_final import KEYS, SETUPS, dynamo_counters, load_policy, make_inputs, timed  # noqa: E402

T_IMPORTED = time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=KEYS, required=True)
    parser.add_argument("--label", required=True, help="e.g. cold, fill, warm")
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True  # as lerobot-eval / lerobot-rollout (--tf32=true)
    torch.backends.cudnn.allow_tf32 = True

    t_load = time.perf_counter()
    policy = load_policy(SETUPS[args.setup])
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    batch, noise = make_inputs(policy)

    def predict():
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())

    _, first_ms = timed(predict)
    t_first_done = time.perf_counter()
    after_first = dynamo_counters()
    second_ms = timed(predict)[1]  # max-autotune records its CUDA graphs over the first few calls
    for _ in range(10):
        timed(predict)
    torch.cuda.reset_peak_memory_stats()
    ms = [timed(predict)[1] for _ in range(args.calls)]
    after_calls = dynamo_counters()
    stats = {
        "setup": args.setup,
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "import_s": round(T_IMPORTED - T_START, 2),
        "load_s": round(load_s, 2),
        "first_call_s": round(first_ms / 1000, 2),
        "second_call_ms": round(second_ms, 1),
        "script_start_to_first_action_s": round(t_first_done - T_START, 2),
        "median_ms_warm_calls": round(statistics.median(ms), 2),
        "warm_calls": args.calls,
        "peak_gpu_mb_allocated": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_mb_reserved": round(torch.cuda.max_memory_reserved() / 2**20),
        "compile_mode": policy.config.compile_mode if SETUPS[args.setup]["compile_model"] else None,
        "dynamo_after_first_call": after_first,
        "dynamo_after_warm_calls": after_calls,
        "recompiled_after_first_call": after_first.get("frames") != after_calls.get("frames"),
        "env": {v: os.environ.get(v) for v in ("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "TORCHINDUCTOR_CACHE_DIR",
                                                "TRITON_CACHE_DIR", "TORCH_LOGS", "HF_HUB_OFFLINE")},
        "torch": torch.__version__,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.setup}_{args.label}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("dynamo")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
