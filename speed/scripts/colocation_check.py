"""Side check: does sharing the process with a compiled B slow the eager A0?

The final session holds all four setups in one process. Its A0 median (~119 ms) is slower than A0 in a fresh
single-model process (~107 ms, startup/A0_eager.json). This runs A0 alone, or A0 with B loaded and compiled
(from a warm compile cache) in the same process, in alternating fresh processes, and times A0 the same way.
Usage: python colocation_check.py --mode alone|with_b --out <file.json>
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from energy_session_final import Gpu, SETUPS, latency_stats, load_policy, make_inputs, timed


def thermal(name="x86_pkg_temp"):
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (zone / "type").read_text().strip() == name:
            return int((zone / "temp").read_text()) / 1000
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alone", "with_b"], required=True)
    parser.add_argument("--calls", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    gpu = Gpu()
    if pids := gpu.foreign_pids():
        raise SystemExit(f"Another compute process is on the GPU: {pids}")
    torch.backends.cuda.matmul.allow_tf32 = True  # as in the session
    torch.backends.cudnn.allow_tf32 = True

    policies = {"A0": load_policy(SETUPS["A0"])}
    if args.mode == "with_b":
        policies["B"] = load_policy(SETUPS["B"])
    predict = {}
    for k, policy in policies.items():
        batch, noise = make_inputs(policy)
        predict[k] = (lambda p=policy, b=batch, n=noise: p.predict_action_chunk(dict(b), noise=n.clone()))
        for _ in range(11):  # first call compiles B (warm cache), then warm-up
            timed(predict[k])
    time.sleep(5)  # let the CPU cool a little after compiling, as the session's settle phase does

    result = {"mode": args.mode, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "cpu_pkg_temp_c_start": thermal(), "children": len(psutil.Process().children(recursive=True))}
    a0 = [timed(predict["A0"])[1] for _ in range(args.calls)]
    result["A0"] = latency_stats(a0)
    if "B" in predict:
        result["B"] = latency_stats([timed(predict["B"])[1] for _ in range(args.calls)])
    result["A0_again"] = latency_stats([timed(predict["A0"])[1] for _ in range(args.calls)])
    result["cpu_pkg_temp_c_end"] = thermal()
    result["foreign_gpu_pids_at_end"] = gpu.foreign_pids()
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    main()
