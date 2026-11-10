"""Follow-up 2: setup B (1 step, no blank slot, cameras batched, whole-model compile) with one torch.compile mode,
in a fresh process, on today's stack (the kept final worktree's venv).

Startup = the first predict_action_chunk call after loading (it includes compiling). Whether that is cold or
reuses a cache is set by the caller (run_task2_compile_modes.sh): caches off in a fresh dir (cold), caches on in
a fresh empty dir (fill), or that dir again (warm). Then: 10 warm-up calls, 200 timed calls (median, p99),
J per decision from NVML's energy counter (idle window, ~6 s of calls, idle window), and torch._dynamo's
counters (frames, graphs, graph breaks; a frame count that changes after the first call = a recompile).
TF32 on, as in the final session. Saves the actions of the first and the last call for parity checks.
Usage: python compile_modes.py --mode default --label cold1 --out-dir <dir>
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from followup_common import (SETUPS, Gpu, Sampler, cpu_pkg_temp, dynamo_summary, energy_block, latency_stats,
                             load_policy, make_inputs, timed, timed_block)

MODES = ["max-autotune", "default", "reduce-overhead"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--energy-seconds", type=float, default=6)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {args.mode} {args.label}: {msg}", flush=True)  # noqa: E731
    gpu = Gpu()
    gpu.wait_until_ours(log)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    temps_start = {"cpu_pkg_c": cpu_pkg_temp(), "gpu_c": gpu.temp()}

    t = time.perf_counter()
    policy = load_policy(SETUPS["B"], compile_mode=args.mode)
    load_s = time.perf_counter() - t
    batch, noise = make_inputs(policy)

    def predict():
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())

    actions, first_ms = timed(predict)
    first_actions = actions.clone().cpu()  # CUDA graphs reuse output buffers, so keep a copy
    after_first = dynamo_summary()
    log(f"first call {first_ms / 1000:.1f} s")
    for _ in range(10):
        timed(predict)

    sampler = Sampler(gpu)
    sampler.start()
    ms = timed_block(gpu, sampler, predict, args.calls, "timed", log)
    stats = latency_stats(ms)
    energy = energy_block(gpu, sampler, predict, args.energy_seconds, stats["median_ms"], "energy", log)
    last_actions = predict().clone().cpu()
    sampler.stop()
    after_all = dynamo_summary()

    result = {
        "mode": args.mode, "label": args.label, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__, "compile_mode_in_config": policy.config.compile_mode,
        "load_s": round(load_s, 2), "first_call_s": round(first_ms / 1000, 2),
        "latency_200_warm_calls": stats, "energy": energy,
        "dynamo_after_first_call": after_first, "dynamo_after_all_calls": after_all,
        "recompiled_after_first_call": after_first["frames"] != after_all["frames"],
        "first_and_last_call_actions_identical": bool(torch.equal(first_actions, last_actions)),
        "temps_start": temps_start, "gpu_cpu_during_timed_calls": sampler.summary("timed"),
        "gpu_cpu_during_energy_block": sampler.summary("energy"),
        "env": {k: __import__("os").environ.get(k) for k in ("TORCHINDUCTOR_FORCE_DISABLE_CACHES",
                                                             "TORCHINDUCTOR_CACHE_DIR")},
    }
    name = f"{args.mode}_{args.label}"
    torch.save(first_actions, out / f"{name}_first_actions.pt")
    torch.save(last_actions, out / f"{name}_last_actions.pt")
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    log(json.dumps({k: result[k] for k in ("first_call_s", "recompiled_after_first_call")}
                   | {"median_ms": stats["median_ms"], "p99_ms": stats["p99_ms"],
                      "j_total": energy["j_per_call_total"], "graphs": after_all["unique_graphs"],
                      "graph_breaks": after_all["graph_break_count"]}))


if __name__ == "__main__":
    main()
