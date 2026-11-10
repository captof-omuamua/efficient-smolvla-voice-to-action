"""Collect follow-up 3 (talk's stack OLD vs today's NEW, alternating fresh processes) into stack_compare_summary.json.

Per setup: every run's median and p99 (200 calls), the NEW/OLD ratio for each adjacent OLD/NEW pair in the
mirrored order (runs 1-2, 3-4, 5-6), A0's parts (prep, vision encoder, embed_prefix, VLM pass, denoise loop),
GPU kernel time and kernel launches per call, and temperatures and clocks during the timed calls.
Usage: python summarize_task3.py
"""

import json
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the repository root
HERE = Path(os.environ.get("OUT_DIR", REPO / "runs")) / "speed" / "followups"  # results read and written
RUNS = HERE / "stack_compare"


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def main():
    runs = [json.loads(p.read_text()) | {"source": os.path.relpath(p, REPO)} for p in sorted(RUNS.glob("*.json"))]
    result = {"what": "Talk's stack (OLD: 8c4f33e5, torch 2.11, transformers 5.5.4) vs today's (NEW: final worktree, "
                      "torch 2.15, transformers 5.22.1), alternating fresh processes, TF32 on for both",
              "stacks": {}, "setups": {}}
    for r in runs:
        result["stacks"].setdefault(r["stack"], {"lerobot_file": r["lerobot_file"], "torch": r["torch"],
                                                 "transformers": r["transformers"],
                                                 "float32_matmul_precision": r["float32_matmul_precision"],
                                                 "cuda_matmul_allow_tf32": r["cuda_matmul_allow_tf32"]})
    fills = {r["stack"]: {"first_call_s (caches on, empty cache)": r["first_call_s"], "source": r["source"]}
             for r in runs if r["label"] == "fill"}
    result["B_compile_first_call_fill_runs"] = fills
    for setup in ("A0", "B"):
        mine = sorted((r for r in runs if r["setup"] == setup and r["label"] != "fill"),
                      key=lambda r: int(r["label"].removeprefix("run")))
        if not mine:
            continue
        rows = [{"label": r["label"], "stack": r["stack"], "median_ms": r["latency"]["median_ms"],
                 "p99_ms": r["latency"]["p99_ms"], "min_ms": r["latency"]["min_ms"],
                 "parts_ms": r["parts_ms"], "gpu_profile": r["gpu_profile"],
                 "temps_start": r["temps_start"], "during_timed_calls": r["gpu_cpu_during_timed_calls"],
                 "source": r["source"]} for r in mine]
        pairs = []
        for a, b in zip(rows[0::2], rows[1::2]):  # mirrored order: (OLD, NEW), (NEW, OLD), (OLD, NEW)
            old, new = (a, b) if a["stack"] == "OLD" else (b, a)
            pairs.append({"runs": [a["label"], b["label"]], "median_new_over_old": round(new["median_ms"] / old["median_ms"], 3),
                          "p99_new_over_old": round(new["p99_ms"] / old["p99_ms"], 3)})
        per_stack = {}
        for stack in ("OLD", "NEW"):
            s = [row for row in rows if row["stack"] == stack]
            entry = {"median_ms_runs": [row["median_ms"] for row in s], "p99_ms_runs": [row["p99_ms"] for row in s],
                     "min_ms_runs": [row["min_ms"] for row in s],
                     "median_ms_mean": mean(row["median_ms"] for row in s), "p99_ms_mean": mean(row["p99_ms"] for row in s),
                     "gpu_kernel_ms_per_call_mean": mean(row["gpu_profile"].get("gpu_kernel_ms_per_call") for row in s),
                     "kernels_per_call": sorted({row["gpu_profile"].get("kernels_per_call") for row in s},
                                                key=lambda v: (v is None, v)),
                     "cpu_pkg_temp_c_start": [row["temps_start"]["cpu_pkg_c"] for row in s],
                     "cpu_pkg_temp_c_mean_during": [row["during_timed_calls"].get("cpu_pkg_temp_c_mean") for row in s],
                     "cpu_mhz_mean_during": [row["during_timed_calls"].get("cpu_mhz_mean") for row in s],
                     "gpu_temp_c_mean_during": [row["during_timed_calls"].get("gpu_temp_c_mean") for row in s],
                     "sm_mhz_mean_during": [row["during_timed_calls"].get("sm_mhz_mean") for row in s]}
            if setup == "A0":
                entry["parts_ms_mean"] = {k: mean(row["parts_ms"][k] for row in s)
                                          for k in ("prep", "vision_encoder", "embed_prefix", "vlm_prefix_pass",
                                                    "denoise_loop_est")}
            if entry["gpu_kernel_ms_per_call_mean"] is not None:
                entry["wall_minus_gpu_kernel_ms"] = round(entry["median_ms_mean"] - entry["gpu_kernel_ms_per_call_mean"], 2)
            per_stack[stack] = entry
        ratio_of_means = round(per_stack["NEW"]["median_ms_mean"] / per_stack["OLD"]["median_ms_mean"], 3)
        result["setups"][setup] = {"runs": rows, "pairs": pairs, "per_stack": per_stack,
                                   "median_new_over_old_mean_of_pairs": round(statistics.mean(
                                       p["median_new_over_old"] for p in pairs), 3),
                                   "median_new_over_old_ratio_of_means": ratio_of_means}
        if setup == "A0":
            result["setups"][setup]["parts_new_minus_old_ms"] = {
                k: round(per_stack["NEW"]["parts_ms_mean"][k] - per_stack["OLD"]["parts_ms_mean"][k], 2)
                for k in per_stack["NEW"]["parts_ms_mean"]}
    (HERE / "stack_compare_summary.json").write_text(json.dumps(result, indent=2))
    for setup, s in result["setups"].items():
        print(setup, "pairs", s["pairs"], "mean ratio", s["median_new_over_old_mean_of_pairs"])
        for stack, e in s["per_stack"].items():
            print("  ", stack, {k: v for k, v in e.items() if not k.startswith(("cpu_", "gpu_temp", "sm_"))})
        if "parts_new_minus_old_ms" in s:
            print("   parts NEW-OLD:", s["parts_new_minus_old_ms"])
    print("fills", fills)


if __name__ == "__main__":
    main()
