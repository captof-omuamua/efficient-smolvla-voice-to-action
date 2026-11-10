"""Collect follow-up 2 (compile modes for B) into compile_modes_summary.json.

Per mode: cold first calls (caches off), the fill run's first call (empty cache, caches on), the warm first call;
median and p99 over 200 warm calls and J per decision for every run; recompiles and graph breaks; action parity
against max-autotune (first 10 steps of the chunk, the ones executed with n_action_steps=10), with two references:
max-autotune against itself in another process, and eager C (same model, no compiler, from the final session).
Usage: python summarize_task2.py
"""

import json
import os
import statistics
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]  # the repository root
HERE = Path(os.environ.get("OUT_DIR", REPO / "runs")) / "speed" / "followups"  # results read and written
RUNS = HERE / "compile_modes"
MODES = ["max-autotune", "default", "reduce-overhead"]
LABELS = ["cold1", "cold2", "fill", "warm"]
REFERENCE = ("max-autotune", "cold1")
EAGER_C = HERE.parent / "session/C_first_call_actions.pt"


def actions(mode, label):
    return torch.load(RUNS / f"{mode}_{label}_last_actions.pt")


def parity(a, b):
    diff = (a[:, :10] - b[:, :10]).abs()
    return {"mean_abs_diff_first_10_steps": round(float(diff.mean()), 5),
            "max_abs_diff_first_10_steps": round(float(diff.max()), 5), "bit_identical": bool(torch.equal(a, b))}


def main():
    runs = {(m, lab): json.loads((RUNS / f"{m}_{lab}.json").read_text())
            for m in MODES for lab in LABELS if (RUNS / f"{m}_{lab}.json").exists()}
    ref = actions(*REFERENCE)
    summary = {}
    for mode in MODES:
        mine = {lab: r for (m, lab), r in runs.items() if m == mode}
        if not mine:
            continue
        per_run = {lab: {"first_call_s": r["first_call_s"], "median_ms": r["latency_200_warm_calls"]["median_ms"],
                         "p99_ms": r["latency_200_warm_calls"]["p99_ms"],
                         "j_per_decision_total": r["energy"]["j_per_call_total"],
                         "j_per_decision_above_idle": r["energy"]["j_per_call_above_idle"],
                         "busy_w": r["energy"]["busy_w"], "idle_w": r["energy"]["idle_w"],
                         "recompiled_after_first_call": r["recompiled_after_first_call"],
                         "unique_graphs": r["dynamo_after_all_calls"]["unique_graphs"],
                         "graph_break_count": r["dynamo_after_all_calls"]["graph_break_count"],
                         "frames": r["dynamo_after_all_calls"]["frames"],
                         "first_and_last_call_actions_identical": r["first_and_last_call_actions_identical"],
                         "cpu_pkg_temp_c_mean_during_calls": r["gpu_cpu_during_timed_calls"].get("cpu_pkg_temp_c_mean"),
                         "gpu_temp_c_mean_during_calls": r["gpu_cpu_during_timed_calls"].get("gpu_temp_c_mean"),
                         "source": os.path.relpath(RUNS / f"{mode}_{lab}.json", REPO)}
                   for lab, r in mine.items()}
        medians = [v["median_ms"] for v in per_run.values()]
        summary[mode] = {
            "cold_first_call_s (caches off)": [per_run[lab]["first_call_s"] for lab in ("cold1", "cold2") if lab in per_run],
            "fill_first_call_s (empty cache, caches on)": per_run.get("fill", {}).get("first_call_s"),
            "warm_first_call_s": per_run.get("warm", {}).get("first_call_s"),
            "median_ms_range_over_runs": [min(medians), max(medians)],
            "median_ms_mean_over_runs": round(statistics.mean(medians), 2),
            "p99_ms_range_over_runs": [min(v["p99_ms"] for v in per_run.values()),
                                       max(v["p99_ms"] for v in per_run.values())],
            "j_per_decision_total_mean": round(statistics.mean(v["j_per_decision_total"] for v in per_run.values()), 3),
            "j_per_decision_above_idle_mean": round(statistics.mean(v["j_per_decision_above_idle"]
                                                                    for v in per_run.values()), 3),
            "any_recompile": any(v["recompiled_after_first_call"] for v in per_run.values()),
            "graph_break_count": sorted({v["graph_break_count"] for v in per_run.values()}),
            "unique_graphs": sorted({v["unique_graphs"] for v in per_run.values()}),
            "parity_vs_max_autotune_cold1": {lab: parity(actions(mode, lab), ref) for lab in mine},
            "runs": per_run,
        }
    eager = torch.load(EAGER_C)
    result = {
        "what": "B with compile_mode max-autotune / default / reduce-overhead on today's stack, fresh process per run",
        "reference_actions": f"{REFERENCE[0]} {REFERENCE[1]} (last call)",
        "parity_reference_eager_C_vs_max_autotune_cold1": parity(eager, ref),
        "modes": summary,
    }
    (HERE / "compile_modes_summary.json").write_text(json.dumps(result, indent=2))
    for mode, s in summary.items():
        print(mode, {k: v for k, v in s.items() if k not in ("runs", "parity_vs_max_autotune_cold1")})
        print("   parity:", {lab: (p["mean_abs_diff_first_10_steps"], p["max_abs_diff_first_10_steps"])
                             for lab, p in s["parity_vs_max_autotune_cold1"].items()})
    print("eager C vs max-autotune:", result["parity_reference_eager_C_vs_max_autotune_cold1"])


if __name__ == "__main__":
    main()
