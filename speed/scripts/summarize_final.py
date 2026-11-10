"""Collect the final numbers into final_numbers.json, with the source file of every number.

Reads session/ and session_pass2/session.json, precision_check/precision_check.json, startup/*.json, the LIBERO runs' eval_info.json
and libero_energy_*.jsonl, and the talk's files for comparison. Paired LIBERO stats use the same formulas as
bench/libero_compare.py (imported from there).
Usage: python summarize_final.py   (writes final_numbers.json next to this file)
"""

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the repository root
HERE = Path(os.environ.get("RESULTS_DIR", Path(os.environ.get("OUT_DIR", REPO / "runs")) / "speed"))  # results read and written
RUNS = HERE / "libero_runs"
LOG_DIR = Path(os.environ.get("LOG_DIR", REPO / "logs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # libero_compare.py sits next to this file
from libero_compare import SUITES, load_successes, mcnemar_p, paired_diff_ci  # noqa: E402

A1_RUN, B_RUN = "final_2026-11-07_A1_eager_steps10", "final_2026-11-07_B_tuned"
TALK_BASELINE_RUN, TALK_B_RUN = "baseline_eager_n10", "batched_compile_model_steps1_noblank"
TALK = {  # the talk's 31 Oct numbers, and the 7 Nov same-session energy re-measure
    "median_ms": {"A0": 101.3, "B": 13.1}, "p99_ms": {"A0": 120.1, "B": 16.3}, "speedup_median": 7.7,
    "energy_same_session_7nov_j_total": {"A0": 10.264, "B": 1.405, "ratio": 7.31},
    "energy_same_session_7nov_j_above_idle": {"A0": 8.094, "B": 1.123, "ratio": 7.21},
    "success_pct": {"A0": 83.75, "B": 83.00}, "startup_b": "71-89 s cold (two runs), 15 s warm",
    "sources": {"speed": "bench/results_v2/SUMMARY.md (pr_eager_steps10_stats.json, "
                         "no_blank_slot/batched_compile_model_steps1_stats.json)",
                "energy": "bench/results_energy_same_session_2026-11-07/summary.json (pass2)",
                "success": "libero_runs/baseline_eager_n10, libero_runs/batched_compile_model_steps1_noblank"},
}


def rel(path):
    return os.path.relpath(Path(path).resolve(), REPO)


def paired(ref_name, run_name):
    """Run vs reference over the same 400 starting scenes: what libero_compare.py prints, as numbers."""
    ref, run = load_successes(RUNS / ref_name), load_successes(RUNS / run_name)
    assert ref.keys() == run.keys(), f"{run_name} has different episodes from {ref_name}"
    b = sum(ref[k] and not run[k] for k in ref)  # reference succeeded, run failed
    c = sum(run[k] and not ref[k] for k in ref)  # run succeeded, reference failed
    diff, low, high = paired_diff_ci(b, c, len(run))
    return {"reference": ref_name, "run": run_name, "episodes": len(run),
            "reference_pct": round(100 * sum(ref.values()) / len(ref), 2),
            "run_pct": round(100 * sum(run.values()) / len(run), 2),
            "only_reference_won": b, "only_run_won": c, "mcnemar_p": round(mcnemar_p(b, c), 3),
            "diff_pp": round(diff, 2), "ci95_pp": [round(low, 1), round(high, 1)]}


def libero_run(name):
    info_path = RUNS / name / "eval_info.json"
    info = json.loads(info_path.read_text())
    successes = load_successes(RUNS / name)
    per_suite = {s: sum(v for (suite, _, _), v in successes.items() if suite == s) for s in SUITES}
    out = {"eval_info": rel(info_path), "pc_success": info["overall"]["pc_success"],
           "n_success": info["overall"]["n_success"], "n_episodes": info["overall"]["n_episodes"],
           "pc_success_ci95_wilson": info["overall"].get("pc_success_ci95"),
           "per_suite_successes_of_100": per_suite, "eval_s": round(info["overall"]["eval_s"], 1)}
    energy_path = HERE / f"libero_energy_{name}.jsonl"
    if energy_path.exists():
        rows = {r["mark"]: r for r in map(json.loads, energy_path.read_text().splitlines())}
        seconds = rows["end"]["unix_s"] - rows["start"]["unix_s"]
        joules = (rows["end"]["energy_mj"] - rows["start"]["energy_mj"]) / 1000
        idle_s = rows["idle_end"]["unix_s"] - rows["idle_start"]["unix_s"]
        idle_w = (rows["idle_end"]["energy_mj"] - rows["idle_start"]["energy_mj"]) / 1000 / idle_s
        out["gpu_energy_whole_run"] = {
            "label": "includes simulator rendering (MuJoCo EGL on the same GPU), model loading and compiling",
            "source": rel(energy_path), "seconds": round(seconds, 1), "kilojoules": round(joules / 1000, 2),
            "mean_watts": round(joules / seconds, 1), "joules_per_episode": round(joules / out["n_episodes"], 1),
            # context only: the two runs' idle readings caught the GPU in different power states
            "idle_watts_20s_before_run": round(idle_w, 1)}
    return out


def session_numbers(session):
    speed = {}
    for k, v in session["summary"].items():
        cpu = [b["gpu"].get("other_cpu_cores_mean") for b in session["blocks"] if b["setup"] == k]
        speed[k] = {"setup": v["setup"], "median_ms": v["latency"]["median_ms"], "p99_ms": v["latency"]["p99_ms"],
                    "calls": v["latency"]["calls"], "block_medians_ms": v["block_median_ms"]["per_block"],
                    "j_per_decision_total": round(v["j_per_call_total"]["mean"], 3),
                    "j_per_decision_total_range": [v["j_per_call_total"]["min"], v["j_per_call_total"]["max"]],
                    "j_per_decision_above_idle": round(v["j_per_call_above_idle"]["mean"], 3),
                    "j_per_decision_above_idle_range": [v["j_per_call_above_idle"]["min"],
                                                        v["j_per_call_above_idle"]["max"]],
                    "busy_w": v["busy_w"]["mean"], "idle_w": v["idle_w"]["mean"],
                    "first_call_s_in_session": v["first_call_s_in_session"],
                    "other_programs_cpu_cores_mean_over_blocks": round(sum(cpu) / len(cpu), 2)}
    return speed


def first_rollout_step(name):
    """The first policy step of the eval, from the tqdm line in the run's log (for B it includes the compile)."""
    log = LOG_DIR / f"final_2026-11-07_libero_{name}.log"
    text = log.read_text(errors="replace").replace("\r", "\n")
    m = re.search(r"Running rollout with at most \d+ steps:\s+0%\|[^|]*\| 1/\d+ \[[\d:]+<[^,]*,\s+([\d.]+)(s/it|it/s)",
                  text)
    if not m:
        return None
    rate = float(m.group(1))
    return {"seconds": round(rate if m.group(2) == "s/it" else 1 / rate, 2), "source": rel(log)}


def main():
    sessions = {"pass1": HERE / "session/session.json", "pass2": HERE / "session_pass2/session.json"}
    loaded = {p: json.loads(path.read_text()) for p, path in sessions.items()}
    session = loaded["pass1"]
    speed = {p: session_numbers(s) for p, s in loaded.items()}
    ratios = {p: s["ratios"] for p, s in loaded.items()}

    startup = {}
    for path in sorted((HERE / "startup").glob("*.json")):
        s = json.loads(path.read_text())
        startup[f"{s['setup']}_{s['label']}"] = {
            "source": rel(path), "first_call_s": s["first_call_s"], "load_s": s["load_s"],
            "script_start_to_first_action_s": s["script_start_to_first_action_s"],
            "peak_gpu_mb_allocated": s["peak_gpu_mb_allocated"], "peak_gpu_mb_reserved": s["peak_gpu_mb_reserved"],
            "median_ms_warm_calls": s["median_ms_warm_calls"],
            "recompiled_after_first_call": s["recompiled_after_first_call"],
            "dynamo_frames": s["dynamo_after_warm_calls"].get("frames"),
            "unique_graphs": s["dynamo_after_warm_calls"].get("stats", {}).get("unique_graphs"),
            "graph_break_reasons": {reason.split("\n")[0] + " | " + reason.split("\n")[1].strip(): n
                                    for reason, n in s["dynamo_after_warm_calls"].get("unimplemented", {}).items()},
            "graph_break_counter": sum(s["dynamo_after_warm_calls"].get("graph_break", {}).values())}

    startup["B_cold_first_call_s_all_runs"] = {
        "session_pass1": loaded["pass1"]["summary"]["B"]["first_call_s_in_session"],
        "session_pass2": loaded["pass2"]["summary"]["B"]["first_call_s_in_session"],
        "startup_cold": startup["B_cold"]["first_call_s"], "startup_fill (caches on, empty cache)":
        startup["B_fill"]["first_call_s"]}
    colocation = {}
    for path in sorted((HERE / "colocation_check").glob("run*.json")):
        c = json.loads(path.read_text())
        colocation[path.stem] = {"A0_median_ms": c["A0"]["median_ms"], "A0_again_median_ms": c["A0_again"]["median_ms"],
                                 "B_median_ms": c.get("B", {}).get("median_ms"),
                                 "cpu_pkg_temp_c": [c["cpu_pkg_temp_c_start"], c["cpu_pkg_temp_c_end"]],
                                 "source": rel(path)}
    precision = json.loads((HERE / "precision_check/precision_check.json").read_text())
    libero = {name: libero_run(name) for name in (A1_RUN, B_RUN) if (RUNS / name / "eval_info.json").exists()}
    for name in libero:
        libero[name]["first_rollout_step"] = first_rollout_step(name)
    compare = {}
    if len(libero) == 2:
        compare["B_vs_A1"] = paired(A1_RUN, B_RUN)
        compare["context_vs_talk_baseline"] = [paired(TALK_BASELINE_RUN, r) for r in (A1_RUN, B_RUN, TALK_B_RUN)]
        compare["context_talk_B_vs_today_B"] = paired(TALK_B_RUN, B_RUN)

    result = {
        "what": "Final numbers for the 10 Nov talk, measured 7-8 Nov 2026 on today's LeRobot",
        "sources": {"speed_energy": {p: rel(path) for p, path in sessions.items()},
                    "precision_check": "bench/results_final_2026-11-07/precision_check/precision_check.json",
                    "startup": "bench/results_final_2026-11-07/startup/", "libero_runs": "libero_runs/"},
        "code": session["code"], "environment": session["environment"],
        "session_settings": {p: {k: s["settings"][k] for k in ("order", "calls_per_block", "window_s_target",
                                                               "settle_median_ms", "inputs_identical_to_A1")}
                             for p, s in loaded.items()},
        "speed_energy": speed,
        "ratios": ratios,
        "recompiled_during_session_blocks": {p: s["dynamo_counters"]["recompiled_during_blocks"]
                                             for p, s in loaded.items()},
        "redone_blocks": {p: s["redone_blocks"] for p, s in loaded.items()},
        "startup_and_memory": startup,
        "precision_check": {k: precision[k] for k in ("results", "median_ratio_off_over_on", "redone_blocks")},
        "colocation_check": colocation,
        "parity": session["parity"],
        "libero": libero,
        "libero_paired": compare,
        "talk": TALK,
    }
    (HERE / "final_numbers.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"speed": speed, "ratios_A0_over_B": {p: r["A0_over_B"] for p, r in ratios.items()},
                      "ratios_A1_over_B": {p: r["A1_over_B"] for p, r in ratios.items()},
                      "startup": startup, "libero": libero, "paired": compare}, indent=1))


if __name__ == "__main__":
    main()
