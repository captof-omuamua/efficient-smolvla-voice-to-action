"""Measured numbers, file paths and simulator facts for the v3 meters clip (final runs of 8 Nov 2026).

Speed and energy: one session on current LeRobot, pass 1, the headline (bench/results_final_2026-11-07).
Episodes and success: the two final closed-loop runs (libero_runs/final_2026-11-07_*).
Every number is read from a file.
"""
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: hide the GPU from this process and from ffmpeg

REPO = Path(__file__).resolve().parents[3]  # the repository root
FINAL = Path(os.environ.get("RESULTS_DIR", REPO / "speed" / "results" / "final_2026-11-07"))  # speed + energy
RUNS = Path(os.environ.get("LIBERO_RUNS", FINAL / "libero_runs"))  # closed-loop evals: eval_info.json, task_descriptions.json, videos/
LOGS = Path(os.environ.get("LOG_DIR", REPO / "logs"))  # the eval logs record each run's policy settings
CLIPS = Path(os.environ.get("OUT_DIR", REPO / "runs")) / "clip"  # everything these scripts make goes here
CLIPS.mkdir(parents=True, exist_ok=True)

# Simulator facts, checked in the final code (lerobot-final-2026-11-07) and the eval logs (see README)
STEP_S = 1 / 20  # the robot runs 20 control steps per second; the videos hold one frame per step
STEPS_PER_DECISION = 10  # n_action_steps=10: the policy makes a new decision every 10 steps
VIDEO_FPS = 20  # since LeRobot #4888 the eval videos are tagged with the control rate, so they play at real time
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_NAMES = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
               "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-Long"}
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # sim/integrations/libero.py

# The scene the clip shows (chosen with check_videos.py; why this one: see README)
SCENE = ("libero_10", 2, 8)  # suite, task, episode: both runs succeed in exactly 242 steps

SESSION = json.loads((FINAL / "session" / "session.json").read_text())  # pass 1, the headline


def load_setup(name, run, key):
    """One setup = its final closed-loop run + its pass-1 speed and energy (key "A1" or "B" in session.json)."""
    numbers = SESSION["summary"][key]
    info = json.loads((RUNS / run / "eval_info.json").read_text())
    return {
        "name": name,
        "run": run,
        "key": key,
        "think_ms": numbers["latency"]["median_ms"],  # median time per decision, batch 1
        "energy_j": numbers["j_per_call_total"]["mean"],  # total GPU energy per decision, NVML energy counter
        "success_pc": info["overall"]["pc_success"],  # LIBERO success over 400 episodes
        "per_task": info["per_task"],  # per-episode successes
        "tasks": json.loads((RUNS / run / "task_descriptions.json").read_text()),  # instruction per task
    }


ORIGINAL = load_setup("Original", "final_2026-11-07_A1_eager_steps10", "A1")
TUNED = load_setup("Tuned", "final_2026-11-07_B_tuned", "B")
RATIOS = SESSION["ratios"]["A1_over_B"]  # original / tuned, per decision, from the same session


def check_sources():
    """Stop if session.json, final_numbers.json, the eval runs and their logs disagree."""
    final = json.loads((FINAL / "final_numbers.json").read_text())
    for s in (ORIGINAL, TUNED):
        summary = final["speed_energy"]["pass1"][s["key"]]
        assert summary["median_ms"] == s["think_ms"] and summary["j_per_decision_total"] == s["energy_j"]
        assert final["libero"][s["run"]]["pc_success"] == s["success_pc"]
    assert final["ratios"]["pass1"]["A1_over_B"] == RATIOS
    # the ratios we show follow from the per-decision numbers we show
    assert round(ORIGINAL["think_ms"] / TUNED["think_ms"], 2) == RATIOS["median_ms"]
    assert round(ORIGINAL["energy_j"] / TUNED["energy_j"], 2) == RATIOS["j_per_call_total"]["overall"]
    # the session measured the same two setups that ran the episodes
    setups = SESSION["settings"]["setups"]
    for s, (compiled, steps, cameras) in ((ORIGINAL, (False, 10, "looped")), (TUNED, (True, 1, "batched"))):
        setup = setups[s["key"]]
        assert (setup["compile_model"], setup["num_steps"], setup["empty_cameras"]) == (compiled, steps, 0)
        assert setup["batched_cameras"] == (cameras == "batched")
        log = (LOGS / f"final_2026-11-07_libero_{s['run']}.log").read_text()
        assert f"SmolVLA cameras: {cameras}; num_steps={steps}, empty_cameras=0, compile_model={compiled}" in log
    assert ORIGINAL["tasks"] == TUNED["tasks"]  # same task list and wording in both runs


def per_decision(setup):
    """Header text, e.g. '111.0 ms, 11.0 J per decision'."""
    return f"{setup['think_ms']:.1f} ms, {setup['energy_j']:.1f} J per decision"


def decisions(steps):
    """Decisions an episode needs: one at step 0, 10, 20, ..."""
    return math.ceil(steps / STEPS_PER_DECISION)


def video_path(setup, suite, task, episode):
    return RUNS / setup["run"] / "videos" / f"{suite}_{task}" / f"eval_episode_{episode}.mp4"


def successes(setup):
    """{(suite, task, episode): True/False}, read from eval_info.json."""
    out = {}
    for t in setup["per_task"]:
        for ep, ok in enumerate(t["metrics"]["successes"]):
            # the video list in the same entry must point at episode `ep`
            assert t["metrics"]["video_paths"][ep].endswith(f"/{t['task_group']}_{t['task_id']}/eval_episode_{ep}.mp4")
            out[(t["task_group"], t["task_id"], ep)] = ok
    return out
