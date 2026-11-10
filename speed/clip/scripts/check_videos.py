"""Check frames vs steps in the final runs' videos, and list scenes for the clip -> episode_steps.json.

For each of the 400 paired episodes in both runs: frame count and fps tag (ffprobe), success (eval_info.json).
Checks: every video is tagged 20 fps; every failed episode has exactly its suite's step limit of frames.
"""
import json
import subprocess

from measured import CLIPS, MAX_STEPS, ORIGINAL, SUITES, TUNED, VIDEO_FPS, check_sources, successes, video_path

PREFERRED = ("libero_10", 8, 0)  # the scene used in v1 and v2


def probe(path):
    """Frame count (by counting packets, no decoding) and fps tag of a video."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=r_frame_rate,nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["nb_read_packets"]), stream["r_frame_rate"]


check_sources()
ok = {s["name"]: successes(s) for s in (ORIGINAL, TUNED)}

# 1. Every episode of both runs: success and steps (= frames), with the two checks
episodes = []
for suite in SUITES:
    for task in range(10):
        for ep in range(10):
            rec = {"suite": suite, "task": task, "episode": ep}
            for s in (ORIGINAL, TUNED):
                frames, rate = probe(video_path(s, suite, task, ep))
                success = ok[s["name"]][(suite, task, ep)]
                assert rate == f"{VIDEO_FPS}/1", (s["run"], suite, task, ep, rate)  # tagged with the control rate
                assert frames <= MAX_STEPS[suite], (s["run"], suite, task, ep, frames)
                assert success or frames == MAX_STEPS[suite], (s["run"], suite, task, ep, frames)  # failures hit the limit
                rec[s["name"].lower()] = {"success": success, "steps": frames}
            episodes.append(rec)
    print(f"{suite}: checked")

# 2. Scenes for the clip: both runs succeed with the same number of steps, 15-25 s at real time
both = [e for e in episodes if e["original"]["success"] and e["tuned"]["success"]]
same = [e for e in both if e["original"]["steps"] == e["tuned"]["steps"]]
fits = [e for e in same if 15 * VIDEO_FPS <= e["original"]["steps"] <= 25 * VIDEO_FPS]
preferred = next(e for e in episodes if (e["suite"], e["task"], e["episode"]) == PREFERRED)
summary = {
    "videos_checked": 2 * len(episodes),
    "fps_tag": f"{VIDEO_FPS}/1 on every video",
    "failed_episodes_at_step_limit": {s["name"].lower(): sum(not e[s["name"].lower()]["success"] for e in episodes)
                                      for s in (ORIGINAL, TUNED)},
    "both_succeeded": len(both),
    "both_succeeded_same_steps": len(same),
    "same_steps_15_to_25_s": [{k: e[k] for k in ("suite", "task", "episode")} | {"steps": e["original"]["steps"],
                              "instruction": ORIGINAL["tasks"][f"{e['suite']}_{e['task']}"]} for e in fits],
    "preferred_scene": preferred,
}
(CLIPS / "episode_steps.json").write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=1))
print(json.dumps(summary, indent=1))
