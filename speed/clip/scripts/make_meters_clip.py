"""v3 meters_side_by_side.mp4 (+ .png still): the final runs of 8 Nov 2026, one LIBERO episode, side by side.

20 output frames per second = 20 control steps per second (the eval videos hold one frame per step): real time.
The simulator pauses while the model thinks, so both robots move at the same pace; only the GPU cost per
decision differs. Counters = decisions so far x the measured per-decision numbers (session.json, pass 1).
"""
import math

from PIL import ImageDraw

import draw as d
from measured import (CLIPS, ORIGINAL, RATIOS, SCENE, STEP_S, STEPS_PER_DECISION, SUITE_NAMES, TUNED, VIDEO_FPS,
                      check_sources, per_decision)

FPS = VIDEO_FPS  # one output frame per control step = real time
INTRO, OUTRO = 20, 60  # still frames before (1 s) and after (3 s) the episode, time to read
ROWS = (744, 810, 888)  # tops of the three counter rows under each video
SIDES = [(ORIGINAL, d.LEFT_X), (TUNED, d.RIGHT_X)]

check_sources()  # the numbers we print must match their source files

# 1. The same episode from both runs: same start state and seed, solved by both, same length
video = d.load_scene([ORIGINAL, TUNED])
n = len(video["Original"])  # control steps in the episode
total = math.ceil(n / STEPS_PER_DECISION)  # decisions in the whole episode, the same on both sides
full_think_s = total * ORIGINAL["think_ms"] / 1000  # the original's totals fill a meter bar,
full_energy = total * ORIGINAL["energy_j"]  # so both sides share one scale
suite, task, episode = SCENE
instruction = ORIGINAL["tasks"][f"{suite}_{task}"]  # LIBERO's wording, from the run's task_descriptions.json

# 2. Draw what stays still once: subtitle, headers, counter labels, footer
bg = d.background(
    f'{SUITE_NAMES[suite]} task {task}, episode {episode} · same start state and seed · "{instruction}"',
    [(s["name"], per_decision(s)) for s, _ in SIDES],
    [
        "Laptop RTX 5090, not Jetson · measured 8 Nov 2026 on current LeRobot · think time at batch 1 · "
        "energy from one session (NVML counter) · simulator pauses while the model thinks",
        "Counters = decisions so far × per-decision numbers above · this scene succeeds in both runs "
        f"(overall LIBERO success {ORIGINAL['success_pc']:.2f}% vs {TUNED['success_pc']:.2f}%)",
    ],
)
draw = ImageDraw.Draw(bg)
for s, x in SIDES:
    for y, label in zip(ROWS, ("Decisions", "GPU think time", "GPU energy")):
        d.text(draw, (x, y + 10), label, 34, d.GREY)
d.text(draw, (d.GAP_X, 170), "Sim time", 30, d.GREY, anchor="ma")
d.text(draw, (d.GAP_X, 272), "real-time playback", 26, d.GREY, anchor="ma")


def frame(i, decided, sim_s, outro=False):
    """One output frame: eval frame i on both sides, after `decided` decisions, at `sim_s` seconds."""
    img = bg.copy()
    draw = ImageDraw.Draw(img)
    for s, x in SIDES:
        img.paste(video[s["name"]][i], (x, d.VID_Y))
        think_s, energy = decided * s["think_ms"] / 1000, decided * s["energy_j"]  # counters
        for y, value in zip(ROWS, (f"{decided}", f"{think_s:.2f} s", f"{energy:.1f} J")):
            d.text(draw, (x + d.VID, y), value, 50, bold=True, anchor="ra")
        d.bar(draw, x, ROWS[1] + 58, think_s / full_think_s, d.COLORS[s["name"]])
        d.bar(draw, x, ROWS[2] + 58, energy / full_energy, d.COLORS[s["name"]])
        if outro:
            d.badge(draw, x + 16, d.VID_Y + d.VID - 62, "Task done", d.WHITE)
    d.text(draw, (d.GAP_X, 208), f"{sim_s:.2f} s", 54, bold=True, anchor="ma")
    if outro:  # the same-session ratios; both sides make `total` decisions, so they also hold for the totals
        for y, ratio, what in ((420, RATIOS["median_ms"], "less GPU think time"),
                               (580, RATIOS["j_per_call_total"]["overall"], "less GPU energy")):
            d.text(draw, (d.GAP_X, y), f"{ratio:.1f}×", 80, bold=True, anchor="ma")
            d.text(draw, (d.GAP_X, y + 92), what, 28, d.GREY, anchor="ma")
    return img


# 3. Write the clip: intro (nothing decided yet), the episode at real time, outro (episode totals)
out = d.Mp4Writer(CLIPS / "meters_side_by_side.mp4", FPS)
for _ in range(INTRO):
    out.write(frame(0, decided=0, sim_s=0.0))
for i in range(n):  # a decision happens at steps 0, 10, 20, ... so after step i: i // 10 + 1 decisions
    out.write(frame(i, decided=i // STEPS_PER_DECISION + 1, sim_s=i * STEP_S))
last = frame(n - 1, decided=total, sim_s=n * STEP_S, outro=True)
for _ in range(OUTRO):
    out.write(last)
out.close()
last.save(CLIPS / "meters_side_by_side.png")  # the still for the PDF deck
print(f"meters_side_by_side: {INTRO + n + OUTRO} frames at {FPS} fps = {(INTRO + n + OUTRO) / FPS:.1f} s; "
      f"{n} steps, {total} decisions each; think {total * ORIGINAL['think_ms'] / 1000:.2f} s vs "
      f"{total * TUNED['think_ms'] / 1000:.2f} s; energy {total * ORIGINAL['energy_j']:.1f} J vs "
      f"{total * TUNED['energy_j']:.1f} J")
