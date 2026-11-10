"""One chart for the talk: the policy's p99 decision time in each regime of a shared-GPU run.

Usage: plot_share.py RUN_DIR OUT_STEM
Reads RUN_DIR/summary.json and RUN_DIR/decisions.csv, writes OUT_STEM.pdf and OUT_STEM.png.
"""
import csv
import json
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

INK, MUTED, FAINT = "#1F2328", "#6B7280", "#D1D5DB"  # the deck's text and line colours
GREY, RED, BLUE = "#6B7280", "#D64545", "#1A8FD0"  # baseline, no turn-taking, turn-taking
DEADLINE_MS = 50  # one control step at 20 Hz

# (group title, colour, [(regime in summary.json, label on the chart)])
GROUPS = [
    ("Voice models loaded, doing nothing", GREY, [("idle", "Idle")]),
    ("Sharing the GPU, no turn-taking", RED, [
        ("whisper_flat", "Speech-to-text, flat out"),
        ("llm_flat", "Language model, flat out"),
        ("user_llm", "Busy user, language model every time"),
        ("user_router", "Busy user, word rules first"),
    ]),
    ("Sharing the GPU, with turn-taking", BLUE, [
        ("flat_gate", "Both, flat out"),
        ("user_llm_gate", "Busy user, language model every time"),
        ("user_router_gate", "Busy user, word rules first"),
    ]),
]


def whole_log(run_dir):
    """Every logged decision: how many, how many late, the longest beat-to-end time, the longest start delay (ms)."""
    rows = [(int(r["beat_ns"]), int(r["start_ns"]), int(r["end_ns"]))
            for r in csv.DictReader(open(f"{run_dir}/decisions.csv"))]
    beat_to_end = [(end - beat) / 1e6 for beat, start, end in rows]
    start_delay = [(start - beat) / 1e6 for beat, start, end in rows]
    return len(rows), sum(ms > DEADLINE_MS for ms in beat_to_end), max(beat_to_end), max(start_delay)


def main(run_dir, out_stem):
    policy = json.load(open(f"{run_dir}/summary.json"))["policy"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y, ticks, names = 0.0, [], []
    for title, colour, rows in GROUPS:
        ax.text(0, y, title, color=INK, fontsize=8, fontweight="bold", va="center",  # group title = the legend
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})  # white behind it hides the grid
        y += 0.9
        for regime, label in rows:
            p99 = policy[regime]["p99"]
            by_round = policy[regime]["p99_by_round"].values()  # the same p99, one per round
            ax.barh(y, p99, height=0.5, color=colour)
            ax.plot([min(by_round), max(by_round)], [y, y], color=INK, linewidth=1)  # how far the rounds differ
            ax.text(max(p99, max(by_round)) + 0.8, y, f"{p99:.1f} ms", color=INK, fontsize=8, va="center")
            ticks.append(y)
            names.append(label)
            y += 1.0
        y += 0.35  # air between groups
    ax.axvline(DEADLINE_MS, color=MUTED, linewidth=1, linestyle="--")  # a real threshold, so dashed
    ax.text(DEADLINE_MS - 0.6, ticks[0], "late: ends more than 50 ms\nafter its beat", color=MUTED, fontsize=7,
            ha="right", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(ticks, names, fontsize=8, color=INK)
    ax.set_ylim(y - 0.6, -0.6)  # first group at the top
    ax.set_xlim(0, 56)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Robot policy decision time, p99 (99 of 100 decisions are faster), ms", fontsize=8, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=FAINT, linewidth=0.5)
    ax.set_axisbelow(True)  # grid behind the bars
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    n, late, closest_ms, delay_ms = whole_log(run_dir)
    fig.text(0.012, 0.015, color=MUTED, fontsize=6.5, va="bottom", s=(
        f"Bars: time of the model call. Thin line: range over the {len(by_round)} rounds. "
        f"Late counts from the beat, which adds a start delay (worst {delay_ms:.1f} ms).\n"
        f"{late} of {n:,} decisions were late; the closest ended {closest_ms:.1f} ms after its beat."))
    fig.tight_layout(rect=(0, 0.085, 1, 1))  # keep the bottom strip free for the footnote
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=300)
    print("wrote", out_stem + ".pdf", "and", out_stem + ".png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
