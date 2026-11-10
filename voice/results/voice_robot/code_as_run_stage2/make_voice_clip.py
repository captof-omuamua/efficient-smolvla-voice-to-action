"""Turn one robot run into a finished clip: robot video, a caption panel and the recorded voice.

Input, in a run folder: frames.mp4 (one frame per simulator step, 20 fps, no captions)
and timeline.json (what was said and when, and the robot's state per step).
Output, in the same folder: clip.mp4 (1280 x 720, H.264, 20 fps, with sound) and clip.png.

  python make_voice_clip.py results/voice_robot/<name>/<run folder>
"""
import functools
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tasks import COMMANDS

# Output frame: 1280 x 720 fills a 16:9 slide. Robot video on the left, captions on the right.
W, H = 1280, 720
BOX = 640  # the robot video is scaled to fit a 640 x 640 square
BOX_X, BOX_Y = 24, 24
PANEL_X = BOX_X + BOX + 32  # left edge of the caption panel
PANEL_W = W - PANEL_X - 24  # its width
RATE = 16000  # the voice recordings: 16 kHz, mono, 16-bit
HOLD_END_S = 1.5  # hold the last frame this long at the end
FOOTER = "Recorded voice, played in at the marked times. Simulator."

# Colours of the talk deck
WHITE, TEXT, MUTED, LINES = (255, 255, 255), (31, 35, 40), (107, 114, 128), (209, 213, 219)
BLUE, RED = (26, 143, 208), (214, 69, 69)
STATE_COLOR = {"WAITING": MUTED, "MOVING": BLUE, "RETURNING": BLUE, "DONE": BLUE, "HOLDING": RED, "FAILED": RED}

# The deck's font (Liberation Sans), as in talk/clips/v3/scripts/draw.py
REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


@functools.lru_cache
def font(size, bold=False):
    """Load a font once per size and style (cached)."""
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def wrap(draw, s, size, bold=False, width=PANEL_W):
    """Split a string into lines that each fit within `width` pixels."""
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font(size, bold)) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def write_lines(draw, y, s, size, color=TEXT, bold=False, x=PANEL_X, width=PANEL_W):
    """Draw wrapped text starting at height y; return the height just below it."""
    for line in wrap(draw, s, size, bold, width):
        draw.text((x, y), line, font=font(size, bold), fill=color)
        y += round(size * 1.25)
    return y


def badge(draw, x, y, s, color, size=36):
    """A filled pill with white text, e.g. MOVING."""
    w = draw.textlength(s, font=font(size, True))
    draw.rounded_rectangle([x, y, x + w + 40, y + size + 24], radius=(size + 24) // 2, fill=color)
    draw.text((x + 20, y + 12), s, font=font(size, True), fill=WHITE)


def command_id(name):
    """The state's command as an id: it may be LIBERO's sentence or the id itself."""
    if name in COMMANDS:
        return COMMANDS.index(name)
    return int(name) if str(name).isdigit() else None


def state_at(timeline, step):
    """The robot's state entry in force at this step (the last one that started at or before it)."""
    states = [s for s in timeline["states"] if s["from_step"] <= step]
    return states[-1] if states else {"state": "WAITING", "command": None}


def took_effect(line, t):
    """Has this line's event taken effect by time t? effect_s is None if it never did before the run ended."""
    return line["effect_s"] is not None and t >= line["effect_s"]


def commands_worked_on(timeline, step, ids):
    """How many of the plan's commands the arm has worked on (moved or held for) up to this step."""
    worked = 0
    for s in timeline["states"]:
        if s["from_step"] <= step and s["state"] in ("MOVING", "HOLDING"):
            cid = command_id(s["command"])
            if cid in ids:
                worked = max(worked, ids.index(cid) + 1)
    return worked


def draw_panel(draw, timeline, t, step, at_end):
    """Everything on the right: state, clock, the latest line and the command(s) in force at time t.
    at_end is True from the last step on, where a lost goal is flagged."""
    state = state_at(timeline, step)
    badge(draw, PANEL_X, 24, state["state"], STATE_COLOR.get(state["state"], MUTED))
    draw.text((W - 24, 34), f"{t:4.1f} s", font=font(30), fill=MUTED, anchor="ra")
    draw.line([PANEL_X, 104, W - 24, 104], fill=LINES, width=2)

    # The voice: "listening" while a line plays, then what was heard and what it meant.
    y = 124
    playing = [line for line in timeline["lines"] if line["start_s"] <= t < line["end_s"]]
    ended = [line for line in timeline["lines"] if line["end_s"] <= t]
    if playing:
        draw.ellipse([PANEL_X, y + 4, PANEL_X + 26, y + 30], fill=BLUE)
        draw.text((PANEL_X + 40, y), "Listening...", font=font(32, True), fill=BLUE)
    elif ended and t < ended[-1]["end_s"] + ended[-1]["speech_ms"] / 1000:
        # Whisper is still working on the line that just ended
        draw.text((PANEL_X, y), "Transcribing…", font=font(32, True), fill=BLUE)
    elif ended:
        line = ended[-1]
        y = write_lines(draw, y, f"Heard: “{line['heard']}”", 30)
        if took_effect(line, t):
            if line.get("file"):  # timings only for a real recording, not for a sentence given as text
                y = write_lines(draw, y + 2, f"Speech {line['speech_ms']:.0f} ms + decision {line['parse_ms']:.1f} ms",
                                22, MUTED)
            result = {"stop": ("STOP", RED), "continue": ("CONTINUE", BLUE),
                      "refuse": ("Refused: not one of the robot's commands", RED)}.get(line["event"])
            if result:
                y = write_lines(draw, y + 10, result[0], 40 if result[0] in ("STOP", "CONTINUE") else 30,
                                result[1], bold=True)
    else:
        draw.text((PANEL_X, y), "Waiting for a voice command", font=font(28), fill=MUTED)

    # The command(s) in force: from the latest plan that has taken effect.
    plans = [line for line in timeline["lines"] if line["event"] == "plan" and took_effect(line, t)]
    if at_end and timeline["success"] and timeline.get("goals_hold_at_end") is False:
        # the goals were reached during the run but were no longer true at the end
        draw.text((PANEL_X, H - 70), "Goal reached, then lost", font=font(26, True), fill=RED)
    if not plans:
        return
    y = max(y + 24, 400)
    draw.line([PANEL_X, y - 16, W - 24, y - 16], fill=LINES, width=2)
    ids = plans[-1]["commands"]
    returning = state["state"] == "RETURNING"
    if len(ids) == 1:
        draw.text((PANEL_X, y), "Command:", font=font(26), fill=MUTED)
        if returning:  # between commands the arm goes back to its ready pose
            write_lines(draw, y + 34, "Returning to the ready pose", 32, BLUE, bold=True)
        else:
            write_lines(draw, y + 34, COMMANDS[ids[0]], 32, TEXT, bold=True)
        return
    draw.text((PANEL_X, y), f"Commands ({len(ids)}, in order):", font=font(26), fill=MUTED)
    y += 38
    current = command_id(state["command"])
    # How many commands of the plan are finished (they are shown greyed, with "(done)").
    if state["state"] == "DONE":
        finished = len(ids)  # all of them
    elif returning:
        finished = commands_worked_on(timeline, step, ids)  # the arm returns after finishing one
    elif current in ids:
        finished = ids.index(current)  # every command before the current one
    else:
        finished = 0
    for i, cid in enumerate(ids):
        now = cid == current and state["state"] in ("MOVING", "HOLDING")
        if now:  # a triangle marks the command being done now
            draw.polygon([(PANEL_X, y + 6), (PANEL_X, y + 30), (PANEL_X + 18, y + 18)], fill=BLUE)
        color = BLUE if now else MUTED if i < finished else TEXT
        label = f"{i + 1}. {COMMANDS[cid]}" + (" (done)" if i < finished else "")
        y = write_lines(draw, y, label, 28, color, bold=now, x=PANEL_X + 28, width=PANEL_W - 28) + 8
    if returning:
        write_lines(draw, y + 8, "Returning to the ready pose", 28, BLUE, bold=True)


def read_frames(path):
    """Every frame of frames.mp4, scaled to fit the BOX x BOX square."""
    with av.open(str(path)) as container:
        frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
    scale = BOX / max(frames[0].size)
    size = (round(frames[0].width * scale), round(frames[0].height * scale))
    return [f.resize(size, Image.LANCZOS) for f in frames]


def make_frame(robot, timeline, i, fps):
    """One finished 1280 x 720 frame for output frame i."""
    # Frame i is the picture at i x 50 ms, i.e. after i steps (frames.mp4 may have one more frame than
    # there are steps: the picture before the first step). During the end hold, the last step's state stays.
    step = min(i, timeline["steps"])
    t = i / fps
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)
    x = BOX_X + (BOX - robot.width) // 2  # centre the robot video in its square
    y = BOX_Y + (BOX - robot.height) // 2
    img.paste(robot, (x, y))
    draw.rectangle([x - 1, y - 1, x + robot.width, y + robot.height], outline=LINES, width=1)
    draw_panel(draw, timeline, t, step, at_end=step >= timeline["steps"])
    draw.text((BOX_X, H - 14), FOOTER, font=font(18), fill=MUTED, anchor="ls")
    return img


def write_audio(timeline, seconds, path):
    """One wav file: silence, with each recorded line placed at its start time."""
    track = np.zeros(round(seconds * RATE), dtype=np.int32)
    for line in timeline["lines"]:
        if not line.get("file") or not Path(line["file"]).exists():
            continue  # a sentence given as text (no recording) adds no sound; its captions still show
        voice = np.frombuffer(Path(line["file"]).read_bytes(), dtype=np.int16)
        start = round(line["start_s"] * RATE)
        voice = voice[: max(0, len(track) - start)]  # cut anything past the end of the clip
        track[start:start + len(voice)] += voice
    track = np.clip(track, -32768, 32767).astype(np.int16)  # overlapping lines must not wrap around
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(track.tobytes())


def main(folder):
    folder = Path(folder)
    timeline = json.loads((folder / "timeline.json").read_text())
    fps = timeline["fps"]
    frames = read_frames(folder / "frames.mp4")
    total = len(frames) + round(HOLD_END_S * fps)  # the end hold repeats the last frame
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "voice.wav"
        write_audio(timeline, total / fps, audio)
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",  # frames in
            "-i", str(audio),  # the voice track
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",  # plays everywhere
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(folder / "clip.mp4"),
        ]
        ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(total):
            ffmpeg.stdin.write(make_frame(frames[min(i, len(frames) - 1)], timeline, i, fps).tobytes())
        ffmpeg.stdin.close()
        assert ffmpeg.wait() == 0, "ffmpeg failed"
    # Still picture: 1 s after the last line took effect (or the middle of the run if nothing was said).
    effects = [line["effect_s"] for line in timeline["lines"] if line["effect_s"] is not None]
    still = min(round((max(effects) + 1.0) * fps) if effects else len(frames) // 2, total - 1)
    make_frame(frames[min(still, len(frames) - 1)], timeline, still, fps).save(folder / "clip.png")
    print(f"Wrote {folder / 'clip.mp4'} ({total / fps:.1f} s) and {folder / 'clip.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
