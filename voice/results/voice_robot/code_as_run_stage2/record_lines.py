"""Record all the spoken lines for the voice-robot runs and show what Whisper hears.

Stage 1 lines go to data/voice_test/line_1.raw ... line_5.raw (where the robot loop expects them);
stage 2 lines go to data/voice_lines/<name>.raw. All are 16 kHz, mono, 16-bit raw, as in speech.py.
An older recording is never overwritten: it is renamed to <name>_attempt1.raw, _attempt2.raw, ...
first, because every attempt is reported. Each folder has a lines.json with what was heard.

Only Whisper is loaded (no router, no language model). Each clip is also trimmed (audio_trim.py)
and transcribed again, because the robot runs use the trimmed clip; a hint says when a redo may help.

  python record_lines.py                      # all 9 lines: stage 1, then stage 2
  python record_lines.py --set stage2         # just one group
  python record_lines.py --only stop          # just one line
  python record_lines.py --line "Go." --name go   # a custom line (saved with the stage 2 lines)
  python record_lines.py --from-file clip.raw # use a clip instead of the microphone (for testing)
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from audio_trim import trim_silence
from speech import RATE, Transcriber, load_raw, record_until_enter

ROOT = Path(__file__).parent
STAGE1 = ROOT / "data" / "voice_test"  # same names and texts as voice_test.py
STAGE2 = ROOT / "data" / "voice_lines"
SETS = {
    "stage1": {
        "line_1": "Turn on the stove.",
        "line_2": "Put the wine bottle on top of the cabinet.",
        "line_3": "Kindly keep the bowl on the plate.",
        "line_4": "Push the plate to the front of the stove.",
        "line_5": "Close the top drawer.",
    },
    "stage2": {
        "stop": "Stop.",
        "continue": "Continue.",
        "two_step_1": "Turn on the stove, then put the wine bottle on top of the cabinet.",
        "two_step_2": "Open the middle drawer, then turn on the stove.",
    },
}
FOLDER = {"stage1": STAGE1, "stage2": STAGE2}
# Every built-in line: name -> (text, folder)
LINES = {name: (text, FOLDER[group]) for group, lines in SETS.items() for name, text in lines.items()}


def keep_old_attempt(path):
    """If a recording already exists, rename it to <name>_attemptN.raw so it is kept."""
    if not path.exists():
        return None
    n = 1
    while (old := path.with_name(f"{path.stem}_attempt{n}.raw")).exists():
        n += 1
    path.rename(old)
    return old


def record_one(transcriber, name, text, folder, from_file):
    """Record one line, transcribe it, and log it in the folder's lines.json."""
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "lines.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    path = folder / f"{name}.raw"
    old = keep_old_attempt(path)
    if old:
        print(f"  (earlier recording kept as {old.name})")
        if name in log:  # keep its entry too, under the new file name
            log[old.stem] = {**log.pop(name), "file": str(old)}
    if from_file:
        shutil.copyfile(from_file, path)  # test mode: a ready-made clip stands in for the microphone
        audio = load_raw(path)
    else:
        input("  Press Enter, then read it aloud. ")
        audio = record_until_enter(path)
    heard, speech_ms = transcriber(audio)
    # The robot runs cut the silence first, so check what Whisper hears on the trimmed clip too.
    trimmed, _, flag = trim_silence(audio)
    heard_trimmed, _ = transcriber(trimmed)
    seconds, trimmed_seconds = len(audio) / RATE, len(trimmed) / RATE
    print(f"  Audio: {seconds:.2f} s recorded, {trimmed_seconds:.2f} s after trimming (Whisper took {speech_ms:.0f} ms)")
    print(f"  Heard (as recorded): {heard}")
    print(f"  Heard (trimmed, as the robot will hear it): {heard_trimmed}")
    problems = []
    if trimmed_seconds > 0.9 * seconds:
        problems.append("almost no silence was trimmed")
    if flag:
        problems.append(f"trimming says: {flag}")
    if words(heard_trimmed) != words(text):
        problems.append("the trimmed clip was not heard word for word")
    if problems:
        print(f"  HINT: {'; '.join(problems)}. You may want to redo it: --only {name}")
    log[name] = {"text": text, "file": str(path), "heard": heard, "seconds_of_audio": round(seconds, 3),
                 "heard_trimmed": heard_trimmed, "seconds_trimmed": round(trimmed_seconds, 3),
                 "trim_flag": flag, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    log_path.write_text(json.dumps(log, indent=1))  # save after every line, so nothing is lost


def words(text):
    """Lowercase and drop punctuation, so "Stop." and "stop" count as the same."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("-", " ")).split()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS, help="record just one group of lines")
    parser.add_argument("--only", choices=LINES, help="record just this one line")
    parser.add_argument("--line", help="text of a custom line (needs --name)")
    parser.add_argument("--name", help="file name for the custom line")
    parser.add_argument("--from-file", help="take this raw clip instead of the microphone (for testing)")
    args = parser.parse_args()

    # Work out which lines to record: name -> (text, folder).
    if args.line or args.name:
        if not (args.line and args.name) or args.name in LINES:
            parser.error("--line and --name go together, and the name must not be a built-in line")
        lines = {args.name: (args.line, STAGE2)}
    elif args.only:
        lines = {args.only: LINES[args.only]}
    elif args.set:
        lines = {name: LINES[name] for name in SETS[args.set]}
    else:
        lines = LINES  # all nine, stage 1 first

    print("Loading Whisper, about 10 seconds...")
    transcriber = Transcriber()
    transcriber(np.zeros(RATE, dtype=np.float32))  # first call is slow, so warm it up once
    print("\nTip: start speaking right after you press Enter, and press Enter again as soon as you finish.")
    print("Silence at the ends is trimmed later, so small pauses are fine.")

    for i, (name, (text, folder)) in enumerate(lines.items()):
        print(f'\nLine {i + 1} of {len(lines)} ({name}): "{text}"')
        record_one(transcriber, name, text, folder, args.from_file)

    print(f"\nSaved in {STAGE1} and {STAGE2}. What was heard is in lines.json in each folder.")


if __name__ == "__main__":
    main()
