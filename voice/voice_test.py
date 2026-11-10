"""Your voice test: read 5 lines aloud and see what the system hears and picks.

Your recordings stay in data/voice_test/ on this laptop and go nowhere else.
"""
import json
import os
from pathlib import Path

import numpy as np

from router import Router
from speech import Transcriber, record_until_enter
from tasks import COMMANDS

ROOT = Path(__file__).parent
LINES = [
    ("Turn on the stove.", 7),
    ("Put the wine bottle on top of the cabinet.", 2),
    ("Kindly keep the bowl on the plate.", 8),
    ("Push the plate to the front of the stove.", 5),
    ("Close the top drawer.", None),  # the robot can't do this, so it should refuse
]

print("Loading the models, about 20 seconds...")
transcriber, router = Transcriber(), Router()
transcriber(np.zeros(16000, dtype=np.float32))  # first call is slow, so it isn't timed
folder = ROOT / "data" / "voice_test"
folder.mkdir(parents=True, exist_ok=True)

results = []
for i, (line, expected) in enumerate(LINES):
    print(f'\nLine {i + 1} of {len(LINES)}: "{line}"')
    input("  Press Enter, then read it aloud. ")
    audio = record_until_enter(folder / f"line_{i + 1}.raw")
    heard, speech_ms = transcriber(audio)
    decision = router.route(heard)
    picked = "refuse" if decision.command_id is None else COMMANDS[decision.command_id]
    correct = decision.command_id == expected
    print(f"  Heard:  {heard}")
    print(f"  Picked: {picked} ({decision.level}, {decision.ms:.1f} ms; speech took {speech_ms:.0f} ms)")
    print("  RIGHT" if correct else "  WRONG")
    results.append({"line": line, "heard": heard, "picked": picked, "level": decision.level, "right": correct,
                    "seconds_of_audio": len(audio) / 16000, "speech_ms": speech_ms, "router_ms": decision.ms})

OUT = Path(os.environ.get("OUT_DIR", ROOT.parent / "runs"))  # new outputs; the committed one is results/voice_test.json
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "voice_test.json").write_text(json.dumps(results, indent=1))
print(f"\nRight: {sum(r['right'] for r in results)} of {len(results)}. Saved to {OUT / 'voice_test.json'}")
