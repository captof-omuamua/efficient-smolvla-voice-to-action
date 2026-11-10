"""Checks for the word rules. Needs no GPU. Run: python tests/test_rules.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from router import match_rules  # noqa: E402
from tasks import COMMANDS  # noqa: E402

# Every command in LIBERO's own wording must be answered by the rules.
for command_id, text in enumerate(COMMANDS):
    assert match_rules(text) == command_id, text

SAME = [
    ("Switch on the gas.", 7),
    ("kindly keep the bowl on the plate", 8),
    ("okay robot, put the bottle of wine on the cupboard", 2),
    ("open the center drawer of the cabinet", 0),
]
for text, command_id in SAME:
    assert match_rules(text) == command_id, text

# When the meaning could differ, the rules must step aside (None sends it to the language model).
STEP_ASIDE = [
    "turn off the stove",
    "close the middle drawer",
    "put the plate on the bowl",  # the thing named first is the thing being moved
    "put the plate on the stove",  # same things as command 5, but not "front"
    "take the bowl off the plate",
    "put the bowl on the wine rack",
    "open the top drawer",  # only part of command 3
    "open the bottom drawer",
    "make me a cup of coffee",
]
for text in STEP_ASIDE:
    assert match_rules(text) is None, text

print("all word-rule checks passed")
