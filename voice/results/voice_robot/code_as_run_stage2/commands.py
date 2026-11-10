"""Turn a heard sentence into an event, and keep the robot's to-do list.

parse() rules, in this order (VOICE_ROBOT_PLAN.md):
  1. stop:     the sentence has the whole word stop, halt or freeze
  2. continue: the sentence is exactly continue, go on, carry on, resume or keep going
  3. two commands: split at the first joining word; both halves give different commands
  4. one command:  the router on the whole sentence (a command, or a refusal)
"""
import re
import time
from dataclasses import dataclass, field

from router import clean
from tasks import COMMANDS

STOP_WORDS = r"\b(stop|halt|freeze)\b"
CONTINUE_SENTENCES = {"continue", "go on", "carry on", "resume", "keep going"}
# Longer joining words first, so "and then" is found before "and" at the same place.
JOIN_WORDS = r"\b(and then|after that|then|and)\b"


@dataclass
class Event:
    kind: str  # "plan", "stop", "continue" or "refuse"
    commands: list[int] = field(default_factory=list)  # for "plan": one or two command ids
    level: str = "rule"  # "rule" or "llm": which level decided
    ms: float = 0.0  # time taken to decide


def parse(text, router):
    """Turn one heard sentence into an Event. router is anything with route(text)."""
    start = time.perf_counter()
    sentence = clean(text)

    # Rule 1: stop. No language model; a false stop is safer than a missed one.
    if re.search(STOP_WORDS, sentence):
        return Event("stop", [], "rule", (time.perf_counter() - start) * 1000)

    # Rule 2: continue, only when the whole sentence is one of the phrases.
    if sentence in CONTINUE_SENTENCES:
        return Event("continue", [], "rule", (time.perf_counter() - start) * 1000)

    # Rule 3: two commands, split at the first joining word.
    join = re.search(JOIN_WORDS, sentence)
    if join:
        first = router.route(sentence[: join.start()])
        second = router.route(sentence[join.end():])
        both = first.command_id is not None and second.command_id is not None
        if both and first.command_id != second.command_id:
            level = "llm" if "llm" in (first.level, second.level) else "rule"
            ms = (time.perf_counter() - start) * 1000
            return Event("plan", [first.command_id, second.command_id], level, ms)

    # Rule 4: one command, or a refusal. The router gets the sentence as heard, as it was tested.
    decision = router.route(text)
    ms = (time.perf_counter() - start) * 1000
    if decision.command_id is None:
        return Event("refuse", [], decision.level, ms)
    return Event("plan", [decision.command_id], decision.level, ms)


class Session:
    """The robot's to-do list: what it is doing now, what is next, and whether it is holding."""

    def __init__(self):
        self.queue = []  # commands still to do after the current one
        self.current = None  # command being done now
        self.holding = False  # True after "stop", until "continue" or a new plan
        self.done = []  # commands finished, in order

    def on_event(self, event):
        if event.kind == "plan":  # a new plan replaces everything
            self.current = event.commands[0]
            self.queue = list(event.commands[1:])
            self.holding = False
        elif event.kind == "stop":
            self.holding = True
        elif event.kind == "continue":
            self.holding = False
        # "refuse" changes nothing

    def on_success(self):
        """The current command reached its goal: mark it done and take the next one."""
        self.done.append(self.current)
        self.current = self.queue.pop(0) if self.queue else None

    def instruction(self):
        """LIBERO's sentence for the current command; None when idle or holding."""
        if self.current is None or self.holding:
            return None
        return COMMANDS[self.current]
