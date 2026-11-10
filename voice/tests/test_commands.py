"""Black-box tests for commands.py, written from VOICE_ROBOT_PLAN.md ("commands.py (fixed interface)").

A fake router stands in for the real one, so no GPU is needed. Run from the voice-robot folder:
  python -m pytest -q tests/test_commands.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from commands import Event, Session, parse  # noqa: E402
from router import clean  # noqa: E402
from tasks import COMMANDS  # noqa: E402


@dataclass
class FakeDecision:
    command_id: int | None  # None means refuse
    level: str
    ms: float


class FakeRouter:
    """Answers from a small dict of cleaned sentences; anything else is refused. Remembers what it was asked."""

    def __init__(self, answers):
        self.answers = {clean(text): command for text, command in answers.items()}
        self.asked = []

    def route(self, text):
        self.asked.append(text)
        return FakeDecision(self.answers.get(clean(text)), "llm", 1.5)


# The sentences the fake router knows (all LIBERO's own wording, plus one extra).
KNOWN = {**{text: i for i, text in enumerate(COMMANDS)}, "open the middle drawer": 0}


def fake():
    return FakeRouter(KNOWN)


# ---------- rule 1: stop ----------

@pytest.mark.parametrize("text", ["Stop.", "stop", "STOP!", "Halt!", "Freeze.",
                                  "Hey robot, please stop right now.", "Okay STOP, that is wrong!"])
def test_stop_words(text):
    router = fake()
    event = parse(text, router)
    assert event.kind == "stop"
    assert event.commands == []
    assert event.level == "rule"
    assert router.asked == []  # no language model for a stop


def test_stop_beats_a_command_in_the_same_sentence():
    assert parse("Turn on the stove, no wait, stop.", fake()).kind == "stop"


def test_stop_must_be_a_whole_word():
    # "stopwatch" is not the word stop, so this is not a stop.
    assert parse("put the stopwatch on the plate", fake()).kind != "stop"


# ---------- rule 2: continue ----------

@pytest.mark.parametrize("text", ["Continue.", "continue", "Go on.", "Carry on!", "Resume.", "Keep going."])
def test_continue_phrases(text):
    router = fake()
    event = parse(text, router)
    assert event.kind == "continue"
    assert event.commands == []
    assert event.level == "rule"
    assert router.asked == []


def test_continue_must_be_exact():
    # Extra words: not a continue; it goes on to the router, which refuses it.
    event = parse("Continue to the stove.", fake())
    assert event.kind == "refuse"


# ---------- rule 4: one command or a refusal ----------

def test_one_command():
    event = parse("Turn on the stove.", fake())
    assert event.kind == "plan"
    assert event.commands == [7]
    assert event.level == "llm"  # the router's own level is passed on
    assert isinstance(event.ms, float) and event.ms >= 0


def test_refusal():
    event = parse("Close the top drawer.", fake())
    assert event.kind == "refuse"
    assert event.commands == []


# ---------- rule 3: two commands ----------

@pytest.mark.parametrize("joiner", [", then", " and then", " and", ", after that,"])
def test_two_commands(joiner):
    text = f"Turn on the stove{joiner} put the wine bottle on top of the cabinet."
    event = parse(text, fake())
    assert event.kind == "plan"
    assert event.commands == [7, 2]


def test_two_commands_keep_their_order():
    event = parse("Open the middle drawer, then turn on the stove.", fake())
    assert event.commands == [0, 7]


def test_two_halves_with_the_same_command_fall_back_to_one():
    # Both halves give 7, so rule 3 does not apply; the whole sentence (known here as 7) decides.
    router = FakeRouter({**KNOWN, "turn on the stove then turn on the stove": 7})
    event = parse("Turn on the stove, then turn on the stove.", router)
    assert event.kind == "plan"
    assert event.commands == [7]


def test_one_half_refused_falls_back_to_one():
    # "close the top drawer" is refused, so the whole sentence goes to the router (which refuses it).
    event = parse("Turn on the stove, then close the top drawer.", fake())
    assert event.kind == "refuse"


def test_one_half_refused_whole_sentence_known():
    router = FakeRouter({**KNOWN, "turn on the stove then close the top drawer": 7})
    event = parse("Turn on the stove, then close the top drawer.", router)
    assert event.kind == "plan"
    assert event.commands == [7]


def test_trained_command_with_and_stays_one():
    # Command 3 has "and" in it. Its halves are refused, so the whole sentence decides: command 3.
    event = parse("Open the top drawer and put the bowl inside.", fake())
    assert event.kind == "plan"
    assert event.commands == [3]


# ---------- Session ----------

def plan(*ids):
    return Event("plan", list(ids), "rule", 0.1)


STOP = Event("stop", [], "rule", 0.1)
CONTINUE = Event("continue", [], "rule", 0.1)
REFUSE = Event("refuse", [], "llm", 0.1)


def test_session_starts_idle():
    session = Session()
    assert session.current is None
    assert session.queue == []
    assert session.holding is False
    assert session.done == []
    assert session.instruction() is None


def test_plan_sets_the_current_command():
    session = Session()
    session.on_event(plan(7))
    assert session.current == 7
    assert session.instruction() == COMMANDS[7]


def test_plan_replaces_everything():
    session = Session()
    session.on_event(plan(7, 2))
    session.on_event(STOP)
    session.on_event(plan(0))
    assert session.current == 0
    assert session.queue == []
    assert session.holding is False
    assert session.instruction() == COMMANDS[0]


def test_stop_holds_and_continue_restores():
    session = Session()
    session.on_event(plan(7))
    session.on_event(STOP)
    assert session.holding is True
    assert session.instruction() is None
    assert session.current == 7  # the command is kept, only paused
    session.on_event(CONTINUE)
    assert session.holding is False
    assert session.instruction() == COMMANDS[7]


def test_success_walks_through_a_two_command_plan():
    session = Session()
    session.on_event(plan(7, 2))
    assert (session.current, session.queue) == (7, [2])
    session.on_success()
    assert (session.current, session.queue, session.done) == (2, [], [7])
    assert session.instruction() == COMMANDS[2]
    session.on_success()
    assert (session.current, session.queue, session.done) == (None, [], [7, 2])
    assert session.instruction() is None


def test_refuse_changes_nothing():
    session = Session()
    session.on_event(plan(7, 2))
    session.on_event(REFUSE)
    assert (session.current, session.queue, session.holding, session.done) == (7, [2], False, [])
    session.on_event(STOP)
    session.on_event(REFUSE)
    assert session.holding is True
    assert session.instruction() is None
