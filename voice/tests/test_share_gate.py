"""Checks for the turn-taking rule. Needs no GPU. Run: python tests/test_share_gate.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from share_gate import MS, PERIOD_NS, wait_for_window, wait_ns  # noqa: E402

GUARD_AFTER = 25 * MS
GUARD_BEFORE = 10 * MS
BUDGET = 100 * MS
LATEST = PERIOD_NS - GUARD_BEFORE - BUDGET  # last phase a 100 ms job may start at: 390 ms
BEAT = 7_123 * PERIOD_NS  # a beat far from zero, like a real monotonic clock reading


def wait(phase_ns, budget_ns=BUDGET):
    """Wait for a job asked for phase_ns after a beat."""
    return wait_ns(BEAT + phase_ns, budget_ns, GUARD_AFTER, GUARD_BEFORE, PERIOD_NS)


# Inside the window: start at once.
assert wait(100 * MS) == 0
assert wait(300 * MS) == 0

# Too soon after a beat: wait until guard_after has passed.
assert wait(0) == 25 * MS
assert wait(10 * MS) == 15 * MS

# The budget would cross into the guard before the next beat: wait for next beat plus guard_after.
assert wait(450 * MS) == 50 * MS + 25 * MS
assert wait(499 * MS) == 1 * MS + 25 * MS

# Exact edges, to the nanosecond.
assert wait(GUARD_AFTER - 1) == 1
assert wait(GUARD_AFTER) == 0
assert wait(LATEST) == 0
assert wait(LATEST + 1) == PERIOD_NS - LATEST - 1 + GUARD_AFTER

# After waiting, the job always lands inside the window.
for phase in range(0, PERIOD_NS, MS):
    start = BEAT + phase + wait(phase)
    assert GUARD_AFTER <= start % PERIOD_NS <= LATEST, phase

# Guards plus budget longer than one period cannot work.
for budget in (466 * MS, 600 * MS):
    try:
        wait_ns(BEAT, budget, GUARD_AFTER, GUARD_BEFORE, PERIOD_NS)
    except ValueError:
        pass
    else:
        raise AssertionError(f"no ValueError for budget {budget}")
assert wait_ns(BEAT + GUARD_AFTER, 465 * MS, GUARD_AFTER, GUARD_BEFORE, PERIOD_NS) == 0  # fits exactly

# Worst wait = guard_after + budget + guard_before (just after the last allowed start; 1 ns short of it).
for budget_ms, worst_ms in ((100, 135), (140, 175)):
    budget = budget_ms * MS
    latest = PERIOD_NS - GUARD_BEFORE - budget
    phases = list(range(0, PERIOD_NS, MS)) + [latest, latest + 1]
    worst = max(wait(phase, budget) for phase in phases)
    assert worst == worst_ms * MS - 1, (budget_ms, worst)

# One real-clock check: after wait_for_window returns, the job is inside the window (390 ms latest start,
# plus 2 ms for the few lines that run after the last clock reading), and it never waited more than one period.
waited_ns = wait_for_window(100, 25)
phase = time.monotonic_ns() % PERIOD_NS
assert 25 * MS <= phase <= 392 * MS, phase
assert 0 <= waited_ns / 1e9 <= 0.51, waited_ns

print("all turn-taking checks passed")
