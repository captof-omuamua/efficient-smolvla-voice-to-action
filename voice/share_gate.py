"""Turn-taking for voice jobs: start only in the gap between the robot policy's beats.

Beats sit at whole multiples of PERIOD_NS on time.monotonic_ns(), the clock every program shares.
A job may start only if it starts at least guard_after after a beat and its time budget ends
at least guard_before before the next beat. Otherwise it waits for the next beat plus guard_after.
"""
import time

PERIOD_NS = 500_000_000  # the policy decides every 500 ms
MS = 1_000_000  # nanoseconds in one millisecond


def wait_ns(now_ns, budget_ns, guard_after_ns, guard_before_ns, period_ns=PERIOD_NS):
    """How many ns a job must wait before it may start (0 = start now)."""
    if guard_after_ns + budget_ns + guard_before_ns > period_ns:
        raise ValueError("guard_after + budget + guard_before is longer than one period")
    phase = now_ns % period_ns  # time since the last beat
    latest_start = period_ns - guard_before_ns - budget_ns  # last phase that still ends in time
    if guard_after_ns <= phase <= latest_start:
        return 0
    if phase < guard_after_ns:
        return guard_after_ns - phase  # the policy may still be running: wait out the guard
    return period_ns - phase + guard_after_ns  # too late in this gap: next beat plus the guard


def wait_for_window(budget_ms, guard_after_ms, guard_before_ms=10):
    """Sleep until a job with this budget may start. Returns the ns spent waiting."""
    budget_ns, after_ns, before_ns = round(budget_ms * MS), round(guard_after_ms * MS), round(guard_before_ms * MS)
    first_ns = now_ns = time.monotonic_ns()
    while True:
        to_wait = wait_ns(now_ns, budget_ns, after_ns, before_ns)
        if to_wait == 0:
            return now_ns - first_ns
        time.sleep(to_wait / 1e9)
        now_ns = time.monotonic_ns()  # check again, because sleep can wake a little late
