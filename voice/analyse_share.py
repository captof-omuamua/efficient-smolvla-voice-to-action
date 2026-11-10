"""Analyse one shared-GPU run (PREREG.md, second half): join decisions to blocks, report the measures and the six bars.

Run: python analyse_share.py results/share/NAME [--calibration] [--min-rounds 3]
Writes NAME/summary.json and NAME/REPORT.md, and prints the report.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PERIOD_NS = 500_000_000  # one decision beat
SKIP_NS = 2_000_000_000  # the first 2 s of every block are dropped
LATE_NS = 50_000_000  # one control step: a decision ending later than this after its beat is late
GPU_KINDS = ("whisper", "llm")  # "rules" runs on the CPU, so it cannot collide with the policy on the GPU
ALL_REGIMES = ["idle", "whisper_flat", "llm_flat", "user_llm", "user_router",
               "user_llm_gate", "user_router_gate", "flat_gate"]
UNPROTECTED = ["whisper_flat", "llm_flat", "user_llm", "user_router"]
FIX_REGIMES = ["user_router_gate", "flat_gate"]
FLAT_REGIMES = ["whisper_flat", "llm_flat", "flat_gate"]
CAL_MEDIAN_MS, CAL_P99_MS = 13.97, 16.01  # the saved 8 Nov alone run c2_paced500
DEFAULT_BUDGET_MS = {"whisper": 100, "llm": 140}
BOOT_BLOCK, BOOT_REPS, BOOT_SEED = 20, 2000, 0
Z95 = 1.96  # two-sided 95%

# Copied from fast-and-on-time/ontime/throttle.py: why the GPU clock is below its maximum.
THROTTLE_BITS = {
    0x1: "gpu_idle",
    0x2: "app_clock_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}


# ---------- reading the run folder ----------

def read_csv(path):
    """Rows of a CSV file as dicts; a missing file gives no rows."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    """A JSON file as Python data; a missing or half-written file gives an empty dict."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def parse_mask(text):
    """A throttle bit mask written as '4' or '0x4'; empty means 0."""
    if not text:
        return 0
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def load_decisions(path):
    """decisions.csv as numpy columns. A row cut short by a crash is counted as bad, not used."""
    names = ("i", "beat", "start", "end", "ms", "sm_mhz", "throttle", "cpu", "pids")
    cols = {name: [] for name in names}
    bad = 0
    for row in read_csv(path):
        try:
            values = (int(row["i"]), int(row["beat_ns"]), int(row["start_ns"]), int(row["end_ns"]),
                      float(row["ms"]), float(row["sm_mhz"] or "nan"), parse_mask(row["throttle"]),
                      int(row["cpu"] or -1), [p for p in row["other_pids"].split(";") if p])
        except (KeyError, TypeError, ValueError, AttributeError):
            bad += 1
            continue
        for name, value in zip(names, values):
            cols[name].append(value)
    dec = {name: np.array(cols[name], dtype=np.int64) for name in ("i", "beat", "start", "end", "throttle", "cpu")}
    dec["ms"] = np.array(cols["ms"], dtype=float)
    dec["sm_mhz"] = np.array(cols["sm_mhz"], dtype=float)
    dec["pids"] = cols["pids"]  # a list of pid strings per decision (filled on every 20th decision)
    dec["bad_rows"] = bad
    return dec


def load_blocks(path):
    """blocks.csv as a list of dicts with numbers converted."""
    blocks = []
    for row in read_csv(path):
        try:
            blocks.append({"round": int(row["round"]), "slot": int(row["slot"]), "regime": row["regime"],
                           "start": int(row["start_ns"]), "end": int(row["end_ns"]), "errors": int(row["errors"] or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    return blocks


def load_jobs(path):
    """jobs.csv as a list of dicts with numbers converted."""
    jobs = []
    for row in read_csv(path):
        try:
            jobs.append({"round": int(row["round"]), "regime": row["regime"], "command": row["command"] or "",
                         "kind": row["kind"], "item": row["item"], "ready": int(row["ready_ns"]),
                         "start": int(row["start_ns"]), "end": int(row["end_ns"]),
                         "wait_ms": float(row["wait_ms"] or 0), "result": row["result"]})
        except (KeyError, TypeError, ValueError):
            continue
    return jobs


def load_phases(path):
    """phases.csv (alone_pre, alone_post) as a list of dicts."""
    phases = []
    for row in read_csv(path):
        try:
            phases.append({"phase": row["phase"], "start": int(row["start_ns"]), "end": int(row["end_ns"])})
        except (KeyError, TypeError, ValueError):
            continue
    return phases


# ---------- small statistics ----------

def in_window(dec, start_ns, end_ns, skip_ns=0):
    """Decisions whose beat is at least skip_ns after the window start and that end by the window end."""
    return (dec["beat"] >= start_ns + skip_ns) & (dec["end"] <= end_ns)


def time_stats(ms):
    """Count, median, p99 and worst of times in ms (None when there are none)."""
    if ms.size == 0:
        return {"n": 0, "median": None, "p99": None, "max": None}
    return {"n": int(ms.size), "median": float(np.median(ms)), "p99": float(np.percentile(ms, 99)),
            "max": float(ms.max())}


def spread_stats(values):
    """Mean, p99 and worst of a set of values (None when there are none)."""
    if values.size == 0:
        return {"mean": None, "p99": None, "max": None}
    return {"mean": float(values.mean()), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def wilson_upper(k, n, z=Z95):
    """Upper end of the Wilson 95% interval for k events out of n (0 of 1,200 gives about 0.32%)."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / (1 + z * z / n)


def late_stats(late):
    """How many decisions were late, their share, and the Wilson upper limit of that share."""
    k, n = int(late.sum()), int(late.size)
    return {"late": k, "late_share": k / n if n else None, "late_upper": wilson_upper(k, n)}


def resampled_p99(groups, rng, reps=BOOT_REPS, block=BOOT_BLOCK):
    """Moving-block bootstrap of the p99: blocks of 20 decisions drawn inside each 2-minute block, `reps` times.

    Slow calls come in runs, so whole runs of decisions are drawn, never mixing two 2-minute blocks.
    """
    pieces = []
    for x in groups:
        n = x.size
        if n == 0:
            continue
        size = min(block, n)
        n_blocks = math.ceil(n / size)
        starts = rng.integers(0, n - size + 1, size=(reps, n_blocks))
        index = (starts[:, :, None] + np.arange(size)).reshape(reps, -1)[:, :n]  # cut back to n decisions
        pieces.append(x[index])
    return np.percentile(np.concatenate(pieces, axis=1), 99, axis=1)


def p99_ratio(groups, idle_groups):
    """p99 of a regime divided by the p99 of idle, with a 95% bootstrap interval (seed 0)."""
    ms, idle_ms = np.concatenate(groups), np.concatenate(idle_groups)
    if ms.size == 0 or idle_ms.size == 0:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    replicates = resampled_p99(groups, rng) / resampled_p99(idle_groups, rng)
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {"ratio": float(np.percentile(ms, 99) / np.percentile(idle_ms, 99)), "low": float(low), "high": float(high)}


def overlaps_any(a_start, a_end, b_start, b_end):
    """For each interval a, True if it shares at least one instant with any interval b (ends included)."""
    hit = np.zeros(a_start.size, dtype=bool)
    if b_start.size == 0 or a_start.size == 0:
        return hit
    order = np.argsort(b_start)
    starts = b_start[order]
    latest_end = np.maximum.accumulate(b_end[order])  # latest end among the b intervals started so far
    last = np.searchsorted(starts, a_end, side="right") - 1  # the last b that starts before a ends
    has = last >= 0
    hit[has] = latest_end[last[has]] >= a_start[has]
    return hit


def skipped_beats(beats):
    """Beats with no decision between consecutive decisions (a slow call skips beats)."""
    if beats.size < 2:
        return 0
    return int((np.diff(np.sort(beats)) // PERIOD_NS - 1).sum())


def gpu_health(dec, mask):
    """Throttle-flag counts, mean GPU clock and the share of decisions on CPU cores 0 to 7."""
    masks = dec["throttle"][mask]
    flags = {name: int(((masks & bit) != 0).sum()) for bit, name in THROTTLE_BITS.items()}
    clocks = dec["sm_mhz"][mask]
    clocks = clocks[~np.isnan(clocks)]
    cpus = dec["cpu"][mask]
    known = cpus[cpus >= 0]
    return {"throttle": {name: count for name, count in flags.items() if count},
            "mean_sm_mhz": float(clocks.mean()) if clocks.size else None,
            "cpu0_7_share": float((known <= 7).mean()) if known.size else None}


# ---------- run structure: rounds, exclusions ----------

def expected_regimes(plan, blocks):
    """Regimes every round must have: all regimes named in plan.json, else every regime seen in blocks.csv."""
    if isinstance(plan, dict):
        rounds = plan.get("rounds", plan.get("order", list(plan.values())))
    else:
        rounds = plan
    names = set()
    for entry in rounds if isinstance(rounds, list) else []:
        if isinstance(entry, dict):
            entry = entry.get("regimes", [])
        if isinstance(entry, list):
            names.update(name for name in entry if isinstance(name, str))
    return names or {b["regime"] for b in blocks}


def loop_gap(block, dec):
    """Why the robot loop's log does not cover the whole block (None = it does).

    Only the log's first beat and last end matter: a slow call skips beats for a real reason, so no count is used.
    """
    if dec["beat"].size == 0:
        return "robot loop logged no decisions"
    if dec["beat"].min() > block["start"]:
        return "robot loop log starts after block start"
    if dec["end"].max() < block["end"]:
        return "robot loop log ends before block"
    return None


def complete_rounds(blocks, expected, dec):
    """Rounds that finished a block for every expected regime while the robot loop ran, and why the others are out."""
    seen, left_out = {}, {}
    for b in blocks:
        seen.setdefault(b["round"], set()).add(b["regime"])
        gap = loop_gap(b, dec)
        if gap:
            left_out.setdefault(b["round"], []).append(f"{gap} (round {b['round']}, {b['regime']})")
    for r, names in seen.items():
        missing = sorted(expected - names)
        if missing:
            left_out.setdefault(r, []).append(f"no block for {', '.join(missing)} (round {r})")
    return sorted(r for r in seen if r not in left_out), left_out


def drop_reasons(block, dec, allowed_pids):
    """Why a block is dropped (empty = kept): a voice error, or an unexpected GPU program seen during it."""
    reasons = []
    if block["errors"] > 0:
        reasons.append(f"{block['errors']} voice error(s)")
    during = np.flatnonzero((dec["start"] >= block["start"]) & (dec["start"] <= block["end"]))
    strangers = sorted({p for k in during for p in dec["pids"][k] if p not in allowed_pids})
    if strangers:
        reasons.append("unexpected GPU pid(s) " + ", ".join(strangers))
    return reasons


def ordered(names):
    """Regime names in PREREG order, unknown ones last."""
    known = [r for r in ALL_REGIMES if r in names]
    return known + sorted(set(names) - set(known))


# ---------- measures ----------

def policy_measures(dec, kept, gpu_jobs):
    """Policy timing per regime from decisions in kept blocks: stats, late share, p99 ratio, cause check."""
    by_regime = {}
    for b in kept:
        by_regime.setdefault(b["regime"], []).append(b)
    idle_groups = [dec["ms"][b["mask"]] for b in by_regime.get("idle", [])]
    job_start = np.array([j["start"] for j in gpu_jobs], dtype=np.int64)
    job_end = np.array([j["end"] for j in gpu_jobs], dtype=np.int64)
    out = {}
    for regime in ordered(by_regime):
        blocks = by_regime[regime]
        mask = np.zeros(dec["ms"].size, dtype=bool)
        for b in blocks:
            mask |= b["mask"]
        ms = dec["ms"][mask]
        late = (dec["end"][mask] - dec["beat"][mask]) > LATE_NS
        busy = overlaps_any(dec["start"][mask], dec["end"][mask], job_start, job_end)
        record = {"blocks": len(blocks), **time_stats(ms), **late_stats(late)}
        groups = [dec["ms"][b["mask"]] for b in blocks]
        record["ratio"] = p99_ratio(groups, idle_groups) if regime != "idle" and idle_groups else None
        record["voice_job_running"] = {**time_stats(ms[busy]), **late_stats(late[busy])}
        record["none_running"] = {**time_stats(ms[~busy]), **late_stats(late[~busy])}
        record["p99_by_round"] = {str(b["round"]): time_stats(dec["ms"][b["mask"]])["p99"] for b in blocks}
        record["skipped_beats"] = sum(skipped_beats(dec["beat"][b["mask"]]) for b in blocks)
        record.update(gpu_health(dec, mask))
        out[regime] = record
    return out


def command_measures(jobs):
    """Per spoken command: added wait (sum of wait_ms) and command time (last end minus first ready)."""
    commands = {}
    for j in jobs:
        if j["command"] != "":
            commands.setdefault((j["round"], j["command"]), []).append(j)
    if not commands:
        return None
    added = np.array([sum(j["wait_ms"] for j in group) for group in commands.values()])
    total = np.array([(max(j["end"] for j in group) - min(j["ready"] for j in group)) / 1e6
                      for group in commands.values()])
    return {"commands": len(commands), "added_wait": spread_stats(added), "command_time": time_stats(total)}


def after_first_2s(jobs, kept):
    """Voice jobs that arrive at least 2 s after their block starts.

    A spoken command is kept or dropped whole, by its arrival (the ready_ns of its first job);
    a flat-out job by its own ready_ns. Jobs must come from kept blocks.
    """
    cutoff = {(b["round"], b["regime"]): b["start"] + SKIP_NS for b in kept}
    arrival = {}  # (round, regime, command) -> earliest ready_ns of that command
    for j in jobs:
        key = (j["round"], j["regime"], j["command"])
        arrival[key] = min(arrival.get(key, j["ready"]), j["ready"])
    kept_jobs = []
    for j in jobs:
        ready = arrival[(j["round"], j["regime"], j["command"])] if j["command"] != "" else j["ready"]
        if ready >= cutoff[(j["round"], j["regime"])]:
            kept_jobs.append(j)
    return kept_jobs


def voice_measures(jobs, kept, budgets):
    """Voice job times, turn-taking waits, command times and flat-out job rates per regime (first 2 s left out)."""
    jobs = after_first_2s(jobs, kept)
    seconds = {}
    for b in kept:
        seconds[b["regime"]] = seconds.get(b["regime"], 0) + (b["end"] - b["start"] - SKIP_NS) / 1e9
    out = {}
    for regime in ordered(seconds):
        regime_jobs = [j for j in jobs if j["regime"] == regime]
        kinds = {}
        for kind in ("whisper", "llm", "rules"):
            kind_jobs = [j for j in regime_jobs if j["kind"] == kind]
            if not kind_jobs:
                continue
            times = np.array([(j["end"] - j["start"]) / 1e6 for j in kind_jobs])
            record = {**time_stats(times), "wait": spread_stats(np.array([j["wait_ms"] for j in kind_jobs]))}
            if kind in budgets:
                record["over_budget"] = int((times > budgets[kind]).sum())
            if regime in FLAT_REGIMES:
                record["per_s"] = len(kind_jobs) / seconds[regime]
            kinds[kind] = record
        out[regime] = {"kinds": kinds, "commands": command_measures(regime_jobs)}
    return out


def answer_check(jobs):
    """Clips or sentences whose answer differs between jobs (should be none).

    llm answers in user_router* pass the router's conflict check, so they are compared only with each other.
    """
    answers = {}
    for j in jobs:
        checked = j["kind"] == "llm" and j["regime"].startswith("user_router")
        answers.setdefault((j["kind"], j["item"], checked), set()).add(j["result"])
    mismatched = [f"{kind} {item}" for (kind, item, _), results in answers.items() if len(results) > 1]
    return {"items": len(answers), "mismatched": mismatched}


def gated_overlaps(jobs, dec, kept):
    """GPU voice jobs in gated regimes that overlap any policy call (every job and call, first 2 s included).

    Also counts the policy calls inside the gated blocks: with none, "0 overlaps" proves nothing.
    """
    gated = [j for j in jobs if j["regime"].endswith("_gate") and j["kind"] in GPU_KINDS]
    hit = overlaps_any(np.array([j["start"] for j in gated], dtype=np.int64),
                       np.array([j["end"] for j in gated], dtype=np.int64), dec["start"], dec["end"])
    regimes = sorted({j["regime"] for j in gated})
    calls = {r: int(sum(in_window(dec, b["start"], b["end"]).sum() for b in kept if b["regime"] == r))
             for r in regimes}
    return {"jobs": len(gated), "overlapping": int(hit.sum()), "regimes": regimes, "policy_calls": calls}


# ---------- bars ----------

def bar(number, name, verdict, numbers):
    """One row of the bar table."""
    return {"bar": number, "name": name, "verdict": verdict, "numbers": numbers}


def ratio_text(r):
    """A p99 ratio with its interval, e.g. '1.31x [1.12, 1.50]'."""
    return "–" if r is None else f"{r['ratio']:.2f}x [{r['low']:.2f}, {r['high']:.2f}]"


def no_decisions(policy, regimes):
    """Regimes that ran but kept no decisions: a bar that needs them cannot be judged."""
    return [r for r in regimes if r in policy and policy[r]["n"] == 0]


def bar_slowdown(policy):
    """Bar 2: p99 ratio >= 1.25 with its interval wholly above 1, in at least one unprotected regime."""
    present = [r for r in UNPROTECTED if r in policy and policy[r]["ratio"]]
    shown = [r for r in present if policy[r]["ratio"]["ratio"] >= 1.25 and policy[r]["ratio"]["low"] > 1]
    numbers = "; ".join(f"{r} {ratio_text(policy[r]['ratio'])}" for r in present)
    empty = no_decisions(policy, ["idle"] + UNPROTECTED)
    if empty and not shown:  # a regime we could not measure might have shown it
        return bar(2, "Sharing slows the policy", "not shown", f"no decisions in {', '.join(empty)}")
    if not present:
        return bar(2, "Sharing slows the policy", "not applicable", "needs idle and an unprotected regime")
    return bar(2, "Sharing slows the policy", "PASS" if shown else "FAIL", numbers)


def bar_fix_speed(policy):
    """Bar 3: in user_router_gate and flat_gate, p99 no more than 10% above idle."""
    empty = no_decisions(policy, ["idle"] + FIX_REGIMES)
    failed = [r for r in FIX_REGIMES if r in policy and policy[r]["ratio"] and policy[r]["ratio"]["ratio"] > 1.10]
    if empty:  # a regime we could measure may still fail the bar outright
        numbers = f"no decisions in {', '.join(empty)}" + "".join(
            f"; {r} {policy[r]['ratio']['ratio']:.2f}x (max 1.10x)" for r in failed)
        return bar(3, "Fix, speed", "FAIL" if failed else "not shown", numbers)
    if not all(r in policy and policy[r]["ratio"] for r in FIX_REGIMES):
        return bar(3, "Fix, speed", "not applicable", "needs idle, user_router_gate and flat_gate")
    ok = all(policy[r]["ratio"]["ratio"] <= 1.10 for r in FIX_REGIMES)
    numbers = "; ".join(f"{r} p99 {policy[r]['p99']:.2f} ms vs idle {policy['idle']['p99']:.2f} ms "
                        f"= {policy[r]['ratio']['ratio']:.2f}x (max 1.10x)" for r in FIX_REGIMES)
    return bar(3, "Fix, speed", "PASS" if ok else "FAIL", numbers)


def bar_fix_deadline(policy):
    """Bar 4: in user_router_gate and flat_gate, 0 late decisions."""
    if not all(r in policy for r in FIX_REGIMES):
        return bar(4, "Fix, deadline", "not applicable", "needs user_router_gate and flat_gate")
    ok = all(policy[r]["late"] == 0 for r in FIX_REGIMES)
    numbers = "; ".join(f"{r} {policy[r]['late']} late of {policy[r]['n']} "
                        f"(upper limit {fmt_pct(policy[r]['late_upper'])})" for r in FIX_REGIMES)
    if ok and no_decisions(policy, FIX_REGIMES):  # never PASS on "0 late of 0"
        return bar(4, "Fix, deadline", "not shown", numbers)
    return bar(4, "Fix, deadline", "PASS" if ok else "FAIL", numbers)


def bar_fix_cost(voice):
    """Bar 5: in user_router_gate, added wait per command: mean <= 60 ms and worst <= 200 ms."""
    commands = voice.get("user_router_gate", {}).get("commands")
    if not commands:
        return bar(5, "Fix, cost", "not applicable", "needs commands in user_router_gate")
    wait = commands["added_wait"]
    ok = wait["mean"] <= 60 and wait["max"] <= 200
    numbers = (f"{commands['commands']} commands: mean {wait['mean']:.1f} ms (max 60 ms), "
               f"worst {wait['max']:.1f} ms (max 200 ms)")
    return bar(5, "Fix, cost", "PASS" if ok else "FAIL", numbers)


def bar_turn_taking(gated):
    """Bar 6: 0 voice jobs in gated regimes overlap a policy call."""
    if not gated["regimes"]:
        return bar(6, "Turn-taking kept", "not applicable", "needs a gated regime")
    empty = [r for r, n in gated["policy_calls"].items() if n == 0]
    if empty and gated["overlapping"] == 0:  # no policy call to collide with, so 0 overlaps proves nothing
        return bar(6, "Turn-taking kept", "not shown", f"no policy calls in {', '.join(empty)}")
    ok = gated["overlapping"] == 0
    calls = ", ".join(f"{r} {n}" for r, n in gated["policy_calls"].items())
    numbers = f"{gated['overlapping']} of {gated['jobs']} GPU jobs overlap a policy call (policy calls: {calls})"
    return bar(6, "Turn-taking kept", "PASS" if ok else "FAIL", numbers)


def judge_bars(policy, voice, gated, n_rounds, min_rounds):
    """Bars 2 to 6 for a shared run (bar 1 belongs to the calibrate run)."""
    bars = [bar(1, "Timer check, alone", "not shown", "measured by the calibrate run (--calibration)")]
    if n_rounds < min_rounds:
        reason = f"only {n_rounds} complete round(s), need {min_rounds}"
        names = ["Sharing slows the policy", "Fix, speed", "Fix, deadline", "Fix, cost", "Turn-taking kept"]
        return bars + [bar(k, name, "not shown", reason) for k, name in enumerate(names, start=2)]
    return bars + [bar_slowdown(policy), bar_fix_speed(policy), bar_fix_deadline(policy),
                   bar_fix_cost(voice), bar_turn_taking(gated)]


def judge_calibration(stats):
    """Bar 1: median within 10% of 13.97 ms and p99 within 15% of 16.01 ms."""
    if stats["n"] == 0:
        return bar(1, "Timer check, alone", "not shown", "no decisions")
    ok = abs(stats["median"] / CAL_MEDIAN_MS - 1) <= 0.10 and abs(stats["p99"] / CAL_P99_MS - 1) <= 0.15
    numbers = (f"median {stats['median']:.2f} ms (allowed {0.9 * CAL_MEDIAN_MS:.2f} to {1.1 * CAL_MEDIAN_MS:.2f} ms), "
               f"p99 {stats['p99']:.2f} ms (allowed {0.85 * CAL_P99_MS:.2f} to {1.15 * CAL_P99_MS:.2f} ms)")
    return bar(1, "Timer check, alone", "PASS" if ok else "FAIL", numbers)


# ---------- the two analyses ----------

def known_pids(run_dir):
    """Pids of our own two programs, as strings (any other GPU program is unexpected)."""
    pids = {read_json(run_dir / name).get("pid") for name in ("policy_ready.json", "voice_ready.json")}
    return {str(p) for p in pids if p is not None}


def unexpected_pids(dec, allowed):
    """Every unexpected GPU pid seen anywhere in the run, with how many samples saw it."""
    counts = {}
    for pids in dec["pids"]:
        for p in pids:
            if p not in allowed:
                counts[p] = counts.get(p, 0) + 1
    return counts


def cpu_cores(run_dir, programs=("policy", "voice")):
    """The CPU cores each program was allowed to use, from its *_ready.json ("not recorded" in older runs)."""
    return {p: read_json(run_dir / f"{p}_ready.json").get("cpu_cores", "not recorded") for p in programs}


def cores_text(cores):
    """CPU cores as text: '8 to 23 (16 cores)' when there are no gaps, else listed one by one; or 'not recorded'."""
    if not isinstance(cores, list) or not cores:
        return "not recorded"
    if cores == list(range(cores[0], cores[-1] + 1)):
        return f"{cores[0]} to {cores[-1]} ({len(cores)} cores)"
    return ", ".join(str(c) for c in cores)


def calibrate(run_dir):
    """Robot loop alone: bar 1 only, from every decision in decisions.csv."""
    dec = load_decisions(run_dir / "decisions.csv")
    everything = np.ones(dec["ms"].size, dtype=bool)
    stats = time_stats(dec["ms"])
    late = (dec["end"] - dec["beat"]) > LATE_NS
    return {"mode": "calibration", "run_dir": str(run_dir), "bad_rows": dec["bad_rows"],
            "cpu_cores": cpu_cores(run_dir, ("policy",)),
            "alone": {**stats, **late_stats(late), "skipped_beats": skipped_beats(dec["beat"]),
                      **gpu_health(dec, everything)},
            "first_call_ms": read_json(run_dir / "policy_ready.json").get("first_call_ms"),
            "unexpected_pids": unexpected_pids(dec, known_pids(run_dir)),
            "bars": [judge_calibration(stats)]}


def analyse(run_dir, min_rounds):
    """Shared run: join, exclusions, every PREREG measure and bars 2 to 6."""
    dec = load_decisions(run_dir / "decisions.csv")
    blocks = load_blocks(run_dir / "blocks.csv")
    jobs = load_jobs(run_dir / "jobs.csv")
    phases = load_phases(run_dir / "phases.csv")
    allowed = known_pids(run_dir)

    expected = expected_regimes(read_json(run_dir / "plan.json"), blocks)
    rounds, left_out = complete_rounds(blocks, expected, dec)
    for b in blocks:
        b["mask"] = in_window(dec, b["start"], b["end"], SKIP_NS)
        b["reasons"] = drop_reasons(b, dec, allowed) if b["round"] in rounds else ["incomplete round"]
    kept = [b for b in blocks if not b["reasons"]]
    kept_keys = {(b["round"], b["regime"]) for b in kept}
    kept_jobs = [j for j in jobs if (j["round"], j["regime"]) in kept_keys]

    in_block = np.zeros(dec["ms"].size, dtype=bool)
    in_kept = np.zeros(dec["ms"].size, dtype=bool)
    for b in blocks:
        in_block |= b["mask"]
    for b in kept:
        in_kept |= b["mask"]
    phase_stats, in_phase = {}, np.zeros(dec["ms"].size, dtype=bool)
    for p in phases:
        mask = in_window(dec, p["start"], p["end"])
        in_phase |= mask
        late = (dec["end"][mask] - dec["beat"][mask]) > LATE_NS
        phase_stats[p["phase"]] = {**time_stats(dec["ms"][mask]), **late_stats(late), **gpu_health(dec, mask)}

    gpu_jobs = [j for j in jobs if j["kind"] in GPU_KINDS]  # every GPU job counts for the cause check
    settings = read_json(run_dir / "voice_ready.json").get("settings", {})
    budgets = {kind: settings.get(f"{kind}_budget_ms", ms) for kind, ms in DEFAULT_BUDGET_MS.items()}
    policy = policy_measures(dec, kept, gpu_jobs)
    voice = voice_measures(kept_jobs, kept, budgets)
    gated = gated_overlaps(kept_jobs, dec, kept)
    return {
        "mode": "shared", "run_dir": str(run_dir), "bad_rows": dec["bad_rows"], "cpu_cores": cpu_cores(run_dir),
        "expected_regimes": ordered(expected), "complete_rounds": rounds,
        "rounds_left_out": {str(r): reasons for r, reasons in sorted(left_out.items())},
        "all_rounds": sorted({b["round"] for b in blocks}), "min_rounds": min_rounds,
        "decisions": {"total": int(dec["ms"].size), "kept_blocks": int(in_kept.sum()),
                      "dropped_or_incomplete_blocks": int((in_block & ~in_kept).sum()),
                      "phases": int(in_phase.sum()), "unassigned": int((~in_block & ~in_phase).sum())},
        "dropped_blocks": [{"round": b["round"], "regime": b["regime"], "reasons": b["reasons"]}
                           for b in blocks if b["reasons"] and b["round"] in rounds],
        "unexpected_pids": unexpected_pids(dec, allowed),
        "skipped_beats_total": read_json(run_dir / "policy_summary.json").get("skipped_beats",
                                                                               skipped_beats(dec["beat"])),
        "budgets_ms": budgets, "phases": phase_stats, "policy": policy, "voice": voice,
        "answers": answer_check(kept_jobs), "gated": gated,
        "bars": judge_bars(policy, voice, gated, len(rounds), min_rounds),
    }


# ---------- the report ----------

def fmt_ms(x):
    """A time in ms, or a dash when missing."""
    return "–" if x is None else f"{x:.2f} ms"


def fmt_pct(x):
    """A share as a percentage, or a dash when missing."""
    return "–" if x is None else f"{100 * x:.2f}%"


def table(headers, rows):
    """A Markdown table as lines of text."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def flags_text(flags):
    """Throttle flags as 'name count, ...', or 'none'."""
    return ", ".join(f"{name} {count}" for name, count in flags.items()) or "none"


def bar_lines(bars):
    """The bar table."""
    return table(["#", "Bar", "Verdict", "Numbers"], [[b["bar"], b["name"], b["verdict"], b["numbers"]] for b in bars])


def calibration_report(s):
    """Report text for a calibrate run."""
    a = s["alone"]
    lines = [f"# Calibration: robot loop alone ({s['run_dir']})", "", "## Bar 1", ""] + bar_lines(s["bars"])
    lines += ["", "## Decisions", ""]
    lines += table(["Decisions", "Median", "p99", "Worst", "Late (> 50 ms)", "Late upper limit (95%)",
                    "Skipped beats", "First call"],
                   [[a["n"], fmt_ms(a["median"]), fmt_ms(a["p99"]), fmt_ms(a["max"]), a["late"],
                     fmt_pct(a["late_upper"]), a["skipped_beats"], fmt_ms(s["first_call_ms"])]])
    clock = "–" if a["mean_sm_mhz"] is None else f"{a['mean_sm_mhz']:.0f} MHz"
    lines += ["", f"- Throttle flags (decisions): {flags_text(a['throttle'])}",
              f"- Mean GPU clock: {clock}; share on CPU cores 0 to 7: {fmt_pct(a['cpu0_7_share'])}",
              f"- CPU cores allowed, robot loop: {cores_text(s['cpu_cores']['policy'])}",
              f"- Unexpected GPU pids (samples): {s['unexpected_pids'] or 'none'}",
              f"- Unreadable decision rows: {s['bad_rows']}"]
    return "\n".join(lines) + "\n"


def policy_lines(policy):
    """Policy tables: timing and late share, cause check, p99 per round, GPU health."""
    lines = ["## Policy decisions per regime", ""]
    lines += table(["Regime", "Decisions", "Median", "p99", "Worst", "Late", "Late upper (95%)", "p99 vs idle [95%]"],
                   [[r, p["n"], fmt_ms(p["median"]), fmt_ms(p["p99"]), fmt_ms(p["max"]), p["late"],
                     fmt_pct(p["late_upper"]), ratio_text(p["ratio"])] for r, p in policy.items()])
    lines += ["", "## Cause check: was a voice job (Whisper or LLM) running during the decision?", ""]
    rows = []
    for r, p in policy.items():
        for label, part in (("yes", p["voice_job_running"]), ("no", p["none_running"])):
            rows.append([r, label, part["n"], fmt_ms(part["median"]), fmt_ms(part["p99"]), fmt_ms(part["max"]),
                         part["late"]])
    lines += table(["Regime", "Voice job running", "Decisions", "Median", "p99", "Worst", "Late"], rows)
    rounds = sorted({k for p in policy.values() for k in p["p99_by_round"]}, key=int)
    lines += ["", "## p99 per round", ""]
    lines += table(["Regime"] + [f"Round {k}" for k in rounds],
                   [[r] + [fmt_ms(p["p99_by_round"].get(k)) for k in rounds] for r, p in policy.items()])
    lines += ["", "## GPU and CPU health per regime", ""]
    lines += table(["Regime", "Throttle flags (decisions)", "Mean GPU clock", "On CPU 0 to 7", "Skipped beats"],
                   [[r, flags_text(p["throttle"]),
                     "–" if p["mean_sm_mhz"] is None else f"{p['mean_sm_mhz']:.0f} MHz",
                     fmt_pct(p["cpu0_7_share"]), p["skipped_beats"]] for r, p in policy.items()])
    return lines


def voice_lines(voice, budgets):
    """Voice tables: job times and waits, commands, flat-out rates."""
    lines = ["## Voice jobs per regime (jobs and commands arriving in a block's first 2 s left out)", ""]
    rows = []
    for r, v in voice.items():
        for kind, k in v["kinds"].items():
            budget = f"{k['over_budget']} (> {budgets[kind]} ms)" if "over_budget" in k else "–"
            rows.append([r, kind, k["n"], fmt_ms(k["median"]), fmt_ms(k["p99"]), fmt_ms(k["max"]), budget,
                         fmt_ms(k["wait"]["mean"]), fmt_ms(k["wait"]["p99"]), fmt_ms(k["wait"]["max"])])
    lines += table(["Regime", "Job", "Jobs", "Median", "p99", "Worst", "Over budget", "Wait mean", "Wait p99",
                    "Wait worst"], rows)
    lines += ["", "## Spoken commands (arrival to routed answer)", ""]
    rows = []
    for r, v in voice.items():
        c = v["commands"]
        if c:
            rows.append([r, c["commands"], fmt_ms(c["added_wait"]["mean"]), fmt_ms(c["added_wait"]["p99"]),
                         fmt_ms(c["added_wait"]["max"]), fmt_ms(c["command_time"]["median"]),
                         fmt_ms(c["command_time"]["p99"]), fmt_ms(c["command_time"]["max"])])
    lines += table(["Regime", "Commands", "Added wait mean", "Added wait p99", "Added wait worst", "Time median",
                    "Time p99", "Time worst"], rows)
    lines += ["", "## Flat-out rates", ""]
    rows = [[r, kind, f"{k['per_s']:.2f} jobs/s"] for r, v in voice.items() for kind, k in v["kinds"].items()
            if "per_s" in k]
    lines += table(["Regime", "Job", "Rate"], rows)
    return lines


def shared_report(s):
    """Report text for a shared run."""
    d = s["decisions"]
    lines = [f"# One GPU, three models: {s['run_dir']}", "",
             "Bars apply to the Tuned run; for the Original contrast they are information only (PREREG).", "",
             "## Bars", ""] + bar_lines(s["bars"])
    lines += ["", "## Run", "",
              f"- Complete rounds: {len(s['complete_rounds'])} {s['complete_rounds']} of rounds seen "
              f"{s['all_rounds']} (need {s['min_rounds']}); regimes per round: {', '.join(s['expected_regimes'])}"]
    left_out = [text for reasons in s["rounds_left_out"].values() for text in reasons]
    lines += [f"- Rounds left out: {len(s['rounds_left_out'])}"] + [f"  - {text}" for text in left_out]
    lines += [f"- CPU cores allowed: robot loop {cores_text(s['cpu_cores']['policy'])}; "
              f"voice program {cores_text(s['cpu_cores']['voice'])}",
              f"- Decisions: {d['total']} total; {d['kept_blocks']} in kept blocks; "
              f"{d['dropped_or_incomplete_blocks']} in dropped or incomplete blocks (not used); "
              f"{d['phases']} in alone phases; {d['unassigned']} unassigned (not used); "
              f"{s['bad_rows']} unreadable rows",
              f"- Skipped beats, whole run: {s['skipped_beats_total']}",
              f"- Unexpected GPU pids (samples): {s['unexpected_pids'] or 'none'}"]
    dropped = [f"round {b['round']} {b['regime']}: {'; '.join(b['reasons'])}" for b in s["dropped_blocks"]]
    lines += [f"- Dropped blocks: {len(dropped)}"] + [f"  - {text}" for text in dropped]
    lines += ["", "## Robot loop alone (phases)", ""]
    lines += table(["Phase", "Decisions", "Median", "p99", "Worst", "Late", "Throttle flags"],
                   [[name, p["n"], fmt_ms(p["median"]), fmt_ms(p["p99"]), fmt_ms(p["max"]), p["late"],
                     flags_text(p["throttle"])] for name, p in s["phases"].items()])
    lines += [""] + policy_lines(s["policy"]) + [""] + voice_lines(s["voice"], s["budgets_ms"])
    a = s["answers"]
    lines += ["", "## Sanity", "",
              f"- Same clip or sentence, same answer in every regime: {len(a['mismatched'])} of {a['items']} "
              f"items differ" + (f" ({', '.join(a['mismatched'][:10])})" if a["mismatched"] else "")]
    return "\n".join(lines) + "\n"


def main():
    """Analyse a run folder, write summary.json and REPORT.md, print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--calibration", action="store_true", help="robot loop alone: bar 1 only")
    parser.add_argument("--min-rounds", type=int, default=3)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if args.calibration:
        summary = calibrate(run_dir)
        report = calibration_report(summary)
    else:
        summary = analyse(run_dir, args.min_rounds)
        report = shared_report(summary)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda v: v.item()) + "\n")
    (run_dir / "REPORT.md").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
