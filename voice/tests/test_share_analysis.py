"""Checks for the shared-GPU analysis on a tiny fake run. Needs no GPU. Run: python tests/test_share_analysis.py"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import analyse_share as a  # noqa: E402

MS = 1_000_000  # ns per ms
BLOCK_NS = 20_000_000_000  # 20 s blocks keep the fake run small
T0 = 1000 * a.PERIOD_NS  # the clock starts on a beat
POLICY_PID, VOICE_PID = 111, 222
DECISION_HEADER = "i,beat_ns,start_ns,end_ns,ms,sm_mhz,mem_mhz,temp_c,power_mw,throttle,cpu,cpu_khz,other_pids"
JOB_HEADER = "round,regime,command,kind,item,ready_ns,start_ns,end_ns,wait_ms,result"


def write_csv(path, header, rows):
    """Write a header line and rows to a CSV file."""
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        csv.writer(f).writerows(rows)


def decision_row(i, beat, call_ms, pids=""):
    """One decisions.csv row: the call starts 1 ms after its beat and lasts call_ms."""
    start = beat + 1 * MS
    end = start + int(call_ms * MS)
    return [i, beat, start, end, call_ms, 2500, 12000, 60, 90000, "0x4", 3, 4000000, pids]


def build_run(folder, late_in_flat_gate=False, stranger=False, early_overlap=False):
    """A fake run: rounds 1 and 2 complete, round 3 only idle. llm_flat is slow when an LLM job overlaps.

    The robot loop also runs alone for 4 beats before the first block and after the last one, as in a real run.
    """
    plan = [["idle", "llm_flat", "user_router_gate", "flat_gate"]] * 3  # plan.json's real shape: a list per round
    order = [(1, r) for r in plan[0]] + [(2, r) for r in plan[1]] + [(3, "idle")]
    run_end = T0 + len(order) * BLOCK_NS
    alone_pre = [T0 - k * a.PERIOD_NS for k in (4, 3, 2, 1)]
    decisions = [decision_row(i, beat, 14.0) for i, beat in enumerate(alone_pre)]
    blocks, jobs = [], []
    for slot, (rnd, regime) in enumerate(order):
        start = T0 + slot * BLOCK_NS
        blocks.append([rnd, slot, regime, start, start + BLOCK_NS, 0])
        for k, beat in enumerate(range(start + a.PERIOD_NS, start + BLOCK_NS, a.PERIOD_NS)):
            call_ms = 14.0 + 0.01 * (k % 7)
            if regime == "llm_flat" and k % 2 == 0:  # an LLM job runs over every other decision
                call_ms = 30.0 + 0.01 * (k % 7)
                jobs.append([rnd, regime, "", "llm", f"s{k % 5}", beat, beat, beat + 100 * MS, 0, str(k % 5)])
            if regime == "user_router_gate" and k % 6 == 0:  # a command: whisper (waits 20 ms), then rules
                ready = beat + 5 * MS
                jobs.append([rnd, regime, k, "whisper", f"c{k % 3}", ready, beat + 25 * MS, beat + 90 * MS, 20,
                             f"h{k % 3}"])
                jobs.append([rnd, regime, k, "rules", f"s{k % 5}", beat + 90 * MS, beat + 90 * MS, beat + 91 * MS, 0,
                             str(k % 5)])
            if regime == "user_router_gate" and k == 2:  # arrives at 1.9 s, its second job is ready at 2.09 s
                jobs.append([rnd, regime, k, "whisper", f"c{k % 3}", beat + 400 * MS, beat + 525 * MS, beat + 590 * MS,
                             300, f"h{k % 3}"])
                jobs.append([rnd, regime, k, "rules", f"s{k % 5}", beat + 590 * MS, beat + 590 * MS, beat + 591 * MS,
                             0, str(k % 5)])
            if regime == "flat_gate":
                kind = "whisper" if k % 2 else "llm"
                item = f"c{k % 3}" if kind == "whisper" else f"s{k % 5}"
                result = f"h{k % 3}" if kind == "whisper" else str(k % 5)
                job_start = beat if early_overlap and rnd == 1 and k == 1 else beat + 25 * MS
                jobs.append([rnd, regime, "", kind, item, beat, job_start, beat + 150 * MS, 24, result])
                if late_in_flat_gate and rnd == 2 and k == 10:
                    call_ms = 60.0
            pids = ""
            if k % 20 == 0:
                pids = f"{VOICE_PID}" + (";999" if stranger and rnd == 2 and regime == "idle" else "")
            decisions.append(decision_row(len(decisions), beat, call_ms, pids))
    for k in (1, 2, 3, 4):
        decisions.append(decision_row(len(decisions), run_end + k * a.PERIOD_NS, 14.0))
    write_csv(folder / "decisions.csv", DECISION_HEADER, decisions)
    write_csv(folder / "blocks.csv", "round,slot,regime,start_ns,end_ns,errors", blocks)
    write_csv(folder / "jobs.csv", JOB_HEADER, jobs)
    write_csv(folder / "phases.csv", "phase,start_ns,end_ns",
              [["alone_pre", alone_pre[0], T0], ["alone_post", run_end, run_end + 5 * a.PERIOD_NS]])
    (folder / "plan.json").write_text(json.dumps(plan))
    (folder / "policy_ready.json").write_text(json.dumps({"pid": POLICY_PID, "first_call_ms": 40000.0,
                                                          "cpu_cores": list(range(8, 24))}))
    (folder / "voice_ready.json").write_text(json.dumps({"pid": VOICE_PID, "settings": {}}))  # an older run: no cores
    return folder


def keep_decisions(folder, keep):
    """Rewrite decisions.csv with only the rows whose beat passes keep(beat_ns)."""
    rows = list(csv.reader(open(folder / "decisions.csv")))
    write_csv(folder / "decisions.csv", ",".join(rows[0]), [r for r in rows[1:] if keep(int(r[1]))])


def block_of(slot):
    """Start and end ns of the block in this slot of the fake run."""
    return T0 + slot * BLOCK_NS, T0 + (slot + 1) * BLOCK_NS


def verdicts(summary):
    """Bar number -> verdict."""
    return {b["bar"]: b["verdict"] for b in summary["bars"]}


def commands_voice(added_waits):
    """A voice summary whose user_router_gate commands had these added waits (one whisper job each)."""
    jobs = [{"round": 1, "regime": "user_router_gate", "command": str(n), "ready": 0, "end": 100 * MS, "wait_ms": w}
            for n, w in enumerate(added_waits)]
    return {"user_router_gate": {"commands": a.command_measures(jobs)}}


def ratio_policy(ratio, low, high):
    """A policy summary with one unprotected regime and this p99 ratio against idle."""
    return {"idle": {"n": 100, "ratio": None},
            "llm_flat": {"n": 100, "ratio": {"ratio": ratio, "low": low, "high": high}}}


# The join: beat at least 2 s after the block start, call finished by the block end (edges included).
dec = {"beat": np.array([0, 2000, 1999, 2000, 2000]) * MS, "end": np.array([10, 2010, 2009, 5000, 5001]) * MS}
assert list(a.in_window(dec, 0, 5000 * MS, a.SKIP_NS)) == [False, True, False, True, False]
assert list(a.in_window(dec, 0, 5000 * MS)) == [True, True, True, True, False]  # phases: no 2 s rule

# The overlap test: touching ends count, a gap does not, a long earlier job still counts.
hit = a.overlaps_any(np.array([10, 30, 100]), np.array([20, 40, 110]), np.array([0, 41, 50]), np.array([10, 45, 105]))
assert list(hit) == [True, False, True]
assert not a.overlaps_any(np.array([1]), np.array([2]), np.array([], dtype=np.int64), np.array([], dtype=np.int64))[0]

# Wilson upper limit for 0 late out of 1,200 is about 0.32%.
assert abs(a.wilson_upper(0, 1200) - 0.0032) < 0.0001
assert a.wilson_upper(0, 0) is None

# Complete rounds: every planned regime, and the robot loop's log covers each block (first beat to last end).
blocks = [{"round": 1, "regime": "idle", "start": 0, "end": 10}, {"round": 1, "regime": "flat_gate", "start": 10,
                                                                    "end": 20},
          {"round": 2, "regime": "idle", "start": 20, "end": 30}]
assert a.complete_rounds(blocks, {"idle", "flat_gate"}, {"beat": np.array([0]), "end": np.array([30])})[0] == [1]
rounds, left_out = a.complete_rounds(blocks, {"idle", "flat_gate"}, {"beat": np.array([0]), "end": np.array([15])})
assert rounds == [] and left_out[1] == ["robot loop log ends before block (round 1, flat_gate)"], left_out
assert a.complete_rounds(blocks, {"idle"}, {"beat": np.array([1]), "end": np.array([30])})[0] == [2]
assert a.complete_rounds(blocks, {"idle"}, {"beat": np.array([], dtype=np.int64),
                                            "end": np.array([], dtype=np.int64)})[0] == []

# plan.json in its real shape: one list of regime names per round.
assert a.expected_regimes([["idle", "flat_gate"], ["flat_gate", "idle"]], []) == {"idle", "flat_gate"}

with tempfile.TemporaryDirectory() as tmp:
    run = build_run(Path(tmp))
    s = a.analyse(run, min_rounds=2)
    assert s["expected_regimes"] == ["idle", "llm_flat", "user_router_gate", "flat_gate"]  # read from the list
    assert s["complete_rounds"] == [1, 2]
    assert s["rounds_left_out"] == {"3": ["no block for flat_gate, llm_flat, user_router_gate (round 3)"]}
    assert s["policy"]["idle"]["blocks"] == 2  # round 3's idle block is not used
    assert s["policy"]["idle"]["n"] == 2 * 36  # 39 beats per 20 s block, minus the 3 in the first 2 s
    assert s["phases"]["alone_pre"]["n"] == 4 and s["phases"]["alone_post"]["n"] == 4
    # Cause check: in llm_flat the slow decisions are exactly those with an LLM job running.
    busy, calm = s["policy"]["llm_flat"]["voice_job_running"], s["policy"]["llm_flat"]["none_running"]
    assert busy["median"] > 29 and calm["median"] < 15, (busy, calm)
    assert busy["n"] + calm["n"] == s["policy"]["llm_flat"]["n"]
    assert verdicts(s) == {1: "not shown", 2: "PASS", 3: "PASS", 4: "PASS", 5: "PASS", 6: "PASS"}, s["bars"]
    # The 2 s rule for voice: 7 commands per block, 1 arrives at 0.5 s and 1 at 1.9 s (its rules job is ready at
    # 2.09 s, but the command is dropped whole); that one waited 300 ms, so the worst kept wait stays 20 ms.
    commands = s["voice"]["user_router_gate"]["commands"]
    assert commands["commands"] == 2 * 6 and commands["added_wait"]["max"] == 20, commands
    assert s["voice"]["user_router_gate"]["kinds"]["rules"]["n"] == 2 * 6
    # Flat-out jobs: 39 per block, the 3 ready before 2 s left out; rate = kept jobs / (20 s - 2 s).
    whisper = s["voice"]["flat_gate"]["kinds"]["whisper"]
    assert whisper["n"] == 2 * 18 and abs(whisper["per_s"] - 1.0) < 1e-9, whisper
    # Bar 6 still uses every gated GPU job: 39 flat_gate jobs and 8 whisper jobs per block, first 2 s included.
    assert s["gated"]["jobs"] == 2 * 39 + 2 * 8 and s["gated"]["overlapping"] == 0, s["gated"]
    assert s["gated"]["policy_calls"] == {"flat_gate": 2 * 39, "user_router_gate": 2 * 39}, s["gated"]
    assert s["answers"]["mismatched"] == []
    # CPU cores: recorded for the robot loop, missing in this (older-style) voice file.
    assert s["cpu_cores"] == {"policy": list(range(8, 24)), "voice": "not recorded"}
    assert "robot loop 8 to 23 (16 cores); voice program not recorded" in a.shared_report(s)
    assert verdicts(a.analyse(run, min_rounds=3))[3] == "not shown"  # too few complete rounds

# Bar 6 counts policy calls in a block's first 2 s too.
with tempfile.TemporaryDirectory() as tmp:
    s = a.analyse(build_run(Path(tmp), early_overlap=True), min_rounds=2)
    assert verdicts(s)[6] == "FAIL" and s["gated"]["overlapping"] == 1, s["gated"]

# The robot loop stops part-way through round 2: rounds 2 and 3 are left out, so too few rounds to judge.
with tempfile.TemporaryDirectory() as tmp:
    run = build_run(Path(tmp))
    stop_ns = block_of(7)[0] + 10_000_000_000  # 10 s into round 2's flat_gate block
    keep_decisions(run, lambda beat: beat < stop_ns)
    s = a.analyse(run, min_rounds=2)
    assert s["complete_rounds"] == [1], s["rounds_left_out"]
    assert s["rounds_left_out"]["2"] == ["robot loop log ends before block (round 2, flat_gate)"]
    assert "robot loop log ends before block (round 3, idle)" in s["rounds_left_out"]["3"]
    assert all(verdicts(s)[k] == "not shown" for k in range(2, 7)), s["bars"]
    assert "robot loop log ends before block (round 2, flat_gate)" in a.shared_report(s)

# A slow policy skips beats inside a block: the round stays complete (no decision count is used).
with tempfile.TemporaryDirectory() as tmp:
    run = build_run(Path(tmp))
    start, _ = block_of(1)  # round 1 llm_flat
    keep_decisions(run, lambda beat: not start + 5_000_000_000 <= beat < start + 8_000_000_000)
    s = a.analyse(run, min_rounds=2)
    assert s["complete_rounds"] == [1, 2] and s["policy"]["llm_flat"]["skipped_beats"] == 6, s["policy"]["llm_flat"]

# A regime with zero decisions (every beat of both flat_gate blocks skipped): no crash, its bars not shown.
with tempfile.TemporaryDirectory() as tmp:
    run = build_run(Path(tmp))
    flat_gate_blocks = [block_of(3), block_of(7)]
    keep_decisions(run, lambda beat: not any(lo <= beat < hi for lo, hi in flat_gate_blocks))
    s = a.analyse(run, min_rounds=2)
    assert s["complete_rounds"] == [1, 2] and s["policy"]["flat_gate"]["n"] == 0
    assert verdicts(s) == {1: "not shown", 2: "PASS", 3: "not shown", 4: "not shown", 5: "PASS",
                           6: "not shown"}, s["bars"]
    assert "flat_gate" in a.shared_report(s)

# Bar 2 needs a ratio of at least 1.25 AND an interval wholly above 1.
assert a.bar_slowdown(ratio_policy(1.30, 1.05, 1.60))["verdict"] == "PASS"
assert a.bar_slowdown(ratio_policy(1.20, 1.10, 1.30))["verdict"] == "FAIL"  # ratio under 1.25
assert a.bar_slowdown(ratio_policy(1.30, 0.98, 1.60))["verdict"] == "FAIL"  # interval includes 1
assert a.bar_slowdown(ratio_policy(1.30, 1.00, 1.60))["verdict"] == "FAIL"  # touching 1 is not wholly above

# Bar 5: added wait per command, mean <= 60 ms and worst <= 200 ms, each checked on its own.
assert a.bar_fix_cost(commands_voice([60] * 10))["verdict"] == "PASS"  # mean exactly at its limit passes
assert a.bar_fix_cost(commands_voice([0] * 9 + [200]))["verdict"] == "PASS"  # worst exactly at its limit passes
assert a.bar_fix_cost(commands_voice([70] * 10))["verdict"] == "FAIL"  # mean 70 ms, worst only 70 ms
assert a.bar_fix_cost(commands_voice([0] * 9 + [210]))["verdict"] == "FAIL"  # worst 210 ms, mean only 21 ms

# Answer check: the same clip or sentence with two answers is flagged; checked and unchecked llm answers are apart.
jobs = [{"kind": "whisper", "item": "c01_0", "regime": "whisper_flat", "result": "aa"},
        {"kind": "whisper", "item": "c01_0", "regime": "flat_gate", "result": "bb"},
        {"kind": "llm", "item": "s7", "regime": "llm_flat", "result": "3"},
        {"kind": "llm", "item": "s7", "regime": "user_router", "result": "refuse"},
        {"kind": "llm", "item": "s8", "regime": "user_llm", "result": "4"},
        {"kind": "llm", "item": "s8", "regime": "user_llm_gate", "result": "5"}]
assert sorted(a.answer_check(jobs)["mismatched"]) == ["llm s8", "whisper c01_0"]

with tempfile.TemporaryDirectory() as tmp:
    s = a.analyse(build_run(Path(tmp), late_in_flat_gate=True, stranger=True), min_rounds=2)
    assert verdicts(s)[4] == "FAIL"  # one decision 61 ms after its beat
    assert verdicts(s)[6] == "FAIL"  # and that long call overlaps a gated job
    assert s["dropped_blocks"] == [{"round": 2, "regime": "idle", "reasons": ["unexpected GPU pid(s) 999"]}]
    assert s["policy"]["idle"]["blocks"] == 1

# Missing regimes give "not applicable", not a crash (smoke tests, the Original contrast).
with tempfile.TemporaryDirectory() as tmp:
    run = build_run(Path(tmp))
    (run / "plan.json").write_text(json.dumps({"rounds": [["idle", "llm_flat"]] * 3}))  # the older dict shape
    rows = list(csv.reader(open(run / "blocks.csv")))
    write_csv(run / "blocks.csv", ",".join(rows[0]), [r for r in rows[1:] if r[2] in ("idle", "llm_flat")])
    assert verdicts(a.analyse(run, min_rounds=2)) == {1: "not shown", 2: "PASS", 3: "not applicable",
                                                      4: "not applicable", 5: "not applicable", 6: "not applicable"}

# Calibration needs only decisions.csv and the policy JSON files.
with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    rows = [decision_row(i, T0 + i * a.PERIOD_NS, 16.0 if i % 60 == 0 else 14.0) for i in range(600)]
    write_csv(folder / "decisions.csv", DECISION_HEADER, rows)
    (folder / "policy_ready.json").write_text(json.dumps({"pid": POLICY_PID, "first_call_ms": 40000.0}))
    c = a.calibrate(folder)
    assert c["alone"]["n"] == 600 and c["bars"][0]["verdict"] == "PASS", c["bars"]
    assert "PASS" in a.calibration_report(c)
    assert c["cpu_cores"] == {"policy": "not recorded"} and "robot loop: not recorded" in a.calibration_report(c)
    (folder / "policy_ready.json").write_text(json.dumps({"pid": POLICY_PID, "cpu_cores": [0, 2, 3]}))
    assert "robot loop: 0, 2, 3" in a.calibration_report(a.calibrate(folder))

print("all share-analysis checks passed")
