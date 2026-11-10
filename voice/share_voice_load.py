"""The voice program that shares the GPU with the robot policy: Whisper and the language model in 8 regimes.

Each round runs every regime once, as one block, in a seeded order. Every job is timed on
time.monotonic_ns(), the clock the robot loop also uses, so the two logs can be joined later.
Sentences come from a scored test set: only their ids are ever logged, never their text.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

# One CPU thread for the maths libraries; this must be set before they are imported. With the default
# 24 threads, Whisper's sound-to-features step took 5 ms or 90 ms at random (PREREG.md, amendment).
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import torch  # noqa: E402

from router import RULES, Router, clean, match_rules, named_things
from share_gate import wait_for_window
from speech import Transcriber, load_raw

ROOT = Path(__file__).parent
REGIMES = ["idle", "whisper_flat", "llm_flat", "user_llm", "user_router", "user_llm_gate", "user_router_gate",
           "flat_gate"]
BLOCK_COLUMNS = ["round", "slot", "regime", "start_ns", "end_ns", "errors"]
JOB_COLUMNS = ["round", "regime", "command", "kind", "item", "ready_ns", "start_ns", "end_ns", "wait_ms", "result"]
WARM_CALLS = 5


class CsvLog:
    """A CSV file with a header row; every row is flushed as soon as it is written."""

    def __init__(self, path, columns):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)
        self.file.flush()

    def write(self, row):
        """Write one row and flush it to disk."""
        self.writer.writerow(row)
        self.file.flush()


class VoiceLoad:
    """The two models, the clips and sentences, and the logs for one run."""

    def __init__(self, args, clips, sentences):
        self.args = args
        self.clips = clips  # list of (clip name, audio)
        self.sentences = sentences  # list of (sentence id, text)
        self.transcriber = Transcriber()
        self.router = Router()
        self.jobs = CsvLog(args.out / "jobs.csv", JOB_COLUMNS)
        self.round = 0
        self.regime = ""

    def warm_up(self):
        """Call each model a few times, so first-call costs do not land inside a block."""
        for i in range(WARM_CALLS):
            self.transcriber(self.clips[i % len(self.clips)][1])
            self.router.llm.choose(self.sentences[i % len(self.sentences)][1])
            torch.cuda.synchronize()

    def run_job(self, kind, item, payload, ready_ns, command=""):
        """Run one job (after waiting for its turn in gated regimes), log it, and return its end time."""
        gated = self.regime.endswith("_gate") and kind != "rules"  # rules use no GPU, so never wait
        budget = self.args.whisper_budget_ms if kind == "whisper" else self.args.llm_budget_ms
        waited_ns = wait_for_window(budget, self.args.guard_after_ms) if gated else 0
        start_ns = time.monotonic_ns()
        result = self.compute(kind, payload)
        end_ns = time.monotonic_ns()
        self.jobs.write([self.round, self.regime, command, kind, item, ready_ns, start_ns, end_ns,
                         f"{waited_ns / 1e6:.3f}", result])
        return end_ns, result

    def compute(self, kind, payload):
        """Do the work of one job and return what goes in the result column."""
        if kind == "whisper":
            text, _ = self.transcriber(payload)  # Transcriber synchronizes the GPU before it returns
            torch.cuda.synchronize()
            return hashlib.sha1(text.encode()).hexdigest()[:8]
        if kind == "llm":
            command = self.router.llm.choose(payload)
            torch.cuda.synchronize()
            if self.regime.startswith("user_router"):
                command = conflict_check(payload, command)
            return as_result(command)
        return as_result(match_rules(payload))  # kind == "rules"

    def run_block(self, end_by_ns, rng):
        """Run the current regime until end_by_ns. Returns the number of errors caught."""
        if self.regime == "idle":
            time.sleep(max(0, end_by_ns - time.monotonic_ns()) / 1e9)
            return 0
        if self.regime.startswith("user_"):
            return self.run_user(end_by_ns, rng)
        return self.run_flat(end_by_ns)

    def run_flat(self, end_by_ns):
        """Run jobs back to back until end_by_ns, cycling through the clips and sentences."""
        kinds = {"whisper_flat": ["whisper"], "llm_flat": ["llm"], "flat_gate": ["whisper", "llm"]}[self.regime]
        errors, count, ready_ns = 0, 0, time.monotonic_ns()
        while ready_ns < end_by_ns:
            kind = kinds[count % len(kinds)]  # flat_gate alternates whisper and llm
            turn = count // len(kinds)  # how many times this kind has run, to cycle its list
            name, payload = self.clips[turn % len(self.clips)] if kind == "whisper" else \
                self.sentences[turn % len(self.sentences)]
            try:
                ready_ns, _ = self.run_job(kind, name, payload, ready_ns)
            except Exception as error:
                errors += 1
                report(error)
                ready_ns = time.monotonic_ns()
            count += 1
        return errors

    def run_user(self, end_by_ns, rng):
        """Start a spoken command at each seeded arrival time until end_by_ns."""
        errors, number = 0, 0
        arrival_ns = time.monotonic_ns() + gap_ns(rng)
        while arrival_ns < end_by_ns:
            clip_name, audio = rng.choice(self.clips)
            sentence_id, text = rng.choice(self.sentences)
            time.sleep(max(0, arrival_ns - time.monotonic_ns()) / 1e9)
            try:  # the command's clock starts at its arrival, even if the one before it ran over
                self.run_command(number, clip_name, audio, sentence_id, text, arrival_ns)
            except Exception as error:
                errors += 1
                report(error)
            number += 1
            arrival_ns += gap_ns(rng)
        return errors

    def run_command(self, number, clip_name, audio, sentence_id, text, ready_ns):
        """One spoken command: Whisper on the clip, then route the sentence (word rules first in user_router*)."""
        ready_ns, _ = self.run_job("whisper", clip_name, audio, ready_ns, number)
        if self.regime.startswith("user_router"):
            ready_ns, result = self.run_job("rules", sentence_id, text, ready_ns, number)
            if result != "refuse":
                return  # the word rules answered, so the language model is skipped
        self.run_job("llm", sentence_id, text, ready_ns, number)


def conflict_check(text, command):
    """Refuse if the model picked a command that leaves out something the person named (as Router.route does)."""
    if command is None:
        return None
    allowed = next(things for c, things, _, _ in RULES if c == command)
    return None if named_things(clean(text)) - allowed else command


def as_result(command):
    """A command id as text, or 'refuse' for None."""
    return "refuse" if command is None else str(command)


def gap_ns(rng):
    """Time to the next spoken command: uniform 2 to 4 s."""
    return round(rng.uniform(2.0, 4.0) * 1e9)


def report(error):
    """Write a caught error to stderr (type and message only; sentence text is never part of it)."""
    print(f"error: {type(error).__name__}: {error}", file=sys.stderr, flush=True)


def load_clips():
    """The 20 flite clips made by make_share_audio.sh, as (name, audio), sorted by name."""
    paths = sorted((ROOT / "data" / "share_audio").glob("c*_*.raw"))
    if len(paths) != 20:
        raise SystemExit(f"expected 20 clips in data/share_audio, found {len(paths)}: run make_share_audio.sh")
    return [(path.stem, load_raw(path)) for path in paths]


def load_sentences():
    """All 130 test sentences as (id, text): the 100 in scope, then the 30 out of scope."""
    data = json.loads((ROOT / "data" / "test.json").read_text())
    return [(s["id"], s["text"]) for s in data["in_scope"] + data["out_of_scope"]]


def make_plan(regimes, rounds, seed):
    """The seeded random order of regimes in each round."""
    rng = random.Random(seed)
    return [rng.sample(regimes, len(regimes)) for _ in range(rounds)]


def parse_args():
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--block-s", type=float, default=120)
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--guard-after-ms", type=float, default=25)
    parser.add_argument("--whisper-budget-ms", type=float, default=100)
    parser.add_argument("--llm-budget-ms", type=float, default=140)
    args = parser.parse_args()
    args.regimes = args.regimes.split(",")
    unknown = set(args.regimes) - set(REGIMES)
    if unknown:
        parser.error(f"unknown regimes: {sorted(unknown)}")
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    load = VoiceLoad(args, load_clips(), load_sentences())
    load.warm_up()
    ready = {"pid": os.getpid(), "ready_ns": time.monotonic_ns(), "settings": settings,
             "cpu_threads": torch.get_num_threads(), "cpu_cores": sorted(os.sched_getaffinity(0))}
    (args.out / "voice_ready.json").write_text(json.dumps(ready, indent=1))
    plan = make_plan(args.regimes, args.rounds, args.seed)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1))
    blocks = CsvLog(args.out / "blocks.csv", BLOCK_COLUMNS)
    for round_number, order in enumerate(plan):
        for slot, regime in enumerate(order):
            load.round, load.regime = round_number, regime
            rng = random.Random(f"{args.seed}:{round_number}")  # user regimes in one round hear the same commands
            start_ns = time.monotonic_ns()
            errors = load.run_block(start_ns + round(args.block_s * 1e9), rng)
            blocks.write([round_number, slot, regime, start_ns, time.monotonic_ns(), errors])
            print(f"round {round_number} slot {slot} {regime}: done, errors {errors}", flush=True)


if __name__ == "__main__":
    main()
