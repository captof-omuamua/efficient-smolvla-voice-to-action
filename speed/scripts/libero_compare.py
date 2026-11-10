"""Compare LIBERO runs episode by episode against a reference run.

Every run used the same seed and the same initial states, so episode k of task t starts from the same
scene in every run. That lets us use McNemar's test: it only looks at the episodes where the two runs
disagree (one succeeded, the other failed) and asks whether that split could be chance.

Usage: python libero_compare.py --reference baseline_eager_n10 trt_steps10 eager_steps5 ...
"""

import argparse
import json
import os
from math import comb
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load_successes(run_dir):
    """Return {(suite, task_id, episode): True/False} for one run."""
    info = json.loads((Path(run_dir) / "eval_info.json").read_text())
    out = {}
    for task in info["per_task"]:
        for k, success in enumerate(task["metrics"]["successes"]):
            out[(task["task_group"], task["task_id"], k)] = bool(success)
    return out


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the two kinds of disagreement (b and c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff_ci(b, c, n):
    """Run minus reference, in percentage points, with a 95% interval for paired data."""
    diff = (c - b) / n
    se = ((b + c) / n - diff**2) ** 0.5 / n**0.5
    return 100 * diff, 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parents[2] / "runs")) / "speed" / "libero_runs"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    ref = load_successes(Path(args.runs_dir) / args.reference)
    print("run | Spatial | Object | Goal | Long | overall | only ref won | only run won | McNemar p | diff [95% CI]")
    for name in [args.reference] + args.runs:
        run = load_successes(Path(args.runs_dir) / name)
        assert run.keys() == ref.keys(), f"{name} has different episodes from the reference"
        per_suite = [sum(v for (s, _, _), v in run.items() if s == suite) for suite in SUITES]  # 100 episodes each
        b = sum(ref[k] and not run[k] for k in ref)  # reference succeeded, this run failed
        c = sum(run[k] and not ref[k] for k in ref)  # this run succeeded, reference failed
        overall = 100 * sum(run.values()) / len(run)
        diff, low, high = paired_diff_ci(b, c, len(run))
        print(
            f"{name} | " + " | ".join(map(str, per_suite))
            + f" | {overall:.2f} | {b} | {c} | {mcnemar_p(b, c):.2f} | {diff:+.2f} [{low:+.1f}, {high:+.1f}]"
        )


if __name__ == "__main__":
    main()
