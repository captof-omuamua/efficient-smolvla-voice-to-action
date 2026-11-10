"""Collect follow-up 1 (PR #4634 test fix) into task1_test_fix.json: test results, the diff, and the source lines
of transformers' DynamicLayer.crop in the PR's pinned version (5.5.4) and today's main's (5.22.1).
Usage: python summarize_task1.py
"""

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the repository root
HERE = Path(os.environ.get("OUT_DIR", REPO / "runs")) / "speed" / "followups"  # results read and written
SITE = "lib/python3.12/site-packages/transformers/cache_utils.py"
OLD_SRC = Path(os.environ.get("OLD_LEROBOT_DIR", "")) / ".venv" / SITE
NEW_SRC = Path(os.environ.get("LEROBOT_DIR", "")) / ".venv" / SITE


def lines(path, first, last):
    text = path.read_text().splitlines()
    return {"file": str(path), "lines": f"{first}-{last}", "text": text[first - 1 : last]}


def pytest_result(name):
    path = HERE / f"pytest_{name}.txt"
    text = path.read_text()
    summary = re.findall(r"=+ (.*(?:passed|failed).*) in [\d.]+s", text)
    return {"file": os.path.relpath(path, REPO), "header": text.splitlines()[:2], "summary": summary[-1] if summary else None,
            "results": re.findall(r"^(PASSED|FAILED|ERROR) (\S+)", text, flags=re.M)}


result = {
    "what": "PR #4634's test_smolvla_tuple_kv_parity: fix for today's main (transformers 5.22) and checks on both stacks",
    "fix_diff": {"file": "bench/results_final_2026-11-07/followups/pr4634_test_fix.diff",
                 "text": (REPO / "speed/patches/pr4634_test_fix.diff").read_text()},
    "why": ("transformers 5.22 changed DynamicLayer.crop's argument from a target length (crop(70) kept 70 tokens; "
            "a negative value removed that many) to a number of tokens to remove: a positive value now raises "
            "ValueError and crop(0) does nothing. Main adapted SmolVLA's denoise_step in #5220 (4598a0bb, 3 Nov 2026) "
            "with `layer.crop(prefix_len - layer.get_seq_length())`, but the PR's test still calls "
            "`dynamic_kv.crop(prefix_len)`. The fix uses the same per-layer form as #5220."),
    "transformers_source": {
        "5.5.4 (PR pin)": lines(OLD_SRC, 139, 151),
        "5.22.1 (main pin)": lines(NEW_SRC, 163, 180),
    },
    "main_adaptation": "4598a0bb Bump transformers version to 5.22 (#5220): modeling_smolvla.py denoise_step "
                       "`past_key_values.crop(prefix_len)` -> `for layer in past_key_values.layers: "
                       "layer.crop(prefix_len - layer.get_seq_length())`",
    "runs": {
        "today's main + PR + patch, unchanged test (7 Nov, transformers 5.22.1)": {
            "file": "bench/results_final_2026-11-07/pytest_smolvla.log",
            "summary": "1 failed, 14 passed (test_smolvla_tuple_kv_parity: ValueError from crop(70))"},
        "today's main + PR + patch, with the fix: test_smolvla_compile_parity.py": pytest_result(
            "new_fixed_compile_parity"),
        "today's main + PR + patch, with the fix: tests/policies/smolvla": pytest_result("new_fixed_smolvla_all"),
        "PR's pinned stack (lerobot-pr4634 venv + worktree at a932c505), unchanged": pytest_result(
            "old_a932c505_unchanged_compile_parity"),
        "PR's pinned stack, PR head 2d200334, unchanged": pytest_result("old_head_unchanged_compile_parity"),
        "PR's pinned stack, PR head 2d200334, with the fix (expected to fail)": pytest_result(
            "old_head_fixed_tuple_kv"),
    },
    "conclusion": ("The test fails only on current main's transformers (5.22). Unchanged, it passes on the PR's own "
                   "pinned stack (torch 2.11, transformers 5.5.4) at a932c505 and at the head 2d200334. With the "
                   "fix, all 15 tests in tests/policies/smolvla pass on today's main. The fix needs transformers "
                   ">= 5.22 (on 5.5.4, crop(0) empties the cache and the test fails with a shape mismatch), just "
                   "like main's own denoise_step since #5220, so it belongs in the rebase onto main."),
}
(HERE / "task1_test_fix.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v["summary"] if isinstance(v, dict) else v for k, v in result["runs"].items()}, indent=1))
