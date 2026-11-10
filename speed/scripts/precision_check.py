"""Side check: does today's TF32 default change the eager baselines' speed?

The talk's eager baseline ran with PyTorch's default float32 matmul precision ("highest"); today's lerobot-eval
and lerobot-rollout turn TF32 on by default (--tf32=true), and the final session runs every setup that way.
This times A0 and A1 both ways in one process, in mirrored rounds of 100-call blocks.
Output: precision_check.json in --out-dir.
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from energy_session_final import Gpu, SETUPS, latency_stats, load_policy, make_inputs, nvidia_smi_foreign_pids, timed


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    gpu = Gpu()
    if pids := gpu.foreign_pids():
        raise SystemExit(f"Another compute process is on the GPU: {pids}")

    policies = {k: load_policy(SETUPS[k]) for k in ("A0", "A1")}
    inputs = {k: make_inputs(policies[k]) for k in policies}
    predict = {k: (lambda k=k: policies[k].predict_action_chunk(dict(inputs[k][0]), noise=inputs[k][1].clone()))
               for k in policies}
    variants = [(k, tf32) for k in policies for tf32 in (True, False)]
    first_actions = {}
    for k, tf32 in variants:
        set_tf32(tf32)
        first_actions[(k, tf32)] = predict[k]().clone().cpu()
        for _ in range(10):
            timed(predict[k])

    ms = {v: [] for v in variants}
    redone = 0
    for r in range(args.rounds):
        for v in variants if r % 2 == 0 else variants[::-1]:
            while True:
                while gpu.foreign_pids() or nvidia_smi_foreign_pids():  # wait until the GPU is ours alone
                    time.sleep(10)
                set_tf32(v[1])
                block = [timed(predict[v[0]])[1] for _ in range(args.calls)]
                if not gpu.foreign_pids():
                    break
                redone += 1  # another compute process showed up: redo the block
            ms[v] += block

    result = {
        "what": "Eager A0 and A1 with TF32 on (today's lerobot-eval/rollout default) vs off (the talk's speed runs)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "rounds": args.rounds, "calls_per_block": args.calls, "redone_blocks": redone,
        "results": {f"{k}_tf32_{'on' if tf32 else 'off'}": latency_stats(ms[(k, tf32)]) for k, tf32 in variants},
        "median_ratio_off_over_on": {k: round(statistics.median(ms[(k, False)]) / statistics.median(ms[(k, True)]), 3)
                                     for k in policies},
        "actions_max_abs_diff_on_vs_off": {k: float((first_actions[(k, True)] - first_actions[(k, False)]).abs().max())
                                           for k in policies},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "precision_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
