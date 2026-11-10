"""Follow-up 3: why are absolute times 10-17% slower than on 31 Oct? One setup on one stack, in a fresh process.

run_task3_stack_compare.sh runs this in alternating OLD/NEW processes:
  OLD = the talk's code: 8c4f33e5 + the talk's camera-batching patch, with the lerobot-pr4634 venv
        (torch 2.11, transformers 5.5.4), via PYTHONPATH.
  NEW = the kept final worktree (main d490e472 + PR #4634 head + batch_cameras.patch) with its venv
        (torch 2.15, transformers 5.22.1).
Both run with TF32 on (allow_tf32=True). B uses the default compile mode (max-autotune) on both.

Measures, after loading, the first call and 10 warm-up calls:
  - 200 timed calls of predict_action_chunk (median, p99);
  - A0 only: the parts of one decision, 100 times each, timed separately (as feasibility_smolvla.prefix_breakdown):
    image prep, the vision encoder alone (embed_image per camera), embed_prefix (vision encoder + language
    embeddings), and the VLM prefix pass. Denoise loop = whole-call median - (prep + embed_prefix + VLM);
  - GPU kernel time and kernel launches per call (torch.profiler over 10 calls). If the kernels take as long on
    both stacks but the wall time differs, the extra time is on the CPU side (Python, dispatch, launches);
  - GPU and CPU temperature and clocks during the timed calls.
Usage: python stack_compare.py --stack OLD --setup A0 --label run1 --out-dir <dir>
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

import lerobot
from followup_common import (SETUPS, Gpu, Sampler, cpu_pkg_temp, dynamo_summary, gpu_kernel_profile, latency_stats,
                             load_policy, make_inputs, timed, timed_block)
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


@torch.no_grad()
def breakdown(policy, batch, n=100):
    """Median ms of each part of the prefix, timed separately (each with a synchronize before and after)."""
    parts = {"prep": [], "vision_encoder": [], "embed_prefix": [], "vlm_prefix_pass": []}
    for _ in range(n):
        (images, img_masks), ms = timed(lambda: policy.prepare_images(batch))
        parts["prep"].append(ms)
        state = policy.prepare_state(batch)
        _, ms = timed(lambda: [policy.model.vlm_with_expert.embed_image(img) for img in images])
        parts["vision_encoder"].append(ms)
        (embs, pad_masks, att_masks), ms = timed(lambda: policy.model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state))
        parts["embed_prefix"].append(ms)
        _, ms = timed(lambda: policy.model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad_masks, att_masks), position_ids=torch.cumsum(pad_masks, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embs, None], use_cache=policy.config.use_cache))
        parts["vlm_prefix_pass"].append(ms)
    return {k: round(statistics.median(v), 2) for k, v in parts.items()} | {"images_per_call": len(images)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["OLD", "NEW"], required=True)
    parser.add_argument("--setup", choices=list(SETUPS), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    log = lambda msg: print(f"[{datetime.now():%H:%M:%S}] {args.stack} {args.setup} {args.label}: {msg}",  # noqa: E731
                            flush=True)
    log(f"lerobot code: {lerobot.__file__} | torch {torch.__version__} | transformers {transformers.__version__}")
    gpu = Gpu()
    gpu.wait_until_ours(log)
    torch.backends.cuda.matmul.allow_tf32 = True  # the same TF32 setting on both stacks
    torch.backends.cudnn.allow_tf32 = True
    temps_start = {"cpu_pkg_c": cpu_pkg_temp(), "gpu_c": gpu.temp()}

    t = time.perf_counter()
    policy = load_policy(SETUPS[args.setup])
    load_s = time.perf_counter() - t
    batch, noise = make_inputs(policy)

    def predict():
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())

    _, first_ms = timed(predict)
    for _ in range(10):
        timed(predict)
    sampler = Sampler(gpu)
    sampler.start()
    ms = timed_block(gpu, sampler, predict, args.calls, "timed", log)
    sampler.phase = "parts"
    gpu.wait_until_ours(log)
    parts = breakdown(policy, dict(batch)) if args.setup == "A0" else None
    sampler.phase = "profile"
    profile = gpu_kernel_profile(predict)
    sampler.stop()
    stats = latency_stats(ms)
    if parts:
        parts["denoise_loop_est"] = round(stats["median_ms"] - parts["prep"] - parts["embed_prefix"]
                                          - parts["vlm_prefix_pass"], 2)

    result = {
        "stack": args.stack, "setup": args.setup, "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lerobot_file": lerobot.__file__, "torch": torch.__version__, "transformers": transformers.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "compile_mode": policy.config.compile_mode if SETUPS[args.setup]["compile_model"] else None,
        "load_s": round(load_s, 2), "first_call_s": round(first_ms / 1000, 2),
        "latency": stats, "parts_ms": parts, "gpu_profile": profile,
        "dynamo": dynamo_summary() if SETUPS[args.setup]["compile_model"] else None,
        "temps_start": temps_start, "gpu_cpu_during_timed_calls": sampler.summary("timed"),
        "env": {k: __import__("os").environ.get(k) for k in ("PYTHONPATH", "TORCHINDUCTOR_CACHE_DIR",
                                                             "TORCHINDUCTOR_FORCE_DISABLE_CACHES")},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.setup}_{args.stack}_{args.label}.json").write_text(json.dumps(result, indent=2))
    log(json.dumps({"median_ms": stats["median_ms"], "p99_ms": stats["p99_ms"], "first_call_s": result["first_call_s"],
                    "parts": parts, "profile": profile, "during": result["gpu_cpu_during_timed_calls"]}))


if __name__ == "__main__":
    main()
