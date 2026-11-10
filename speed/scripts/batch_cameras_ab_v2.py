"""Same-process A/B for the SmolVLA camera-batching PR, rebased on upstream main.

One process, one set of weights (lerobot/smolvla_libero), batch size 1, eager PyTorch.
"Looped" is LeRobot's code before the PR: one vision-encoder call per camera, put back by overriding
`_embed_images` on the model instance. "Batched" is the PR's class method: one call for all cameras.
All settings take turns in rounds (order reversed every other round, so slow clock drift hits all of
them equally); warm-up calls are not timed.

Timed rows:
  e2e     predict_action_chunk (chunk 50): 2 cameras + 1 blank slot (empty_cameras=1, as in LeRobot's
          LIBERO eval) and 2 cameras, each at 1 and 10 denoising steps.
  vision  `_embed_images` alone (vision encoder + connector) for 2 and 3 camera slots.
Parity (same input, same starting noise): max abs difference of the image embeddings and of the
actions, batched vs looped, next to a rounding-only reference (the same input run as a batch of 2)
and a determinism check (the looped path run twice).

Run from the PR worktree:
  HF_HUB_OFFLINE=1 uv run --no-sync python batch_cameras_ab_v2.py --out-dir <dir>
With --dump-actions <file> it only saves the looped-path actions; run it on unpatched main too, to check
that the restored "before" path gives exactly main's actions.
"""

import argparse
import functools
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import try_to_load_from_cache

import lerobot
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed

MODEL = "lerobot/smolvla_libero"
E2E = [(blank, steps) for blank in (1, 0) for steps in (1, 10)]  # (blank camera slots, denoising steps)
VISION = (2, 3)  # camera slots fed to the vision encoder alone
EXECUTED_STEPS = 10  # LIBERO eval runs n_action_steps=10 of each 50-step chunk


def make_observation(config, device):
    """One fake observation: random camera images in [0, 1], a random state, an instruction."""
    obs = {"task": ["Pick up the red cube and place it in the box"]}
    for name, feature in config.input_features.items():
        if "empty_camera" in name:
            continue  # the env never sends the blank slot; prepare_images adds it
        if feature.type == FeatureType.VISUAL:
            obs[name] = torch.rand(1, *feature.shape, device=device)
        elif feature.type == FeatureType.STATE:
            obs[name] = torch.randn(1, *feature.shape, device=device)
    return obs


def doubled(batch):
    """The same sample twice in one batch: identical maths, different GPU kernel shapes and rounding."""
    return {k: torch.cat([v, v]) if torch.is_tensor(v) else v for k, v in batch.items()}


def timed_ms(fn):
    """Milliseconds for one call; synchronize() waits for the GPU before and after."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def summarize(samples):
    p = statistics.quantiles(samples, n=100, method="inclusive")  # linear interpolation, like numpy
    return {
        "n": len(samples),
        "median_ms": round(statistics.median(samples), 2),
        "p99_ms": round(p[98], 2),
        "p90_ms": round(p[89], 2),
        "mean_ms": round(statistics.mean(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
    }


def action_diff(a, b):
    """Max abs difference over the whole chunk, plus the mean and the max over the executed steps."""
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": round(float(d.max()), 6),
        "mean_abs": round(float(d.mean()), 6),
        f"max_abs_first_{EXECUTED_STEPS}_steps": round(float(d[:, :EXECUTED_STEPS].max()), 6),
    }


def embedding_diff(a, b):
    """Max and mean abs difference over every camera's image embeddings."""
    d = torch.cat([(x.float() - y.float()).abs().flatten() for x, y in zip(a, b, strict=True)])
    return {"max_abs": round(float(d.max()), 6), "mean_abs": round(float(d.mean()), 6)}


def nvidia_smi(query):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True
    )
    return out.stdout.strip()


def git(*args):
    repo = Path(lerobot.__file__).resolve().parents[2]
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout


def environment():
    weights = try_to_load_from_cache(MODEL, "model.safetensors")
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_state": nvidia_smi("driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "lerobot_src": str(Path(lerobot.__file__).parent),
        "base_sha": git("rev-parse", "HEAD").strip(),
        "pr_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "model": MODEL,
        "model_revision": Path(weights).parent.name if isinstance(weights, str) else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cpu_count": os.cpu_count(),
        "load_avg_1_5_15": os.getloadavg(),
    }


def main(rounds, calls, warmup, out_dir, dump_actions=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()

    config = PreTrainedConfig.from_pretrained(MODEL)
    config.empty_cameras = 1  # as in LeRobot's LIBERO eval; switched off per setting below
    policy = SmolVLAPolicy.from_pretrained(MODEL, config=config).to("cuda").eval()
    model = policy.model
    assert policy.config is model.config  # empty_cameras and num_steps are read from both
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=MODEL)
    env["vision_attention"] = model.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation
    env["weight_dtypes"] = sorted({str(p.dtype) for p in policy.parameters()})

    set_seed(0)
    batch = preprocessor(make_observation(policy.config, "cuda"))
    noise = torch.randn(
        1, config.chunk_size, config.max_action_dim, device="cuda", generator=torch.Generator("cuda").manual_seed(1234)
    )

    def looped(images):  # LeRobot's code before this PR
        return [model.vlm_with_expert.embed_image(img) for img in images]

    def use_path(batched):
        if batched:
            vars(model).pop("_embed_images", None)  # back to the PR's class method: one batched call
        else:
            model._embed_images = looped  # the instance attribute shadows the class method

    @torch.no_grad()
    def predict(blank, steps, batched, b=batch, n=noise):
        policy.config.empty_cameras, policy.config.num_steps = blank, steps
        use_path(batched)
        return policy.predict_action_chunk(dict(b), noise=n.clone())

    @torch.no_grad()
    def embed(images, batched):
        use_path(batched)
        return model._embed_images(images)

    def camera_slots(blank, b=batch):
        policy.config.empty_cameras = blank
        images, _ = policy.prepare_images(b)  # the images the model sees: 512x512, range [-1, 1]
        return images

    if dump_actions is not None:
        # Looped-path actions for every e2e setting. On unpatched main the override is never called, so
        # this saves main's own actions: comparing the two files checks the restored "before" path.
        actions = {f"blank{b}_steps{s}": predict(b, s, batched=False).cpu() for b, s in E2E}
        torch.save({"environment": env, "actions": actions}, dump_actions)
        print(f"saved looped-path actions to {dump_actions} (lerobot from {env['lerobot_src']})")
        return

    slots = {2: camera_slots(0), 3: camera_slots(1)}  # 3 = two cameras + LIBERO's blank slot
    assert [len(slots[n]) for n in VISION] == list(VISION)

    jobs = {}
    for blank, steps in E2E:
        for batched in (False, True):
            name = f"e2e_{2 + blank}slots_blank{blank}_steps{steps}_{'batched' if batched else 'looped'}"
            jobs[name] = functools.partial(predict, blank, steps, batched)
    for n in VISION:
        for batched in (False, True):
            jobs[f"vision_{n}slots_{'batched' if batched else 'looped'}"] = functools.partial(embed, slots[n], batched)

    # Timing: warm up every job, then take turns in rounds
    for fn in jobs.values():
        for _ in range(warmup):
            fn()
    env["gpu_state_before_timing"] = nvidia_smi("clocks.sm,temperature.gpu,power.draw")
    samples = {name: [] for name in jobs}
    order = list(jobs)
    start = time.perf_counter()
    for r in range(rounds):
        for name in order if r % 2 == 0 else order[::-1]:
            samples[name] += [timed_ms(jobs[name]) for _ in range(calls)]
        print(f"round {r + 1}/{rounds} done", flush=True)
    env["timing_wall_s"] = round(time.perf_counter() - start, 1)
    env["gpu_state_after_timing"] = nvidia_smi("clocks.sm,temperature.gpu,power.draw")
    env["load_avg_after_timing"] = os.getloadavg()

    timing = {name: summarize(s) for name, s in samples.items()}
    rows = {}
    for name in jobs:
        if name.endswith("_looped"):
            base = name.removesuffix("_looped")
            before, after = timing[name], timing[base + "_batched"]
            rows[base] = {
                "looped_median_ms": before["median_ms"],
                "batched_median_ms": after["median_ms"],
                "speedup_median": round(before["median_ms"] / after["median_ms"], 3),
                "looped_p99_ms": before["p99_ms"],
                "batched_p99_ms": after["p99_ms"],
                "speedup_p99": round(before["p99_ms"] / after["p99_ms"], 3),
            }
            print(base, rows[base], flush=True)

    # Parity: same input, same starting noise
    @torch.no_grad()
    def parity():
        out = {"embeddings": {}, "actions": {}}
        for n in VISION:
            images = slots[n]
            reference = embed(images, batched=False)
            out["embeddings"][f"{n}slots"] = {
                "batched_vs_looped": embedding_diff(reference, embed(images, batched=True)),
                "rounding_reference": embedding_diff(
                    reference, [e[:1] for e in looped([torch.cat([img, img]) for img in images])]
                ),
                "looped_twice": embedding_diff(reference, embed(images, batched=False)),
                "embedding_scale_max_abs": round(max(float(e.float().abs().max()) for e in reference), 4),
            }
        for blank, steps in E2E:
            reference = predict(blank, steps, batched=False)
            out["actions"][f"{2 + blank}slots_blank{blank}_steps{steps}"] = {
                "batched_vs_looped": action_diff(reference, predict(blank, steps, batched=True)),
                "rounding_reference": action_diff(
                    reference, predict(blank, steps, False, doubled(batch), torch.cat([noise, noise]))[:1]
                ),
                "looped_twice": action_diff(reference, predict(blank, steps, batched=False)),
                "action_scale_max_abs": round(float(reference.float().abs().max()), 4),
            }
        return out

    results = {
        "method": {
            "batch_size": 1,
            "rounds": rounds,
            "calls_per_round": calls,
            "timed_calls_per_setting": rounds * calls,
            "warmup_calls_per_setting": warmup,
            "order": "all settings take turns in rounds; order reversed every other round",
            "p99": "statistics.quantiles(method='inclusive'), linear interpolation",
            "actions_compared": "normalized action chunk from predict_action_chunk (50 steps x 7 dims)",
        },
        "environment": env,
        "rows": rows,
        "timing": timing,
    }
    results["parity_bf16_as_deployed"] = parity()
    policy.to(torch.float32)
    results["parity_float32"] = parity()
    use_path(batched=True)

    (out_dir / "batch_cameras_ab.json").write_text(json.dumps(results, indent=2))
    (out_dir / "timing_samples_ms.json").write_text(
        json.dumps({k: [round(x, 4) for x in v] for k, v in samples.items()})
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dump-actions", type=Path, help="only save the looped-path actions to this file")
    args = parser.parse_args()
    main(args.rounds, args.calls, args.warmup, args.out_dir, args.dump_actions)
