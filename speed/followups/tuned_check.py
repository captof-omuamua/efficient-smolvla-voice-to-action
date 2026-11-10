"""Tuned SmolVLA (compile_model=True, num_steps=1, empty_cameras=0), one fresh process per run.
Reuses the 8 Nov measuring code (followups/followup_common.py): same inputs, same timed() and latency_stats().
--cameras native keeps the imported code's own behaviour; looped puts the per-camera loop back (as A0/A1 did).
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # followup_common.py sits next to this file
import torch
import lerobot
from followup_common import SETUPS, Gpu, Sampler, cpu_pkg_temp, dynamo_summary, latency_stats, load_policy, make_inputs, timed, timed_block
import lerobot.policies.smolvla.modeling_smolvla as m

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--cameras", choices=["native", "looped"], default="native")
p.add_argument("--calls", type=int, default=300)
p.add_argument("--out-dir", required=True)
a = p.parse_args()
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {a.label}: {s}", flush=True)
src = Path(m.__file__).read_text()
info = {"lerobot_file": lerobot.__file__, "modeling_file": m.__file__,
        "has_pr4634_compile_denoise": "compile_denoise" in src, "has_batching_patch__embed_images": "_embed_images" in src}
log(json.dumps(info))
gpu = Gpu(); gpu.wait_until_ours(log)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
temps = {"cpu_pkg_c": cpu_pkg_temp(), "gpu_c": gpu.temp()}
setup = dict(SETUPS["B"]); setup["batched_cameras"] = a.cameras == "native"  # False = put the loop back
policy = load_policy(setup)
batch, noise = make_inputs(policy)
predict = lambda: policy.predict_action_chunk(dict(batch), noise=noise.clone())
actions, first_ms = timed(predict)
first_actions = actions.clone().cpu()
after_first = dynamo_summary()
log(f"first call {first_ms / 1000:.1f} s")
for _ in range(10):
    timed(predict)
sampler = Sampler(gpu); sampler.start()
ms = timed_block(gpu, sampler, predict, a.calls, "timed", log)
sampler.stop()
after_all = dynamo_summary()
stats = latency_stats(ms)
res = {"label": a.label, "cameras": a.cameras, **info, "torch": torch.__version__,
       "config": {k: getattr(policy.config, k, None) for k in ("compile_model", "compile_mode", "num_steps", "empty_cameras", "compile_denoise")},
       "first_call_s": round(first_ms / 1000, 2), "latency": stats,
       "dynamo_after_first_call": after_first, "dynamo_after_all_calls": after_all,
       "recompiled_after_first_call": after_first["frames"] != after_all["frames"],
       "temps_start": temps, "during_timed": sampler.summary("timed")}
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
torch.save(first_actions, out / f"{a.label}_first_actions.pt")
(out / f"{a.label}.json").write_text(json.dumps(res, indent=2))
(out / f"{a.label}_ms.json").write_text(json.dumps(ms))
log(json.dumps({"first_call_s": res["first_call_s"], "median_ms": stats["median_ms"], "p99_ms": stats["p99_ms"],
                "recompiled": res["recompiled_after_first_call"], "graphs": after_all["unique_graphs"],
                "graph_breaks": after_all["graph_break_count"], "frames": after_all["frames"]}))
