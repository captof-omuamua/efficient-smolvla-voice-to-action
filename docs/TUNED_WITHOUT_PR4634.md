# Does Tuned need pull request #4634? (measured 10 Nov 2026, 19:41–19:51 IST)

**Result:** Tuned settings compile and run on plain LeRobot main. Median 15.4–15.5 ms per decision on plain main, against 13.8–14.5 ms in the final tested checkout (PR #4634 + camera-batching patch). With the PR but cameras looped: 15.1–15.2 ms.

Tuned settings: `compile_model=True` (mode `max-autotune`), `num_steps=1`, `empty_cameras=0`, `lerobot/smolvla_libero` (cached snapshot `a24f1f38`), batch 1, laptop RTX 5090, TF32 on. `compile_denoise` (the PR's new flag) is off in all runs.

## Results

One fresh process per run; first call, 10 warm-up calls, then 300 timed calls. Order: P, M, L, then P, M, L again. Repeat 1 is a cold compile (empty cache folder); repeat 2 reuses that configuration's cache.

| Configuration | Code imported | First call (cold / warm) | Median ms (r1 / r2) | 99th-percentile latency (p99) ms (r1 / r2) | Recompiles |
|---|---|---|---|---|---|
| P: with PR, cameras batched (= Tuned, setup B) | final tested checkout | 164.3 s / 6.1 s | 14.50 / 13.76 | 18.58 / 16.61 | none |
| M: plain main (no PR, no batching patch) | upstream clone `81b8c5b` | 156.9 s / 5.7 s | 15.54 / 15.38 | 20.43 / 18.20 | none |
| L: with PR, cameras looped | final tested checkout, loop put back | 148.4 s / 5.9 s | 15.20 / 15.14 | 19.95 / 17.92 | none |

Reference: P reproduces the 8 Nov number (14.4 ms, repeat 14.9 ms).

## Reading

- Plain main is 0.9–1.8 ms (6–13%) slower than Tuned. Most of that gap is camera batching: taking batching away from the PR code (P → L) costs 0.6–1.4 ms.
- The PR's part (M vs L) is 0.2–0.4 ms, about 2%. That is smaller than the spread between the two P repeats (0.74 ms), so this test cannot tell it from zero.
- Why the PR's fixed key/value cache does little here: with `compile_denoise` off, the PR turns the cache into a tuple of tensors before the single denoising step. At one step there is only one read of it.

## Compiler behaviour

- All six runs: exit 0, no recompile (Dynamo's frame count did not change after the first call; no recompile line with `TORCH_LOGS=recompiles,graph_breaks`).
- Graphs / graph breaks / frames OK of total: P 11 / 6 / 18 of 20; M 9 / 4 / 15 of 18; L 9 / 5 / 17 of 19.
- Same causes as on 8 Nov, all inside transformers: `aten.nonzero` in SmolVLM's vision embeddings, and two dictionary iterators Dynamo cannot rebuild.
- One difference on plain main: the `aten.nonzero` break falls inside main's per-camera `for` loop, so Dynamo skips the frame `embed_prefix` (`modeling_smolvla.py` line 604) and runs that function's own Python eagerly: "torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager". The functions it calls (vision encoder, language embedding) are still compiled. It works; it is just less tidy.
- Cold runs each log one "Runtime error during autotuning: out of resource: triton_mm ... Ignoring this choice." That is the autotuner dropping one candidate kernel, in all three configurations.

## Actions

Same inputs for all runs. First-call actions: P r1 and P r2 identical. Largest difference from P: M 0.057, L 0.065 (action scale about 2). Same size as the rounding differences noted on 8 Nov. Closed-loop success on plain main was not run.

## Not checked

- Closed-loop LIBERO success on plain main (only speed and first-call actions here).
- Energy per decision on plain main.
- L puts the loop back with a list comprehension on the loaded model (as A0/A1 did on 8 Nov). That is not identical to main's own loop code, which is why `embed_prefix` is skipped in M but not in L. So "M vs L" is the PR plus this small difference in loop code, not the PR alone.
- Only 2 repeats per configuration, 300 calls each, in one 10-minute window. CPU package was 59 °C at the first run and 70–78 °C at the later ones.
- Plain main here is `81b8c5b` (a clone from 10 Nov), not `d490e472`; the SmolVLA files are the same in both.

## How it was run

- Python: the final tested checkout's venv (torch 2.15.0, transformers 5.22.1).
- P and L: no `PYTHONPATH`; imported the final tested checkout's `src/lerobot` (main `d490e472` + PR head `2d200334` merged uncommitted + `batch_cameras.patch`). Checked: the file contains `compile_denoise` and `_embed_images`.
- M: `PYTHONPATH=<clone>/src`; imported a clean clone of LeRobot main at `81b8c5b`. Checked: the file contains neither `compile_denoise` nor `_embed_images`.
- Measuring code: `speed/followups/followup_common.py` (`load_policy`, `make_inputs`, `timed`, `timed_block`, `latency_stats`, `dynamo_summary`), same inputs as 8 Nov (seed 0 observation, noise seed 1234).
- Scripts: `speed/followups/tuned_check.py` (one run) and `speed/followups/run_tuned_check.sh` (all six). Per-run results, per-call times, first-call actions and logs: `speed/results/tuned_without_pr4634_2026-11-10/`.
- Command per run:

```
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_LOGS=recompiles,graph_breaks \
PYTHONPATH=<empty, or clone/src for M> TORCHINDUCTOR_CACHE_DIR=<cache>/<P|M|L> \
flock $GPU_LOCK $PYTHON tuned_check.py --label <name> --cameras <native|looped> --out-dir <out>
```
