# Speed results: SmolVLA on LIBERO (measured 7–8 Nov 2026)

Everything ran on a laptop RTX 5090, between 7 Nov 23:58 and 8 Nov 02:08 IST. The success counts (328/400 for both runs, 22 flips each way), the medians (A0 118.93, A1 110.99, C 27.89, B 14.41 ms), the energy ratios (A0→B 7.96×, A1→B 7.64×, with per-round ranges) and the failing test's error message were recomputed from the raw files in a separate check.

These numbers replace an earlier measurement of 31 Oct on an older stack (PR #4634 head `8c4f33e5`, torch 2.11, transformers 5.5.4). Section 2 compares the two.

Short forms:
- TensorFloat-32 (TF32): a faster float32 matrix mode.
- NVIDIA Management Library (NVML): the driver's counters.
- 99th-percentile latency (p99): 1 call in 100 is slower.
- percentage points (pp); joules (J).
- "LeRobot CI" means LeRobot's continuous-integration workflows. Statistical intervals are always written "95% interval".

In the main README, **Original** is A1 and **Tuned** is B.

## 1. Headline table

Setups (`lerobot/smolvla_libero`, batch 1 for speed):
- **A0, earlier baseline definition:** eager, 10 denoising steps, 1 blank camera slot, cameras one call each.
- **A1, today's default baseline (Original):** eager, 10 steps, no blank slot (the checkpoint's default), cameras one call each.
- **C, no compiler:** eager, 1 step, no blank slot, cameras batched.
- **B, tuned (Tuned):** whole-model compile (`compile_model=True`, default mode `max-autotune`), 1 step, no blank slot, cameras batched.

Speed and energy come from one session where all four setups take turns (pass 1, 8 Nov 00:19–00:28). Numbers in brackets come from a repeat session (pass 2, 01:39–01:51).

| Setup | Median ms | p99 ms | J per decision | J above idle | Startup (first call) | Peak GPU memory | LIBERO success, 400 episodes |
|---|---|---|---|---|---|---|---|
| A0 | 118.9 (118.8) | 147.0 (144.5) | 11.47 (11.36) | 8.54 (8.54) | 0.49 s | 946 MB | not run |
| A1 | 111.0 (112.9) | 147.0 (137.0) | 11.00 (10.89) | 8.39 (8.21) | 0.48 s | 936 MB | 82.00% |
| C | 27.9 (27.5) | 39.2 (37.5) | 2.83 (2.82) | 2.18 (2.15) | 0.43 s | 936 MB | not run |
| **B** | **14.4 (14.9)** | **16.8 (17.2)** | **1.44 (1.47)** | **1.11 (1.12)** | **152–161 s cold, 6.5–6.7 s warm** | **897 MB** | **82.00%** |

Ratios (pass 1, pass 2 in brackets). Energy ranges are over the 6 rounds of a session.

| Comparison | Median | p99 | Energy per decision | Energy above idle |
|---|---|---|---|---|
| A0 → B | 8.25× (7.99×) | 8.75× (8.42×) | 7.96×, range 7.50–8.40 (7.73×, 7.28–8.19) | 7.68×, 7.22–8.32 (7.61×, 7.10–8.18) |
| A1 → B | 7.70× (7.60×) | 8.75× (7.98×) | 7.64×, range 7.45–8.17 (7.42×, 6.90–7.92) | 7.55×, 7.26–8.22 (7.33×, 6.70–8.01) |
| A0 → C | 4.26× (4.32×) | 3.75× (3.85×) | 4.06× (4.03×) | 3.92× (3.97×) |
| A1 → C | 3.98× (4.10×) | 3.75× (3.65×) | 3.89× (3.86×) | 3.85× (3.82×) |
| A0 → A1 (blank slot only) | 1.07× (1.05×) | 1.00× (1.05×) | 1.04× (1.04×) | 1.02× (1.04×) |

**Closed loop, B against A1, on the same 400 starting scenes:** 82.00% vs 82.00%.
- Paired difference: +0.00 pp, 95% interval −3.3 to +3.3.
- McNemar p = 1.00, with 22 episodes flipped each way.

## 2. Against the earlier 31 Oct numbers

| Headline | 31 Oct stack | Final (7–8 Nov) |
|---|---|---|
| Speed | 101.3 → 13.1 ms, 7.7× | vs A0: 118.9 → 14.4 ms, 8.25× (repeat 7.99×). vs A1: 111.0 → 14.4 ms, 7.70× (repeat 7.60×) |
| Energy, same session | 10.26 → 1.41 J, 7.31× (7.21× above idle) | vs A0: 11.47 → 1.44 J, 7.96× (7.68× above idle). vs A1: 11.00 → 1.44 J, 7.64× (7.55×) |
| Success | 83.75% vs 83.00%, −0.75 pp, 95% interval −4.3 to +2.8 | A1 82.00% vs B 82.00%, +0.00 pp, 95% interval −3.3 to +3.3, p = 1.00 |
| Startup of B | 71–89 s cold, 15 s warm | 152–161 s cold (4 runs), 6.5–6.7 s warm (2 runs) |
| Peak memory | 929 MB baseline, 897 MB B | 946 MB A0, 936 MB A1, 897 MB B |

In plain words:
- **The speed-up holds:** about 8× against the earlier baseline definition, and about 7.6–7.7× against today's default baseline.
- **The energy saving holds:** about 7.7–8.0× against A0 and about 7.4–7.6× against A1.
- **Accuracy:** no change between A1 and B on the same 400 scenes.
- **Startup:** a cold start now takes about twice as long (152–161 s). A warm start is faster than before (6.5 s instead of 15 s).
- **Absolute times:** every setup is slower in ms than on 31 Oct (A0 +17%, B +10%).
  - I did not isolate it here; the follow-up in section 9 splits it into software stack and measurement conditions.
  - A0's median was 107–119 ms across the saved runs of 7–8 Nov, and single 11 s blocks reached 133 ms.
  - The CPU package went from 56 to 95 °C during the co-location check.

## 3. Closed loop in detail

| Run | Spatial | Object | Goal | Long | Overall |
|---|---|---|---|---|---|
| Final A1 (`final_2026-11-07_A1_eager_steps10`) | 79 | 89 | 87 | 73 | 82.00% |
| Final B (`final_2026-11-07_B_tuned`) | 80 | 89 | 87 | 72 | 82.00% |
| 31 Oct baseline (`baseline_eager_n10`, old stack, blank slot) | 86 | 90 | 86 | 73 | 83.75% |
| 31 Oct B (`batched_compile_model_steps1_noblank`, old stack) | 84 | 88 | 87 | 73 | 83.00% |

**Context against the 31 Oct runs.** These use the same seed and starting states, but different software.
- Final A1 vs 31 Oct baseline: −1.75 pp, 95% interval −5.1 to +1.6, p = 0.38.
- Final B vs 31 Oct baseline: −1.75 pp, 95% interval −5.2 to +1.7, p = 0.39.
- Final B vs 31 Oct B: −1.00 pp, 95% interval −4.3 to +2.3, p = 0.66.
- None of these is significant.
- The Spatial suite is lower now (79–80 vs 84–86). I did not isolate why. Candidates: the blank slot (only the 31 Oct baseline had it), the new simulator code with MuJoCo 3.8.1, or the PyTorch stack.

**GPU energy for a whole 400-episode run** (includes simulator rendering):
- A1: 62.6 kJ in 27.8 min (37.5 W mean).
- B: 62.3 kJ in 30.4 min (34.1 W mean). This includes B's cold compile: its first policy step took 179 s at batch 10.
- The two are about equal. Ten simulators step and render on the same GPU, and the policy runs only once every 10 steps, so the simulator dominates the run's time and energy. The per-decision savings above are about the policy alone.

**Eval loop time:** A1 1637 s, B 1792 s (179 s of that is B's first step). No other compute process appeared on the GPU during either run.

## 4. Startup, memory and compiler behaviour

**Eager setups.** The first call takes 0.43–0.49 s. From process start to the first action takes about 9 s: 6.8 s of Python imports and 1.7 s to load the model.

**B, cold first call** (no compile cache):
- 155.6 s in a fresh process with caches off;
- 158.1 s in a fresh process with an empty cache;
- 161.3 s and 152.1 s as the first call of session pass 1 and pass 2;
- 178.9 s in the closed-loop eval, at batch 10.

**B, warm first call** (reusing the cache an earlier start filled): 6.66 s and 6.54 s, or 6.18 s with compiler logging on. From process start to the first action: 17.3 s.

**Peak GPU memory** (PyTorch's allocated peak over 50 warm calls, one model per process):
- A0 946 MB, A1 936 MB, C 936 MB, B 897 MB.
- Reserved: 970, 966, 1010 and 1054 MB.

**Graph breaks.** B's "whole-model" compile is really 11 graphs (Dynamo compiled 20 frames, 18 of them OK). Both causes are inside transformers, not LeRobot:
- `aten.nonzero`, from boolean-mask indexing in SmolVLM's vision embeddings (`modeling_smolvlm.py` line 130). Its output shape depends on the data.
- Two dictionary iterators Dynamo cannot rebuild, in transformers' output-recording wrapper (`utils/output_capturing.py` lines 257 and 272). Dynamo skips that wrapper and compiles the functions it calls.

**Recompiles: none.**
- Dynamo's frame count did not change during any measured block, in either session, or over 50 warm calls.
- With `TORCH_LOGS=recompiles,graph_breaks`, neither the diagnostic run nor the 400-episode eval of B logged a recompile.

## 5. What changed against the 31 Oct stack

| | 31 Oct | Final (7–8 Nov) |
|---|---|---|
| LeRobot code | PR #4634 head `8c4f33e5` (on main `146d8474`). Batched rows added the camera-batching patch | main `d490e472` (6 Nov) + PR #4634 head `2d200334` merged with `--no-commit --no-ff` (library code equals `a932c505`) + `batch_cameras.patch` |
| torch | 2.11.0+cu128 | 2.15.0+cu132 |
| transformers | 5.5.4 | 5.22.1 |
| MuJoCo | 3.3.7 | 3.8.1 |
| triton | 3.6.0 | 3.9.0 |
| CUDA runtime, cuDNN | 12.8, 9.19 | 13.2, 9.26 |
| gymnasium, torchvision | 1.3.0, 0.26.0 | 1.4.0, 0.29.2 |
| Blank camera slot | The 31 Oct baseline used `--policy.empty_cameras=1`, as LeRobot CI's LIBERO eval did until 5 Nov | The checkpoint's default is `empty_cameras: 0`. #4888 (`d100b030`, 5 Nov) removed the flag from LeRobot CI's eval. A0 keeps the earlier definition; A1 is today's default |
| float32 matmul precision | The eager speed runs used PyTorch's default ("highest"). `compile_model` switched TF32 on itself. The closed-loop eval already had TF32 on for all setups | TF32 on for every setup. Since #5096, lerobot-eval and lerobot-rollout default to `--tf32=true`, and `compile_model` no longer sets it. Side check: TF32 changes eager medians by under 1% |
| LIBERO eval | lerobot-eval, simulator in the same process | #4888's new simulator core. It still runs in the same process by default, and the earlier flags still work. LeRobot CI now serves the simulator from a Docker image and maps cameras with `--rename_map`; that is a 30-step smoke test, not a success-rate run |
| Checkpoint | `lerobot/smolvla_libero`, cached snapshot `a24f1f38` | The same snapshot, now pinned with `--policy.pretrained_revision` (LeRobot CI pins the same one) |
| Speed method | One process per setup, 200 calls | One session with all four setups in mirrored rounds, 6 blocks each, plus a repeat session |

Unchanged: robosuite 1.4.0, hf-libero 0.1.4, numpy 2.2.6, Python 3.12.13, driver 595.99.02, power profile "balanced", AC power, enforced GPU power limit 110 W (default 95 W). Exact versions are in `speed/results/final_2026-11-07/versions.json`.

## 6. How it was measured

**Code under test:** a LeRobot checkout, kept as the "final tested" code.
- `git worktree add --detach … d490e472`, then `git merge --no-commit --no-ff 2d200334`: clean, no conflicts.
- `git apply batch_cameras.patch`: clean.
- Nothing is committed in that checkout. The merge leaves the PR's 4 files staged; the patch also changes `modeling_smolvla.py` and adds an untracked test file.

**Environment:**
- `uv sync --locked --extra smolvla --extra libero --extra test --extra evaluation --extra training`.
- The training extra brings `nvidia-ml-py`, which reads the energy counter.
- LeRobot CI's policy image also uses smolvla, evaluation and training; LIBERO lives in its separate simulator image.
- Settings: `MUJOCO_GL=egl`, `HF_HUB_OFFLINE=1`.

**Inputs for the speed runs:** a random observation is made right after `set_seed(0)`, and the starting noise comes from a generator seeded with 1234. All four setups get identical inputs (checked).

**Per-camera loop:** for A0 and A1, the loop is put back on the loaded instance (`policy.model._embed_images`). The A1 closed-loop run does the same.

**Session** (`energy_session_final.py`):
- Order: A0 A1 C B, then B C A1 A0, three times, so every setup gets 6 blocks.
- A block lasts about 11 s: 100 A0 calls, or as many calls of the others as fit in that time (pass 1: A1 104, C 405, B 765; pass 2: 110, 435, 773).
- An idle window of the same length comes before every block, plus one after the last.
- Energy comes from NVML's cumulative counter. Idle power for each block comes from the windows on both sides of it.
- Before every idle window and block, `nvidia-smi --query-compute-apps=pid` had to list only this process. NVML also checked every 0.5 s. No block needed a redo.
- B's first call is a cold compile: `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, in a fresh cache folder.

**Startup** (`startup_final.py`, run by `run_startup.sh`), one fresh process per run:
- B cold has caches off.
- B fill starts with an empty cache and caches on.
- B warm and warm2 reuse that cache.

**Closed loop** (`run_libero_final.sh`, `libero_eval_final.py`): LeRobot's lerobot-eval with the same protocol as on 31 Oct. Exact A1 command:

```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 python libero_eval_final.py --cameras looped --policy.num_steps=10 --policy.empty_cameras=0 \
  --policy.path=lerobot/smolvla_libero --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d \
  --policy.n_action_steps=10 --policy.device=cuda \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 --env.max_parallel_tasks=1 \
  --eval.batch_size=10 --eval.n_episodes=10 --eval.use_async_envs=true \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --seed=1000 --output_dir=$OUT_DIR/speed/libero_runs/final_2026-11-07_A1_eager_steps10
# B: --cameras batched --policy.compile_model=true --policy.num_steps=1 --policy.empty_cameras=0,
#    output final_2026-11-07_B_tuned, with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 (fresh cache folder)
#    and TORCH_LOGS=recompiles,graph_breaks
```

- The NVML energy counter was read at the start and end of each whole run, after a 20 s idle reading.
- A background loop watched for other compute processes during each run. None appeared.
- Paired stats come from `libero_compare.py`: an exact McNemar test and a paired 95% interval.

## 7. Checks, failures and caveats

**One sanity test fails.** `pytest tests/policies/smolvla -q` on the GPU gives 14 passed, 1 failed. The failure is `test_smolvla_tuple_kv_parity`, PR #4634's own test.
- The test calls `DynamicCache.crop(prefix_len)` with a positive length. transformers 5.22.1 rejects that: "Calling `crop` with a positive value (`70`) is not supported."
- The PR's library code never calls `crop` this way; main's `denoise_step` passes a negative count.
- I changed only that one call, to `crop(prefix_len - get_seq_length())`, in a copy outside the checkout (`tuple_kv_parity_check.py`). The comparison then passes bit for bit: the two key/value cache types give identical velocities, and the prefix stays unchanged.
- So the PR's test needs a one-line update for current main, while its library code needed no change. The follow-up in section 9 checks the fix on both stacks.

**TF32 check** (`precision_check.py`): with TF32 off, the eager medians were 1.008× (A0) and 1.004× (A1) of the TF32-on medians. TF32 does not explain the slower eager times.

**Co-location check** (`colocation_check.py`): A0 alone vs A0 next to a compiled B, in 6 alternating fresh processes.
- A0 alone: 111.4–114.7 ms. A0 next to B: 113.1–113.6 ms.
- Sharing the process does not slow the eager setup. The spread across runs follows the laptop's CPU temperature.

**Eager tails:** eager p99 is 137–147 ms.
- Other programs (a desktop chat app and GNOME Shell) used about 0.6 CPU cores during the eager blocks.
- The eager setups are CPU-bound, so their tails are wider than in an earlier clean pass on 7 Nov, which had about 0.1 cores of other load.
- B's tail stays tight (16.8–17.2 ms).

**Actions:** the first 10 executed steps match the 31 Oct saved actions up to rounding.
- The largest difference is 0.019–0.044, on an action scale of about 2.
- Later steps of the 50-step chunk can differ more (up to 1.7 for A0), because small rounding changes grow there.
- Only the first 10 steps are executed (`n_action_steps=10`).

**Scope:** this is a laptop RTX 5090, not a Jetson. Speed is measured at batch 1; the eval runs at batch 10.

## 8. Every number and its source file

All paths are under `speed/results/`. `final_2026-11-07/final_numbers.json` collects every number; `speed/scripts/summarize_final.py` builds it. Run logs are not in the repository; a re-run writes them to `$LOG_DIR`.

| Numbers | Source |
|---|---|
| Speed, energy, ratios, session first calls (pass 1) | `final_2026-11-07/session/session.json`; per-call times `session/block_latencies_ms.json`; GPU clocks `session/gpu_samples.csv` |
| Same, pass 2 | `final_2026-11-07/session_pass2/` |
| Startup, peak memory, graph breaks, recompiles | `final_2026-11-07/startup/{A0,A1,C}_eager.json` and `B_{cold,fill,warm,warm2,diag}.json` |
| TF32 check | `final_2026-11-07/precision_check/precision_check.json` |
| Co-location check | `final_2026-11-07/colocation_check/run1–6_*.json` |
| Closed-loop success, per episode | `final_2026-11-07/libero_runs/final_2026-11-07_A1_eager_steps10/eval_info.json` and `final_2026-11-07/libero_runs/final_2026-11-07_B_tuned/eval_info.json` |
| Paired stats | `final_2026-11-07/libero_compare_B_vs_A1.txt`, `libero_compare_vs_talk_baseline.txt`, `libero_compare_B_today_vs_talk_B.txt` |
| Closed-loop GPU energy | `final_2026-11-07/libero_energy_final_2026-11-07_{A1_eager_steps10,B_tuned}.jsonl` |
| Versions | `final_2026-11-07/versions.json` |
| Sanity tests | `final_2026-11-07/pytest_smolvla.log`; `speed/scripts/tuple_kv_parity_check.py` |
| 31 Oct numbers | `reference_2026-10-31/` (per-setup stats and first-call actions, and the two 31 Oct closed-loop runs' `eval_info.json`) |

Scripts in `speed/scripts/`:
- `run_all.sh` runs everything below in order, under the GPU lock.
- `run_session.sh` runs `energy_session_final.py`, then `precision_check.py`.
- `run_startup.sh` runs `startup_final.py`.
- `run_libero_final.sh` runs `libero_eval_final.py` and reads the energy counter with `nvml_energy.py`.
- `run_colocation_check.sh` runs `colocation_check.py`.
- `summarize_final.py` builds `final_numbers.json`.

## 9. Follow-up checks (8 Nov 2026, 02:35–03:37 IST)

Same laptop. Fixes and old code ran in separate scratch checkouts; the final tested checkout was not edited. Results: `speed/results/followups_2026-11-08/` (`SUMMARY_followups.txt` is the original short report). Scripts: `speed/followups/`.

**1. PR #4634 test fix** (`speed/patches/pr4634_test_fix.diff`, `task1_test_fix.json`)
- Change in `test_smolvla_tuple_kv_parity`: `dynamic_kv.crop(prefix_len)` becomes `for layer in dynamic_kv.layers: layer.crop(prefix_len - layer.get_seq_length())`. This is the same form main uses in `denoise_step` since #5220 (`4598a0bb`).
- Why: transformers 5.22 changed `DynamicLayer.crop`'s argument from a target length to a number of tokens to remove. A positive value now raises `ValueError` ("Calling `crop` with a positive value ... is not supported"), and `crop(0)` does nothing. Source: `cache_utils.py` lines 163–180 in 5.22.1, and 139–151 in 5.5.4.
- Results:
  - Current main + PR + patch, with the fix: 2/2 in `test_smolvla_compile_parity.py`, and 15/15 in `tests/policies/smolvla`.
  - The PR's own stack (torch 2.11, transformers 5.5.4), unchanged: passes at `a932c505` and at the head `2d200334`. So the test fails only on current main's transformers.
  - The fix needs transformers 5.22 or newer. On 5.5.4, `crop(0)` empties the cache and the test fails with a shape mismatch.

**2. Faster-startup compile modes for B, current stack** (`compile_modes_summary.json`)

| Mode | Cold first call (caches off) | Warm | Median ms (4 runs) | p99 ms | J/decision (above idle) |
|---|---|---|---|---|---|
| max-autotune | 158.9, 155.5 s (fill 158.5) | 5.9 s | 14.02 (13.46–14.59) | 16.6–18.7 | 1.45 (1.16) |
| default | 31.0, 30.1 s (fill 32.0) | 5.6 s | 15.43 (14.86–16.04) | 16.5–20.7 | 1.53 (1.19) |
| reduce-overhead | 28.1, 27.7 s (fill 30.1) | 5.6 s | 14.84 (14.37–15.35) | 16.4–18.4 | 1.54 (1.20) |

- Every mode: 11 graphs, 6 graph breaks, no recompiles.
- Parity against max-autotune, first 10 steps (mean / max abs difference): default 0.0055–0.0062 / up to 0.024; reduce-overhead 0.0055 / up to 0.016. For scale: max-autotune against itself in another process gives 0.005–0.008 / 0.018–0.021, and eager C gives 0.0076 / 0.032.
- reduce-overhead starts cold about 5.6× faster (28 s vs 157 s). It costs about 6% in median time and in energy per decision. Warm starts are about 5.6–5.9 s in every mode.
- Closed-loop success was not run for the other modes.

**3. The 31 Oct stack (OLD) vs the current one (NEW), same conditions** (`stack_compare_summary.json`)
- Setup: alternating fresh processes, mirrored order OLD NEW NEW OLD OLD NEW, 200 calls each, TF32 on for both. Cool-down between processes: at least 45 s, and until the CPU package is at most 62 °C.

| Setup | OLD median (3 runs) | NEW median (3 runs) | NEW/OLD by pair | p99 mean OLD → NEW |
|---|---|---|---|---|
| A0 | 107.0 (107.0, 106.6, 107.4) | 111.0 (110.3, 109.4, 113.2) | 1.030, 1.027, 1.054 | 152 → 156 ms |
| B | 13.50 (13.49, 13.38, 13.64) | 13.58 (13.54, 13.53, 13.66) | 1.004, 1.011, 1.001 | 15.9 → 16.7 ms |

- Where A0's extra time is:
  - GPU kernel time fell (41.8 → 37.0 ms per call). The vision encoder got faster (19.6 → 16.5 ms).
  - CPU-side time grew (wall minus kernel time 65.2 → 74.0 ms). Most of it is in the 10-step denoise loop (78.5 → 85.3 ms). The VLM pass changed little (8.7 → 9.2 ms). Kernel launches per call: 14,190 → 14,403.
  - So the newer software adds CPU overhead to the eager path, not slower kernels.
- Temperature does not explain the OLD/NEW gap. Both stacks ran at about the same CPU package temperature (76–81 °C mean during the calls) and GPU temperature (56–58 °C).
- Cold compile of B (fresh empty cache): OLD 96 s, NEW 168 s. The startup regression comes from the stack.
- Meaning for the final numbers:
  - The stack explains about +4% of A0's +17% and about +0.5% of B's +10%.
  - The rest comes from measurement conditions. The old stack itself ran 107.0 ms on 8 Nov vs 101.3 ms on 31 Oct. The four-setup session added about 7% more for eager (118.9 vs 111.0 ms in fresh processes).
  - The A0/B speed-up is stable: 8.2× in fresh processes on the current stack, and 8.0–8.25× in the sessions.

**4. Does Tuned need PR #4634?** No: 15.4 ms on plain main. See [TUNED_WITHOUT_PR4634.md](TUNED_WITHOUT_PR4634.md).

## 10. Camera batching patch check (7 Nov 2026)

`speed/patches/batch_cameras.patch` is my port of the camera-batching pattern from LeRobot's merged PI0 change (PR #4426) to SmolVLA: both camera images go through the vision encoder in one call instead of one call each. It was not proposed upstream. The A/B timing script is `speed/scripts/batch_cameras_ab_v2.py`; its results (timings and checks) are in `speed/results/batch_cameras_2026-11-07/`.
