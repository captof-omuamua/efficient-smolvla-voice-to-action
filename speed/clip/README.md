# Demo 1 clip: speed and energy meters, side by side

The clip in `media/meters_side_by_side_clean.mp4` (still: `media/meters_side_by_side_clean.png`) shows one LIBERO episode from both final closed-loop runs, with counters for model think time and GPU energy. The scripts here build it from the measured result files and the runs' episode videos.

- **Original** = A1: eager, 10 denoising steps, no blank camera slot, cameras encoded one call each. Run `final_2026-11-07_A1_eager_steps10`.
- **Tuned** = B: whole-model `torch.compile` (LeRobot's default mode, `max-autotune`), 1 denoising step, no blank slot, cameras batched. Run `final_2026-11-07_B_tuned`.
- Both runs: checkpoint `lerobot/smolvla_libero` (pinned revision), `n_action_steps=10`, seed 1000, the same 400 start states.
- Both use the same code (the patched LeRobot checkout, see the main README) and MuJoCo 3.8.1.
- The earlier baseline definition with a blank camera slot appears in `session.json` as A0 (118.93 ms, 11.468 J). It is not shown here.

## Files

| File | What it is |
|---|---|
| `scripts/measured.py` | Numbers, file paths and checks (`check_sources()`) |
| `scripts/draw.py` | Layout |
| `scripts/check_videos.py` | Checks all 800 episode videos and writes `episode_steps.json` |
| `scripts/make_meters_clip.py` | Builds the clip and its last-frame still |
| `episode_steps.json` | Steps (= frames), success and fps check for all 400 episodes of both runs; used to choose the scene |

The scripts write `meters_side_by_side.mp4` (16.1 s, 1920×1080, 20 fps, one frame per control step = real time, H.264, CPU-encoded) and `meters_side_by_side.png`. The file in `media/` is a later "clean" version of that clip; the edit that made it is not in this repository.

## On screen, and where each number comes from

All paths are under `speed/results/final_2026-11-07/`.

| Shown | Value used | Source |
|---|---|---|
| Original 111.0 ms per decision | 110.99 ms | `session/session.json`, `summary.A1.latency.median_ms` (pass 1) |
| Original 11.0 J per decision | 11.002 J | same file, `summary.A1.j_per_call_total.mean` |
| Tuned 14.4 ms | 14.41 ms | same file, `summary.B.latency.median_ms` |
| Tuned 1.4 J | 1.44 J | same file, `summary.B.j_per_call_total.mean` |
| 7.7× less GPU think time | 7.7 | same file, `ratios.A1_over_B.median_ms` |
| 7.6× less GPU energy | 7.64 | same file, `ratios.A1_over_B.j_per_call_total.overall` |
| 82.00% vs 82.00% overall success | 82.0, 82.0 | each run's `eval_info.json`, `overall.pc_success` (400 episodes) |
| Instruction text | | each run's `task_descriptions.json` (identical in both) |

**Episode totals (25 decisions each):**
- GPU think time: 2.77 s vs 0.36 s.
- GPU energy: 275.1 J vs 36.0 J.

Counters at eval frame i: decisions = i // 10 + 1; think time = decisions × median ms; energy = decisions × J per decision. Labels are rounded; all sums use the unrounded JSON values.

`check_sources()` in `scripts/measured.py` stops the build if any of these fail:
- `session.json` and `final_numbers.json` (`speed_energy.pass1`, `ratios.pass1.A1_over_B`, `libero`) disagree;
- the shown ratios don't follow from the shown per-decision values: 110.99 / 14.41 must round to 7.70, and 11.002 / 1.44 to 7.64;
- the session's A1 and B settings (compile, denoising steps, blank slot, camera batching) differ from what each LIBERO run's log reports;
- the two runs' task lists differ.

Footer line 1 is set in Liberation Sans Narrow, at 28 px height, so it fits the 1920 px width.

## The chosen episode

**LIBERO-Long (`libero_10`) task 2, episode 8: "turn on the stove and put the moka pot on it".**
- The first-choice scene (task 8, episode 0) succeeds in both runs, but takes 422 vs 382 steps. The shorter side would freeze 2 s early, which could read as "the tuned robot moves faster".
- 306 episodes succeed in both runs, but only 44 of them take the same number of steps, and none of those lasts 15–25 s (300–500 steps) at real time. This one is the longest of the 44.
- Both runs succeed in exactly 242 steps, so both make 25 decisions, and the only difference on screen is the cost per decision. At real time it lasts 12.1 s; with the 1 s intro and 3 s outro stills, the clip is 16.1 s.
- Same start state: the first frames differ by 0.56 out of 255 on average, which is compression noise.
- It is easy to follow: the burner visibly turns red, then the moka pot goes onto it. The two runs track each other closely all the way.

## Frames and steps on the new simulator core (LeRobot #4888)

**Still one video frame per control step (50 ms), with a decision every 10 frames. The files are tagged 20 fps, so they play at real time.** (Eval videos from before #4888 were tagged 80 fps.)
- `scripts/lerobot_eval.py` in the patched LeRobot checkout:
  - one frame after `env.reset()` (lines 233–235) and one after every `env.step()` (lines 341–343);
  - each video is cut to `frames[: done_index + 1]` (line 668);
  - it is written at `env.unwrapped.metadata["render_fps"]` (line 669).
- `sim/local_vector_env.py`:
  - `render_fps` is now the env's `fps`, the 20 Hz control rate (line 304);
  - finished environments are no longer stepped;
  - frames come from each environment's last camera observation (`sim/types.py`, `frames()`).
- `sim/integrations/libero.py`: the step limits are 280 / 280 / 300 / 520 (line 134). The 10 no-op settle steps run inside `reset()` (line 387) and are not in the videos.
- SmolVLA's action queue (`modeling_smolvla.py`, lines 278–288) refills every 10 steps.
- Both eval logs show `'fps': 20`, `n_action_steps: 10`, seed 1000. Neither run logged another process on the GPU.
- Data check (`check_videos.py`): all 800 videos are tagged 20/1 fps. In each run, all 72 failed episodes have exactly their suite's step limit as their frame count.

## Caveats

- Laptop RTX 5090, not Jetson.
- Think time and energy are batch-1 medians and means from one session (pass 1, ABBA-style blocks, NVML energy counter). The closed-loop eval ran at batch 10.
- p99 is 146.98 vs 16.8 ms.
- "Total" energy includes the GPU's idle power. Above idle, the ratio is 7.55×. Across rounds the total-energy ratio ranged 7.45–8.17×.
- The counters are decisions × measured per-decision numbers, not measurements taken during this episode.
- The simulator pauses while the model thinks, so both robots run on the same clock. The robot itself is not faster.
- The scene was picked because both runs succeed with equal step counts. Overall success is 82.00% vs 82.00% (paired over 400 episodes: 22 episodes flip each way).

## Rebuild (CPU only, about 1 min)

The episode videos are not in this repository (they are large). Re-run the two LIBERO evals first (main README, Reproduce), then point the scripts at that run folder and its logs:

```bash
cd speed/clip
export LIBERO_RUNS=$OUT_DIR/speed/libero_runs   # holds final_2026-11-07_*/videos/
export LOG_DIR=$LOG_DIR                         # holds final_2026-11-07_libero_*.log
CUDA_VISIBLE_DEVICES="" nice -n 19 $PYTHON scripts/check_videos.py       # ~25 s
CUDA_VISIBLE_DEVICES="" nice -n 19 $PYTHON scripts/make_meters_clip.py   # ~10 s
```

The scripts need only av, numpy and Pillow; they never import torch. Output goes to `$OUT_DIR/clip/` (default `runs/clip/`).
