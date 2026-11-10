# One GPU, three models: plan and results (10 Nov 2026)

A separate timing test from the voice-robot runs. Code: `voice/run_share.sh`, `voice/share_policy_loop.py`, `voice/share_voice_load.py`, `voice/share_gate.py`, `voice/analyse_share.py`, `voice/plot_share.py`, `voice/make_share_audio.sh`. Results: `voice/results/share/`. Chart: `voice/figures/share_p99.png`.

Words: **99th-percentile latency (p99)** = the slow end: 99 of 100 decisions are faster than this. **Late** = a decision that finishes more than 50 ms after its beat (one control step at 20 Hz); that time includes the short delay before the model call starts (median 0.2 ms, worst 3.1 ms). **Turn-taking** = a voice job may start only in the gap between two policy decisions, and only if it will finish before the next one (`share_gate.py`, 35 lines).

## Results

**Question.** The robot policy (Tuned SmolVLA) decides on a fixed beat, once every 500 ms. If speech-to-text (Whisper) and a language model (Qwen3.5-4B) work on the same GPU from a second program, do the policy's decisions get slower or late? Does turn-taking fix it, and what does it cost the voice side?

Timing only: a fixed synthetic observation, no simulator, no robot. Bars, regimes and rules were written first (below, "Pass bars"). Registered run: `voice/results/share/tuned_main` (03:17 to 04:43 IST, 5 of 5 rounds, 40 two-minute blocks, 0 errors, 0 blocks dropped). The timing numbers below were recomputed from the raw logs in a separate check; all matched. What that check did not cover is listed under Caveats.

| # | Bar | Passes if | Result |
|---|---|---|---|
| 1 | Timer check, robot loop alone | median within ±10% of 13.97 ms, p99 within ±15% of 16.01 ms | **Pass**, twice: 14.89 and 16.41 ms on any CPU core; 13.70 and 14.87 ms on fast cores only |
| 2 | Sharing slows the policy | p99 at least 1.25x `idle`, interval wholly above 1 | **Pass**: 2.02x to 2.80x in all four regimes without turn-taking |
| 3 | Fix, speed | p99 no more than 10% above `idle` | **Pass**: 14.78 ms and 14.85 ms against 14.83 ms |
| 4 | Fix, deadline | 0 late decisions | **Pass**: 0 of 1,164 and 0 of 1,180 |
| 5 | Fix, cost | added wait per spoken command: mean at most 60 ms, worst at most 200 ms | **Pass**: mean 23.3 ms, worst 172.9 ms (197 commands) |
| 6 | Turn-taking kept | 0 voice jobs overlap a policy call | **Pass**: 0 of 6,645 |

**Policy decision time (model call), per regime.**

| What the voice program does | Turn-taking | Decisions | Median | p99 | Worst | Late | p99 against idle [95%] |
|---|---|---|---|---|---|---|---|
| Nothing (models loaded): `idle` | – | 1,180 | 14.35 ms | 14.83 ms | 15.35 ms | 0 | – |
| Whisper back to back | no | 1,180 | 31.18 ms | 40.57 ms | 42.09 ms | 0 | 2.74x [2.71, 2.75] |
| Language model back to back | no | 1,181 | 39.18 ms | 41.47 ms | 42.09 ms | 0 | 2.80x [2.78, 2.81] |
| Busy user, language model every time | no | 1,163 | 14.38 ms | 34.61 ms | 36.10 ms | 0 | 2.33x [2.29, 2.37] |
| Busy user, word rules first | no | 1,163 | 14.33 ms | 29.99 ms | 35.99 ms | 0 | 2.02x [1.41, 2.26] |
| Busy user, language model every time | yes | 1,164 | 14.35 ms | 14.86 ms | 15.99 ms | 0 | 1.00x [1.00, 1.00] |
| Busy user, word rules first | yes | 1,164 | 14.34 ms | 14.78 ms | 15.30 ms | 0 | 1.00x [0.99, 1.00] |
| Whisper and language model back to back | yes | 1,180 | 14.34 ms | 14.85 ms | 15.08 ms | 0 | 1.00x [1.00, 1.01] |

"Busy user" = a spoken command every 2 to 4 s. Chart: `voice/figures/share_p99.png` (made by `plot_share.py`). Its bars are the p99 column above; the thin line on each bar is the range of the five rounds.

**What the numbers say.**
- **Sharing slows the policy.** With a voice model working flat out beside it, p99 went from 14.83 ms to 40.57 ms (Whisper) and 41.47 ms (language model). The median rose too, to 31.2 and 39.2 ms.
- **A busy user is enough.** p99 was 34.61 ms with the language model answering every command, and about 30 ms with word rules first. That second number is shaky (18.6 to 32.8 ms from round to round), because only 2.9% of decisions met a voice job.
- **The slowdown sits in the decisions that met a voice job.** Busy user, language model every time: the 66 decisions that met a running voice job had a median of 32.4 ms; the 1,097 that met none had a median of 14.4 ms and a p99 of 14.9 ms, the same as `idle`. With word rules first: 24.0 ms for the 34 that met one. The 1,129 that met none had a median of 14.3 ms, with a slightly longer slow end than `idle` (p99 15.2 ms and worst 16.6 ms, against 14.8 and 15.4 ms). The control: with turn-taking no decision met a voice job, and none slowed down.
- **It was never late, but only just.** 0 of all 10,279 logged decisions finished more than 50 ms after their beat. The closest finished 48.9 ms after it. The five slowest calls (42.7 to 48.65 ms) all came in the first 2 s of a language-model block, which the 2-second rule leaves out of the table; the cause is not known.
- **Turn-taking fixes it.** p99 back to 14.78 to 14.86 ms in all three turn-taking regimes, against 14.83 ms for `idle`. 0 of 6,645 voice jobs overlapped a policy call. The closest one started 9.2 ms after a decision ended, and every one ended at least 48 ms before the next decision.
- **What the fix costs.** A spoken command waited 23.3 ms longer on average and 172.9 ms at worst (41.1 and 172.4 ms with the language model every time). The rule's own worst case is about 175 ms. Flat out, the voice side finished 5.0 Whisper and 5.0 language-model jobs per second and was working about 78% of the time, against about 100% without the rule.
- **Word rules help, but are not the fix.** They cut the decisions that met a voice job from 66 to 34 of 1,163. 27 of those 34 met Whisper, which still runs for every command. Their real gain is speed for the user: median time from a command's arrival to its routed answer was 155 ms with the language model every time and 60 ms with word rules first (they answered 142 of 197 commands). Turn-taking barely moved that median (62 ms) but raised the worst from 178 to 321 ms.

**Original policy, for contrast.** The same test with the Original SmolVLA (no compile, 10 denoising steps): run `voice/results/share/original_contrast` (04:46 to 05:17 IST, 3 of 3 rounds, 12 two-minute blocks, 0 errors). No pass bars apply to this run; its report prints bar lines, and they are information only.

| What the voice program does | Turn-taking | Decisions | Median | p99 | Worst | Late | p99 against idle [95%] |
|---|---|---|---|---|---|---|---|
| Nothing (models loaded): `idle` | – | 708 | 88.84 ms | 109.71 ms | 132.78 ms | all | – |
| Language model back to back | no | 708 | 145.56 ms | 155.73 ms | 160.88 ms | all | 1.42x [1.25, 1.52] |
| Busy user, language model every time | no | 695 | 89.93 ms | 131.38 ms | 141.05 ms | all | 1.20x [1.06, 1.29] |
| Busy user, word rules first | yes | 696 | 88.42 ms | 97.53 ms | 114.53 ms | all | 0.89x [0.78, 0.97] |

- **It is late with or without sharing.** All 3,607 logged decisions finished more than 50 ms after their beat. With the robot loop alone the median was 88.9 ms.
- **Sharing costs it more time, at a smaller ratio.** With the language model flat out, the median rose by 57 ms (88.8 to 145.6 ms), against 25 ms for the tuned policy (14.4 to 39.2 ms). The p99 ratio is smaller (1.42x against 2.80x) only because the starting point is larger: p99 rose by 46 ms here and by 27 ms for the tuned policy. Busy user: the 56 decisions that met a voice job had a median of 111.5 ms, against 89.7 ms for the 639 that met none.
- **Turn-taking still keeps the two apart, at about four times the average cost.** 0 of 148 voice jobs overlapped a policy call, and the policy always finished inside its guard (latest 114.7 ms after the beat; guard 170 ms; without turn-taking its calls reached 161 ms). A spoken command waited 100.4 ms longer on average and 317.1 ms at worst (119 commands), against 23.3 and 172.9 ms beside the tuned policy. Those 119 commands are the same sentences, in the same order, as in the first three tuned rounds; on those the tuned average was 23.0 ms, so the ratio is 4.4x. 79 of 119 commands had to wait, against 61 of 197. The reason is in the rule: the guard after each beat is 1.25 × the policy's slowest saved call (fixed in the pass bars), so 170 ms instead of 25 ms. A Whisper job may then start in only 220 ms of each 500 ms beat instead of 365 ms (language model: 180 instead of 325 ms). Median time from a command's arrival to its answer: 162 ms, against 62 ms beside the tuned policy.
- **The 0.89x is not read as a speed-up.** The medians are the same (88.4 against 88.8 ms). The slow end was shorter with turn-taking in this run: 4 of 696 calls took over 100 ms, against 17 of 708 in `idle` and 12 of 720 with the robot loop alone, and its p99 (97.5 ms) is below `idle` (109.7 ms) and the robot loop alone (106.5 and 111.7 ms). The reason is not known and no bar was set for it, so nothing is claimed from it. Each p99 here rests on the slowest 7 or so of about 700 decisions.

**Found on the way (method).**
- **Whisper had a slow mode.** Alone, with default settings, Whisper took about 60 ms or about 150 ms at random. Cause: a CPU step (turning sound into features) used 24 threads and sometimes took 90 ms instead of 5 ms. Fix: 1 CPU thread for the voice program. Found and fixed before the first shared run (`voice/diagnostics/whisper_slow_mode_2026-11-10.txt`).
- **The kind of CPU core matters.** The first timer check showed the policy's median at 13.65 ms on the 8 fast cores and 15.51 ms on the 16 slow ones. That 14% gap could have hidden or faked an effect, so each program got its own fast cores (robot loop 0 to 3, voice 4 to 7) and the timer check was repeated.
- **The power cap alone does not explain it.** The GPU reported its software power cap in every flat-out block. The cap is not needed for the slowdown: in the busy-user regimes, decisions that met a voice job with the cap flag off were still slow (median 26.7 ms, worst 36.1 ms, 76 decisions). And it does not slow the policy on its own: decisions with the flag on and no voice job were not slow (median 14.4 ms, 318 decisions), and flat out with turn-taking the flag was on for 97% of decisions while p99 stayed at 14.85 ms. Whether the cap adds to the slowdown is not settled: the 24 decisions that met a voice job with the flag on had a median of 33.8 ms, but 21 of them came from the regime with the language model every time, where such decisions are slower anyway. The flag is read once, just after each call, so it shows that moment and not the whole call.
- **One pattern is not explained.** Even with the robot loop alone, the call time moves between two levels (about 13.5 and 14.4 ms). It does not follow the CPU core, CPU speed, GPU clock, temperature or the start delay. The higher level is more common when the power-cap flag is on (88% of 136 `idle` decisions, against 57% of 1,044 with the flag off), but both levels appear either way and the cause is not known. This is why a difference in medians under 7% is not read as an effect: `idle` (14.35 ms, voice models loaded) against the robot loop alone (13.62 and 13.71 ms) is a 5% gap, so nothing is claimed from it. The flag does not account for that gap: with the flag off, the higher level was still more common in `idle` (57%) than with the robot loop alone (43% before the shared blocks, 39% after). One candidate was not tested: the desktop draws on this GPU too (see Caveats).

**Caveats.**
- One laptop GPU (RTX 5090 laptop, 24 GB), not a Jetson. One night, one 86-minute run. The 95% intervals cover variation inside this run only. Each p99 rests on the slowest 12 or so of about 1,180 decisions.
- Timing only. The observation is fixed and synthetic, the speech is computer-generated and there is no simulator. Nothing here says whether 15 ms against 40 ms changes robot task success.
- Two separate programs under the GPU's default sharing: it switches between programs in short time slices (the slice length was not measured). Other ways to share were not tested and may protect the policy at a lower cost: NVIDIA's Multi-Process Service, a shorter time slice, or all three models in one program (threads, or stream priorities).
- Each program had its own CPU cores. With shared cores the slowdown may be larger.
- The language model ran slow reference code for some layers (two speed-up packages are not installed), and its job here is a 4-token choice. A faster stack would hold the GPU for less time. Free-text answers would hold it for longer and would need splitting to fit the gap.
- Each busy-user regime had 197 commands; the worst waits come from that many.
- Turn-taking needs a shared clock. Here both programs read the same machine clock and the beat is fixed. On a real robot the policy would have to publish its schedule.
- Turn-taking is voluntary; the GPU does not enforce it. It held here because the policy always finished inside the 25 ms guard (worst call 16 ms) and no voice job ran over its budget (worst 87 of 100 ms and 107 of 140 ms). A voice job that ran far over its budget would meet the next decision, and nothing in the running programs would stop it.
- Only compute programs on the GPU were watched (that is what the GPU library NVML lists). The desktop also draws on this GPU: two spot checks after the runs (05:24 and 05:25 IST) showed `gnome-shell` holding 6 MiB there, and the GPU 1 to 6% busy with no test running. How much it used during the runs is not known; it was not logged.
- Code state: the robot loop ran on the 31 Oct policy stack (a LeRobot checkout at PR #4634 head `8c4f33e5` with uncommitted changes, in that PR's pinned venv with torch 2.11), not on the final tested checkout used for the speed and voice-robot results. The voice folder was not under version control at the time. That the bars came first rests on file times (thresholds in `analyse_share.py` since 03:02:56, before the first two-program test at 03:11; the pass-bar file last saved at 03:16:59, 5 s before the registered run) and on the sha256 fingerprints in the run folder.
- The separate check recomputed the timing numbers of the registered run and the two timer checks from the raw logs. It checked the Whisper slow-mode figures against the diagnostics file only. It did not re-run the unit tests, re-check the 8 Nov baseline used in bar 1, or check the hardware description. It also recomputed the Original contrast run from its raw logs, and the numbers matched; one interval differs by resampling noise (lower limit 0.80 against the 0.78 printed here). It checked the chart as a PNG, not the PDF. Its scripts are not in this repository.
- The Original contrast has 3 rounds and 119 commands per busy-user regime, against 5 rounds and 197 for the tuned policy, and it ran after the tuned run, not mixed with it. The two runs are compared side by side, not in one randomised design. The commands are the same sentences as in the first three tuned rounds, but they arrive at different points of the beat.

Other run folders in `voice/results/share/`: `cal_2026-11-10` and `cal_pinned_2026-11-10` (the two timer checks), `smoke_1` and `smoke_original` (smoke tests, not results).

## Pass bars (written 10 Nov 2026, 02:35 IST, before any run with two programs on the GPU)

This published copy has edited paths and wording; the bars and times are unchanged. The original plan files (`PREREG.md`, `SHARE_PLAN.md`) are not in this repository, so their sha256 values in the run folders' `fingerprints.txt` will not match this copy.

**Set-up.**
- Robot loop: SmolVLA, compiled with compile mode `default` (Tuned in the speed results uses `max-autotune`), 1 denoising step, on the 31 Oct stack (see Caveats), the same code and settings as a saved 8 Nov tail-latency run (`c2_paced500`, not in this repository; fixed synthetic observation, so this times the model call only, not cameras or a simulator). One decision every 500 ms, because the LIBERO runs used 10-step action chunks at 20 Hz.
- Voice load, in a second program: Whisper large-v3-turbo and Qwen3.5-4B, with the frozen `router.py` (sha256 in [ROUTER_PREREG.md](ROUTER_PREREG.md), unchanged).
- Laptop RTX 5090. GPU sharing mode `Default` (programs take turns), time slice `Default`. No setting is changed.
- Both programs read the same system clock (`CLOCK_MONOTONIC`). Beats sit at whole multiples of 500 ms on it.

**Regimes** (what the voice program does in a block).

| Name | Voice work | Turn-taking |
|---|---|---|
| `idle` | models loaded, no work | – |
| `whisper_flat` | Whisper back to back | no |
| `llm_flat` | language model back to back | no |
| `user_llm` | a spoken command about every 3 s: Whisper, then the language model every time | no |
| `user_router` | the same, but word rules first (the router) | no |
| `user_llm_gate` | `user_llm` | yes |
| `user_router_gate` | `user_router` (the full fix, realistic load) | yes |
| `flat_gate` | Whisper and language model alternating back to back (the full fix, worst load) | yes |

- Commands arrive 2 to 4 s apart (uniform, seeded): a deliberately busy user, so collisions are common enough to measure.
- Audio is computer-generated speech (flite, `make_share_audio.sh`: the 10 LIBERO-Goal command sentences from `tasks.py`, each plain and with "please" in front; 20 clips in `voice/data/share_audio/`). It is there only to make Whisper do real work. The sentence that gets routed is drawn (seeded) from the 130 router test sentences, so the mix of rule answers and language-model answers is the measured one. Nothing here re-scores that test set.
- Word rules skip only the language model. Whisper still uses the GPU for every command.

**Turn-taking rule (fixed now).** A voice job may start only if it starts at least `guard_after` after a beat and its time budget ends at least 10 ms before the next beat. Otherwise it waits for the next beat plus `guard_after`. Budgets: Whisper 100 ms, language model 140 ms. `guard_after` = 1.25 × the policy's saved worst alone time, rounded up to 5 ms: 25 ms for Tuned (worst 18.07 ms), 170 ms for Original (worst 132.36 ms). If the smoke test shows a job type over its budget in more than 1% of jobs, that budget is raised once to 1.2 × its smoke p99 (rounded up to 10 ms) and noted here before the main run.

**Layout.** 5 rounds. Each round runs all 8 regimes once, 2 minutes each, in a seeded random order (seed 20261110), so heat and drift spread evenly. The first 2 s of each block are dropped. Before and after: 3 minutes of the robot loop alone. If the run is cut short, only complete rounds count, and at least 3 are needed.

**Measures, per regime.**
- Policy: model-call time (the 8 Nov method): median, p99, worst. Late share: decisions that finish more than 50 ms (one control step) after their beat, with a 95% upper limit (Wilson).
- p99 ratio against `idle`, with a 95% interval (moving-block bootstrap: blocks of 20 decisions, resampled inside each 2-minute block, 2,000 resamples, seed 0).
- Cause check: decisions split into "a voice job was running" and "none was running".
- Voice: job time per model, wait caused by turn-taking (mean, p99, worst), time from command arrival to routed answer, jobs per second in flat-out regimes.
- Sanity: answers for the same clip or sentence must be identical in every regime.

**Bars (Tuned).**

| # | Bar | Passes if |
|---|---|---|
| 1 | Timer check, robot loop alone (600 decisions, before any sharing) | median within ±10% of 13.97 ms and p99 within ±15% of 16.01 ms |
| 2 | "Sharing slows the policy" counts as shown | p99 ratio ≥ 1.25 in an unprotected regime, with its interval wholly above 1 |
| 3 | Fix, speed | in `user_router_gate` and `flat_gate`: p99 no more than 10% above `idle` |
| 4 | Fix, deadline | in `user_router_gate` and `flat_gate`: 0 late decisions |
| 5 | Fix, cost | in `user_router_gate`: added wait per command, mean ≤ 60 ms and worst ≤ 200 ms (predicted from the rule: mean about 26 ms, worst about 175 ms) |
| 6 | Turn-taking kept | 0 voice jobs in gated regimes overlap a policy call |

**If a bar fails.** Bar 1: check for other GPU programs and throttle flags, fix what is found, re-run once; if it still fails, go on with the alone numbers of the night as the baseline and report the mismatch. Bars 2 to 6: reported as failed. A "no" on bar 2 is a result too: it would mean sharing is safe for this policy on this GPU.

**Exclusions (fixed now).** A block is dropped only if an unexpected GPU program was seen during it or the voice program logged an error in it. Dropped blocks are listed. No decision is dropped as an outlier.

**Original contrast.** After the Tuned run, if time allows: eager, 10 steps, 3 rounds of `idle`, `llm_flat`, `user_llm`, `user_router_gate`. No pass bars; it is a contrast. Original alone already takes about 91 to 132 ms, so it cannot meet 50 ms.

**Limits.** Laptop GPU, not a Jetson. One machine, one night. Timing only, no robot success. Synthetic observation and synthetic voice. Turn-taking here uses a shared clock; on a real robot the policy would publish its schedule.

**Fingerprints.** Each run folder stores the sha256 of the pass-bar file and of every script used, taken at the start of the run (`fingerprints.txt`). The pass-bar and build-plan files named there are the originals of this document and are not in the repository; the scripts are.

**Amendment before the first shared run (10 Nov 2026, 02:45 IST).** Whisper alone, with default settings, took about 60 ms or about 150 ms at random: 67 of 150 calls went over its 100 ms budget. Cause: the CPU step that turns sound into features used 24 threads and sometimes took about 90 ms instead of 5 ms. The GPU part was steady (median 57 ms, worst 63 ms). Fix: the voice program's maths libraries use 1 CPU thread. Alone after the fix: median 67 ms, p99 75 ms, worst 82 ms, 0 of 165 calls over 100 ms. The robot loop, the regimes, the budgets and the bars are unchanged. Evidence: `voice/diagnostics/whisper_slow_mode_2026-11-10.txt`.

**Second amendment before the first shared run (10 Nov 2026, 03:00 IST).** The timer check (bar 1) passed with the robot loop alone and free to run on any CPU core: median 14.89 ms, p99 16.41 ms, 600 decisions (`voice/results/share/cal_2026-11-10`). Its log also showed that the call time depends on the kind of CPU core the loop is on: median 13.65 ms on the 8 fast cores (206 decisions) and 15.51 ms on the 16 slow cores (394 decisions). The system moved the loop from fast to slow cores about 100 s into the run; the GPU clock did not change. That 14% gap is larger than bar 3's 10% allowance, so a regime could look slower or faster than `idle` only because of where the system put the loop. Change: each program keeps to its own fast cores (robot loop on cores 0 to 3, voice program on cores 4 to 7), set with `taskset` in `run_share.sh` and recorded in each run folder. The two programs then share the GPU, not CPU cores. The timer check is repeated with the loop pinned and judged by bar 1 again; both checks are reported. Regimes, budgets and bars are unchanged.

**Three readings made explicit (same time; the first two come from the code review).**
- The smoke-test budget rule looks at voice jobs in the turn-taking regimes only, because that is where a budget is used. In the other regimes a voice job can run at the same moment as the policy, which makes it slower for a different reason.
- Complete round: a round counts only if the robot loop's log covers every block in it from start to end. A run where the robot loop stops part-way is a run cut short. Blocks where a slow policy skips beats stay in.
- "The first 2 s of each block are dropped" also applies to voice statistics (by when the job or command arrives). The overlap checks (cause check, bar 6) use every voice job and every policy call in the block.

**Pinned timer check (10 Nov 2026, 03:06 IST; noted 03:11 IST, before the first shared run).** With the robot loop on cores 0 to 3: median 13.70 ms, p99 14.87 ms, worst 15.17 ms, 0 late, 600 decisions (`voice/results/share/cal_pinned_2026-11-10`). Bar 1 passed again. One thing seen and not explained: the call time moves between two levels about 7% apart (about 13.5 ms and about 14.4 ms), in stretches of about 12 to 100 s. It does not follow the CPU core, CPU speed, GPU clock, GPU power, temperature or how late the call starts after its beat. Nothing is changed for it. The stretches are shorter than a 2-minute block, the block order is random over 5 rounds, and the bars use p99, which was steady (worst per 100 decisions: 14.76 to 15.17 ms). Reading rule, fixed now: a difference in medians between two regimes of less than 7% is not read as an effect.

**Smoke test (10 Nov 2026, 03:11 to 03:16 IST; `voice/results/share/smoke_1`, 1 round of 30 s blocks).** It checked that the set-up runs end to end. In the turn-taking regimes 0 of 172 Whisper jobs and 0 of 163 language-model jobs ran over budget (worst 76.8 ms and 111.0 ms), so the budgets stay at 100 ms and 140 ms. The smoke test is not part of the results. Nothing in the pass bars changed after this point.

## How the parts fit (build plan, summarised)

- `share_policy_loop.py` (robot loop): loads and warms up the policy, writes `policy_ready.json`, then decides once at every whole multiple of 500 ms on `time.monotonic_ns()`. Each decision is a row in `decisions.csv` (`i,beat_ns,start_ns,end_ns,ms,sm_mhz,mem_mhz,temp_c,power_mw,throttle,cpu,cpu_khz,other_pids`). It stops when the `STOP` file appears and writes `policy_summary.json`.
- `share_gate.py`: `wait_ns(...)` returns how long a job must wait under the turn-taking rule; `wait_for_window(...)` sleeps until a job may start.
- `share_voice_load.py`: loads Whisper and the router, writes `voice_ready.json` and `plan.json` (the seeded regime order), then runs the blocks. Job rows go to `jobs.csv` (`round,regime,command,kind,item,ready_ns,start_ns,end_ns,wait_ms,result`, sentence ids only, never sentence text) and block rows to `blocks.csv`. Its maths libraries use 1 CPU thread.
- `run_share.sh NAME MODE` (MODE = `tuned`, `original` or `calibrate`): takes the GPU lock, stores fingerprints, the GPU sharing mode and the mains state, starts both programs on their own CPU cores, runs the alone phases (`phases.csv`) and the blocks, stops both, and runs `analyse_share.py`.
- `analyse_share.py RUN_DIR [--calibration] [--min-rounds 3]`: joins decisions to blocks (2 s rule), applies the complete-rounds rule, and writes `summary.json` and `REPORT.md` with the bar verdicts.
- Tests (no GPU): `voice/tests/test_share_gate.py` and `voice/tests/test_share_analysis.py`.

## Re-running it

- The robot loop of the registered run used the 31 Oct policy stack (see Caveats). `run_share.sh` reads `ROBOT_PYTHON` (that stack's Python), `ROBOT_PYTHONPATH` (its `src` folder) and `ROBOT_LEROBOT_DIR` (its checkout, for the recorded commit). Left unset, the robot loop runs in the patched LeRobot checkout's environment instead, which is a different stack from the registered run.
- `share_voice_load.py` draws its routed sentences from `voice/data/test.json`, which is held back ([ROUTER_PREREG.md](ROUTER_PREREG.md)). A re-run needs a sentence file of your own in the same format at that path.
- New runs go to `$OUT_DIR/share/<name>/`; the script refuses to write into an existing run folder.
- The script's comments name `PREREG.md` and `SHARE_PLAN.md`, the original plan files; their content is in this document and [ROUTER_PREREG.md](ROUTER_PREREG.md). A re-run's `fingerprints.txt` will list those two files as missing; the run itself is not affected.
