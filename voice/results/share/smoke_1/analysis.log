# One GPU, three models: <project>/voice-robot/results/share/smoke_1

Bars apply to the Tuned run; for the Original contrast they are information only (PREREG).

## Bars

| # | Bar | Verdict | Numbers |
|---|---|---|---|
| 1 | Timer check, alone | not shown | measured by the calibrate run (--calibration) |
| 2 | Sharing slows the policy | PASS | whisper_flat 2.68x [2.64, 2.71]; llm_flat 2.79x [2.75, 2.82]; user_llm 2.36x [1.09, 2.23]; user_router 1.01x [1.00, 1.01] |
| 3 | Fix, speed | PASS | user_router_gate p99 14.56 ms vs idle 14.69 ms = 0.99x (max 1.10x); flat_gate p99 14.47 ms vs idle 14.69 ms = 0.99x (max 1.10x) |
| 4 | Fix, deadline | PASS | user_router_gate 0 late of 56 (upper limit 6.42%); flat_gate 0 late of 56 (upper limit 6.42%) |
| 5 | Fix, cost | PASS | 11 commands: mean 18.3 ms (max 60 ms), worst 101.1 ms (max 200 ms) |
| 6 | Turn-taking kept | PASS | 0 of 335 GPU jobs overlap a policy call (policy calls: flat_gate 60, user_llm_gate 60, user_router_gate 60) |

## Run

- Complete rounds: 1 [0] of rounds seen [0] (need 1); regimes per round: idle, whisper_flat, llm_flat, user_llm, user_router, user_llm_gate, user_router_gate, flat_gate
- Rounds left out: 0
- CPU cores allowed: robot loop 0 to 3 (4 cores); voice program 4 to 7 (4 cores)
- Decisions: 591 total; 449 in kept blocks; 0 in dropped or incomplete blocks (not used); 80 in alone phases; 62 unassigned (not used); 0 unreadable rows
- Skipped beats, whole run: 0
- Unexpected GPU pids (samples): none
- Dropped blocks: 0

## Robot loop alone (phases)

| Phase | Decisions | Median | p99 | Worst | Late | Throttle flags |
|---|---|---|---|---|---|---|
| alone_pre | 40 | 13.29 ms | 13.71 ms | 13.74 ms | 0 | sw_power_cap 1 |
| alone_post | 40 | 13.43 ms | 14.47 ms | 14.47 ms | 0 | none |

## Policy decisions per regime

| Regime | Decisions | Median | p99 | Worst | Late | Late upper (95%) | p99 vs idle [95%] |
|---|---|---|---|---|---|---|---|
| idle | 56 | 14.33 ms | 14.69 ms | 14.72 ms | 0 | 6.42% | – |
| whisper_flat | 56 | 30.57 ms | 39.42 ms | 39.68 ms | 0 | 6.42% | 2.68x [2.64, 2.71] |
| llm_flat | 56 | 38.81 ms | 41.05 ms | 41.30 ms | 0 | 6.42% | 2.79x [2.75, 2.82] |
| user_llm | 57 | 13.60 ms | 34.60 ms | 35.36 ms | 0 | 6.31% | 2.36x [1.09, 2.23] |
| user_router | 56 | 13.87 ms | 14.77 ms | 14.81 ms | 0 | 6.42% | 1.01x [1.00, 1.01] |
| user_llm_gate | 56 | 13.54 ms | 14.47 ms | 14.49 ms | 0 | 6.42% | 0.99x [0.98, 0.99] |
| user_router_gate | 56 | 13.52 ms | 14.56 ms | 14.66 ms | 0 | 6.42% | 0.99x [0.98, 1.00] |
| flat_gate | 56 | 13.65 ms | 14.47 ms | 14.52 ms | 0 | 6.42% | 0.99x [0.98, 0.99] |

## Cause check: was a voice job (Whisper or LLM) running during the decision?

| Regime | Voice job running | Decisions | Median | p99 | Worst | Late |
|---|---|---|---|---|---|---|
| idle | yes | 0 | – | – | – | 0 |
| idle | no | 56 | 14.33 ms | 14.69 ms | 14.72 ms | 0 |
| whisper_flat | yes | 56 | 30.57 ms | 39.42 ms | 39.68 ms | 0 |
| whisper_flat | no | 0 | – | – | – | 0 |
| llm_flat | yes | 56 | 38.81 ms | 41.05 ms | 41.30 ms | 0 |
| llm_flat | no | 0 | – | – | – | 0 |
| user_llm | yes | 3 | 34.00 ms | 35.34 ms | 35.36 ms | 0 |
| user_llm | no | 54 | 13.58 ms | 15.82 ms | 16.11 ms | 0 |
| user_router | yes | 1 | 14.33 ms | 14.33 ms | 14.33 ms | 0 |
| user_router | no | 55 | 13.87 ms | 14.77 ms | 14.81 ms | 0 |
| user_llm_gate | yes | 0 | – | – | – | 0 |
| user_llm_gate | no | 56 | 13.54 ms | 14.47 ms | 14.49 ms | 0 |
| user_router_gate | yes | 0 | – | – | – | 0 |
| user_router_gate | no | 56 | 13.52 ms | 14.56 ms | 14.66 ms | 0 |
| flat_gate | yes | 0 | – | – | – | 0 |
| flat_gate | no | 56 | 13.65 ms | 14.47 ms | 14.52 ms | 0 |

## p99 per round

| Regime | Round 0 |
|---|---|
| idle | 14.69 ms |
| whisper_flat | 39.42 ms |
| llm_flat | 41.05 ms |
| user_llm | 34.60 ms |
| user_router | 14.77 ms |
| user_llm_gate | 14.47 ms |
| user_router_gate | 14.56 ms |
| flat_gate | 14.47 ms |

## GPU and CPU health per regime

| Regime | Throttle flags (decisions) | Mean GPU clock | On CPU 0 to 7 | Skipped beats |
|---|---|---|---|---|
| idle | none | 1702 MHz | 100.00% | 0 |
| whisper_flat | sw_power_cap 56 | 1920 MHz | 100.00% | 0 |
| llm_flat | sw_power_cap 56 | 1777 MHz | 100.00% | 0 |
| user_llm | sw_power_cap 4 | 1723 MHz | 100.00% | 0 |
| user_router | sw_power_cap 1 | 1689 MHz | 100.00% | 0 |
| user_llm_gate | sw_power_cap 5 | 1734 MHz | 100.00% | 0 |
| user_router_gate | none | 1713 MHz | 100.00% | 0 |
| flat_gate | sw_power_cap 56 | 2001 MHz | 100.00% | 0 |

## Voice jobs per regime (jobs and commands arriving in a block's first 2 s left out)

| Regime | Job | Jobs | Median | p99 | Worst | Over budget | Wait mean | Wait p99 | Wait worst |
|---|---|---|---|---|---|---|---|---|---|
| whisper_flat | whisper | 432 | 63.41 ms | 82.92 ms | 87.01 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| llm_flat | llm | 261 | 104.16 ms | 122.66 ms | 124.24 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | whisper | 11 | 57.81 ms | 72.49 ms | 73.88 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | llm | 11 | 95.34 ms | 110.98 ms | 110.99 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | whisper | 11 | 56.14 ms | 59.71 ms | 59.86 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | llm | 2 | 95.59 ms | 96.61 ms | 96.63 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | rules | 11 | 0.03 ms | 0.43 ms | 0.46 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm_gate | whisper | 11 | 55.64 ms | 59.51 ms | 59.58 ms | 0 (> 100.0 ms) | 16.83 ms | 97.02 ms | 98.14 ms |
| user_llm_gate | llm | 11 | 93.95 ms | 97.21 ms | 97.26 ms | 0 (> 140.0 ms) | 18.79 ms | 105.23 ms | 105.70 ms |
| user_router_gate | whisper | 11 | 56.99 ms | 60.53 ms | 60.55 ms | 0 (> 100.0 ms) | 18.27 ms | 99.82 ms | 101.09 ms |
| user_router_gate | llm | 2 | 95.01 ms | 95.97 ms | 95.99 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router_gate | rules | 11 | 0.03 ms | 0.03 ms | 0.03 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |
| flat_gate | whisper | 140 | 61.57 ms | 76.38 ms | 76.78 ms | 0 (> 100.0 ms) | 18.87 ms | 102.52 ms | 104.22 ms |
| flat_gate | llm | 140 | 94.27 ms | 102.40 ms | 102.90 ms | 0 (> 140.0 ms) | 24.44 ms | 133.25 ms | 134.37 ms |

## Spoken commands (arrival to routed answer)

| Regime | Commands | Added wait mean | Added wait p99 | Added wait worst | Time median | Time p99 | Time worst |
|---|---|---|---|---|---|---|---|
| user_llm | 11 | 0.00 ms | 0.00 ms | 0.00 ms | 152.73 ms | 170.87 ms | 171.06 ms |
| user_router | 11 | 0.00 ms | 0.00 ms | 0.00 ms | 56.78 ms | 153.34 ms | 153.64 ms |
| user_llm_gate | 11 | 35.62 ms | 105.23 ms | 105.70 ms | 154.55 ms | 256.17 ms | 256.74 ms |
| user_router_gate | 11 | 18.27 ms | 99.82 ms | 101.09 ms | 60.62 ms | 230.74 ms | 238.85 ms |

## Flat-out rates

| Regime | Job | Rate |
|---|---|---|
| whisper_flat | whisper | 15.40 jobs/s |
| llm_flat | llm | 9.31 jobs/s |
| flat_gate | whisper | 5.00 jobs/s |
| flat_gate | llm | 5.00 jobs/s |

## Sanity

- Same clip or sentence, same answer in every regime: 0 of 161 items differ
