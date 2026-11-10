# One GPU, three models: <project>/voice-robot/results/share/tuned_main

Bars apply to the Tuned run; for the Original contrast they are information only (PREREG).

## Bars

| # | Bar | Verdict | Numbers |
|---|---|---|---|
| 1 | Timer check, alone | not shown | measured by the calibrate run (--calibration) |
| 2 | Sharing slows the policy | PASS | whisper_flat 2.74x [2.71, 2.75]; llm_flat 2.80x [2.78, 2.81]; user_llm 2.33x [2.29, 2.37]; user_router 2.02x [1.41, 2.26] |
| 3 | Fix, speed | PASS | user_router_gate p99 14.78 ms vs idle 14.83 ms = 1.00x (max 1.10x); flat_gate p99 14.85 ms vs idle 14.83 ms = 1.00x (max 1.10x) |
| 4 | Fix, deadline | PASS | user_router_gate 0 late of 1164 (upper limit 0.33%); flat_gate 0 late of 1180 (upper limit 0.32%) |
| 5 | Fix, cost | PASS | 197 commands: mean 23.3 ms (max 60 ms), worst 172.9 ms (max 200 ms) |
| 6 | Turn-taking kept | PASS | 0 of 6645 GPU jobs overlap a policy call (policy calls: flat_gate 1200, user_llm_gate 1184, user_router_gate 1184) |

## Run

- Complete rounds: 5 [0, 1, 2, 3, 4] of rounds seen [0, 1, 2, 3, 4] (need 3); regimes per round: idle, whisper_flat, llm_flat, user_llm, user_router, user_llm_gate, user_router_gate, flat_gate
- Rounds left out: 0
- CPU cores allowed: robot loop 0 to 3 (4 cores); voice program 4 to 7 (4 cores)
- Decisions: 10279 total; 9375 in kept blocks; 0 in dropped or incomplete blocks (not used); 720 in alone phases; 184 unassigned (not used); 0 unreadable rows
- Skipped beats, whole run: 0
- Unexpected GPU pids (samples): none
- Dropped blocks: 0

## Robot loop alone (phases)

| Phase | Decisions | Median | p99 | Worst | Late | Throttle flags |
|---|---|---|---|---|---|---|
| alone_pre | 360 | 13.62 ms | 14.96 ms | 15.17 ms | 0 | none |
| alone_post | 360 | 13.71 ms | 14.68 ms | 14.80 ms | 0 | sw_power_cap 51 |

## Policy decisions per regime

| Regime | Decisions | Median | p99 | Worst | Late | Late upper (95%) | p99 vs idle [95%] |
|---|---|---|---|---|---|---|---|
| idle | 1180 | 14.35 ms | 14.83 ms | 15.35 ms | 0 | 0.32% | – |
| whisper_flat | 1180 | 31.18 ms | 40.57 ms | 42.09 ms | 0 | 0.32% | 2.74x [2.71, 2.75] |
| llm_flat | 1181 | 39.18 ms | 41.47 ms | 42.09 ms | 0 | 0.32% | 2.80x [2.78, 2.81] |
| user_llm | 1163 | 14.38 ms | 34.61 ms | 36.10 ms | 0 | 0.33% | 2.33x [2.29, 2.37] |
| user_router | 1163 | 14.33 ms | 29.99 ms | 35.99 ms | 0 | 0.33% | 2.02x [1.41, 2.26] |
| user_llm_gate | 1164 | 14.35 ms | 14.86 ms | 15.99 ms | 0 | 0.33% | 1.00x [1.00, 1.00] |
| user_router_gate | 1164 | 14.34 ms | 14.78 ms | 15.30 ms | 0 | 0.33% | 1.00x [0.99, 1.00] |
| flat_gate | 1180 | 14.34 ms | 14.85 ms | 15.08 ms | 0 | 0.32% | 1.00x [1.00, 1.01] |

## Cause check: was a voice job (Whisper or LLM) running during the decision?

| Regime | Voice job running | Decisions | Median | p99 | Worst | Late |
|---|---|---|---|---|---|---|
| idle | yes | 0 | – | – | – | 0 |
| idle | no | 1180 | 14.35 ms | 14.83 ms | 15.35 ms | 0 |
| whisper_flat | yes | 1180 | 31.18 ms | 40.57 ms | 42.09 ms | 0 |
| whisper_flat | no | 0 | – | – | – | 0 |
| llm_flat | yes | 1181 | 39.18 ms | 41.47 ms | 42.09 ms | 0 |
| llm_flat | no | 0 | – | – | – | 0 |
| user_llm | yes | 66 | 32.37 ms | 36.02 ms | 36.10 ms | 0 |
| user_llm | no | 1097 | 14.36 ms | 14.85 ms | 16.18 ms | 0 |
| user_router | yes | 34 | 24.04 ms | 35.84 ms | 35.99 ms | 0 |
| user_router | no | 1129 | 14.31 ms | 15.22 ms | 16.64 ms | 0 |
| user_llm_gate | yes | 0 | – | – | – | 0 |
| user_llm_gate | no | 1164 | 14.35 ms | 14.86 ms | 15.99 ms | 0 |
| user_router_gate | yes | 0 | – | – | – | 0 |
| user_router_gate | no | 1164 | 14.34 ms | 14.78 ms | 15.30 ms | 0 |
| flat_gate | yes | 0 | – | – | – | 0 |
| flat_gate | no | 1180 | 14.34 ms | 14.85 ms | 15.08 ms | 0 |

## p99 per round

| Regime | Round 0 | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|---|
| idle | 14.75 ms | 14.84 ms | 14.70 ms | 14.61 ms | 14.86 ms |
| whisper_flat | 40.82 ms | 39.93 ms | 40.62 ms | 39.64 ms | 39.84 ms |
| llm_flat | 41.77 ms | 41.17 ms | 41.40 ms | 41.39 ms | 41.53 ms |
| user_llm | 34.96 ms | 34.48 ms | 34.10 ms | 32.82 ms | 34.46 ms |
| user_router | 32.77 ms | 18.62 ms | 27.22 ms | 29.59 ms | 20.40 ms |
| user_llm_gate | 14.59 ms | 14.93 ms | 14.87 ms | 14.75 ms | 14.84 ms |
| user_router_gate | 14.78 ms | 14.83 ms | 14.80 ms | 14.59 ms | 14.73 ms |
| flat_gate | 14.90 ms | 14.75 ms | 14.76 ms | 14.76 ms | 14.80 ms |

## GPU and CPU health per regime

| Regime | Throttle flags (decisions) | Mean GPU clock | On CPU 0 to 7 | Skipped beats |
|---|---|---|---|---|
| idle | sw_power_cap 136 | 1657 MHz | 100.00% | 0 |
| whisper_flat | sw_power_cap 1180 | 1898 MHz | 100.00% | 0 |
| llm_flat | sw_power_cap 1181 | 1764 MHz | 100.00% | 0 |
| user_llm | sw_power_cap 224 | 1710 MHz | 100.00% | 0 |
| user_router | sw_power_cap 118 | 1696 MHz | 100.00% | 0 |
| user_llm_gate | sw_power_cap 202 | 1704 MHz | 100.00% | 0 |
| user_router_gate | sw_power_cap 378 | 1682 MHz | 100.00% | 0 |
| flat_gate | sw_power_cap 1150 | 1914 MHz | 100.00% | 0 |

## Voice jobs per regime (jobs and commands arriving in a block's first 2 s left out)

| Regime | Job | Jobs | Median | p99 | Worst | Over budget | Wait mean | Wait p99 | Wait worst |
|---|---|---|---|---|---|---|---|---|---|
| whisper_flat | whisper | 8986 | 64.07 ms | 84.61 ms | 88.50 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| llm_flat | llm | 5485 | 104.54 ms | 123.39 ms | 136.01 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | whisper | 197 | 58.68 ms | 76.42 ms | 78.96 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | llm | 197 | 95.71 ms | 113.50 ms | 116.34 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | whisper | 197 | 58.39 ms | 74.26 ms | 78.89 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | llm | 55 | 96.01 ms | 111.19 ms | 111.73 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router | rules | 197 | 0.03 ms | 0.04 ms | 0.45 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm_gate | whisper | 197 | 57.58 ms | 62.42 ms | 68.74 ms | 0 (> 100.0 ms) | 18.31 ms | 131.26 ms | 134.22 ms |
| user_llm_gate | llm | 197 | 95.18 ms | 99.42 ms | 103.97 ms | 0 (> 140.0 ms) | 22.84 ms | 165.13 ms | 172.38 ms |
| user_router_gate | whisper | 197 | 57.67 ms | 62.84 ms | 65.80 ms | 0 (> 100.0 ms) | 18.37 ms | 130.50 ms | 131.81 ms |
| user_router_gate | llm | 55 | 94.77 ms | 97.30 ms | 97.33 ms | 0 (> 140.0 ms) | 17.59 ms | 166.61 ms | 172.87 ms |
| user_router_gate | rules | 197 | 0.03 ms | 0.03 ms | 0.04 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |
| flat_gate | whisper | 2950 | 61.52 ms | 76.99 ms | 86.91 ms | 0 (> 100.0 ms) | 18.54 ms | 102.24 ms | 108.25 ms |
| flat_gate | llm | 2949 | 94.68 ms | 102.63 ms | 107.20 ms | 0 (> 140.0 ms) | 24.24 ms | 136.60 ms | 169.60 ms |

## Spoken commands (arrival to routed answer)

| Regime | Commands | Added wait mean | Added wait p99 | Added wait worst | Time median | Time p99 | Time worst |
|---|---|---|---|---|---|---|---|
| user_llm | 197 | 0.00 ms | 0.00 ms | 0.00 ms | 155.04 ms | 173.09 ms | 176.35 ms |
| user_router | 197 | 0.00 ms | 0.00 ms | 0.00 ms | 59.88 ms | 172.62 ms | 178.41 ms |
| user_llm_gate | 197 | 41.15 ms | 165.13 ms | 172.38 ms | 156.52 ms | 313.83 ms | 328.83 ms |
| user_router_gate | 197 | 23.29 ms | 157.28 ms | 172.87 ms | 61.67 ms | 313.85 ms | 321.05 ms |

## Flat-out rates

| Regime | Job | Rate |
|---|---|---|
| whisper_flat | whisper | 15.23 jobs/s |
| llm_flat | llm | 9.29 jobs/s |
| flat_gate | whisper | 5.00 jobs/s |
| flat_gate | llm | 5.00 jobs/s |

## Sanity

- Same clip or sentence, same answer in every regime: 0 of 285 items differ
