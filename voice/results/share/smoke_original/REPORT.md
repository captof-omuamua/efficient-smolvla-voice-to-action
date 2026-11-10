# One GPU, three models: <project>/voice-robot/results/share/smoke_original

Bars apply to the Tuned run; for the Original contrast they are information only (PREREG).

## Bars

| # | Bar | Verdict | Numbers |
|---|---|---|---|
| 1 | Timer check, alone | not shown | measured by the calibrate run (--calibration) |
| 2 | Sharing slows the policy | PASS | llm_flat 1.58x [1.53, 1.58]; user_llm 1.22x [1.12, 1.22] |
| 3 | Fix, speed | not applicable | needs idle, user_router_gate and flat_gate |
| 4 | Fix, deadline | not applicable | needs user_router_gate and flat_gate |
| 5 | Fix, cost | FAIL | 7 commands: mean 116.4 ms (max 60 ms), worst 270.0 ms (max 200 ms) |
| 6 | Turn-taking kept | PASS | 0 of 9 GPU jobs overlap a policy call (policy calls: user_router_gate 38) |

## Run

- Complete rounds: 1 [0] of rounds seen [0] (need 1); regimes per round: idle, llm_flat, user_llm, user_router_gate
- Rounds left out: 0
- CPU cores allowed: robot loop 0 to 3 (4 cores); voice program 4 to 7 (4 cores)
- Decisions: 224 total; 140 in kept blocks; 0 in dropped or incomplete blocks (not used); 39 in alone phases; 45 unassigned (not used); 0 unreadable rows
- Skipped beats, whole run: 0
- Unexpected GPU pids (samples): none
- Dropped blocks: 0

## Robot loop alone (phases)

| Phase | Decisions | Median | p99 | Worst | Late | Throttle flags |
|---|---|---|---|---|---|---|
| alone_pre | 20 | 87.08 ms | 98.47 ms | 99.35 ms | 20 | sw_power_cap 16 |
| alone_post | 19 | 85.26 ms | 89.15 ms | 89.74 ms | 19 | none |

## Policy decisions per regime

| Regime | Decisions | Median | p99 | Worst | Late | Late upper (95%) | p99 vs idle [95%] |
|---|---|---|---|---|---|---|---|
| idle | 36 | 84.97 ms | 86.19 ms | 86.23 ms | 36 | 100.00% | – |
| llm_flat | 36 | 128.33 ms | 135.89 ms | 137.37 ms | 36 | 100.00% | 1.58x [1.53, 1.58] |
| user_llm | 34 | 87.75 ms | 105.15 ms | 107.60 ms | 34 | 100.00% | 1.22x [1.12, 1.22] |
| user_router_gate | 34 | 85.28 ms | 89.69 ms | 91.13 ms | 34 | 100.00% | 1.04x [1.00, 1.06] |

## Cause check: was a voice job (Whisper or LLM) running during the decision?

| Regime | Voice job running | Decisions | Median | p99 | Worst | Late |
|---|---|---|---|---|---|---|
| idle | yes | 0 | – | – | – | 0 |
| idle | no | 36 | 84.97 ms | 86.19 ms | 86.23 ms | 36 |
| llm_flat | yes | 36 | 128.33 ms | 135.89 ms | 137.37 ms | 36 |
| llm_flat | no | 0 | – | – | – | 0 |
| user_llm | yes | 2 | 103.88 ms | 107.52 ms | 107.60 ms | 2 |
| user_llm | no | 32 | 87.45 ms | 89.77 ms | 89.90 ms | 32 |
| user_router_gate | yes | 0 | – | – | – | 0 |
| user_router_gate | no | 34 | 85.28 ms | 89.69 ms | 91.13 ms | 34 |

## p99 per round

| Regime | Round 0 |
|---|---|
| idle | 86.19 ms |
| llm_flat | 135.89 ms |
| user_llm | 105.15 ms |
| user_router_gate | 89.69 ms |

## GPU and CPU health per regime

| Regime | Throttle flags (decisions) | Mean GPU clock | On CPU 0 to 7 | Skipped beats |
|---|---|---|---|---|
| idle | sw_power_cap 29 | 1660 MHz | 100.00% | 0 |
| llm_flat | sw_power_cap 36 | 1746 MHz | 100.00% | 0 |
| user_llm | sw_power_cap 23 | 1749 MHz | 100.00% | 0 |
| user_router_gate | sw_power_cap 24 | 1671 MHz | 100.00% | 0 |

## Voice jobs per regime (jobs and commands arriving in a block's first 2 s left out)

| Regime | Job | Jobs | Median | p99 | Worst | Over budget | Wait mean | Wait p99 | Wait worst |
|---|---|---|---|---|---|---|---|---|---|
| llm_flat | llm | 158 | 106.12 ms | 152.81 ms | 157.55 ms | 22 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | whisper | 7 | 59.33 ms | 98.19 ms | 99.37 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | llm | 7 | 97.61 ms | 129.89 ms | 131.95 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router_gate | whisper | 7 | 55.34 ms | 57.71 ms | 57.74 ms | 0 (> 100.0 ms) | 116.37 ms | 269.24 ms | 269.99 ms |
| user_router_gate | llm | 2 | 94.75 ms | 95.16 ms | 95.17 ms | 0 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router_gate | rules | 7 | 0.03 ms | 0.45 ms | 0.47 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |

## Spoken commands (arrival to routed answer)

| Regime | Commands | Added wait mean | Added wait p99 | Added wait worst | Time median | Time p99 | Time worst |
|---|---|---|---|---|---|---|---|
| user_llm | 7 | 0.00 ms | 0.00 ms | 0.00 ms | 157.44 ms | 194.55 ms | 194.75 ms |
| user_router_gate | 7 | 116.37 ms | 269.24 ms | 269.99 ms | 226.88 ms | 402.35 ms | 407.28 ms |

## Flat-out rates

| Regime | Job | Rate |
|---|---|---|
| llm_flat | llm | 8.74 jobs/s |

## Sanity

- Same clip or sentence, same answer in every regime: 0 of 142 items differ
