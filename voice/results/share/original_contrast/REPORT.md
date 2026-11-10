# One GPU, three models: <project>/voice-robot/results/share/original_contrast

Bars apply to the Tuned run; for the Original contrast they are information only (PREREG).

## Bars

| # | Bar | Verdict | Numbers |
|---|---|---|---|
| 1 | Timer check, alone | not shown | measured by the calibrate run (--calibration) |
| 2 | Sharing slows the policy | PASS | llm_flat 1.42x [1.25, 1.52]; user_llm 1.20x [1.06, 1.29] |
| 3 | Fix, speed | not applicable | needs idle, user_router_gate and flat_gate |
| 4 | Fix, deadline | not applicable | needs user_router_gate and flat_gate |
| 5 | Fix, cost | FAIL | 119 commands: mean 100.4 ms (max 60 ms), worst 317.1 ms (max 200 ms) |
| 6 | Turn-taking kept | PASS | 0 of 148 GPU jobs overlap a policy call (policy calls: user_router_gate 708) |

## Run

- Complete rounds: 3 [0, 1, 2] of rounds seen [0, 1, 2] (need 3); regimes per round: idle, llm_flat, user_llm, user_router_gate
- Rounds left out: 0
- CPU cores allowed: robot loop 0 to 3 (4 cores); voice program 4 to 7 (4 cores)
- Decisions: 3607 total; 2807 in kept blocks; 0 in dropped or incomplete blocks (not used); 720 in alone phases; 80 unassigned (not used); 0 unreadable rows
- Skipped beats, whole run: 0
- Unexpected GPU pids (samples): none
- Dropped blocks: 0

## Robot loop alone (phases)

| Phase | Decisions | Median | p99 | Worst | Late | Throttle flags |
|---|---|---|---|---|---|---|
| alone_pre | 360 | 88.89 ms | 106.50 ms | 131.53 ms | 360 | none |
| alone_post | 360 | 88.91 ms | 111.66 ms | 128.97 ms | 360 | sw_power_cap 138 |

## Policy decisions per regime

| Regime | Decisions | Median | p99 | Worst | Late | Late upper (95%) | p99 vs idle [95%] |
|---|---|---|---|---|---|---|---|
| idle | 708 | 88.84 ms | 109.71 ms | 132.78 ms | 708 | 100.00% | – |
| llm_flat | 708 | 145.56 ms | 155.73 ms | 160.88 ms | 708 | 100.00% | 1.42x [1.25, 1.52] |
| user_llm | 695 | 89.93 ms | 131.38 ms | 141.05 ms | 695 | 100.00% | 1.20x [1.06, 1.29] |
| user_router_gate | 696 | 88.42 ms | 97.53 ms | 114.53 ms | 696 | 100.00% | 0.89x [0.78, 0.97] |

## Cause check: was a voice job (Whisper or LLM) running during the decision?

| Regime | Voice job running | Decisions | Median | p99 | Worst | Late |
|---|---|---|---|---|---|---|
| idle | yes | 0 | – | – | – | 0 |
| idle | no | 708 | 88.84 ms | 109.71 ms | 132.78 ms | 708 |
| llm_flat | yes | 708 | 145.56 ms | 155.73 ms | 160.88 ms | 708 |
| llm_flat | no | 0 | – | – | – | 0 |
| user_llm | yes | 56 | 111.53 ms | 138.66 ms | 141.05 ms | 56 |
| user_llm | no | 639 | 89.71 ms | 102.67 ms | 128.51 ms | 639 |
| user_router_gate | yes | 0 | – | – | – | 0 |
| user_router_gate | no | 696 | 88.42 ms | 97.53 ms | 114.53 ms | 696 |

## p99 per round

| Regime | Round 0 | Round 1 | Round 2 |
|---|---|---|---|
| idle | 103.12 ms | 123.95 ms | 101.06 ms |
| llm_flat | 154.97 ms | 155.63 ms | 155.31 ms |
| user_llm | 131.06 ms | 126.27 ms | 132.06 ms |
| user_router_gate | 97.82 ms | 94.20 ms | 103.87 ms |

## GPU and CPU health per regime

| Regime | Throttle flags (decisions) | Mean GPU clock | On CPU 0 to 7 | Skipped beats |
|---|---|---|---|---|
| idle | none | 1976 MHz | 100.00% | 0 |
| llm_flat | sw_power_cap 708 | 1808 MHz | 100.00% | 0 |
| user_llm | sw_power_cap 163 | 1927 MHz | 100.00% | 0 |
| user_router_gate | sw_power_cap 144 | 1869 MHz | 100.00% | 0 |

## Voice jobs per regime (jobs and commands arriving in a block's first 2 s left out)

| Regime | Job | Jobs | Median | p99 | Worst | Over budget | Wait mean | Wait p99 | Wait worst |
|---|---|---|---|---|---|---|---|---|---|
| llm_flat | llm | 3020 | 108.39 ms | 155.61 ms | 167.20 ms | 482 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | whisper | 119 | 60.13 ms | 90.96 ms | 93.99 ms | 0 (> 100.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_llm | llm | 119 | 99.15 ms | 140.83 ms | 142.83 ms | 2 (> 140.0 ms) | 0.00 ms | 0.00 ms | 0.00 ms |
| user_router_gate | whisper | 119 | 58.77 ms | 65.05 ms | 85.37 ms | 0 (> 100.0 ms) | 89.37 ms | 278.24 ms | 279.67 ms |
| user_router_gate | llm | 29 | 96.82 ms | 106.60 ms | 108.06 ms | 0 (> 140.0 ms) | 45.12 ms | 305.91 ms | 317.11 ms |
| user_router_gate | rules | 119 | 0.03 ms | 0.09 ms | 0.48 ms | – | 0.00 ms | 0.00 ms | 0.00 ms |

## Spoken commands (arrival to routed answer)

| Regime | Commands | Added wait mean | Added wait p99 | Added wait worst | Time median | Time p99 | Time worst |
|---|---|---|---|---|---|---|---|
| user_llm | 119 | 0.00 ms | 0.00 ms | 0.00 ms | 160.40 ms | 204.39 ms | 207.74 ms |
| user_router_gate | 119 | 100.36 ms | 279.45 ms | 317.11 ms | 161.94 ms | 434.67 ms | 475.23 ms |

## Flat-out rates

| Regime | Job | Rate |
|---|---|---|
| llm_flat | llm | 8.52 jobs/s |

## Sanity

- Same clip or sentence, same answer in every regime: 0 of 256 items differ
