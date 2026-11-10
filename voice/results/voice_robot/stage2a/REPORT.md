# Voice robot: stage2a (10 Nov 2026 15:05 IST)

Hold after stop: variant A. Development test (amendment 13:55): starts 5-7, injected stop and continue: variant A largest overrun 11.47 mm (11.47, 11.10, 10.67), hold offset 2.11, 2.70, 1.43 mm, 3 of 3 succeed; variant B largest overrun 12.35 mm (12.35, 11.95, 11.33), hold offset 2.08, 2.32, 1.34 mm, 3 of 3 succeed. A has the smaller largest overrun, so A is used.
"Stop." starts 1.0 s after the first movement; "Continue." 3.0 s after the stop takes effect (amendment 14:05).

Policy model-call time: one select_action call that computes a new chunk of 10 actions, timed with CUDA synchronised before and after.

Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, from the end of the trimmed clip.

| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | Stop delay ms | Overrun mm | Hold offset mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line1_cmd7_stop_s0 | success | True | True | True | 248 | [7] | [7] | 12.6 | 407.44 | 170 (70) yes | 12.44 | 0.86 |
| line1_cmd7_stop_s1 | success | True | True | False | 235 | [7] | [7] | 14.1 | 398.18 | 170 (70) yes | 14.06 | 1.74 |
| line1_cmd7_stop_s2 | success | True | True | True | 232 | [7] | [7] | 13.1 | 398.10 | 170 (70) yes | 10.86 | 0.55 |
| line2_cmd2_stop_s0 | success | True | True | True | 274 | [2] | [2] | 13.6 | 296.62 | 170 (70) yes | 3.19 | 0.54 |
| line2_cmd2_stop_s1 | failed: 300 acting steps | False | False | False | 472 | [2] | [] | 12.5 | 334.67 | 170 (70) yes | 1.88 | 0.09 |
| line2_cmd2_stop_s2 | success | True | True | False | 286 | [2] | [2] | 12.1 | 292.11 | 170 (70) yes | 1.11 | 0.64 |
| line3_cmd8_stop_s0 | success | True | True | True | 244 | [8] | [8] | 13.8 | 350.38 | 170 (70) yes | 5.16 | 0.51 |
| line3_cmd8_stop_s1 | success | True | True | True | 251 | [8] | [8] | 13.8 | 347.03 | 170 (70) yes | 1.74 | 0.44 |
| line3_cmd8_stop_s2 | success | True | True | False | 249 | [8] | [8] | 13.6 | 346.54 | 170 (70) yes | 3.58 | 0.56 |
| line4_cmd5_stop_s0 | success | True | True | True | 311 | [5] | [5] | 13.4 | 457.37 | 170 (70) yes | 6.39 | 0.26 |
| line4_cmd5_stop_s1 | success | True | True | True | 310 | [5] | [5] | 13.3 | 460.59 | 170 (70) yes | 5.50 | 0.80 |
| line4_cmd5_stop_s2 | success | True | True | True | 291 | [5] | [5] | 12.2 | 433.15 | 170 (70) yes | 8.28 | 1.80 |

| Rule | Numbers | Verdict |
|---|---|---|
| S2-1a overrun | [12.44, 14.06, 10.86, 3.19, 1.88, 1.11, 5.16, 1.74, 3.58, 6.39, 5.5, 8.28] mm; stop not exercised in 0 (bar <= 15 mm in all 12) | PASS |
| S2-1b hold | [0.86, 1.74, 0.55, 0.54, 0.09, 0.64, 0.51, 0.44, 0.56, 0.26, 0.8, 1.8] mm from 1 s after the stop (bar <= 5 mm in all 12) | PASS |
| S2-2 stop delay | from the end of the word [170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0, 170.0] ms; from the end of the clip [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0] ms (bar <= 300 ms from the word, all 12) | PASS |
| S2-3 continue works | 11 of 12 did the expected command (11 reached their plan) (bar 10 of 12; all 12 must exist) | PASS |

All policy calls: median 13.19 ms, 148 calls.
