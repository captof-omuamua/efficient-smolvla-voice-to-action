# Voice robot: stage2a (10 Nov 2026 15:31 IST)

Hold after stop: variant A. Development test (amendment 13:55): starts 5-7, injected stop and continue: variant A largest overrun 11.47 mm (11.47, 11.10, 10.67), hold offset 2.11, 2.70, 1.43 mm, 3 of 3 succeed; variant B largest overrun 12.35 mm (12.35, 11.95, 11.33), hold offset 2.08, 2.32, 1.34 mm, 3 of 3 succeed. A has the smaller largest overrun, so A is used.
"Stop." starts 1.0 s after the first movement; "Continue." 3.0 s after the stop takes effect (amendment 14:05).

Policy model-call time: one select_action call that computes a new chunk of 10 actions, timed with CUDA synchronised before and after.

Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, from the end of the trimmed clip.

| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | Stop delay ms | Overrun mm | Hold offset mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line1_cmd7_stop_s0 | success | True | True | True | 213 | [7] | [7] | 12.6 | 412.35 | 190 (90) yes | 8.54 | 0.43 |
| line1_cmd7_stop_s1 | success | True | True | False | 203 | [7] | [7] | 14.6 | 402.25 | 190 (90) yes | 9.94 | 0.90 |
| line1_cmd7_stop_s2 | success | True | True | True | 199 | [7] | [7] | 13.0 | 404.15 | 190 (90) yes | 8.42 | 0.03 |
| line2_cmd2_stop_s0 | success | True | True | True | 244 | [2] | [2] | 12.6 | 292.01 | 190 (90) yes | 5.06 | 5.05 |
| line2_cmd2_stop_s1 | success | True | True | True | 251 | [2] | [2] | 14.3 | 313.01 | 190 (90) yes | 6.63 | 0.81 |
| line2_cmd2_stop_s2 | success | True | True | False | 251 | [2] | [2] | 14.2 | 293.50 | 190 (90) yes | 9.23 | 1.47 |
| line3_cmd8_stop_s0 | failed: 300 acting steps | False | False | False | 438 | [8] | [] | 12.9 | 350.51 | 190 (90) yes | 8.58 | 1.22 |
| line3_cmd8_stop_s1 | success | True | True | False | 233 | [8] | [8] | 12.4 | 342.15 | 190 (90) yes | 10.91 | 1.77 |
| line3_cmd8_stop_s2 | success | True | True | True | 234 | [8] | [8] | 13.6 | 340.11 | 190 (90) yes | 11.82 | 2.07 |
| line4_cmd5_stop_s0 | success | True | True | True | 309 | [5] | [5] | 12.4 | 461.89 | 190 (90) yes | 7.04 | 1.81 |
| line4_cmd5_stop_s1 | success | True | True | True | 296 | [5] | [5] | 12.5 | 462.03 | 190 (90) yes | 7.45 | 1.93 |
| line4_cmd5_stop_s2 | success | True | True | True | 275 | [5] | [5] | 13.7 | 446.37 | 190 (90) yes | 5.29 | 1.16 |

| Rule | Numbers | Verdict |
|---|---|---|
| S2-1a overrun | [8.54, 9.94, 8.42, 5.06, 6.63, 9.23, 8.58, 10.91, 11.82, 7.04, 7.45, 5.29] mm; stop not exercised in 0 (bar <= 15 mm in all 12) | PASS |
| S2-1b hold | [0.43, 0.9, 0.03, 5.05, 0.81, 1.47, 1.22, 1.77, 2.07, 1.81, 1.93, 1.16] mm from 1 s after the stop (bar <= 5 mm in all 12) | FAIL |
| S2-2 stop delay | from the end of the word [190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0, 190.0] ms; from the end of the clip [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0] ms (bar <= 300 ms from the word, all 12) | PASS |
| S2-3 continue works | 11 of 12 did the expected command (11 reached their plan) (bar 10 of 12; all 12 must exist) | PASS |

All policy calls: median 12.85 ms, 156 calls.
