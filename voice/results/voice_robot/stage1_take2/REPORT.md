# Voice robot: stage1 (10 Nov 2026 15:30 IST)

Policy model-call time: one select_action call that computes a new chunk of 10 actions, timed with CUDA synchronised before and after.

Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, from the end of the trimmed clip.

| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | Stop delay ms | Overrun mm | Hold offset mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line1_cmd7_s0 | success | True | True | True | 124 | [7] | [7] | 12.6 | 404.50 | nan (nan)  | nan | nan |
| line1_cmd7_s1 | success | True | True | True | 119 | [7] | [7] | 13.6 | 399.61 | nan (nan)  | nan | nan |
| line1_cmd7_s2 | success | True | True | True | 119 | [7] | [7] | 12.4 | 399.58 | nan (nan)  | nan | nan |
| line2_cmd2_s0 | success | True | True | True | 163 | [2] | [2] | 13.5 | 292.92 | nan (nan)  | nan | nan |
| line2_cmd2_s1 | failed: 300 acting steps | False | False | False | 357 | [2] | [] | 12.6 | 431.45 | nan (nan)  | nan | nan |
| line2_cmd2_s2 | success | True | True | True | 172 | [2] | [2] | 13.8 | 291.03 | nan (nan)  | nan | nan |
| line3_cmd8_s0 | success | True | True | True | 315 | [8] | [8] | 12.6 | 359.67 | nan (nan)  | nan | nan |
| line3_cmd8_s1 | success | True | True | True | 156 | [8] | [8] | 13.8 | 356.15 | nan (nan)  | nan | nan |
| line3_cmd8_s2 | success | True | True | True | 157 | [8] | [8] | 12.2 | 345.74 | nan (nan)  | nan | nan |
| line4_cmd5_s0 | success | True | True | True | 207 | [5] | [5] | 12.7 | 457.09 | nan (nan)  | nan | nan |
| line4_cmd5_s1 | success | True | True | True | 204 | [5] | [5] | 12.6 | 458.52 | nan (nan)  | nan | nan |
| line4_cmd5_s2 | success | True | True | True | 197 | [5] | [5] | 12.3 | 438.59 | nan (nan)  | nan | nan |
| line5_refuse_s0 | ended: idle | False | False | False | 136 | [] | [] | nan | 0.36 | nan (nan)  | nan | nan |

| Rule | Numbers | Verdict |
|---|---|---|
| S1-0 tuned settings | median call 12.64 ms over 154 calls (bar 20 ms) | PASS |
| S1-1 line 1 | heard ['Turn on the stove.'], got [('plan', (7,))] | PASS |
| S1-1 line 2 | heard ['put the wine bottle on top of the cabinet.'], got [('plan', (2,))] | PASS |
| S1-1 line 3 | heard ['Kindly keep the bowl on the plate.'], got [('plan', (8,))] | PASS |
| S1-1 line 4 | heard ['Push the plate to the front of the stove.'], got [('plan', (5,))] | PASS |
| S1-1 line 5 | heard ['Close the top drawer.'], got [('refuse', ())] | PASS |
| S1-1 right routing | 5 of 5 lines (bar 4) | PASS |
| S1-2 robot does the task | 11 of 12 did the expected command (11 reached their plan) (bar 10 of 12) | PASS |
| S1-3 refusal | largest move in the 3 s after the refusal took effect: [0.346] mm (whole run [0.359] mm; creep after 60 and 100 idle steps [(0.346, 0.359)] mm) (bar < 2 mm) | PASS |

All policy calls: median 12.64 ms, 154 calls.
