REHEARSAL: starts or audio overridden, not a registered run

# Voice robot: stage1 (10 Nov 2026 15:34 IST)

Policy model-call time: one select_action call that computes a new chunk of 10 actions, timed with CUDA synchronised before and after.

Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, from the end of the trimmed clip.

| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | Stop delay ms | Overrun mm | Hold offset mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| line1_cmd7_s0 | success | True | True | True | 124 | [7] | [7] | 13.7 | 404.50 | nan (nan)  | nan | nan |
| line2_cmd2_s0 | success | True | True | True | 162 | [2] | [2] | 13.0 | 292.92 | nan (nan)  | nan | nan |
| line3_cmd8_s0 | success | True | True | True | 315 | [8] | [8] | 12.9 | 359.67 | nan (nan)  | nan | nan |
| line4_cmd5_s0 | success | True | True | True | 207 | [5] | [5] | 13.4 | 457.09 | nan (nan)  | nan | nan |
| line5_refuse_s0 | ended: idle | False | False | False | 144 | [] | [] | nan | 0.36 | nan (nan)  | nan | nan |

| Rule | Numbers | Verdict |
|---|---|---|
| S1-0 tuned settings | median call 12.88 ms over 55 calls (bar 20 ms) | PASS |
| S1-1 line 1 | heard ['Turn on the stove.'], got [('plan', (7,))] | PASS |
| S1-1 line 2 | heard ['put the wine bottle on top of the cabinet.'], got [('plan', (2,))] | PASS |
| S1-1 line 3 | heard ['Kindly keep the bowl on the plate.'], got [('plan', (8,))] | PASS |
| S1-1 line 4 | heard ['Push the plate to the front of the stove.'], got [('plan', (5,))] | PASS |
| S1-1 line 5 | heard ['Kindly keep the ball on the plate.'], got [('refuse', ())] | PASS |
| S1-1 right routing | 5 of 5 lines (bar 4) | PASS |
| S1-2 robot does the task | 4 of 4 did the expected command (4 reached their plan) (bar 10 of 12) | FAIL |
| S1-3 refusal | largest move in the 3 s after the refusal took effect: [0.346] mm (whole run [0.359] mm; creep after 60 and 100 idle steps [(0.346, 0.359)] mm) (bar < 2 mm) | PASS |

All policy calls: median 12.88 ms, 55 calls.
