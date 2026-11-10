# Voice robot: stage2b (10 Nov 2026 15:06 IST)

Ready-pose return between the two commands: True (amendment 13:47; development test: with 6 of 10, without 0 of 10). Return variant B. Development test (amendment 13:55): starts 5-9, injected sentences, both goals still true at the end (both reached): variant A: 7 then 2: 0 of 5 (1), 0 then 7: 5 of 5 (5), total 5; variant B: 7 then 2: 2 of 5 (4), 0 then 7: 5 of 5 (5), total 7. B gives more, so B is used.

Policy model-call time: one select_action call that computes a new chunk of 10 actions, timed with CUDA synchronised before and after.

Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, from the end of the trimmed clip.

| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | Stop delay ms | Overrun mm | Hold offset mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two_step_1_s0 | failed: 300 acting steps | False | False | False | 526 | [7, 2] | [7] | 12.3 | 394.16 | nan (nan)  | nan | nan |
| two_step_1_s1 | success | True | True | True | 342 | [7, 2] | [7, 2] | 12.4 | 386.09 | nan (nan)  | nan | nan |
| two_step_1_s2 | failed: 300 acting steps | False | False | False | 416 | [7, 2] | [] | 12.3 | 387.82 | nan (nan)  | nan | nan |
| two_step_1_s3 | success | True | True | False | 336 | [7, 2] | [7, 2] | 12.4 | 398.04 | nan (nan)  | nan | nan |
| two_step_1_s4 | success | True | True | True | 344 | [7, 2] | [7, 2] | 12.6 | 403.68 | nan (nan)  | nan | nan |
| two_step_2_s0 | success | True | True | True | 368 | [0, 7] | [0, 7] | 12.3 | 407.89 | nan (nan)  | nan | nan |
| two_step_2_s1 | success | True | True | False | 379 | [0, 7] | [0, 7] | 12.4 | 406.48 | nan (nan)  | nan | nan |
| two_step_2_s2 | success | True | True | True | 527 | [0, 7] | [0, 7] | 12.4 | 408.81 | nan (nan)  | nan | nan |
| two_step_2_s3 | success | True | True | True | 371 | [0, 7] | [0, 7] | 12.4 | 377.51 | nan (nan)  | nan | nan |
| two_step_2_s4 | success | True | True | True | 379 | [0, 7] | [0, 7] | 12.4 | 375.25 | nan (nan)  | nan | nan |

| Rule | Numbers | Verdict |
|---|---|---|
| S2-4 split two_step_1 | plans [[7, 2], [7, 2], [7, 2], [7, 2], [7, 2]] (expected [7, 2]) | PASS |
| S2-5 shown two_step_1 | first goal still true when the second was reached: 3 of 5 (both reached in order: 3; both still true 1 s later: 2) (bar 4 of 5) | FAIL |
| S2-4 split two_step_2 | plans [[0, 7], [0, 7], [0, 7], [0, 7], [0, 7]] (expected [0, 7]) | PASS |
| S2-5 shown two_step_2 | first goal still true when the second was reached: 5 of 5 (both reached in order: 5; both still true 1 s later: 4) (bar 4 of 5) | PASS |

All policy calls: median 12.38 ms, 242 calls.
