# Voice-commanded robot: results (10 Nov 2026)

Plan, pass rules and every dated change: [VOICE_PLAN.md](VOICE_PLAN.md). Raw results: `voice/results/voice_robot/`. Clips: `media/`.

**What it is.** A recorded spoken line is turned into text (Whisper large-v3-turbo), then into one of the robot's 10 commands, a stop, a continue or a refusal (word rules first, then the language model Qwen3.5-4B). The tuned SmolVLA policy carries the command out in the LIBERO-Goal simulator. All three models run in one program on one laptop GPU, one after another.

**What it is not.** Not a live microphone: the lines were recorded first and played into the running system at set times. Not a real robot and not a Jetson. Commands outside the 10 are refused, not learned. Changing a command mid-way and undoing an action were not tested. GPU sharing between programs is a separate timing test ([GPU_SHARING.md](GPU_SHARING.md)).

## Second-take recordings (used in the demo clips)

The demo clips use only the second-take recordings, which were spoken so that listeners can follow them easily. The first take is the registered result and is reported in full below. The table counts the second-take runs that used second-take audio only; the three runs of line 3 that fell back to a first-take recording are left out.

| What | Second-take result |
|---|---|
| Spoken lines understood and routed right | 8 of 9. "Kindly keep the bowl on the plate" was heard as "…ball…" and refused |
| Robot does the task (3 command lines × 3 starting layouts) | 8 of 9 |
| After "stop": overrun | 5.1 to 9.9 mm in 9 runs |
| After "stop": holds its place | within 1.9 mm in 8 runs, 5.05 mm in one |
| Stop delay after the word | 190 ms (one recording, measured the same way each time) |
| Task done after "continue" | 9 of 9 |
| "Open the middle drawer, then turn on the stove" | 5 of 5 |
| "Turn on the stove, then put the wine bottle on top of the cabinet" | 3 of 5 (fails the 4-of-5 bar) |
| "bowl" heard as "ball" | refused in 3 of 3 tries; the arm moved 0.35 mm |

Judged by the registered rules on second-take recordings alone, two would fail: "robot does the task" needs 10 of 12 and the misheard line makes three of those runs refusals, and "holds within 5 mm" was missed by 0.05 mm in one run.

All results below this section are the registered first take, in my own voice. Every pass rule was recomputed from the raw files in a separate check, with separate code; all matched (one value differs by rounding, 8.29 against 8.28 mm).

## Stage 1: one command per spoken line

Run: `voice/results/voice_robot/stage1` (14:49 to 14:51 IST). 5 lines. Each of the 4 command lines ran on 3 starting layouts (12 robot runs); the fifth line must be refused.

| # | Rule | Bar | Result |
|---|---|---|---|
| S1-0 | Tuned policy settings are active | median model call at most 20 ms | **Pass**: 12.8 ms over 154 calls |
| S1-1 | The line is routed to the right result | at least 4 of 5 lines | **Pass**: 5 of 5 |
| S1-2 | The robot does the task that was asked | at least 10 of 12 runs | **Pass**: 11 of 12 |
| S1-3 | After a refusal the arm stays still | under 2 mm in 3 s | **Pass**: 0.35 mm |

| Line as spoken | Heard (on the trimmed clip) | Decided by | Result | Robot runs |
|---|---|---|---|---|
| Turn on the stove. | "Turn on the stuff." | language model, 106 to 115 ms | turn on the stove | 3 of 3 |
| Put the wine bottle on top of the cabinet. | "Put the wine bottle on the top of the cabinet" | word rules, 0.1 ms | put the wine bottle on top of the cabinet | 2 of 3 |
| Kindly keep the bowl on the plate. | "Kindly keep the bowl on the plate." | word rules, 0.1 ms | put the bowl on the plate | 3 of 3 |
| Push the plate to the front of the stove. | "Push the plate to the front of the stove." | word rules, 0.1 ms | push the plate to the front of the stove | 3 of 3 |
| Close the top drawer. | "close the top drawer" | language model, 112 ms | refused | arm moved 0.35 mm |

- Whisper took 54 to 76 ms per line. The robot's first movement came about 0.2 s after the last word when the word rules decided, and 0.3 s when the language model decided. Both include 0.1 s of audio that trimming keeps after the word.
- The one failed run: the wine-bottle task on starting layout 1 used all 300 steps without reaching its goal.
- Successful runs took 3.65 to 11.9 s of robot movement (median about 4.2 s).
- Line 1 shows the two levels at work. On the full recording Whisper heard "stove" (`voice/results/voice_test.json`); on the trimmed clip it heard "stuff". The trim kept all of the speech, so this is Whisper behaving differently on the shorter clip. The word rules could not match "stuff", and the language model still chose the right command.

## Stage 2a: "stop" and "continue" while the robot works

Run: `voice/results/voice_robot/stage2a` (15:04 to 15:05 IST). The 12 stage 1 runs again. "Stop." is played 1.0 s after the robot's first movement; "Continue." 3.0 s after the stop takes effect.

| # | Rule | Bar | Result |
|---|---|---|---|
| S2-1a | After the stop the arm does not overrun far | at most 15 mm in all 12 runs | **Pass**: 1.1 to 14.1 mm |
| S2-1b | It then holds its place | within 5 mm, from 1 s after the stop | **Pass**: within 1.8 mm |
| S2-2 | Delay from the end of the word to the first held step | at most 300 ms in all 12 runs | **Pass**: 170 ms |
| S2-3 | After "continue" the robot still does the task | at least 10 of 12 runs | **Pass**: 11 of 12 |

- **How the stop works.** "Stop" is a word rule, never the language model. The loop drops the policy's queued movements and steers the arm back to the pose it had at that moment. Sending zero movements instead let the arm coast 33 mm in a test, so that was replaced before any registered run.
- **The 170 ms is one measurement, repeated.** All 12 runs use the same recorded "Stop." and it always lands at the same point between two control steps. It is made of 100 ms of audio kept after the word, about 50 ms of speech-to-text, and the wait for the next 50 ms control step. A live microphone also needs time to notice that the speaker has finished; that is not in this number.
- **Overrun by task:** stove 10.9 to 14.1 mm, push the plate 5.5 to 8.3 mm, bowl 1.7 to 5.2 mm, wine bottle 1.1 to 3.2 mm. The largest is close to the bar, and the bar was set after development tests.
- The one failed run is the same wine-bottle layout that failed in stage 1.
- **No "continue" reaction time is claimed for the first take.** The recorded "Continue." was too quiet for the trimming rule, so the clip kept 1.1 s of silence after the word and the robot resumed about 1.2 s after it. The holds lasted 4.95 s instead of about 3.9 s, which makes S2-1b the stricter way round.

## Stage 2b: two commands in one sentence

Run: `voice/results/voice_robot/stage2b` (15:05 to 15:06 IST). Two sentences, each on starting layouts 0 to 4. The sentence is split at "then"; a plain to-do list runs the two commands in order; the language model is used only if the word rules cannot read a half.

| Sentence | Split right (S2-4) | Both done, first still true when the second finished (S2-5, bar 4 of 5) | Shown in a demo clip |
|---|---|---|---|
| Open the middle drawer, then turn on the stove. | 5 of 5 | **5 of 5: Pass** | yes |
| Turn on the stove, then put the wine bottle on top of the cabinet. | 5 of 5 | **3 of 5: Fail** | no, reported only |

- **Returning to the ready pose is what makes it work.** Without it, 0 of 10 development runs finished the second command: the policy was only ever trained from the ready pose. With a scripted return (let go, lift 5 cm, go back) the drawer-then-stove sentence worked 5 of 5 in development and 5 of 5 here.
- The failing sentence: on layout 0 the stove was done and the wine bottle ran out of steps; on layout 2 the stove itself was not done.
- In the second half of the first sentence Whisper again heard "stuff" for "stove", and the language model chose the right command.
- Both recordings kept 1.1 to 1.5 s of silence after the words (too quiet to trim). The simulator is paused until the sentence has been understood, so this only delays the start on the clock; it does not change what the robot does.

## Second take (15:14 to 15:33 IST)

I recorded all nine lines again, louder, so that the silence trimming would work on every line. It did: no recording was flagged. The first take stays the registered result; the second take is reported in full, including the one rule it failed.

**Which recordings.** Eight lines are second-take recordings. "Continue." needed one more try: the first was heard as "Okay? Continue." and is not accepted as a continue. Line 3 is the first-take recording, because Whisper heard "ball" for "bowl" in three second-take tries (see the next section). The recordings used are in `voice/data/voice_test/` and `voice/data/voice_lines/`.

| Rule | First take (registered) | Second take |
|---|---|---|
| S1-1 lines routed right | 5 of 5 | 5 of 5 |
| S1-2 robot does the task | 11 of 12 | 11 of 12, the same runs step for step |
| S1-3 refusal, arm movement in 3 s | 0.35 mm | 0.35 mm |
| S2-1a overrun after the stop (bar 15 mm) | 1.1 to 14.1 mm: pass | 5.1 to 11.8 mm: pass |
| S2-1b hold (bar 5 mm) | within 1.8 mm: pass | **5.05 mm in one run: fail**; the other 11 within 2.1 mm |
| S2-2 stop delay after the word (bar 300 ms) | 170 ms: pass | 190 ms: pass |
| S2-3 task done after "continue" | 11 of 12: pass | 11 of 12: pass |
| S2-5 drawer, then stove | 5 of 5: pass | 5 of 5: pass |
| S2-5 stove, then wine bottle | 3 of 5: fail | 3 of 5: fail |

- **The failed rule.** In one wine-bottle run the arm stopped 5.06 mm past the stop point and stayed there, 0.05 mm over the bar. Over both takes: 24 stops, overrun 1.1 to 14.1 mm, and the arm then held within 2.1 mm in 23 of them.
- **Which runs fail after "continue" changes between takes.** First take: wine bottle, layout 1. Second take: bowl on plate, layout 0, while the wine-bottle run succeeded. A pause moves the robot onto a slightly different path. Over both takes, 22 of 24.
- **"Continue" reaction, second take only:** the robot resumed 170 ms after the end of the word. Like the stop delay, this is one recording measured the same way in 12 runs.
- Stage 1 and the two-step runs came out the same as in the first take, because the simulator is paused until the first line is understood and the seeds are the same.

Runs: `voice/results/voice_robot/stage1_take2`, `stage2a_take2`, `stage2b_take2`.

## A misheard word is refused, not guessed

In the second take I said "Kindly keep the bowl on the plate" three times, and Whisper heard "Kindly keep the ball on the plate" each time. None of the robot's 10 commands has a ball, so the language model answered "can't do that" all three times and the robot did nothing. The first-take recording of the same sentence was heard correctly.

- The refusal comes from the list of commands, not from looking at the scene. No camera is involved in that decision.
- This is the design choice from the router work ([ROUTER_PREREG.md](ROUTER_PREREG.md)): a wrong command is worse than no command.
- It also shows the cost: a person had to repeat themselves, and here repeating did not help.

Clip: `media/refusal_bowl_heard_as_ball.mp4`, made by playing the first misheard recording into the system (run `voice/results/voice_robot/ball_refusal`, audio in `voice/data/ball_demo/`).

## Read these with any of the results

- **Small numbers.** 12 runs per stage on 3 starting layouts, 5 per two-step sentence. This shows the chain works; it is not a success rate to quote next to the 400-episode numbers.
- **Start state.** The simulator is paused while the first line is spoken, so the policy starts from the same starting layout, after the same 10 settle steps, as in the 8 Nov runs. Each run is seeded on its own, so only rates compare with 8 Nov, not which layouts succeed.
- **"Still true 1 s later" is stricter than LIBERO's success and gives lower counts.** LIBERO counts a task done the first moment its goal is true, often with the object still in the gripper. Holding the arm for 1 s and checking again: stage 1, 11 of 11 successes still true (re-run `stage1_endcheck`); stage 2a, 8 of 11; stage 2b, 4 of 5 and 2 of 3. Clips come only from runs where everything still holds at the end.
- **Rules changed after rehearsal, before the runs they judge.** The plan's Amendments list each one with its time and reason: silence trimming, the hold after stop and its two bars, the stop delay bar (200 to 300 ms, when the 100 ms of kept audio was counted), the stop time (2.0 to 1.0 s), the pause, the return to the ready pose, and what S2-5 counts. Under the earlier S2-5 definition (both goals still true 1 s later) the counts are 4 of 5 and 2 of 5, so neither verdict changes.
- **Stage 1's first report has a flawed column.** Its "still true" column shows two wine-bottle runs losing their goal. That was the check letting the arm coast with the bottle still gripped (26 mm), not the robot. The four stage 1 rules do not use that column.
- **Code versions.** Stage 1 ran one version of `voice_robot.py` and stage 2 a later one (two changes, both in the Amendments). The stage 1 version was not kept; a re-run of stage 1 with the final code (`stage1_final_code`) gave the same result and the same number of moving steps in all 13 runs. The final code as run is saved in `voice/results/voice_robot/code_as_run_stage2/`.
- **Not covered by the separate check:** the contents of the clips, and the exact pooled median behind S1-0 (per-run medians are 11.8 to 13.8 ms).

## Clips

Second take (tighter, because every line trimmed properly), in `media/`:

| File | Length | What it shows |
|---|---|---|
| `take2_stop_continue_stove.mp4` | 12.2 s | "Turn on the stove", then "Stop", then "Continue" |
| `take2_two_step_drawer_then_stove.mp4` | 18.0 s | "Open the middle drawer, then turn on the stove", with the return to the ready pose |
| `refusal_bowl_heard_as_ball.mp4` | 8.8 s | "bowl" heard as "ball": refused, the arm stays still (0.35 mm in 3 s) |
| `take2_single_stove.mp4` | 7.8 s | One command |
| `take2_refused_top_drawer.mp4` | 8.4 s | "Close the top drawer" is refused |

First-take clips (registered runs) are not included in this repository. `voice/make_voice_clip.py` builds a clip from the run folder of a new run; the committed result folders do not keep the frame videos.

Every clip carries the line "Recorded voice, played in at the marked times. Simulator." Each comes from the lowest-numbered starting layout whose run did what was asked and still held at the end. Second-take clips are reported here next to both takes' numbers.
