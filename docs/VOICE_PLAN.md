# Voice-commanded robot: plan and pass rules (10 Nov 2026)

Written at 13:25 IST, before any voice-to-robot run. Nothing above the "Amendments" heading changed after the first stage 1 robot run. This published copy has edited paths and wording (for example, the original's working rules are summarised as "Run settings"); the rules, bars and times are unchanged. The sha256 of the original file is in each run's `fingerprints.txt`, so it will not match this copy. Later changes are under "Amendments" with the time and the reason. Results: [RESULTS_VOICE.md](RESULTS_VOICE.md).

## Goal

A recorded spoken line is turned into text (Whisper), then into a command (word rules, then Qwen3.5-4B), and the tuned SmolVLA policy carries it out in the LIBERO-Goal simulator. Stage 2 adds "stop", "continue" and two commands in a row.

Not claimed and not tested: a live microphone, changing a command mid-way, undo, commands outside the 10, a real robot, a Jetson. GPU sharing between programs is a separate timing test ([GPU_SHARING.md](GPU_SHARING.md)); here the models run one after another in one program.

## What was checked before writing this (13:23 IST)

- The simulator starts from the patched LeRobot checkout's environment. One simulator step takes 6.6 ms (median of 40).
- One scene can check the goal of each of the 10 commands (`_eval_predicate` on the goal read from each task's BDDL file). All 10 are false at the start.
- The hold action is `[0, 0, 0, 0, 0, 0, gripper]`. Over 40 hold steps at the start pose the arm moved 0.95 mm.
- Tuned policy on these 10 commands, 8 Nov, 10 tries each: 10, 10, 10, 8, 9, 10, 4, 10, 9, 8 of 10 for commands 0 to 9 (`speed/results/final_2026-11-07/libero_runs/final_2026-11-07_B_tuned/eval_info.json`).

## Files

All in `voice/`.

| File | Job |
|---|---|
| `commands.py` | Turns a heard sentence into an event, and keeps the robot's to-do list |
| `tests/test_commands.py` | Unit tests for `commands.py` |
| `record_lines.py` | Records named lines from the microphone and prints what Whisper heard |
| `voice_robot.py` | The robot loop: recorded lines in, robot runs and a results table out |
| `make_voice_clip.py` | Frames + captions + the recorded voice into one mp4 |
| `run_voice_robot.sh` | Sets the environment, takes the GPU lock, logs to `$LOG_DIR/voice_robot.log` |
| `results/voice_robot/<name>/` | `runs.csv`, `events.csv`, `REPORT.md`, `fingerprints.txt`, and per run `timeline.json` (new runs go to `$OUT_DIR/voice_robot/<name>/` and also keep frame videos) |

## `commands.py` (fixed interface)

```python
@dataclass
class Event:
    kind: str            # "plan", "stop", "continue" or "refuse"
    commands: list[int]  # for "plan": one or two command ids; otherwise empty
    level: str           # "rule" or "llm": which level decided
    ms: float            # time taken to decide

def parse(text, router) -> Event   # router = router.Router (frozen), or any object with route(text)

class Session:
    queue: list[int]      # commands still to do after the current one
    current: int | None   # command being done now
    holding: bool         # True after "stop", until "continue" or a new plan
    done: list[int]       # commands finished, in order
    def on_event(self, event): ...
    def on_success(self): ...            # the current command's goal is reached
    def instruction(self) -> str | None  # LIBERO's sentence for the current command; None when idle or holding
```

`parse` rules, in this order:
1. **Stop.** After `router.clean`, if the sentence has the whole word stop, halt or freeze: "stop". No language model. A false stop is safer than a missed one.
2. **Continue.** The cleaned sentence is exactly one of: continue, go on, carry on, resume, keep going: "continue". No language model.
3. **Two commands.** If the sentence has a joining word (and then, after that, then, and), split at its first occurrence. Route each half with `router.route`. If both halves give commands and they differ: "plan" with both. Otherwise go to rule 4.
4. **One command.** `router.route` on the whole sentence: a command gives "plan" with one id; a refusal gives "refuse".

`Session` rules: a "plan" replaces everything and clears holding. "stop" sets holding. "continue" clears holding. "refuse" changes nothing. `on_success` moves the current command to `done` and takes the next one from the queue.

## The robot loop (`voice_robot.py`)

- One process loads Whisper, the router and the tuned SmolVLA. Policy settings are exactly those of the 8 Nov tuned run: `lerobot/smolvla_libero` at revision `a24f1f380461b665158d52acd140152c79e8b36d`, `num_steps=1`, `compile_model=true`, `empty_cameras=0`, `n_action_steps=10`, cameras `agentview_image → camera1` and `robot0_eye_in_hand_image → camera2`, batched cameras (the patched checkout's default). The loop mirrors `eval_main` and `rollout` in `lerobot/scripts/lerobot_eval.py`; the only change is that `observation["task"]` is set from `Session.instruction()`.
- Simulator clock: step k is k × 50 ms after the episode starts (after reset and LIBERO's 10 settle steps).
- A run has a script of spoken lines: (start time, audio file). A line is "heard" at start + its audio length. Whisper and `parse` then run for real and are timed. The event takes effect at the first step at or after start + audio length + measured time. Both models are warmed up once before any timed call.
- Each step: holding or idle gives the hold action (zeros, last gripper value; −1 before the first policy action). Otherwise the policy acts. `policy.reset()` is called when a command starts and on "stop", so queued movements are dropped.
- After each step the current command's goal is checked in the simulator. A command gets at most 300 acting steps; hold steps do not count. A run succeeds when every command of its plan reached its goal.
- The simulator environment is built for the last command of the plan, so its own success signal cannot end the run early. Trap: LeRobot's vector environments reset on the step after a success, so the loop never steps after the last success.
- Video: one frame per step, with captions (said, heard, command, MOVING or HOLDING), and the recorded voice at its start time.

## Stage 1: one command per line

Lines: the 5 recordings from `voice_test.py` (`data/voice_test/line_1.raw` to `line_5.raw`): commands 7, 2, 8, 5, and one that must be refused.
Runs: each of the 4 command lines on starts 0, 1, 2 (12 runs); the refusal line once. The line starts at 0 s.

| # | Rule | Passes if |
|---|---|---|
| S1-0 | Tuned settings are active | median model-call time in the loop is at most 20 ms (8 Nov: 14.4 ms) |
| S1-1 | Right routing | at least 4 of the 5 lines give the expected result |
| S1-2 | Robot does the task | at least 10 of the 12 runs succeed |
| S1-3 | Refusal | the arm moves less than 2 mm in 3 s |

A misheard line may be recorded again once; both attempts are reported.

## Stage 2: stop, continue, two commands

Extra recordings (`record_lines.py`, saved in `data/voice_lines/`): "Stop." "Continue." "Turn on the stove, then put the wine bottle on top of the cabinet." "Open the middle drawer, then turn on the stove."

**2a. Stop and continue.** The 12 stage 1 runs again. "Stop." starts 2.0 s after the robot's first movement. "Continue." starts 3.0 s after the stop takes effect.

| # | Rule | Passes if |
|---|---|---|
| S2-1 | Stop holds the arm | in all 12 runs the arm moves less than 5 mm between the stop and the continue taking effect |
| S2-2 | Stop delay | from the end of the word to the first held step: at most 200 ms in all 12 runs |
| S2-3 | Continue works | at least 10 of the 12 runs still succeed |

**2b. Two commands in a row.** The two sentences, each on starts 0 to 4 (10 runs).

| # | Rule | Passes if |
|---|---|---|
| S2-4 | Split | both sentences give the two expected commands, in order (7 then 2; 0 then 7) |
| S2-5 | Shown in the talk | a pair is shown only if at least 4 of its 5 runs finish both commands; otherwise it is reported, not shown |

## Clips

One clip per capability that passed, from the lowest-numbered start that succeeded. The counts (for example 11 of 12) are shown with the clip. Caption line on every clip: "Recorded voice, played in at the marked times. Simulator."

## Run settings

- Python from the patched LeRobot checkout's venv, with `HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1`, the three thread variables set to 1, and a compile cache in `voice/.cache/inductor` (all set by `run_voice_robot.sh`). Every GPU job runs under the GPU lock.
- `router.py`, `speech.py` and `tasks.py` are frozen (they come from the router work, [ROUTER_PREREG.md](ROUTER_PREREG.md)). The router's held-back sentence sets (`data/test.json`, `data/dev.json`) are never read here.
- Code: small, readable, explainable line by line.
- Time box: stage 1 results by 14:30 IST, stage 2 by 15:30 IST. At 15:30 whatever has no numbers is left out.

## Amendments

The times below are approximate (within about 3 minutes). Each amendment was written before the first run that used a recording; the recordings' file times and the run log show the order.

**13:31 IST, before any robot run. Silence is trimmed from the recordings.**
Reason: I start and stop each recording with the Enter key, so every clip has silence before and after the words. Without trimming, the stop delay (S2-2) would count the time taken to press Enter, not the robot's reaction.
- Rule: cut the clip into 20 ms frames and take each frame's loudness (root mean square). A frame is speech if it is louder than 10% of the clip's 95th-percentile frame. Keep from 100 ms before the first speech frame to 100 ms after the last.
- The trimmed clip is what Whisper hears, what plays in the timeline and what is in the video. Both lengths (recorded and trimmed) are logged.
- Limit to state with S2-2: a live system must also detect that the speaker has finished, which usually adds a few tenths of a second. That time is not in these numbers.

**13:50 IST, before any robot run with a recording. Short clicks do not count as speech when trimming.**
Reason: the Enter key that starts and ends a recording makes a short, loud click next to the laptop microphone. Under the 13:31 rule one loud 20 ms frame counts as speech, so a click would keep the silence between the last word and the key press. A 3 s check of the room at 13:48 gave a background loudness of about 300 to 400 (of 32,768) with short peaks near 30,000.
- Added to the 13:31 rule: speech frames count only in runs of at least 5 loud frames in a row (100 ms). The clip is kept from 100 ms before the first such run to 100 ms after the last one. If a clip has no such run, it is used untrimmed and the run is flagged.

**13:47 IST, before any stage 1 or stage 2 run. "Stop" pulls the arm back to where it was; S2-1 is split in two.**
Reason: function tests showed that the plan's hold action (zeros) does not brake a moving arm. In the injected stop test (development run `dev_inject`, `inject_stop_s0`) the arm travelled 32.7 mm after the stop and stayed 27.8 mm away. A scripted probe (arm driven for 1 s, then held; five motions) gave 7 to 48 mm with zeros.
- New hold after "stop": aim at the pose the arm had when the stop took effect. Position part of the action = (stop position − current position) / 0.05 m, limited to ±1. Rotation part = the rotation still needed to get back to the stop orientation, as an axis-angle vector / 0.5 rad, limited to ±1. Gripper = last commanded value. In the same probe, with the position part only, this gave peaks of 5 to 13 mm and 0.1 to 2.4 mm after 1 s for four motions at about 7.7 mm per step; the fifth, at the controller's top speed, peaked at 27 mm and was still 27 mm away after 1 s.
- Idle (before the first command, after the last one, after a refusal): zeros, as before.
- S2-1 becomes two rules. **S2-1a, overrun:** the arm never gets more than 15 mm from the stop point, in all 12 runs. **S2-1b, hold:** from 1 s after the stop takes effect until the continue takes effect, it stays within 5 mm of the stop point, in all 12 runs.
- These two bars were set after seeing the tests above. No registered run existed yet.

**14:58 IST, before any stage 2 run. S2-5 counts what the return can undo, not the 1 s end check.**
Reason: with the arm now held during the end check (it moved only 0.5 to 8 mm), a rehearsal on starts 5 to 9 (computer voice) still lost 5 of 20 single-command goals in that second: stove 1, wine bottle 2, bowl on plate 2. LIBERO counts a success the first moment the goal is true, which can be while the object is still gripped; freezing the robot at that moment is not a fair test of whether the result lasts. The end check was added at 13:55 to catch one thing: the return to the ready pose undoing the first command. That is what S2-5 should count.
- **S2-5 now:** a run counts if both goals were reached in order and the first goal was still true at the moment the second was reached. Bar unchanged: at least 4 of 5 for a pair to be shown.
- "Still true 1 s later, arm held" is still reported for every run, as information, with the note above.
- Clips are still taken only from runs where everything is still true at the end, so a clip never ends on a result that did not last.
- This changes a bar's definition after rehearsal data on other starts and before any registered stage 2 run. Both counts are reported so the effect of the change is visible.

**14:53 IST, after the registered stage 1 run, before any stage 2 run. The arm is held, not left to coast, during the 1 s end check.**
Reason: in registered stage 1 two wine-bottle runs (starts 0 and 2) reached their goal and then lost it during the 1 s end check. The log shows why: under zero actions the arm, still gripping the bottle, coasted 25.8 mm in those 20 steps. The 13:42 probe had already shown that zero actions do not brake a moving arm; the end check (13:55, item 1) wrongly used them. So "still true" was testing my idle action, not the result.
- Change: during the 20 end steps the arm is held at the pose it had when the last goal was reached (the same "aim at a pose" action as the hold after "stop"), gripper kept.
- Stage 1 verdicts are not affected: S1-0 to S1-3 do not use "still true", and the registered stage 1 report stands as it is, with its "still true" column explained by this note. Stage 1 is run once more under a different name (`stage1_endcheck`) only to fill that column and to pick clips; it is not a second attempt at the pass rules.
- Stage 2 (not yet run) uses the corrected end check; S2-5 depends on it.

**14:20 IST, before any registered run. The simulator is paused while the first line is spoken.**
Reason: on 8 Nov the policy's first action came straight after LIBERO's 10 settle steps. Here the robot idled with zero actions while the line played (about 20 to 70 steps), which the code review listed as a difference that could change behaviour. A probe showed what changes during that idling: the gripper opens another 1.2 mm, the joints come fully to rest, and the wrist camera picture shifts slightly (mean pixel change about 1.3 of 255). In the dress rehearsal (computer voice, start 5, not a result) "turn on the stove" succeeded after 22 idle steps and failed after 69; one case, so not evidence by itself.
- Change: until the first line's event takes effect, the loop does not step the simulator; the picture stays as it was after reset and the clock runs on. The first policy action therefore starts from the same state as on 8 Nov. After a refusal the simulator steps with zero actions as before, so S1-3 is still measured on real steps. This removes the first of the two "differences that stay" listed at 14:05; the seeding difference remains.
- This is adopted by design, not by a test. For information only, a development test compares the two ways (commands 7 and 2, starts 5 to 9, 60 idle steps against none); its counts are reported whatever they show.

**14:05 IST, before any registered run. Corrections from a separate code review.**
The review covered the plan and all the code and found places where the code did not measure what a rule says, and places where the plan was unclear. No recording existed yet. Changes:
1. **S1-3** is measured over the 3 s (60 steps) after the refusal takes effect, as the rule says. The largest move over the whole run is also reported.
2. **A run counts for S1-2 and S2-3 only if the robot did the expected command.** A misheard line that sends the robot to a different task is not a success, even if that task is done.
3. **S2-2 is measured from the end of the word** (the last speech frame), so it includes the 100 ms that trimming keeps after it. That 100 ms was not in mind when the 200 ms bar was written. New bar: at most 300 ms in all 12 runs (100 ms kept + recognition + up to one 50 ms step).
4. **"Stop." starts 1.0 s after the robot's first movement** (was 2.0 s). The shortest task, turn on the stove, took 73 steps (3.65 s) in a test, and a spoken stop needs about 0.7 s more to take effect, so 2.0 s could arrive after the task is done. A stop or continue that arrives after the last goal is reached is ignored. A run where that happens is marked "stop not exercised" and counts as a fail for S2-1a, S2-1b and S2-2.
5. **Goals are not checked during the scripted return** to the ready pose, so a bump on the way cannot count as the policy doing a command.
6. **Recordings come from `record_lines.py`** (all nine lines; the five stage 1 lines are saved under the same names as before). S1-1 is judged on the take I kept for each line. Every earlier take is reported with what was heard.
7. **Clips:** a clip comes from the lowest-numbered start whose run did the expected commands and whose goals still hold at the end. The final badge is DONE only in that case. The clip shows "Transcribing…" until Whisper has finished. One frame is written before the first step, so every picture matches its clock time.
8. **Fingerprints** of the code are written before a stage starts, not after.
9. S2-3 needs all 12 runs to exist.

Two differences from the 8 Nov runs that stay, and are stated with any result: the robot waits about 1.5 to 2.5 s of idle steps while the line is spoken before its first action; and each run is seeded on its own, so only success rates compare with 8 Nov, not which starts succeed.

**13:55 IST, before any registered run. Three refinements after the development tests of 13:50; each has its decision rule fixed here, before its test.**
Development results so far (starts 5 to 9, injected sentences, tuned settings): new hold on start 5: overrun 11.5 mm, hold offset 2.1 mm, run still succeeded. Two commands in a row: without the return 0 of 10; with the return 5 of 5 for "0 then 7" and 1 of 5 for "7 then 2". In two "7 then 2" runs the returning arm dragged the stove knob and the burner went off again.
1. **A goal must still hold at the end.** After the last goal is reached the loop runs 20 more idle steps (1 s), then checks every goal of the plan again. Every run reports both "reached" and "still true at the end". Stage 1 and 2a rules keep counting "reached", as LIBERO does, so they compare with 8 Nov. **S2-5 now counts a run only if both goals are still true at the end.**
2. **Hold: brake first, then straighten.** Variant A applies the rotation part from the first held step (as tested). Variant B applies it from the 11th held step. Test: injected stop runs on starts 5, 6 and 7 for each variant. The variant with the smaller largest overrun is used, provided its hold offset stays within 5 mm in all three; a tie keeps A.
3. **Return: let go and lift first.** Variant A is the return as tested. Variant B first opens the gripper and lifts the arm 5 cm (at most 20 steps), then goes to the ready pose. Test: both pairs on starts 5 to 9 for each variant, counted by "both goals still true at the end". B is used only if it gives more such runs than A, summed over both pairs.

**13:47 IST, before any stage 2 run. A test decides whether the arm returns to its ready pose between two commands.**
Reason: one injected two-command try (7 then 2, start 0, uncompiled) did the first command and failed the second. The policy's training demonstrations all begin from the ready pose, and after the first command the arm is somewhere else.
- Ready-pose return: after a command's goal is reached and another is queued, a scripted move takes the arm back to the pose it had at the start of the run, with the gripper open, using the same "aim at a pose" action as the hold. It ends when the arm is within 5 mm and 0.05 rad, or after 100 steps. State name: RETURNING.
- Decision rule, fixed now: a development test on starts 5 to 9 (not used in the registered runs) runs both pairs with injected sentences, with and without the return. Stage 2b uses the return only if, summed over the two pairs, it gives more successes than without. Otherwise stage 2b runs as first planned. The development counts are reported either way.

Development runs (`dev_*`), rehearsals and preflight checks are not included in this repository; only the registered runs and the second take are.
