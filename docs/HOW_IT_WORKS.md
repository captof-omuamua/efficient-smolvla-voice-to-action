# How it works, and why it is built this way

Numbers and their sources are in [RESULTS_SPEED.md](RESULTS_SPEED.md), [RESULTS_VOICE.md](RESULTS_VOICE.md) and [GPU_SHARING.md](GPU_SHARING.md).

## The robot model

SmolVLA is a vision-language-action model: it takes camera pictures and an instruction and outputs robot movements. The checkpoint here, `lerobot/smolvla_libero`, was fine-tuned by LeRobot on LIBERO's demonstrations. I did not train anything.

- It reads two camera pictures, the arm's state (8 numbers) and the instruction text. That reading is the "prefix".
- An action expert then produces the next 50 small arm movements (7 numbers each) by denoising: it starts from random numbers and nudges them towards a good action. The default is 10 nudges (flow matching, Euler steps).
- The robot uses 10 of those 50 movements, which is half a second at 20 Hz, and then decides again. One model call is one decision.

## What Tuned changes

Same model, same weights.

1. **One denoising step instead of 10.** About ten times less work in the action part. It is cruder, so I checked accuracy on 400 episodes. (Existing LeRobot setting, `num_steps`.)
2. **Whole-model compile.** `torch.compile` in `max-autotune` mode builds one optimised program instead of running operation by operation. The first start takes about 2.5 minutes; later starts load the result in about 6.5 s. (Existing LeRobot setting, `compile_model`.)
3. **Both cameras in one vision call**, instead of one call per camera. This is my port to SmolVLA of the pattern merged for PI0/PI0.5 in LeRobot PR #4426. It is worth 0.6 to 1.4 ms.

The steps add up as: 111.0 ms (Original), 27.9 ms with one step and batched cameras, 14.4 ms with compile on top.

LeRobot PR #4634 by NotNANtoN keeps the stored reading of the scene (the key/value cache) a fixed shape, so that the 10-step denoise loop can be compiled. It was part of the code I measured. Its gain is at 10 steps, where it gave 2.2× end to end on this laptop; at one step there is little loop left to speed up, so the Tuned settings also run on plain LeRobot main. See [TUNED_WITHOUT_PR4634.md](TUNED_WITHOUT_PR4634.md).

## The voice pipeline

- **Speech to text.** Whisper large-v3-turbo (`voice/speech.py`), about 60 ms.
- **Router** (`voice/router.py`). Word rules match clear sentences in under a millisecond. Unclear sentences go to Qwen3.5-4B, which answers a numbered multiple-choice question ("which of these 10 commands, or none?") in at most 4 new tokens, about 100 ms. An embedding match was tried first and dropped: it answered only 23% of rephrased commands safely on the development sentences.
- **Commands** (`voice/commands.py`). "Stop" is a word rule and "continue" is an exact phrase. A sentence with a joining word such as "then" is split, and if the two halves route to different commands they run in order.
- **Robot loop** (`voice/voice_robot.py`). The simulator clock runs in 50 ms steps. Recorded lines are played in at set times. Whisper and the router run for real and are timed, and the command takes effect at the first step after the end of speech plus the measured time.

## Design decisions

| Decision | Why |
|---|---|
| Measure speed, accuracy and energy together, on the same setup | A speed-up that costs accuracy or energy is a trade-off |
| Setups take turns in one session, with a repeat session | Heat and clock drift affect every setup equally |
| The same 400 starting scenes for both setups, with a paired test | It measures the difference per episode, not two noisy totals |
| Check one step on 400 episodes | Fewer denoising steps is cruder and could have lowered success |
| Keep compile despite the slow first start | Each decision is about twice as fast, and a warm start takes about 6.5 s |
| Quote the gain per decision | The simulator waits for the model, so the gain is reaction time and GPU room, not a task finished 7.7× sooner |
| Only 10 commands | The robot model only learned those 10 sentences in this scene |
| Word rules first, language model only when unsure | 88% of rephrased commands are answered without the language model, and the language model can answer "none" |
| Refuse instead of guessing | For a robot, doing nothing is safer than doing the wrong thing |
| Stop steers back to the pose at the stop | Zero actions do not brake LIBERO's arm; it coasted 22–48 mm |
| Return to the ready pose between two commands | The policy only works from its training start pose: 0 of 10 without the return |
| Recorded voice on a simulated clock | Runs are repeatable, with real measured Whisper and router times |
| Pass rules written before the runs, amendments dated | Results cannot be bent to fit afterwards |
| Turn-taking for the voice models | Sharing the GPU made the policy's slowest decisions 2 to 2.8× slower |
