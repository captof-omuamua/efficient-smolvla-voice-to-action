"""The robot loop: recorded spoken lines in, robot runs and a results table out (VOICE_ROBOT_PLAN.md).

One process loads Whisper, the router and the tuned SmolVLA. Each run:
  - resets one LIBERO-Goal scene (start k: LIBERO's starting layout k, seeds 1000 + k),
  - plays a script of recorded lines on the simulator clock (step k is k x 50 ms),
  - when a line ends, Whisper and parse() run for real and are timed; the event takes effect at the
    first step at or after (end of the line + measured time),
  - each step either holds the arm or lets the policy act on Session.instruction(),
  - after each step the current command's goal is checked in the simulator.
The policy, its processors and its observations are built as in LeRobot's lerobot_eval.py (eval_main
and rollout); the only change is that observation["task"] comes from Session.instruction().

Usage: python voice_robot.py --stage stage1 --name stage1      (see STAGES at the bottom)
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import libero.libero.envs.bddl_utils as BDDLUtils
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from libero.libero import benchmark, get_libero_path

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import policy_output_to_transition, transition_to_policy_action
from lerobot.sim.integrations import libero as libero_sim
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

from audio_trim import trim_silence
from commands import Session, parse
from router import Router
from speech import RATE, Transcriber, load_raw
from tasks import COMMANDS

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("OUT_DIR", ROOT.parent / "runs")) / "voice_robot"  # new runs; committed ones are in results/
FPS = 20  # LIBERO steps at 20 Hz, so one step is 50 ms
STEP_S = 1 / FPS
MAX_ACTING_STEPS = 300  # per command; hold steps do not count
IDLE_END_STEPS = 60  # with nothing left to do or hear, the run ends after 3 s
MAX_RUN_STEPS = 1500  # safety cap: 75 s of simulator time
BASE_SEED = 1000  # as on 8 Nov
MAX_MOVE_M = 0.05  # the controller's position change for an action of 1 (per step)
MAX_TURN_RAD = 0.5  # the controller's rotation change for an action of 1 (per step)
HOLD_CHECK_S = 1.0  # hold offset (S2-1b) is measured from 1 s after the stop takes effect
RETURN_MAX_STEPS = 100  # a ready-pose return ends after 100 steps at most
RETURN_NEAR_MM = 5.0  # ... or when the arm is within 5 mm
RETURN_NEAR_RAD = 0.05  # ... and 0.05 rad of the ready pose
GRIPPER_OPEN = -1.0
# Pause the simulator until the first line's event takes effect, so the policy starts from the same state as
# on 8 Nov, straight after LIBERO's 10 settle steps (amendment 14:20). The clock and the frames run on.
PAUSE_UNTIL_FIRST_EVENT = True
END_CHECK_STEPS = 20  # after the last goal: 1 s of idle steps, then every goal is checked again (13:55)
HOLD_BRAKE_STEPS = 10  # hold variant B: rotation part only from the 11th held step
LIFT_M = 0.05  # return variant B: lift 5 cm straight up first ...
LIFT_MAX_STEPS = 20  # ... for at most 20 steps, or until within 5 mm (RETURN_NEAR_MM)

# Chosen by the development tests of the 13:55 amendment (results/voice_robot/dev_hold, dev_return).
HOLD_VARIANT = "A"  # "A": rotation part from the first held step; "B": brake first, then straighten
RETURN_VARIANT = "B"  # "A": straight to the ready pose; "B": open the gripper and lift 5 cm first
DEV_COUNTS_HOLD = ("starts 5-7, injected stop and continue: variant A largest overrun 11.47 mm "
                   "(11.47, 11.10, 10.67), hold offset 2.11, 2.70, 1.43 mm, 3 of 3 succeed; variant B largest "
                   "overrun 12.35 mm (12.35, 11.95, 11.33), hold offset 2.08, 2.32, 1.34 mm, 3 of 3 succeed. "
                   "A has the smaller largest overrun, so A is used.")
DEV_COUNTS_RETURN = ("starts 5-9, injected sentences, both goals still true at the end (both reached): "
                     "variant A: 7 then 2: 0 of 5 (1), 0 then 7: 5 of 5 (5), total 5; variant B: 7 then 2: "
                     "2 of 5 (4), 0 then 7: 5 of 5 (5), total 7. B gives more, so B is used.")

# The 8 Nov tuned run's settings (logs/final_2026-11-07_libero_final_2026-11-07_B_tuned.log, line 5).
TUNED_ARGS = [
    "--policy.path=lerobot/smolvla_libero",
    "--policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d",
    "--policy.compile_model=true",
    "--policy.num_steps=1",
    "--policy.empty_cameras=0",
    "--policy.n_action_steps=10",
    "--policy.device=cuda",
    "--env.type=libero",
    "--env.task=libero_goal",
    '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}',
]


# ---------- the recorded lines ----------




def save_raw(audio, path):
    """Write floats in [-1, 1] back as 16 kHz mono 16-bit raw (exact for audio read by load_raw)."""
    path.write_bytes(np.round(audio * 32768).clip(-32768, 32767).astype(np.int16).tobytes())


@dataclass
class Line:
    """One spoken line in a run's script."""

    file: str | None = None  # a recording (16 kHz mono 16-bit raw)
    trigger: str = "start"  # "start", "first_action" or "stop_effect"
    delay_s: float = 0.0  # seconds after the trigger
    text: str | None = None  # test only: an injected sentence instead of a recording (no Whisper)


@dataclass
class RunSpec:
    run_id: str
    start: int  # LIBERO starting layout; seeds are BASE_SEED + start
    env_task: int  # the scene is built for this command (the plan's last command)
    lines: list
    expected: list | None = None  # expected plan, for the report (None: a refusal is expected)
    ready_return: bool = False  # between two commands, move back to the ready pose first (amendment 13:47)
    hold_variant: str = ""  # "" uses HOLD_VARIANT; "A" or "B" only in the development test
    return_variant: str = ""  # "" uses RETURN_VARIANT; "A" or "B" only in the development test
    idle_end_steps: int = IDLE_END_STEPS  # idle steps before the run ends (refusal run: 100, to log creep)
    pause: bool | None = None  # None uses PAUSE_UNTIL_FIRST_EVENT; True or False only in the development test


# ---------- loading the models ----------


def tuned_config(compile_model=True):
    """LeRobot reads its settings from the command line, so hand it the 8 Nov run's arguments."""
    arguments = list(TUNED_ARGS)
    if not compile_model:  # development only; every reported run uses the tuned settings
        arguments[2] = "--policy.compile_model=false"
    saved, sys.argv = sys.argv, ["voice_robot"] + arguments

    @parser.wrap()
    def read(cfg: EvalPipelineConfig):
        return cfg

    try:
        return read()
    finally:
        sys.argv = saved


class Models:
    """Whisper, the router and the tuned policy with its processors, built as eval_main builds them."""

    def __init__(self, cfg):
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True  # as eval_main
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
        self.transcriber = Transcriber()
        self.router = Router()
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        self.policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        self.policy.eval()
        overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            pretrained_revision=cfg.policy.pretrained_revision,
            preprocessor_overrides=overrides,
        )
        config = self.policy.config
        print(f"SmolVLA: num_steps={config.num_steps}, empty_cameras={config.empty_cameras}, "
              f"compile_model={config.compile_model}, n_action_steps={config.n_action_steps}, "
              f"observation {cfg.env.observation_width}x{cfg.env.observation_height}", flush=True)

    def warm_up(self, scene):
        """Run each model before any timed call: the first calls are slow (and compile the policy)."""
        torch.backends.cudnn.benchmark = False  # the voice models run without cuDNN autotuning (see run_once)
        self.transcriber(np.zeros(RATE, dtype=np.float32))
        warm_up_router(self.router)
        torch.backends.cudnn.benchmark = True  # the policy keeps LeRobot's setting
        obs, _ = scene.env.reset(seed=BASE_SEED)
        for _ in range(3):  # three new chunks (a chunk is computed on every call after reset)
            self.policy.reset()
            self.act(obs, COMMANDS[0])
        self.policy.reset()

    def act(self, raw_obs, instruction):
        """One policy step, as in rollout. Returns (action for the simulator, ms if a new chunk was computed)."""
        observation = prepare_canonical_observation(batch_of_one(libero_sim.canonical_observation(raw_obs)))
        observation["task"] = [instruction]  # the one change from rollout: the task comes from the Session
        observation = self.env_pre(observation)
        observation = self.pre(observation)
        new_chunk = self.policy._check_get_actions_condition()  # True when the action queue is empty
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            action = transition_to_policy_action(policy_output_to_transition(self.policy.select_action(observation)))
        torch.cuda.synchronize()
        call_ms = (time.perf_counter() - begin) * 1000 if new_chunk else None
        action = self.post(action)
        action = self.env_post({ACTION: action})[ACTION]
        return action.to("cpu").numpy()[0], call_ms


WARM_UP_SENTENCES = ["could you hand me the spoon", "Close the top drawer.",  # all go to the language model
                     "Please take the cream cheese out of the bowl now.",
                     "Would you kindly move the wine bottle away from the rack and set it on the floor?"]


def warm_up_router(router):
    """Run the language model on a few sentence lengths before any timed call."""
    for sentence in WARM_UP_SENTENCES:
        router.route(sentence)


def batch_of_one(observation):
    """Add the batch dimension, as LeRobot's vector env does for a batch of one env (np.stack)."""
    return {key: np.stack([value]) for key, value in observation.items()}


# ---------- the simulator scene ----------


class Scene:
    """One LIBERO-Goal scene, built as LeRobot's vector env builds it, plus the goal of each command."""

    def __init__(self, cfg, task_id):
        settings = dataclasses.replace(cfg.env, task_ids=[task_id]).sim_config()
        factories = libero_sim.env_fns(task="libero_goal", n_envs=1, **settings.kwargs)
        self.env = factories["libero_goal"][task_id][0]()
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        self.goals = []
        for i in range(len(COMMANDS)):
            task = suite.get_task(i)
            path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self.goals.append(BDDLUtils.robosuite_parse_problem(path)["goal_state"])

    def reset(self, start):
        self.env.init_state_id = start  # LIBERO's starting layout number `start`
        obs, _ = self.env.reset(seed=BASE_SEED + start)
        return obs

    def goal_reached(self, command):
        inner = self.env._env.env
        return all(inner._eval_predicate(goal) for goal in self.goals[command])

    def frame(self):
        """A 640x480 picture from the agentview camera; the policy's own camera images are not touched."""
        image = self.env._env.sim.render(camera_name="agentview", width=640, height=480)
        return image[::-1, ::-1]  # flipped as LiberoEnv.render flips it

    def close(self):
        self.env.close()


# ---------- one run ----------


def arm_position(obs):
    return np.asarray(obs["robot_state"]["eef"]["pos"], dtype=float)


def arm_rotation(obs):
    """The gripper's orientation as a 3x3 rotation matrix (world frame)."""
    return np.asarray(obs["robot_state"]["eef"]["mat"], dtype=float)


def turn_needed(rotation, current):
    """The rotation still needed to get from `current` to `rotation`, as an axis-angle vector (radians)."""
    return T.quat2axisangle(T.mat2quat(rotation @ current.T))


def toward(obs, position, rotation):
    """The six movement numbers that aim the arm at a pose (amendment 13:47).

    Position: (target - current) / 0.05 m. Rotation: the turn still needed / 0.5 rad. Each limited to +-1.
    The controller moves the goal by these amounts in the world frame, so this pulls the arm to the pose.
    """
    move = (position - arm_position(obs)) / MAX_MOVE_M
    turn = turn_needed(rotation, arm_rotation(obs)) / MAX_TURN_RAD
    return np.clip(np.concatenate([move, turn]), -1, 1)


def run_once(models, scene, spec, folder):
    """Play one run's script on the simulator clock. Writes frames.mp4 and timeline.json to folder."""
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(BASE_SEED + spec.start)
    models.policy.reset()
    models.pre.reset()
    models.post.reset()
    obs = scene.reset(spec.start)
    session = Session()

    # Load and trim each recorded line; injected test lines have no audio.
    lines = []
    for i, line in enumerate(spec.lines):
        entry = {"line": line, "start_s": None, "end_s": None, "event": None, "effect_s": None, "ignored": False}
        if line.file is not None:
            recorded = load_raw(line.file)
            entry["audio"], entry["speech_end_s"], entry["trim_flag"] = trim_silence(recorded)
            if entry["trim_flag"]:
                print(f"  {spec.run_id} TRIM FLAG {line.file}: {entry['trim_flag']}", flush=True)
            entry["recorded_s"] = len(recorded) / RATE
            entry["length_s"] = len(entry["audio"]) / RATE
            entry["file"] = str((folder / f"line_{i + 1}_trimmed.raw").resolve())
            save_raw(entry["audio"], Path(entry["file"]))
        else:
            entry.update(audio=None, recorded_s=0.0, length_s=0.0, file=None, trim_flag="", speech_end_s=0.0)
        if line.trigger == "start":
            entry["start_s"] = line.delay_s
            entry["end_s"] = line.delay_s + entry["length_s"]
        lines.append(entry)

    writer = imageio.get_writer(str(folder / "frames.mp4"), fps=FPS, codec="libx264", pixelformat="yuv420p")
    writer.append_data(scene.frame())  # frame k is the picture at k x 50 ms, so frame 0 is before any step
    positions = [arm_position(obs)]  # positions[k] is the arm before step k
    ready_pose = (arm_position(obs), arm_rotation(obs))  # where the arm starts: the ready pose
    stop_pose = None  # the pose when the stop took effect
    hold_variant = spec.hold_variant or HOLD_VARIANT
    return_variant = spec.return_variant or RETURN_VARIANT
    held_steps = 0  # held steps since the stop took effect
    returning = None  # "lift" or "ready" while moving back between two commands; None otherwise
    returning_steps = 0  # steps in the current part of the return
    lift_pose = None  # return variant B: 5 cm above where the arm let go
    end_steps = None  # idle steps after the last goal was reached; None before that
    goals_hold_at_end = False
    end_pose = None  # the pose when the last goal was reached; held during the end check (amendment 14:53)
    end_check_move_mm = math.nan  # the arm's largest distance from end_pose during the end check
    earlier_goals_hold = False  # when the last goal was reached, were all earlier goals still true? (14:58)
    states = []
    call_ms = []
    gripper = -1.0  # the hold action's gripper value before the first policy action
    acting_steps = 0  # policy steps for the current command
    moving_steps = 0  # policy steps in the whole run
    first_action_s = None
    stop_effect_s = None
    stop_step = continue_step = None
    refuse_step = None  # the step at which a refusal took effect
    pause = PAUSE_UNTIL_FIRST_EVENT if spec.pause is None else spec.pause
    first_event_done = False  # becomes True when the first line's event takes effect
    idle_steps = 0
    result = "running"
    step = 0

    while result == "running":
        now = step * STEP_S

        # Lines whose trigger happened get their start time.
        for entry in lines:
            if entry["start_s"] is None:
                trigger_s = {"first_action": first_action_s, "stop_effect": stop_effect_s}[entry["line"].trigger]
                if trigger_s is not None:
                    entry["start_s"] = trigger_s + entry["line"].delay_s
                    entry["end_s"] = entry["start_s"] + entry["length_s"]

        # A line that has ended is heard now: Whisper and parse run for real and are timed.
        for entry in lines:
            if entry["end_s"] is not None and entry["event"] is None and entry["end_s"] <= now:
                # cuDNN autotuning off for the voice models: with it on, every new sentence length re-tunes
                # (measured: first calls 407-910 ms against 95-306 ms); back on for the policy, as eval_main.
                torch.backends.cudnn.benchmark = False
                if entry["audio"] is not None:
                    entry["heard"], entry["speech_ms"] = models.transcriber(entry["audio"])
                else:
                    entry["heard"], entry["speech_ms"] = entry["line"].text, 0.0
                entry["event"] = parse(entry["heard"], models.router)
                torch.backends.cudnn.benchmark = True
                entry["due_s"] = entry["end_s"] + (entry["speech_ms"] + entry["event"].ms) / 1000
                print(f"  {spec.run_id} t={now:.2f}s heard {entry['heard']!r} -> {entry['event']}", flush=True)

        # Events take effect at the first step at or after their due time.
        for entry in lines:
            if entry["event"] is not None and entry["effect_s"] is None and not entry["ignored"] \
                    and entry["due_s"] <= now + 1e-9:
                event = entry["event"]
                if end_steps is not None and event.kind in ("stop", "continue"):
                    entry["ignored"] = True  # the last goal was already reached: nothing left to stop
                    print(f"  {spec.run_id} t={now:.2f}s {event.kind} ignored: the last goal was already reached",
                          flush=True)
                    continue
                entry["effect_s"] = now
                first_event_done = True
                session.on_event(event)
                if event.kind in ("plan", "stop"):
                    models.policy.reset()  # a command starts, or a stop: drop queued movements
                if event.kind == "plan":
                    acting_steps = 0
                    returning = None
                if event.kind == "stop":
                    stop_pose = (arm_position(obs), arm_rotation(obs))  # hold here
                    held_steps = 0
                if event.kind == "stop" and stop_step is None:
                    stop_effect_s, stop_step = now, step
                if event.kind == "continue" and stop_step is not None and continue_step is None:
                    continue_step = step
                if event.kind == "refuse" and refuse_step is None:
                    refuse_step = step

        # Finish, hold, wait, return to the ready pose, or act.
        instruction = session.instruction()
        if end_steps is not None:  # all goals reached: hold the arm for 1 s, then check them again
            move = toward(obs, *end_pose)  # the same hold as after "stop": zeros would let the arm coast
            if hold_variant == "B" and end_steps < HOLD_BRAKE_STEPS:
                move[3:] = 0
            action = np.array([*move, gripper], dtype=np.float32)
            state = "DONE"
        elif session.holding:  # aim at the pose the arm had when the stop took effect
            move = toward(obs, *stop_pose)
            if hold_variant == "B" and held_steps < HOLD_BRAKE_STEPS:
                move[3:] = 0  # variant B: brake first, straighten from the 11th held step
            action = np.array([*move, gripper], dtype=np.float32)
            held_steps += 1
            state = "HOLDING"
        elif instruction is None:  # idle: zeros, as before
            action = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)
            state = "WAITING"
        elif returning is not None:  # scripted move between two commands, gripper open
            target = lift_pose if returning == "lift" else ready_pose
            action = np.array([*toward(obs, *target), GRIPPER_OPEN], dtype=np.float32)
            gripper = GRIPPER_OPEN
            returning_steps += 1
            state = "RETURNING"
        else:
            action, ms = models.act(obs, instruction)
            if ms is not None:
                call_ms.append(ms)
            gripper = float(action[-1])
            acting_steps += 1
            moving_steps += 1
            if first_action_s is None:
                first_action_s = now
            state = "MOVING"
        command = None if session.current is None else COMMANDS[session.current]
        if not states or (states[-1]["state"], states[-1]["command"]) != (state, command):
            states.append({"from_step": step, "state": state, "command": command})

        if pause and not first_event_done:
            positions.append(positions[-1])  # paused: the simulator is not stepped; the clock runs on
        else:
            obs, _, _, _, _ = scene.env.step(action)
            positions.append(arm_position(obs))
        writer.append_data(scene.frame())  # while paused this is the unchanged picture
        step += 1

        # After the last goal: 1 s idle, then check every goal of the plan again (amendment 13:55).
        # The scene is a plain LiberoEnv, which never resets itself, so stepping after a success is safe.
        if end_steps is not None:
            end_steps += 1
            if end_steps >= END_CHECK_STEPS:
                still_true = {c: scene.goal_reached(c) for c in session.done}
                goals_hold_at_end = all(still_true.values())
                held = positions[-END_CHECK_STEPS:]  # the arm after each end-check step
                end_check_move_mm = max(float(np.linalg.norm(p - end_pose[0])) * 1000 for p in held)
                print(f"  {spec.run_id} step {step}: goals still true after 1 s: {still_true} "
                      f"(arm at most {end_check_move_mm:.2f} mm from where the last goal was reached)", flush=True)
                result = "success"
            continue

        # The lift (variant B) ends within 5 mm or after 20 steps; then the move to the ready pose starts.
        if returning == "lift":
            near_mm = np.linalg.norm(arm_position(obs) - lift_pose[0]) * 1000
            if near_mm <= RETURN_NEAR_MM or returning_steps >= LIFT_MAX_STEPS:
                print(f"  {spec.run_id} step {step}: lifted after {returning_steps} steps ({near_mm:.1f} mm off)",
                      flush=True)
                returning, returning_steps = "ready", 0
        # The return ends near the ready pose, or after 100 steps; then the next command starts afresh.
        elif returning == "ready":
            near_mm = np.linalg.norm(arm_position(obs) - ready_pose[0]) * 1000
            near_rad = np.linalg.norm(turn_needed(ready_pose[1], arm_rotation(obs)))
            if (near_mm <= RETURN_NEAR_MM and near_rad <= RETURN_NEAR_RAD) or returning_steps >= RETURN_MAX_STEPS:
                print(f"  {spec.run_id} step {step}: back at the ready pose after {returning_steps} steps "
                      f"({near_mm:.1f} mm, {near_rad:.3f} rad)", flush=True)
                returning = None
                models.policy.reset()

        # Check the current command's goal.
        # Not during the scripted return: a bump on the way must not count as the policy doing a command.
        if session.current is not None and not session.holding and returning is None \
                and scene.goal_reached(session.current):
            print(f"  {spec.run_id} step {step}: reached '{COMMANDS[session.current]}'", flush=True)
            session.on_success()
            models.policy.reset()
            acting_steps = 0
            if session.current is None and not session.queue:
                end_steps = 0  # every goal reached: start the 1 s end check
                end_pose = (arm_position(obs), arm_rotation(obs))
                earlier_goals_hold = all(scene.goal_reached(c) for c in session.done[:-1])  # True for one command
            elif spec.ready_return and return_variant == "B":
                lift_pose = (arm_position(obs) + np.array([0, 0, LIFT_M]), arm_rotation(obs))
                returning, returning_steps = "lift", 0  # let go and lift first, then the ready pose
            elif spec.ready_return:
                returning, returning_steps = "ready", 0  # another command is next: ready pose first
        elif acting_steps >= MAX_ACTING_STEPS:
            result = "failed: 300 acting steps"

        # With nothing left to do or hear, end after 3 s.
        # (A line whose trigger has not happened cannot happen while idle or holding with nothing coming.)
        waiting = any(e["start_s"] is not None and e["effect_s"] is None for e in lines)
        idle = session.current is None or session.holding
        idle_steps = idle_steps + 1 if idle and not waiting else 0
        if result == "running" and idle_steps >= spec.idle_end_steps:
            result = "ended: idle"
        if result == "running" and step >= MAX_RUN_STEPS:
            result = "failed: run step cap"
    writer.close()

    success = result == "success"  # every goal of the plan was reached
    plan = next((e["event"].commands for e in lines if e["event"] and e["event"].kind == "plan"), [])
    did_expected = success and plan == spec.expected  # it did the commands the line asked for
    # Final badge: DONE only if it did the expected commands and they still hold; a run with no plan and
    # no expected command (a refusal) ends WAITING; anything else is FAILED.
    if did_expected and goals_hold_at_end:
        final_state = "DONE"
    elif not plan and spec.expected is None:
        final_state = "WAITING"
    else:
        final_state = "FAILED"
    states.append({"from_step": step, "state": final_state, "command": None})
    start_pos = positions[0]
    run = {
        "run_id": spec.run_id, "start": spec.start, "result": result, "success": success,
        "did_expected": did_expected, "steps": step,
        "moving_steps": moving_steps, "goals_hold_at_end": goals_hold_at_end,
        "end_check_move_mm": end_check_move_mm, "earlier_goals_hold": earlier_goals_hold,
        "plan": plan, "done": list(session.done), "expected": spec.expected,
        "policy_calls": len(call_ms), "call_ms_median": statistics.median(call_ms) if call_ms else math.nan,
        "max_move_mm": max(np.linalg.norm(p - start_pos) for p in positions) * 1000,
        "refusal_move_mm": math.nan, "creep_60_mm": math.nan, "creep_100_mm": math.nan,
        "stop_exercised": "", "stop_delay_ms": math.nan, "stop_delay_clip_ms": math.nan,
        "overrun_mm": math.nan, "hold_offset_mm": math.nan, "call_ms": call_ms,
    }
    if refuse_step is not None:
        # S1-3: the largest arm move in the 60 steps (3 s) after the refusal took effect.
        after = [float(np.linalg.norm(p - positions[refuse_step]) * 1000) for p in positions[refuse_step:]]
        run["refusal_move_mm"] = max(after[: 60 + 1])
        run["creep_60_mm"] = after[min(60, len(after) - 1)]  # where the arm is after 60 idle steps
        run["creep_100_mm"] = after[min(100, len(after) - 1)]  # ... and after 100
        print(f"  {spec.run_id}: refusal at step {refuse_step}; largest move in 60 steps "
              f"{run['refusal_move_mm']:.3f} mm; creep {run['creep_60_mm']:.3f} mm after 60 steps, "
              f"{run['creep_100_mm']:.3f} mm after {min(100, len(after) - 1)} steps", flush=True)
    # A run whose script plays a stop (stage 2a: the line timed from the first movement), or that heard one.
    tries_stop = any(line.trigger == "first_action" for line in spec.lines) or \
        any(e["event"] is not None and e["event"].kind == "stop" for e in lines)
    stop_entry = next((e for e in lines if e["event"] and e["event"].kind == "stop" and e["effect_s"] is not None), None)
    if tries_stop:
        run["stop_exercised"] = "yes" if stop_entry is not None else "stop not exercised"
    if stop_entry is not None:
        word_end_s = stop_entry["start_s"] + stop_entry["speech_end_s"]  # end of the last speech frame
        run["stop_delay_ms"] = (stop_entry["effect_s"] - word_end_s) * 1000  # S2-2: from the end of the word
        run["stop_delay_clip_ms"] = (stop_entry["effect_s"] - stop_entry["end_s"]) * 1000  # from the clip's end
        last = continue_step if continue_step is not None else step
        held = positions[stop_step: last + 1]
        drift_mm = [float(np.linalg.norm(p - positions[stop_step]) * 1000) for p in held]
        run["overrun_mm"] = max(drift_mm)  # S2-1a: from the stop until the continue takes effect
        settled = drift_mm[round(HOLD_CHECK_S * FPS):]  # S2-1b: from 1 s after the stop
        run["hold_offset_mm"] = max(settled) if settled else math.nan
        # How far the arm is from where it was at the stop, at each held step (kept to explain the number).
        (folder / "hold_drift_mm.json").write_text(json.dumps([round(d, 3) for d in drift_mm]))

    timeline = {
        "fps": FPS, "steps": step, "success": success, "goals_hold_at_end": goals_hold_at_end,
        "expected": spec.expected, "did_expected": did_expected, "earlier_goals_hold": earlier_goals_hold, "plan": plan, "done": list(session.done),
        "lines": [
            {"file": e["file"], "recorded_s": e["recorded_s"], "trim_flag": e["trim_flag"],
             "speech_end_s": e["speech_end_s"], "start_s": e["start_s"], "end_s": e["end_s"],
             "heard": e["heard"], "event": e["event"].kind, "commands": e["event"].commands,
             "level": e["event"].level, "speech_ms": e["speech_ms"], "parse_ms": e["event"].ms,
             "effect_s": e["effect_s"]}
            for e in lines if e["event"] is not None and not e["ignored"]
        ],
        "states": states,
    }
    (folder / "timeline.json").write_text(json.dumps(timeline, indent=1))
    print(f"  {spec.run_id}: {result}, {step} steps, plan {plan}, done {session.done}, "
          f"median call {run['call_ms_median']:.1f} ms over {len(call_ms)} calls", flush=True)
    return run, lines


# ---------- the stages ----------


def share_clip(command, k=0):
    """Computer-generated stand-in clips: k=0 is LIBERO's wording, k=1 adds "please"."""
    return str(ROOT / "data" / "share_audio" / f"c{command:02d}_{k}.raw")


STAGE1_LINES = [(1, 7), (2, 2), (3, 8), (4, 5)]  # (line number, expected command); line 5 must be refused


DATA = ROOT / "data"  # the recordings: data/voice_test/ and data/voice_lines/


def stage1_runs(audio_root=DATA, starts=(0, 1, 2)):
    folder = audio_root / "voice_test"
    runs = []
    for number, command in STAGE1_LINES:
        for start in starts:
            line = Line(file=str(folder / f"line_{number}.raw"))
            runs.append(RunSpec(f"line{number}_cmd{command}_s{start}", start, command, [line], [command]))
    refusal = RunSpec(f"line5_refuse_s{starts[0]}", starts[0], 0, [Line(file=str(folder / "line_5.raw"))], None)
    refusal.idle_end_steps = 100  # idle 100 steps after the refusal, so the arm's creep can be logged
    runs.append(refusal)
    return runs


STOP_AFTER_FIRST_MOVE_S = 1.0  # "Stop." starts 1.0 s after the robot's first movement (amendment 14:05)
CONTINUE_AFTER_STOP_S = 3.0  # "Continue." starts 3.0 s after the stop takes effect


def stage2a_runs(audio_root=DATA, starts=(0, 1, 2)):
    """The 12 stage 1 command runs again, with "Stop." and "Continue." played in."""
    folder = audio_root / "voice_test"
    voice = audio_root / "voice_lines"
    runs = []
    for number, command in STAGE1_LINES:
        for start in starts:
            lines = [Line(file=str(folder / f"line_{number}.raw")),
                     Line(file=str(voice / "stop.raw"), trigger="first_action", delay_s=STOP_AFTER_FIRST_MOVE_S),
                     Line(file=str(voice / "continue.raw"), trigger="stop_effect", delay_s=CONTINUE_AFTER_STOP_S)]
            runs.append(RunSpec(f"line{number}_cmd{command}_stop_s{start}", start, command, lines, [command]))
    return runs


# Ready-pose return in stage 2b: set by the development test on starts 5 to 9 (amendment 13:47).
STAGE2B_READY_RETURN = True  # development test: 6 of 10 with the return, 0 of 10 without
TWO_STEP_SENTENCES = {  # the stage 2 sentences and their expected plans
    "two_step_1": ("Turn on the stove, then put the wine bottle on top of the cabinet.", [7, 2]),
    "two_step_2": ("Open the middle drawer, then turn on the stove.", [0, 7]),
}


def stage2b_runs(audio_root=DATA, starts=(0, 1, 2, 3, 4)):
    voice = audio_root / "voice_lines"
    runs = []
    for name, (_, plan) in TWO_STEP_SENTENCES.items():
        for start in starts:
            line = Line(file=str(voice / f"{name}.raw"))
            runs.append(RunSpec(f"{name}_s{start}", start, plan[-1], [line], plan, STAGE2B_READY_RETURN))
    return runs


def dev_twostep_runs():
    """Development test for the ready-pose return: injected sentences, starts 5 to 9, with and without."""
    runs = []
    for ready_return in (False, True):
        mode = "return" if ready_return else "direct"
        for name, (sentence, plan) in TWO_STEP_SENTENCES.items():
            for start in range(5, 10):
                runs.append(RunSpec(f"{name}_{mode}_s{start}", start, plan[-1], [Line(text=sentence)], plan,
                                    ready_return))
    return runs


def dev_hold_runs():
    """Development test for the hold (amendment 13:55): injected stop runs, starts 5 to 7, variants A and B."""
    runs = []
    for variant in "AB":
        for start in (5, 6, 7):
            lines = [Line(file=share_clip(7)), Line(text="Stop.", trigger="first_action", delay_s=2.0),
                     Line(text="Continue.", trigger="stop_effect", delay_s=3.0)]
            runs.append(RunSpec(f"hold{variant}_stop_s{start}", start, 7, lines, [7], hold_variant=variant))
    return runs


def dev_return_runs():
    """Development test for the return (amendment 13:55): both pairs, starts 5 to 9, variants A and B."""
    runs = []
    for variant in "AB":
        for name, (sentence, plan) in TWO_STEP_SENTENCES.items():
            for start in range(5, 10):
                runs.append(RunSpec(f"{name}_return{variant}_s{start}", start, plan[-1], [Line(text=sentence)],
                                    plan, ready_return=True, return_variant=variant))
    return runs


def inject_stop5_runs():
    """Injected stop/continue on start 5: the stop at +1.0 s (stage 2a timing), and at +1.7 s (about what
    a spoken "Stop." adds: its own length plus recognition)."""
    runs = []
    for delay_s in (1.0, 1.7):
        lines = [Line(file=share_clip(7)), Line(text="Stop.", trigger="first_action", delay_s=delay_s),
                 Line(text="Continue.", trigger="stop_effect", delay_s=CONTINUE_AFTER_STOP_S)]
        runs.append(RunSpec(f"inject_stop_{delay_s:.1f}s_s5", 5, 7, lines, [7]))
    return runs


def inject_refuse_runs():
    """Refusal path with an injected sentence (no stand-in recording of a refused line exists)."""
    return [RunSpec("inject_refuse_s0", 0, 0, [Line(text="Close the top drawer.")], None, idle_end_steps=100)]


def dev_pause_runs():
    """Information only (amendment 14:20): first line 3.0 s in, commands 7 and 2, starts 5 to 9, pause on/off."""
    runs = []
    for pause in (True, False):
        mode = "pause" if pause else "idle"
        for command in (7, 2):
            for start in range(5, 10):
                line = Line(text=COMMANDS[command], delay_s=3.0)
                runs.append(RunSpec(f"c{command:02d}_{mode}_s{start}", start, command, [line], [command], pause=pause))
    return runs


def func_runs():
    """Functional test (not a result): stand-in clip for command 7 on start 0."""
    return [RunSpec("c07_s0", 0, 7, [Line(file=share_clip(7))], [7])]


def repeat_runs():
    """Repeatability: the same run twice."""
    return [RunSpec(f"c07_s0_{tag}", 0, 7, [Line(file=share_clip(7))], [7]) for tag in "ab"]


def inject_runs():
    """Stop/continue and two-commands path, with injected sentences (no stand-in recordings exist for these)."""
    return [
        RunSpec("inject_stop_s0", 0, 7, [Line(file=share_clip(7)),
                                         Line(text="Stop.", trigger="first_action", delay_s=2.0),
                                         Line(text="Continue.", trigger="stop_effect", delay_s=3.0)], [7]),
        RunSpec("inject_two_s0", 0, 2,
                [Line(text="Turn on the stove, then put the wine bottle on top of the cabinet.")], [7, 2]),
    ]


STAGES = {"stage1": stage1_runs, "stage2a": stage2a_runs, "stage2b": stage2b_runs,
          "func": func_runs, "repeat": repeat_runs, "inject": inject_runs,
          "dev_twostep": dev_twostep_runs, "inject_stop5": inject_stop5_runs,
          "dev_hold": dev_hold_runs, "dev_return": dev_return_runs, "inject_refuse": inject_refuse_runs,
          "dev_pause": dev_pause_runs}


# ---------- tables and report ----------


def write_tables(out, runs, all_lines):
    columns = ["run_id", "start", "result", "success", "did_expected", "earlier_goals_hold", "goals_hold_at_end", "end_check_move_mm", "steps", "moving_steps",
               "plan", "done", "expected", "policy_calls", "call_ms_median", "max_move_mm", "refusal_move_mm",
               "creep_60_mm", "creep_100_mm", "stop_exercised", "stop_delay_ms", "stop_delay_clip_ms",
               "overrun_mm", "hold_offset_mm"]
    with open(out / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for run in runs:
            writer.writerow([run[c] for c in columns])
    with open(out / "events.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "recording", "recorded_s", "trimmed_s", "start_s", "end_s", "heard", "event",
                         "commands", "level", "speech_ms", "parse_ms", "effect_s", "speech_end_s", "trim_flag",
                         "ignored"])
        for run_id, lines in all_lines:
            for e in lines:
                event = e["event"]
                writer.writerow([run_id, e["line"].file or f"injected: {e['line'].text}", e["recorded_s"],
                                 e["length_s"], e["start_s"], e["end_s"], e.get("heard"),
                                 event.kind if event else "not heard", event.commands if event else "",
                                 event.level if event else "", e.get("speech_ms"), event.ms if event else "",
                                 e["effect_s"], e["speech_end_s"], e["trim_flag"], e["ignored"]])


def verdict(passed):
    return "PASS" if passed else "FAIL"


def write_report(out, stage, runs, all_lines, rehearsal=False):
    calls = [ms for run in runs for ms in run["call_ms"]]
    median_ms = statistics.median(calls) if calls else math.nan

    # The header, in order: title, then (stage 2) which development-tested variants are used and why.
    rows = []
    if rehearsal:
        rows += ["REHEARSAL: starts or audio overridden, not a registered run", ""]
    rows += [f"# Voice robot: {stage} ({time.strftime('%d %b %Y %H:%M IST')})", ""]
    if stage == "stage2a":
        rows += [f"Hold after stop: variant {HOLD_VARIANT}. Development test (amendment 13:55): {DEV_COUNTS_HOLD}",
                 f"\"Stop.\" starts {STOP_AFTER_FIRST_MOVE_S} s after the first movement; \"Continue.\" "
                 f"{CONTINUE_AFTER_STOP_S} s after the stop takes effect (amendment 14:05).", ""]
    if stage == "stage2b":
        rows += [f"Ready-pose return between the two commands: {STAGE2B_READY_RETURN} (amendment 13:47; "
                 f"development test: with 6 of 10, without 0 of 10). Return variant {RETURN_VARIANT}. "
                 f"Development test (amendment 13:55): {DEV_COUNTS_RETURN}", ""]
    rows += ["Policy model-call time: one select_action call that computes a new chunk of 10 actions, "
             "timed with CUDA synchronised before and after.", "",
             "Reached: every goal of the plan was reached (as LIBERO counts). Expected: reached, and the plan was "
             "the commands the line asked for. Earlier held: when the last goal was reached, every earlier goal "
             "was still true (amendment 14:58). Still true: every goal still held after 1 s with the arm held at "
             "the end (amendments 13:55, 14:53; information). Stop delay: from the end of the word (last speech frame); in brackets, "
             "from the end of the trimmed clip.", "",
             "| Run | Result | Expected | Earlier held | Still true | Steps | Plan | Done | Median call ms | Max arm move mm | "
             "Stop delay ms | Overrun mm | Hold offset mm |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        rows.append(f"| {r['run_id']} | {r['result']} | {r['did_expected']} | {r['earlier_goals_hold']} | "
                    f"{r['goals_hold_at_end']} | "
                    f"{r['steps']} | {r['plan']} | {r['done']} | {r['call_ms_median']:.1f} | "
                    f"{r['max_move_mm']:.2f} | {r['stop_delay_ms']:.0f} ({r['stop_delay_clip_ms']:.0f}) "
                    f"{r['stop_exercised']} | {r['overrun_mm']:.2f} | {r['hold_offset_mm']:.2f} |")

    rows += ["", "| Rule | Numbers | Verdict |", "|---|---|---|"]
    if stage == "stage1":
        rows.append(f"| S1-0 tuned settings | median call {median_ms:.2f} ms over {len(calls)} calls (bar 20 ms) | "
                    f"{verdict(median_ms <= 20)} |")
        right = 0
        for number, command in STAGE1_LINES + [(5, None)]:
            kinds = [(e["event"].kind, e["event"].commands, e["heard"]) for run_id, lines in all_lines
                     if run_id.startswith(f"line{number}_") for e in lines if e["event"]]
            expected = ("refuse", []) if command is None else ("plan", [command])
            line_right = bool(kinds) and all((k, c) == expected for k, c, _ in kinds)
            right += line_right
            rows.append(f"| S1-1 line {number} | heard {sorted({h for _, _, h in kinds})}, got "
                        f"{sorted({(k, tuple(c)) for k, c, _ in kinds})} | {verdict(line_right)} |")
        rows.append(f"| S1-1 right routing | {right} of 5 lines (bar 4) | {verdict(right >= 4)} |")
        commands = [r for r in runs if r["expected"]]
        reached = sum(r["success"] for r in commands)
        did = sum(r["did_expected"] for r in commands)  # S1-2 counts only the expected command (14:05)
        rows.append(f"| S1-2 robot does the task | {did} of {len(commands)} did the expected command "
                    f"({reached} reached their plan) (bar 10 of 12) | {verdict(did >= 10 and len(commands) == 12)} |")
        refusal = [r for r in runs if r["expected"] is None]
        moves = [r["refusal_move_mm"] for r in refusal]
        whole = [round(float(r["max_move_mm"]), 3) for r in refusal]
        creep = [(round(r["creep_60_mm"], 3), round(r["creep_100_mm"], 3)) for r in refusal]
        rows.append(f"| S1-3 refusal | largest move in the 3 s after the refusal took effect: "
                    f"{[round(m, 3) for m in moves]} mm (whole run {whole} mm; creep after 60 and 100 idle steps "
                    f"{creep} mm) (bar < 2 mm) | {verdict(bool(moves) and all(m < 2 for m in moves))} |")
    if stage == "stage2a":
        overruns = [r["overrun_mm"] for r in runs]
        offsets = [r["hold_offset_mm"] for r in runs]
        delays = [r["stop_delay_ms"] for r in runs]
        clip_delays = [r["stop_delay_clip_ms"] for r in runs]
        not_exercised = sum(r["stop_exercised"] != "yes" for r in runs)
        did = sum(r["did_expected"] for r in runs)  # S2-3 counts only the expected command (14:05)
        reached = sum(r["success"] for r in runs)
        all_12 = len(runs) == 12
        rows.append(f"| S2-1a overrun | {[round(d, 2) for d in overruns]} mm; stop not exercised in "
                    f"{not_exercised} (bar <= 15 mm in all 12) | "
                    f"{verdict(all_12 and all(d <= 15 for d in overruns))} |")
        rows.append(f"| S2-1b hold | {[round(d, 2) for d in offsets]} mm from 1 s after the stop "
                    f"(bar <= 5 mm in all 12) | {verdict(all_12 and all(d <= 5 for d in offsets))} |")
        rows.append(f"| S2-2 stop delay | from the end of the word {[round(d, 1) for d in delays]} ms; from the "
                    f"end of the clip {[round(d, 1) for d in clip_delays]} ms (bar <= 300 ms from the word, all 12) | "
                    f"{verdict(all_12 and all(d <= 300 for d in delays))} |")
        rows.append(f"| S2-3 continue works | {did} of {len(runs)} did the expected command ({reached} reached "
                    f"their plan) (bar 10 of 12; all 12 must exist) | {verdict(all_12 and did >= 10)} |")
    if stage == "stage2b":
        for name, (_, plan) in TWO_STEP_SENTENCES.items():
            mine = [r for r in runs if r["run_id"].startswith(name)]
            split_ok = bool(mine) and all(r["plan"] == plan for r in mine)
            counted = sum(r["did_expected"] and r["earlier_goals_hold"] for r in mine)  # S2-5 counts this (14:58)
            reached = sum(r["did_expected"] for r in mine)
            later = sum(r["did_expected"] and r["goals_hold_at_end"] for r in mine)  # information
            rows.append(f"| S2-4 split {name} | plans {[r['plan'] for r in mine]} (expected {plan}) | "
                        f"{verdict(split_ok)} |")
            rows.append(f"| S2-5 shown {name} | first goal still true when the second was reached: {counted} of "
                        f"{len(mine)} (both reached in order: {reached}; both still true 1 s later: {later}) "
                        f"(bar 4 of 5) | {verdict(counted >= 4)} |")
    rows += ["", f"All policy calls: median {median_ms:.2f} ms, {len(calls)} calls."]
    (out / "REPORT.md").write_text("\n".join(rows) + "\n")


def write_fingerprints(out):
    """sha256 of the plan and the code, written before a stage starts (amendment 14:05)."""
    names = ["VOICE_ROBOT_PLAN.md", "commands.py", "voice_robot.py", "run_voice_robot.sh", "audio_trim.py",
             "router.py", "speech.py", "tasks.py", "make_voice_clip.py", "record_lines.py"]
    rows = [f"{hashlib.sha256((ROOT / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (out / "fingerprints.txt").write_text("\n".join(rows) + "\n")


# The nine recordings, for the pre-flight audio check: (file, expected event, expected commands).
NINE_LINES = [(f"voice_test/line_{n}.raw", "plan", [c]) for n, c in STAGE1_LINES] + [
    ("voice_test/line_5.raw", "refuse", []),
    ("voice_lines/stop.raw", "stop", []),
    ("voice_lines/continue.raw", "continue", []),
    ("voice_lines/two_step_1.raw", "plan", [7, 2]),
    ("voice_lines/two_step_2.raw", "plan", [0, 7]),
]


def check_audio(out, audio_root):
    """Pre-flight, no robot: trim each recording that exists, transcribe it and parse it."""
    torch.backends.cudnn.benchmark = False  # as for the voice models in the robot loop
    transcriber, router = Transcriber(), Router()
    transcriber(np.zeros(RATE, dtype=np.float32))  # the first calls are slow, so warm up first
    warm_up_router(router)
    rows = []
    for name, kind, commands in NINE_LINES:
        path = audio_root / name
        if not path.exists():
            print(f"  {name}: not recorded yet", flush=True)
            continue
        recorded = load_raw(path)
        trimmed, speech_end_s, flag = trim_silence(recorded)
        heard, speech_ms = transcriber(trimmed)
        event = parse(heard, router)
        right = event.kind == kind and event.commands == commands
        rows.append([name, len(recorded) / RATE, len(trimmed) / RATE, speech_end_s, flag, heard, event.kind,
                     event.commands, event.level, round(speech_ms, 1), round(event.ms, 2), kind, commands, right])
        print(f"  {name}: recorded {len(recorded) / RATE:.2f} s, trimmed {len(trimmed) / RATE:.2f} s, "
              f"flag {flag!r}, heard {heard!r} -> {event.kind} {event.commands} ({'right' if right else 'WRONG'})",
              flush=True)
    with open(out / "audio_check.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "recorded_s", "trimmed_s", "speech_end_s", "trim_flag", "heard", "event",
                         "commands", "level", "speech_ms", "parse_ms", "expected_event", "expected_commands",
                         "right"])
        writer.writerows(rows)


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--stage", required=True, choices=sorted(STAGES) + ["check_audio"])
    arguments.add_argument("--name", required=True, help="output folder under $OUT_DIR/voice_robot/")
    arguments.add_argument("--no-compile", action="store_true", help="development only: skip torch.compile")
    arguments.add_argument("--audio-root", default=None,
                           help="rehearsal: a folder holding voice_test/ and voice_lines/ (default data/)")
    arguments.add_argument("--starts", default=None, help="rehearsal: comma-separated start numbers, e.g. 5,6")
    args = arguments.parse_args()

    # Overrides make a rehearsal: it must not use a registered stage's name.
    audio_root = Path(args.audio_root) if args.audio_root else DATA
    starts = tuple(int(k) for k in args.starts.split(",")) if args.starts else None
    rehearsal = args.audio_root is not None or starts is not None
    if rehearsal and args.name in ("stage1", "stage2a", "stage2b"):
        raise SystemExit("with --audio-root or --starts this is a rehearsal: give it another --name")
    if starts is not None and args.stage not in ("stage1", "stage2a", "stage2b"):
        raise SystemExit("--starts works only with stage1, stage2a or stage2b")

    out = RESULTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_fingerprints(out)  # before anything runs (amendment 14:05)
    if args.stage == "check_audio":
        check_audio(out, audio_root)
        print(f"{time.strftime('%H:%M:%S')} done: {out}", flush=True)
        return

    for name, command in enumerate(COMMANDS):  # tasks.py must match LIBERO's own wording
        assert benchmark.get_benchmark_dict()["libero_goal"]().get_task(name).language == command
    if args.stage in ("stage1", "stage2a", "stage2b"):
        options = {"audio_root": audio_root} if starts is None else {"audio_root": audio_root, "starts": starts}
        specs = STAGES[args.stage](**options)
    elif args.audio_root is not None:
        raise SystemExit("--audio-root works only with check_audio, stage1, stage2a or stage2b")
    else:
        specs = STAGES[args.stage]()
    for spec in specs:  # fail early if a recording is missing
        for line in spec.lines:
            if line.file is not None and not Path(line.file).exists():
                raise SystemExit(f"missing recording: {line.file}")

    print(f"{time.strftime('%H:%M:%S')} loading models...", flush=True)
    cfg = tuned_config(compile_model=not args.no_compile)
    models = Models(cfg)
    scenes = {}
    for spec in specs:
        if spec.env_task not in scenes:
            scenes[spec.env_task] = Scene(cfg, spec.env_task)
    print(f"{time.strftime('%H:%M:%S')} warming up...", flush=True)
    models.warm_up(next(iter(scenes.values())))

    runs, all_lines = [], []
    for spec in specs:
        print(f"{time.strftime('%H:%M:%S')} run {spec.run_id}", flush=True)
        run, lines = run_once(models, scenes[spec.env_task], spec, out / "runs" / spec.run_id)
        runs.append(run)
        all_lines.append((spec.run_id, lines))
    for scene in scenes.values():
        scene.close()
    write_tables(out, runs, all_lines)
    write_report(out, args.stage, runs, all_lines, rehearsal)
    print(f"{time.strftime('%H:%M:%S')} done: {out}", flush=True)


if __name__ == "__main__":
    main()
