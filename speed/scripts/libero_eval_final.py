"""Run today's lerobot-eval with SmolVLA's cameras encoded one vision-encoder call each, or batched.

The final worktree carries the camera-batching patch, so batching is the default there. `--cameras looped`
puts the per-camera loop back on the loaded policy instance (LeRobot's behaviour without the patch), as
energy_session_final.py does for A0 and A1. Everything else is plain lerobot-eval.

Usage: python libero_eval_final.py --cameras looped|batched <the usual lerobot-eval arguments>
"""

import sys

import lerobot.scripts.lerobot_eval as lerobot_eval


def main():
    i = sys.argv.index("--cameras")
    cameras = sys.argv[i + 1]
    del sys.argv[i : i + 2]  # lerobot-eval must not see our extra flag
    if cameras not in ("looped", "batched"):
        raise SystemExit(f"--cameras must be looped or batched, got {cameras}")

    make_policy = lerobot_eval.make_policy

    def make_policy_with_cameras(*args, **kwargs):
        policy = make_policy(*args, **kwargs)
        if cameras == "looped":
            model = policy.model  # LeRobot before the patch: one vision-encoder call per camera
            model._embed_images = lambda images: [model.vlm_with_expert.embed_image(img) for img in images]
        print(f"SmolVLA cameras: {cameras}; num_steps={policy.config.num_steps}, "
              f"empty_cameras={policy.config.empty_cameras}, compile_model={policy.config.compile_model}, "
              f"compile_mode={policy.config.compile_mode}, n_action_steps={policy.config.n_action_steps}", flush=True)
        return policy

    lerobot_eval.make_policy = make_policy_with_cameras  # eval_main looks this name up when it runs
    lerobot_eval.main()


if __name__ == "__main__":
    main()
