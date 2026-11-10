# File notes: what is here, what changed, what was left out

This repository was assembled by copying files from my working project folders. In recorded result files the project folder appears as `<project>`, a temporary scratch folder as `<scratch>`, and the home folder as `~`.

## Where the files came from

| In this repository | Copied from (`<project>/…`) |
|---|---|
| `speed/scripts/` | `bench/results_final_2026-11-07/*.sh, *.py`, plus `bench/libero_compare.py` and `bench/batch_cameras_ab_v2.py` |
| `speed/patches/batch_cameras.patch` | `bench/results_pr_batch_cameras_2026-11-07/batch_cameras.patch` |
| `speed/patches/pr4634_test_fix.diff` | `bench/results_final_2026-11-07/followups/pr4634_test_fix.diff` |
| `speed/followups/` (8 Nov scripts) | `bench/results_final_2026-11-07/followups/*.py, *.sh` |
| `speed/followups/tuned_check.py`, `run_tuned_check.sh` | the 10 Nov scratch folder (`tuned_check.py`, `run_all.sh`) |
| `speed/results/final_2026-11-07/` | `bench/results_final_2026-11-07/` (result files) and `libero_runs/final_2026-11-07_{A1_eager_steps10,B_tuned}/` (`eval_info.json`, `task_descriptions.json` only) |
| `speed/results/reference_2026-10-31/` | the 31 Oct files the final session compares with: `bench/results_v2/` (5 stats and 4 first-call-action files) and `libero_runs/{baseline_eager_n10,batched_compile_model_steps1_noblank}/eval_info.json` |
| `speed/results/batch_cameras_2026-11-07/` | `bench/results_pr_batch_cameras_2026-11-07/` (all but the patch and two raw action dumps) |
| `speed/results/followups_2026-11-08/` | `bench/results_final_2026-11-07/followups/` (result files) |
| `speed/results/tuned_without_pr4634_2026-11-10/` | the 10 Nov scratch folder `out/` (per-run json, per-call times, first-call actions, run logs) |
| `speed/clip/` | `talk/clips/v3/` (scripts, `episode_steps.json`; README rewritten) |
| `voice/` code, `tests/`, `diagnostics/`, `figures/` | `voice-robot/` |
| `voice/data/voice_test/line_1–5.raw`, `voice/data/voice_lines/{stop,continue,two_step_1,two_step_2}.raw` | `voice-robot/data/take2/` (the second-take recordings; line 3 is the first-take recording the second take fell back to, see `voice/results/voice_robot/take2_line3_source.txt`). They sit where `voice_robot.py` reads them by default, as in the second-take runs |
| `voice/data/ball_demo/voice_test/line_1–5.raw` | `voice-robot/data/ball_demo/voice_test/` (all five were read by the `ball_refusal` run: `--audio-root data/ball_demo --starts 0`, 5 runs) |
| `voice/data/share_audio/` | `voice-robot/data/share_audio/`: synthetic speech made by `make_share_audio.sh` (flite) from the 10 command sentences in `tasks.py`; no human voice |
| `voice/results/router/` | `voice-robot/results/{test,dev}_{router,llm}.json`, `summary` key only (see below) |
| `voice/results/voice_robot/` | `voice-robot/results/voice_robot/` (registered and second-take runs, `ball_refusal`, `stage1_final_code`, `stage1_endcheck`, `code_as_run_stage2`) |
| `voice/results/share/` | `voice-robot/results/share/` (all six run folders) |
| `slides/` | `talk/latex/slides_v2.tex`, `slides_v2.pdf` (the PDF as presented; in the `.tex`, only the two header comments and the media paths, now `../media/`, changed) |
| `media/` | `talk/latex/media/` (Demo 1 clip and still, the four voice posters, `pr4634_table.png` used by the slides) and `voice-robot/clips/` (the four second-take clips and `refusal_bowl_heard_as_ball.mp4`) |
| `LICENSE` | the Apache-2.0 `LICENSE` text of the LeRobot checkout, with the copyright line changed to this repository's author |
| `requirements-lock.txt` | `uv pip freeze` of the LeRobot checkout's venv, without the editable LeRobot line |

## Changes to code (paths only)

No logic was changed. Every script now finds its files through the variables in the README (`LEROBOT_DIR`, `PYTHON`, `OUT_DIR`, `LOG_DIR`, `SCRATCH`, `GPU_LOCK`) and through paths relative to its own folder. New outputs go to `$OUT_DIR` (default `runs/`), so a re-run never overwrites the committed results. Where a script now writes into a folder that may not exist yet, a `mkdir -p` (or `mkdir(parents=True, exist_ok=True)`) was added for it.

| File | What changed |
|---|---|
| `speed/scripts/run_all.sh` | Results, scratch, log and lock paths → `HERE` (script folder), `SCRATCH`, `LOG_DIR`, `GPU_LOCK` |
| `speed/scripts/run_session.sh` | Checkout, Python, results, log and scratch paths → `LEROBOT_DIR`, `PYTHON`, `HERE`, `LOG_DIR`, `SCRATCH`; TF32 check output → `$OUT_DIR/speed/precision_check` |
| `speed/scripts/run_startup.sh` | Same variables; output → `$OUT_DIR/speed/startup` |
| `speed/scripts/run_libero_final.sh` | Same variables; run folder → `$OUT_DIR/speed/libero_runs/<name>` (or `RUNS`); energy file → `$OUT_DIR/speed/` |
| `speed/scripts/run_colocation_check.sh` | Same variables; output → `$OUT_DIR/speed/colocation_check/` |
| `speed/scripts/energy_session_final.py` | Patch path → `speed/patches/`; the 31 Oct comparison files → `speed/results/reference_2026-10-31/results_v2`; default `--out-dir` → `$OUT_DIR/speed/session`; one docstring path |
| `speed/scripts/summarize_final.py` | Reads and writes `RESULTS_DIR` (default `$OUT_DIR/speed`); LIBERO runs under it; logs from `LOG_DIR`; imports `libero_compare.py` from its own folder; source paths written relative to the repository root (`os.path.relpath`) |
| `speed/scripts/libero_compare.py` | Default `--runs-dir` → `$OUT_DIR/speed/libero_runs` |
| `speed/followups/run_task1_test_fix.sh` | Paths → the common variables plus `FIX_DIR`, `PR_HEAD_DIR` (scratch checkouts), `OLD_LEROBOT_DIR` / `OLD_PYTHON` (the PR's pinned checkout and venv); the diff is read from `speed/patches/`; outputs → `$OUT_DIR/speed/followups` |
| `speed/followups/run_task2_compile_modes.sh` | Paths → the common variables; outputs → `$OUT_DIR/speed/followups/compile_modes` |
| `speed/followups/run_task3_stack_compare.sh` | Paths → the common variables plus `OLD_SRC` (the 31 Oct code) and `OLD_PYTHON`; outputs → `$OUT_DIR/speed/followups/stack_compare` |
| `speed/followups/summarize_task1.py`, `summarize_task2.py`, `summarize_task3.py` | Read and write `$OUT_DIR/speed/followups`; the venvs' `cache_utils.py` from `OLD_LEROBOT_DIR` and `LEROBOT_DIR`; the diff from `speed/patches/`; source paths relative to the repository root |
| `speed/followups/run_tuned_check.sh` | Scratch, Python and plain-main paths → common variables and `PLAIN_MAIN_SRC`; outputs → `$OUT_DIR/speed/tuned_without_pr4634` (was a scratch `out/` folder); caches → `$SCRATCH/tuned_check_cache` |
| `speed/followups/tuned_check.py` | Imports `followup_common.py` from its own folder |
| `speed/clip/scripts/measured.py` | Results → `RESULTS_DIR` (default `speed/results/final_2026-11-07`); LIBERO runs → `LIBERO_RUNS`; logs → `LOG_DIR`; clip output → `$OUT_DIR/clip` |
| `voice/run_voice_robot.sh` | Python, log and lock paths → `PYTHON`, `LOG_DIR`, `GPU_LOCK`; exports `OUT_DIR` |
| `voice/voice_robot.py` | Results folder → `$OUT_DIR/voice_robot` (was `results/voice_robot`); `--name` help text |
| `voice/run_share.sh` | Project, Python, output, log and lock paths → `PYTHON`, `ROBOT_PYTHON`, `ROBOT_PYTHONPATH`, `ROBOT_LEROBOT_DIR`, `OUT_DIR`, `LOG_DIR`, `GPU_LOCK`; outputs → `$OUT_DIR/share/<name>` |
| `voice/share_policy_loop.py` | The checkout whose commit is recorded → `ROBOT_LEROBOT_DIR` |
| `voice/score_router.py` | Output → `$OUT_DIR/router/` |
| `voice/voice_test.py` | Output → `$OUT_DIR/voice_test.json` |
| `voice/diagnostics/whisper_cores.py`, `whisper_threads.py` | `voice/` and `voice/data/share_audio` found relative to the script |
| `voice/split_sentences.py` | Docstring wording only ("generated sentences") |

`voice/router.py` is unchanged; its sha256 still matches the frozen value in [ROUTER_PREREG.md](ROUTER_PREREG.md). The run folders' `fingerprints.txt` hold the hashes of the code as run, so the edited files above no longer match those hashes; `voice/results/voice_robot/code_as_run_stage2/` keeps the voice-robot code exactly as run (only its path text changed, as for result files).

## Changes to recorded result files

Only text replacements and line endings, nothing else (line endings: CRLF → LF in every `voice/results/voice_robot/*/events.csv`, CR → LF in `voice/results/share/*/policy.log`):
- the scratch-folder prefix (up to and including its `scratchpad` part) → `<scratch>`;
- the project folder → `<project>`; any other home-folder path → `~`;
- in `speed/results/final_2026-11-07/startup/{A0,A1,C}_eager.json`, the default compile-cache folder name, which contained the user name, → `/tmp/torchinductor_<user>`.

`voice/results/router/*.json` keep only the `summary` key of the original files (extracted by a script that never printed the per-sentence rows). Empty `STOP` marker files were dropped from `voice/results/share/`.

## Left out, and why

| Left out | Why |
|---|---|
| `data/test.json`, `data/dev.json`, `data/sentences_raw.json`, and the `rows` of the router result files | Held-back router sentences, so future router versions can be tested on unseen sentences ([ROUTER_PREREG.md](ROUTER_PREREG.md)) |
| `data/take1/`, the `*_attempt*.raw` retries and `lines.json` files in `data/voice_test/` and `data/voice_lines/` | First take and retries; only the recordings the runs used are kept |
| Frame videos (`frames.mp4`), per-run clips and stills, trimmed audio (`line_*_trimmed.raw`) inside `results/voice_robot/` | Bulky and rebuilt by a re-run; summaries and per-run `timeline.json` / `hold_drift_mm.json` are kept |
| First-take clips (`stage1_*`, `stage2_*`) | The demos use the second take; the first-take numbers are reported in full |
| Development, rehearsal and preflight runs (`dev_*`, `func_*`, `repeat_tuned`, `rehearsal*`, `preflight*`) | Not registered results |
| LIBERO episode videos (`libero_runs/*/videos/`) | 800 videos; rebuilt by a re-run |
| All `logs/` folders (including the Demo 1 build logs) | Logs are machine-specific; a re-run writes them to `$LOG_DIR` |
| `voice-robot/scratch_2026-11-10/` | Working files and separate check scripts of the night |
| `.cache/` (compile caches), `__pycache__/`, venvs, model weights | Rebuilt or downloaded |
| The earlier slide decks (`slides.tex`, `slides.pdf`, other `slides_v2_*.pdf`) | Replaced by the deck as presented |
| The original markdown reports (`SUMMARY_final.md`, `VOICE_ROBOT_RESULTS.md`, `VOICE_ROBOT_PLAN.md`, `PREREG.md`, `RESULTS.md`, `SHARE_PLAN.md`, the 10 Nov follow-up note) and the copy of the plan inside `code_as_run_stage2/` | Replaced by the documents in `docs/`, which keep every number and rule and drop internal process notes |
| `actions_pr_looped_path.pt`, `actions_unpatched_main.pt` from the camera-batching check | Raw action dumps whose pickled data holds a machine path; their comparison is kept in `looped_path_vs_unpatched_main.json` and `checks_summary.json` |
| `talk/clips/v3/meters_side_by_side.mp4/.png` | The final clip used is the "clean" version in `media/` |

## Re-running: what is not self-contained

- **Speed follow-ups** need extra checkouts that are not in this repository: the PR #4634 pinned checkout and venv (torch 2.11, transformers 5.5.4), a checkout at `8c4f33e5` with the 31 Oct batching patch, and a plain LeRobot main clone. Each script's header names the variable for each.
- **`summarize_final.py`** reads run logs (for B's first eval step) and the two 31 Oct LIBERO runs; copy `speed/results/reference_2026-10-31/libero_runs/*` into `$OUT_DIR/speed/libero_runs/` first.
- **Demo 1 clip** needs the LIBERO episode videos from a re-run.
- **GPU-sharing test**: the robot loop of the registered run used the 31 Oct policy stack, and the voice load reads the held-back `data/test.json` ([GPU_SHARING.md](GPU_SHARING.md)).
- **`speed/clip/README.md`**: the "clean" Demo 1 clip in `media/` was edited from the scripts' output; that edit is not scripted here.
