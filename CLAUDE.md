# CLAUDE.md — NIDAR Human (Survivor) Detection

> Project context for Claude Code (the only AI coding tool used on this repo) and for new chats. Read it fully before doing any task.
> Keep it current: when a decision, status line or open question changes, edit this file in the same commit.
> Last reviewed: 1 Nov 2026.

## 1. What this repo is

Software for **one subsystem** of the NIDAR team's AirMouse 2026–27 entry: **human (survivor) detection** from the drone's camera video. Owner: Aryan (GitHub: AryanGoyal17). Repo: `github.com/AryanGoyal17/NIDAR-Automation` (a rename to something like `NIDAR-Human-Detection` is under consideration).

AirMouse: a fully autonomous drone flies inside a GPS-denied indoor maze, finds up to six survivors, and reports them on a real-time 2D map, with no manual navigation help.

**In scope:** detecting people in colour frames, estimating each detection's distance and 3D position in the camera frame from the aligned depth image, tracking detections across frames, confirming each survivor once, measuring accuracy and speed, exporting the model for onboard use, and the detection output format.

**Out of scope (other people's work, do not build or modify):** path planning and navigation, mapping and survivor tagging (turning a detection into a map position), obstacle avoidance, flight control, ground-station application. A "controls team" will integrate everything, so the handoff interface (Section 8) matters.

## 2. Deadline and definition of done

The original deadline (22 Oct 2026) has passed. The new deadline is **TBD**. "Done" means all of these exist, run from a fresh clone, and are honest about what is tested:

1. Trained models for YOLO11n and YOLOv8n (weights in `models/`, not in git).
2. Evaluation results on a small **real** test set, reported separately from simulation results.
3. A stable detection API (Section 8) with an NCNN export and a Raspberry Pi inference script (`deploy/pi/`).
4. README (algorithm specification, hardware requirements, results with only measured numbers) and a handoff note for the controls team.

## 3. Hardware targets and constraints

See `docs/decisions/006_pi_rgbd_hardware.md`.

- **Development laptop:** Windows (PowerShell), Python 3.12, PyTorch 2.6 with CUDA 12.4, Ultralytics 8.4, NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM (`nvidia-smi`: 6144 MiB). Bigger training runs may use Kaggle.
- **Onboard computer:** Raspberry Pi, CPU only. ASSUMPTION: Pi 5, 8 GB, with active cooling. Exact model, OS and Python version TBD. Inference uses Ultralytics with an **NCNN** export. No speed has been measured on the Pi yet.
- **Camera:** an **RGB-D** camera. Model TBD. Colour and depth resolution, frame rate, field of view, depth range, mounting height and tilt are all ASSUMPTION or TBD, and must stay configurable. Colour and depth must be aligned and share a capture timestamp.
- The Jetson Nano is **dropped**. There is no Jetson, ONNX-to-TensorRT or Python 3.6 deployment path.
- **No internet during a mission.** No cloud APIs. All weights and dependencies must be local files.
- Detection shares the board with mapping, planning and obstacle avoidance, so keep it light. Detection must never be inside the flight-control loop.
- The dev machine is Windows: write cross-platform Python scripts (use `pathlib`, no hard-coded separators). If a shell script is unavoidable, also provide a PowerShell version.
- PyBullet has no prebuilt package for Python 3.12 on some systems; in this `.venv` it was built from source (pybullet 3.2.7). If a build fails, use a separate Python 3.10 or 3.11 environment for simulation only.

## 4. Decisions already made (do not re-open without being asked)

- **Models:** train **YOLO11n (primary)** and **YOLOv8n (safe fallback and baseline)** on identical data, imgsz 416 (drop to 320 if the Pi is too slow), single class `person`. Export to NCNN for the Pi.
- **Not doing:** RT-DETR (too heavy for the Pi CPU), larger YOLO sizes (s, m), a custom YOLO+ViT hybrid. YOLO11 has a lightweight partial spatial-attention block (C2PSA) in its deepest backbone stage. It is **not** a Vision Transformer, so never describe it as a ViT hybrid. Differences between YOLO11n and YOLOv8n cannot be credited to attention alone.
- **Depth:** the detector uses only the colour image. Depth is used after detection: estimate each survivor's distance from the aligned depth inside the box, compute a 3D point in the camera frame from the camera intrinsics, and reject boxes whose real-world size is implausible for a person.
- **Data strategy:** no real labelled arena data yet. Training data = PyBullet renders of the arena (domain-randomised, procedural survivors) + real-person cutouts composited onto empty PyBullet frames + a small share of real aerial images (SARD, from the NIDARX Kaggle dataset) + negatives. Split by **sequence/group, never by frame**. A small **real test set** (`data/test_real/`) is never used for training.
- **Improvement ladder:** baseline (pretrained YOLOv8) → fine-tuned on our data → tracking + multi-frame confirmation → YOLO11. The final system is the best measured candidate plus tracking and confirmation, exported for the Pi.
- **Confirmation rule:** a track is confirmed when it is matched in at least N of the last M frames; choose N and M from measured per-frame rates, not guesses.
- **Metrics:** recall is the priority (a missed survivor is worse than a false alarm). Report precision, recall, F1, mAP@0.5, and survivor-level false survivors and double counts. Speed in ms per frame **measured on the Pi**. Simulation scores are always reported separately from real-test scores.

- **Config defaults vs decisions:** `configs/default.yaml` currently holds the defaults of the baseline (pretrained) script: imgsz 640, conf 0.5, device `cuda:0`. The 416 size applies to the fine-tuned models and deployment (`configs/train.yaml`, planned). The confidence threshold is chosen by the threshold sweep (plan step 6.4); the 0.4 in Section 8 is only an example. Do not change `default.yaml` to match a decision unless asked.

## 5. Open questions (do not assume; ask or mark as an assumption)

- Exact Raspberry Pi model and memory (assumed Pi 5, 8 GB).
- Exact RGB-D camera model, and its colour and depth resolution, frame rate, field of view, depth range, mounting height and tilt.
- Are survivors real people or dummies? (A COCO "person" model may miss dummies.)
- Required detection frame rate (to be set with the controls team).
- Arena dimensions: corridor width, wall height and colours, lighting, ceiling.
- New deadline.
- Whether the repo stays public or becomes private, and its licence (Ultralytics is AGPL-3.0).

## 6. Status (update as you go)

**Done:**
- Foundation: folder structure, config loader, logger, environment checks, conventions, diary (days 1–2), decision records 001–006.
- Baseline YOLOv8n detection script (`cv/detectors/detect_survivors.py`).
- Weight downloader (`scripts/download_weights.py`; it prints sha256 but does not verify it).
- PyBullet smoke test (`scripts/check_sim.py`).
- Weights are **not** in git (`models/` is gitignored).

**Simulator (`sim/`, exists, runs, not yet usable for training data):**
- `maze.py`: maze builder from ASCII grids. `layout_gen.py`: random maze generator. There are 6 layouts in `sim/layouts/`.
- `camera.py`: pinhole camera returning colour, depth buffer and segmentation mask. `render_segmentation` renders the mask on a canvas extended by a margin on each side, with the same focal length in pixels; its frame crop equals the frame mask (0 px difference, tested). `project_aabb(clip=False)` / `project_box` give unclipped rectangles.
- `trajectory.py`: pose and sequence sampler.
- `survivor.py`: procedural survivors in 6 postures.
- `randomize.py`: textures, lighting and frame augmentation.
- `labels.py`: YOLO labels from segmentation masks.
- `scripts/sim_preview.py`, `configs/arena.yaml`, `configs/randomization.yaml`.
- pytest, 1 Nov 2026 (branch `chore/check-full`, commit `5164606`: `main` plus test-only changes, same test count): 160 passed and 11 skipped by default; the 11 are the heavy scene checks, run with `NIDAR_SCENE_TESTS=1` (171 passed).

Evidence for the items below: local files `outputs/scratch/scan_report.md` (30 Oct) and `outputs/scratch/fix_status_report.md` (1 Nov), gitignored, only on the owner's machine.

**Fixed on `main`:**
- Light direction: `arena.yaml` and the randomiser now put the light above the floor (untextured floor grey 112, was 39). The specular coefficient now reaches the renderer.
- Lint: ruff check 0 errors, ruff format 0 unformatted (47 files, measured 1 Nov 2026 on branch `chore/check-full`). Enforced by the pre-commit hook (ruff 0.16.10, run `pre-commit install` once per clone) and `scripts/check.ps1`.
- Textured surfaces darkened by the body colour: `apply_to_sim` now resets the floor and wall `rgbaColor` to white, so the texture alone sets the albedo (it was multiplied by 0.45 on the floor, 0.80 on walls). Normal-light floor grey over 200 randomised scenes: mean 40.5 → 91.0 (branch `fix/floor-brightness`, measured 1 Nov 2026). Tested in `TestTexturedSurfaceBrightness`.
- Dark floor texture ranges: `randomization.yaml` solid `floor_color` (was 0.20–0.48) and noise `floor_base` (was 0.22–0.45) widened to 0.30–0.75 and 0.30–0.70 (ASSUMPTION, arena floor colour unknown). With the reset, normal-light floor grey over 200 scenes: mean 91.0 → 109.5, share below 60: 18.3 % → 0.6 %, above 200: 0 %.
- Build-specific render pin: `TestRenderPinned` compared sha256 hashes recorded with this laptop's pybullet build. It now compares each pinned pose with a render made from an independently computed view matrix in the same run (per-pose BGR mean absolute difference < 0.05; pins the view/axis maths, not lighting or projection). Fails on a roll sign flip, an up-vector swap and a 0.5° yaw error (branch `chore/check-full`, 1 Nov 2026).
- Collection errors without pybullet: test modules that import pybullet (directly or through `sim.camera` / `sim.maze`) use `pytest.importorskip("pybullet")`, and `test_config.py` uses `pytest.importorskip("yaml")`. Global Python 3.12 without pybullet or yaml: 37 passed, 8 skipped, 0 collection errors (was 8 collection errors; branch `chore/check-full`, 1 Nov 2026).
- Slow scene checks easy to forget: `scripts/check.ps1` prints the Python it uses and the pytest skip count, and the default run ends with "scene tests skipped: run scripts/check.ps1 -Full before merging". `-Full` sets `NIDAR_SCENE_TESTS=1`, runs pytest with `-rs`, fails with "pybullet missing: activate .venv and run again" when pybullet is absent, and fails on any skip whose reason is not in `$FullSkipAllowlist` (empty).

**Only on the unmerged branch `fix/trajectory-clearance` (commit `0483528`), not on `main`; replaced by `fix/trajectory-v2` below:**
- Survivor 3D clearance in `TrajectorySampler` (applies only when the caller passes the survivors), and survivors never placed on the start cell.
- Known regression on that branch: with survivors passed, 15 of 160 test sequences collapse to 1 frame, and repeated positions rise from 18 to 1356. A survivor plus 0.6 m clearance blocks a 1.5 m corridor.

**Merged into `main` from `fix/trajectory-v2` (redo of `0483528`; commits `43061f9` to `d53b98d`; the branch is deleted locally and on `origin`):**
- Survivors never placed on the start cell (`SurvivorPlacer.place`; 22 of 160 test scenes on `main`, 0 on the branch). Tested in `TestPlacement`.
- `SurvivorInfo.aabb`: each survivor's real 3D box (union of its parts' PyBullet AABBs, any posture). Tested in `TestAABB`.
- `sample_poses` (the source of training frames) takes the survivor boxes (`survivor_boxes=[s.aabb ...]`) and enforces `arena.yaml trajectory.survivor_clearance_m` (0.35 m, ASSUMPTION) and `max_survivor_box_fraction` (0.5, ASSUMPTION). It returns exactly n unique poses or raises `PoseSamplingError` with rejection counts by reason; the counts are logged at INFO on every call. Tested in `test_trajectory.py` and over 8 layouts x 20 seeds in `test_trajectory_scenes.py`.
- `generate_sequence` is for replay and tracking tests only: a self-avoiding walk over the corridor graph (no cell re-entered). At a survivor it stops at the last accepted pose and, by default, adds up to 12 in-place yaw frames of 10 degrees. It ends when there is no unvisited neighbour, and it returns at least `trajectory.min_sequence_frames` (20) frames or raises `SequenceSkipped` with a logged reason. Seeded start-cell retries are deterministic. All four values are in `arena.yaml` as ASSUMPTION. Measured 1 Nov 2026, 8 layouts x 20 seeds, 3 survivors, 60 frames: 0 skipped, 13/160 shorter than 60 (min 39), repeated (x, y, yaw) 0, 0 poses within 0.35 m of a survivor box (`0483528`: 15 one-frame sequences, 1356 repeated positions).
- `camera.py`: axis maths shared through `camera_axes` (renders byte-identical before and after, pinned by `TestRenderPinned`), and `project_aabb` / `projected_box_fraction` (box to image rectangle, within 2 px of the rendered mask box in `TestBoxProjection`).

**On the unmerged branch `fix/labels` (C4, issue 2, issue 8, N7), not on `main` yet:**
- Measured truncation and occlusion. `SilhouetteRenderer` copies the survivor parts generically from PyBullet shape data (sphere, box, cylinder, capsule; any other shape raises `UnsupportedShapeError` naming the type) into a second DIRECT client and renders each survivor alone, on a canvas extended where its part boxes overhang the frame (capped at `arena.yaml labels.max_canvas_area_factor`, 4.0, ASSUMPTION). Truncation = alone pixels outside the frame / all alone pixels; occlusion = share of the in-frame alone pixels hidden in the real render. A survivor beyond the cap or behind the near plane gets `truncation_exact=False`. `compute_survivor_labels(silhouettes=...)` applies `max_truncation` 0.85 and `max_occlusion` 0.90 (ASSUMPTION, old defaults) to the measured values. Tested in `test_label_renders.py` (copy 0 px different for each shape type, plate truncation within the derived pixel tolerance, wall and survivor occlusion) and, with `NIDAR_SCENE_TESTS=1`, in `test_label_scenes.py` (alone masks equal a reference rendered in the main scene with all other bodies moved away, 6 layouts at 640x360).
- Thresholds at training resolution (issue 8): `arena.yaml labels` gives `min_box_height_px` 10, `min_box_width_px` 6 and `presence_min_area_px` 1.6 (px² at `imgsz` 416), all ASSUMPTION. `LabelConfig.thresholds(width, height)` converts them with scale = imgsz / max(width, height) (min box height 30.8 px at 1280x720, 15.4 px at 640x360; tested at both sizes). `label_frame(...)` returns `FrameLabels(labels, dropped)`, where each dropped label carries its measured values and every reason (`box_too_short`, `box_too_narrow`, `truncated`, `truncation_unbounded`, `occluded`). Survivors under the presence floor are absent, not dropped. The old render-pixel defaults (6 px box, 15 px) remain only in the low-level `compute_box_for_body_ids` / `compute_survivor_labels`.

**Known open defects on `main`:**
- Broken postures: sitting, supine and prone body parts are misaligned or detached. Lateral pokes into walls. The config posture list is ignored.
- Truncation/occlusion always 0: `labels.py` never computes them, so its filters never fire (fixed on `fix/labels`, above).
- Saturated floor colours: texture colours are sampled per RGB channel independently, so the wider floor ranges also give strongly tinted floors (for example purple, green).
- Walls darker than floors under the overhead light: lighting gain about 0.40 on walls vs 0.97 on the floor (default light, measured 1 Nov 2026). This comes from the light geometry; the light direction is unchanged.
- Raw non-metric depth: `camera.py` returns the raw depth buffer, not metres, although documented as linear.
- Texture memory growth: `randomize.py` loads new textures every sequence and never frees them.
- No frame/dataset generator: nothing yet passes survivors to the sampler or writes a dataset.
- `check.ps1 -Full` fails on a fresh clone: the 4 live-output tests in `test_detection_output.py` skip ("No live detections.jsonl found") until the baseline detector has written its output under the gitignored `outputs/` (seen in a `git archive` copy, 1 Nov 2026).

**Removed:** `gcs/` (ground-station stubs, out of scope). `AGENTS.md` (this file is the only agent context).

**Empty placeholders:** `cv/utils/`, `interfaces/schemas/`, `tests/test_interfaces/` (`.gitkeep` only).

**README:** `README.md` exists but is out of date: it still describes the Jetson, RT-DETR and an RGB-only camera. A rewrite is planned.

**Planned (not built yet):**
- Frame/dataset generator and clutter (`sim/`).
- `data_tools/` (cutouts, compositor, dataset builder) and the real test set (`data/test_real/`).
- `cv/train/`, `cv/eval/`, `cv/tracking/`, the depth module, the detection API.
- NCNN export and benchmark scripts, `deploy/pi/`, the handoff note.

Mark anything not run as **[PLANNED]**. Never write a result into the README unless it was measured by us.

## 7. Repo layout and commands

```
configs/            [exists] default.yaml, arena.yaml, randomization.yaml, maze_layouts/. [PLANNED] composite.yaml, train.yaml, tracking.yaml, depth settings
cv/                 [exists] config.py, logger.py, detectors/, utils/ (empty). [PLANNED] eval/, tracking/, train/, depth, detector_api.py
sim/                [exists] maze, layout_gen, camera, trajectory, survivor, randomize, labels, layouts/. [PLANNED] generator, clutter
data_tools/         [PLANNED] cutouts, compositor, dataset builder, inspection
deploy/             [PLANNED] pi/
interfaces/schemas/ [exists, only .gitkeep] the detection JSON schema is PLANNED
docs/               [exists] conventions.md, arena_questions.md, decisions/ (ADRs 001–006), diary/, setup/
scripts/            [exists] check.ps1 (ruff check + ruff format --check + pytest; -Full adds the scene checks), check_env.py, verify_environment.py, download_weights.py, check_sim.py, sim_preview.py, setup scripts. [PLANNED] export, benchmark
requirements/       [exists] requirements-base.txt, requirements-cv.txt, requirements-sim.txt (pybullet), requirements-pi.txt (Pi runtime, TBD items), requirements-dev.txt (requirements.txt + sim + ruff, pytest, pre-commit)
requirements.txt    [exists] base + cv
.pre-commit-config.yaml  [exists] ruff-check and ruff-format hooks (ruff-pre-commit v0.16.10)
notebooks/          [exists, empty]
tests/              [exists] test_cv/, test_sim/, test_interfaces/ (empty)
data/  models/  outputs/   [exist, gitignored, never committed]. data/test_real/ is PLANNED; models/ holds local weights.
```

```bash
pip install -r requirements/requirements-dev.txt              # dev laptop setup (install torch first, see README §4)
pre-commit install                                            # once per clone: ruff hook on every commit
python scripts/check_env.py                                   # environment health
python -m cv.detectors.detect_survivors --source data/samples/ # baseline detector
python -m ruff check . ; python -m ruff format --check .      # lint and format
pytest -q                                                      # tests (skips the heavy scene checks)
$env:NIDAR_SCENE_TESTS = "1"; pytest -q tests/test_sim/test_trajectory_scenes.py   # 8 layouts x 20 seeds trajectory checks
.\scripts\check.ps1                                           # fast: ruff check + ruff format --check + pytest, scene checks skipped (before every commit)
.\scripts\check.ps1 -Full                                     # everything incl. scene checks; fails on any non-allowlisted skip (before asking to merge)
```

## 8. Handoff interface (what the controls team will call)

```python
# backend: "pytorch" (dev laptop) or "ncnn" (Pi). conf=0.4 is only an example (see Section 4).
detector = SurvivorDetector(backend="ncnn", weights=..., imgsz=416, conf=0.4)
detections = detector.detect(frame_bgr, depth=None)  # list of Detection
```

Each `Detection` has:
- `x1, y1, x2, y2`: float pixels, `xyxy`, origin top-left.
- `confidence`: 0–1. `class_name`: `"person"`.
- `distance_m`: distance in metres from the aligned depth inside the box, or `None`.
- `point_cam_xyz_m`: 3D point in the camera frame in metres, or `None`. ASSUMPTION: OpenCV axes (x right, y down, z forward), to be confirmed with the mapping team.
- `depth_valid`: `False` when no depth was given or the box has no valid depth.

Every result carries the frame's **capture timestamp** (when the camera captured the frame; UTC ISO 8601 with milliseconds and `Z`), because mapping needs it. Colour and depth must share that timestamp. How the timestamp is passed into `detect()` is TBD (to be agreed with the controls team).

Per-frame JSON output follows README Section 8 (frame_id, timestamps, frame size, inference_ms, detections). That section will be updated with the capture timestamp and depth fields.

## 9. Conventions

- Pixel origin top-left, x right, y down. Boxes are `xyxy` in float pixels. Timestamps UTC ISO 8601 with milliseconds and `Z`.
- Settings come from `configs/*.yaml` through `cv/config.py`. Log with `cv/logger.py`, never raw `print()` in library code.
- Seeds everywhere; every generated dataset and training run records its seed, config and git commit.
- Conventional commits (`feat(sim): ...`, `fix(eval): ...`, `docs: ...`), small and atomic, on a feature branch.
- Large files (`.pt`, `.onnx`, NCNN exports, videos, datasets, generated images) are never committed. Record sha256 checksums instead.

## 10. Rules for AI agents working in this repo

1. Read this file first. Do the task asked, inside the scope named. Do not refactor unrelated code.
2. For any change touching more than one file, **propose a plan first** and wait for approval.
3. Run the tests or commands you wrote and show the real output. Never claim something works without running it. Say clearly what you could not test (for example hardware you do not have).
4. **Never invent numbers, metrics, benchmark results, or facts about the rulebook.** If a value is unknown, write TBD or an `# ASSUMPTION` comment and add it to Section 5.
5. Do not add cloud calls, telemetry, or downloads at runtime. Weights and datasets are fetched once during setup.
6. One task = one branch, created from `main` unless the owner names another base. Never commit to a branch that holds other tasks, and never commit directly on `main`. The owner merges into `main`.
7. Before every commit, `ruff check .`, `ruff format --check .` and `pytest` must pass. This is enforced now by the pre-commit hook (ruff) and `scripts/check.ps1` (ruff + pytest).
8. Never commit secrets, datasets, weights, or personal information.
9. Keep README claims aligned with reality: [DONE] only if it exists and ran, otherwise [PLANNED].
10. If you discover a contradiction between this file and the repo, tell me and fix the file, not the other way round, unless I say otherwise.
11. Update Section 6 (Status) in the same commit as the change that alters the status.
12. Run `scripts/check.ps1 -Full` before asking the owner to merge, and show its output. The default `check.ps1` skips the heavy scene checks.
