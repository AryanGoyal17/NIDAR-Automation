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
- `camera.py`: pinhole camera returning colour, depth buffer and segmentation mask.
- `trajectory.py`: pose and sequence sampler.
- `survivor.py`: procedural survivors in 6 postures.
- `randomize.py`: textures, lighting and frame augmentation.
- `labels.py`: YOLO labels from segmentation masks.
- `scripts/sim_preview.py`, `configs/arena.yaml`, `configs/randomization.yaml`.
- pytest, 1 Nov 2026: 123 passed (branch `fix/floor-brightness`; `main` before it: 120).

Evidence for the items below: local files `outputs/scratch/scan_report.md` (30 Oct) and `outputs/scratch/fix_status_report.md` (1 Nov), gitignored, only on the owner's machine.

**Fixed on `main`:**
- Light direction: `arena.yaml` and the randomiser now put the light above the floor (untextured floor grey 112, was 39). The specular coefficient now reaches the renderer.
- Lint: ruff check 0 errors, ruff format 0 unformatted (46 files, measured 1 Nov 2026 on `main`). Enforced by the pre-commit hook (ruff 0.16.10, run `pre-commit install` once per clone) and `scripts/check.ps1`.
- Textured surfaces darkened by the body colour: `apply_to_sim` now resets the floor and wall `rgbaColor` to white, so the texture alone sets the albedo (it was multiplied by 0.45 on the floor, 0.80 on walls). Normal-light floor grey over 200 randomised scenes: mean 40.5 → 91.0 (branch `fix/floor-brightness`, measured 1 Nov 2026). Tested in `TestTexturedSurfaceBrightness`.

**Only on the unmerged branch `fix/trajectory-clearance` (commit `0483528`), not on `main`:**
- Survivor 3D clearance in `TrajectorySampler` (applies only when the caller passes the survivors), and survivors never placed on the start cell.
- Known regression on that branch: with survivors passed, 15 of 160 test sequences collapse to 1 frame, and repeated positions rise from 18 to 1356. A survivor plus 0.6 m clearance blocks a 1.5 m corridor.

**Known open defects on `main`:**
- Camera inside survivors: no survivor clearance check, and survivors can sit on the start cell (22 of 160 test scenes).
- Broken postures: sitting, supine and prone body parts are misaligned or detached. Lateral pokes into walls. The config posture list is ignored.
- Truncation/occlusion always 0: `labels.py` never computes them, so its filters never fire.
- Dark floor texture ranges: `randomization.yaml` solid `floor_color` 0.20–0.48 and noise `floor_base` 0.22–0.45 leave 37 % and 40 % of normal-light solid and noise floors below grey 60 (200 scenes, measured 1 Nov 2026).
- Walls darker than floors under the overhead light: lighting gain about 0.40 on walls vs 0.97 on the floor (default light, measured 1 Nov 2026). This comes from the light geometry; the light direction is unchanged.
- Raw non-metric depth: `camera.py` returns the raw depth buffer, not metres, although documented as linear.
- Texture memory growth: `randomize.py` loads new textures every sequence and never frees them.
- No frame/dataset generator: nothing yet passes survivors to the sampler or writes a dataset.

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
scripts/            [exists] check.ps1 (ruff check + ruff format --check + pytest), check_env.py, verify_environment.py, download_weights.py, check_sim.py, sim_preview.py, setup scripts. [PLANNED] export, benchmark
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
pytest -q                                                      # tests
.\scripts\check.ps1                                           # ruff check + ruff format --check + pytest (PowerShell)
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
