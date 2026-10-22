# CLAUDE.md — NIDAR Human (Survivor) Detection

> Project context for AI coding agents (Claude Code, Antigravity) and for new chats. Read it fully before doing any task.
> Keep it current: when a decision, status line or open question changes, edit this file in the same commit.
> Last reviewed: 21 Oct 2026.

## 1. What this repo is

Software for **one subsystem** of the NIDAR team's AirMouse 2026–27 entry: **human (survivor) detection** from the drone's camera video. Owner: Aryan (GitHub: AryanGoyal17). Repo: `github.com/AryanGoyal17/NIDAR-Automation` (a rename to something like `NIDAR-Human-Detection` is under consideration).

AirMouse: a fully autonomous drone flies inside a GPS-denied indoor maze, finds up to six survivors, and reports them on a real-time 2D map, with no manual navigation help.

**In scope:** detecting people in frames, tracking them across frames, confirming each survivor once, measuring accuracy and speed, exporting the model for onboard use, and the detection output format.

**Out of scope (other people's work, do not build or modify):** path planning and navigation, mapping and survivor tagging (turning a detection into a map position), obstacle avoidance, flight control, ground-station application. A "controls team" will integrate everything, so the handoff interface (Section 8) matters.

## 2. Deadline and definition of done

The finished detection algorithm is due by the **evening of 22 Oct 2026**. "Done" means all of these exist, run from a fresh clone, and are honest about what is tested:

1. Trained models for YOLO11n and YOLOv8n (weights in `models/`, not in git).
2. Evaluation results on a small **real** test set, reported separately from simulation results.
3. A stable detection API (Section 8) with ONNX and NCNN exports, and a Jetson TensorRT inference script.
4. README (algorithm specification, hardware requirements, results with only measured numbers) and a handoff note for the controls team.

## 3. Hardware targets and constraints

- **Development laptop:** Windows (PowerShell), Python 3.12, PyTorch 2.6 with CUDA 12.4, Ultralytics 8.4, NVIDIA RTX 3050 Laptop GPU (usually 4 GB VRAM; verify with `nvidia-smi`).
- **Onboard computer (not final):** most probably a **Raspberry Pi 5 (8 or 16 GB, CPU only)**, otherwise the **original Jetson Nano** (4 GB, 128-core Maxwell GPU, JetPack 4.6.x, Ubuntu 18.04, **Python 3.6**, Ultralytics cannot be installed there).
- The Pi path uses Ultralytics with an NCNN export. The Jetson path uses an ONNX export (opset 12, static shapes), a TensorRT FP16 engine built **on the Nano itself**, and a standalone inference script.
- Code under `deploy/jetson/` must run on **Python 3.6**: no dataclasses, no walrus operator, no `X | Y` type syntax, no positional-only arguments, no `from __future__ import annotations`.
- **No internet during a mission.** No cloud APIs. All weights and dependencies must be local files.
- Detection shares the board with mapping, planning and obstacle avoidance, so keep it light. Detection must never be inside the flight-control loop.
- The dev machine is Windows: write cross-platform Python scripts (use `pathlib`, no hard-coded separators). If a shell script is unavoidable, also provide a PowerShell version.
- PyBullet may fail to build on some Python and OS combinations. If it does, use a separate Python 3.10 or 3.11 environment for simulation only.

## 4. Decisions already made (do not re-open without being asked)

- **Models:** train **YOLO11n (primary)** and **YOLOv8n (safe fallback and baseline)** on identical data, imgsz 416 (revisit if the board is fast enough), single class `person`.
- **Not doing:** RT-DETR (too heavy for the boards), larger YOLO sizes (s, m), a custom YOLO+ViT hybrid. YOLO11 has a lightweight partial spatial-attention block (C2PSA) in its deepest backbone stage. It is **not** a Vision Transformer, so never describe it as a ViT hybrid. Differences between YOLO11n and YOLOv8n cannot be credited to attention alone.
- **Data strategy:** no real labelled arena data yet. Training data = PyBullet renders of the arena (domain-randomised) + real-person cutouts composited onto empty PyBullet frames + a small share of real aerial images (SARD, from the NIDARX Kaggle dataset) + negatives. Split by **sequence/group, never by frame**. A small **real test set** (`data/test_real/`) is never used for training.
- **Improvement ladder:** baseline (pretrained YOLOv8) → fine-tuned (simulation data) → tracking + multi-frame confirmation → YOLO11 → (RT-DETR dropped). The final system is the best measured candidate plus tracking and confirmation, exported for the board.
- **Confirmation rule:** a track is confirmed when it is matched in at least N of the last M frames; choose N and M from measured per-frame rates, not guesses.
- **Metrics:** recall is the priority (a missed survivor is worse than a false alarm). Report precision, recall, F1, mAP@0.5, and survivor-level false survivors and double counts. Speed in ms per frame **on the target board** when available.

- **Config defaults vs decisions:** `configs/default.yaml` currently holds the defaults of the baseline (pretrained) script: imgsz 640 and conf 0.5. The 416 size applies to the fine-tuned models and deployment (`configs/train.yaml`, planned). The confidence threshold is chosen by the threshold sweep (plan step 6.4); the 0.4 in Section 8 is only an example. Do not change `default.yaml` to match a decision unless asked.

## 5. Open questions (do not assume; ask or mark as an assumption)

- Are survivors real people or dummies? (A COCO "person" model may miss dummies; thermal sees nothing unless they are heated.)
- Camera model, resolution, field of view, mounting height and tilt. Is there a depth camera? (Not in the plan.)
- Final onboard computer: Raspberry Pi 5 or original Jetson Nano.
- Required detection frame rate (to be set with the controls team).
- Arena dimensions: corridor width, wall height and colours, lighting, ceiling.
- Whether the repo stays public, and its licence (Ultralytics is AGPL-3.0).

## 6. Status (update as you go)

**Done:** Day-1 foundation (folder structure, config loader, logger, environment checks, decision records 001–005, diary, conventions); baseline YOLOv8n detection script (`cv/detectors/detect_survivors.py`); weight downloader (`scripts/download_weights.py`); PyBullet simulation smoke test (`scripts/check_sim.py`). Weights are **not** in git (`models/` is gitignored).

**Planned (not built yet):** committing the new detection-focused README (plan step 1.1), `sim/` (PyBullet data generation), `data_tools/` (cutouts, compositor, dataset builder), `cv/train/`, `cv/eval/`, `cv/tracking/`, export scripts, `deploy/pi/`, `deploy/jetson/`, the detection API, handoff note.

Mark anything not run as **[PLANNED]**. Never write a result into the README unless it was measured by us.

## 7. Repo layout and commands

```
configs/            [exists] default.yaml, arena.yaml. [PLANNED] randomization.yaml, composite.yaml, train.yaml, tracking.yaml
cv/                 [exists] config.py, logger.py, detectors/, utils/. [PLANNED] eval/, tracking/, train/, detector_api.py
sim/                [PLANNED] PyBullet maze, camera, survivors, randomisation, generator, compositor
data_tools/         [PLANNED] cutouts, dataset builder, inspection
deploy/             [PLANNED] pi/, jetson/
interfaces/schemas/ [exists, only .gitkeep] the detection JSON schema is PLANNED
docs/               [exists] conventions.md, decisions/ (ADRs), diary/, setup/
scripts/            [exists] check_env.py, verify_environment.py, download_weights.py, check_sim.py, setup scripts. [PLANNED] export, benchmark
requirements/       [exists] modular requirements
notebooks/          [exists]
tests/              [exists] test_cv/, test_interfaces/
gcs/                [exists but OUT OF SCOPE] backend/ and frontend/ stubs; to be removed (plan step 1.2). Do not build on it.
data/  models/  outputs/   [exist, gitignored, never committed]. data/test_real/ is PLANNED; models/ holds local weights.
```

```bash
python scripts/check_env.py                                   # environment health
python -m cv.detectors.detect_survivors --source data/samples/ # baseline detector
pytest -q                                                      # tests
```

## 8. Handoff interface (what the controls team will call)

```python
detector = SurvivorDetector(backend="pytorch|onnx|ncnn|trt", weights=..., imgsz=416, conf=0.4)
detections = detector.detect(frame_bgr)   # list of Detection (conf=0.4 above is an example; the final value comes from the threshold sweep)
# Detection: x1, y1, x2, y2 (pixels, xyxy, origin top-left), confidence (0-1), class_name="person"
```

Per-frame JSON output follows README Section 8 (frame_id, timestamps, frame size, inference_ms, detections). Timestamps are UTC ISO 8601. A capture timestamp (when the camera captured the frame) is required because mapping needs it.

## 9. Conventions

- Pixel origin top-left, x right, y down. Boxes are `xyxy` in pixels. Timestamps UTC ISO 8601.
- Settings come from `configs/*.yaml` through `cv/config.py`. Log with `cv/logger.py`, never raw `print()` in library code.
- Seeds everywhere; every generated dataset and training run records its seed, config and git commit.
- Conventional commits (`feat(sim): ...`, `fix(eval): ...`, `docs: ...`), small and atomic, on a feature branch.
- Large files (`.pt`, `.onnx`, `.engine`, videos, datasets) are never committed. Record sha256 checksums instead.

## 10. Rules for AI agents working in this repo

1. Read this file first. Do the task asked, inside the scope named. Do not refactor unrelated code.
2. For any change touching more than one file, **propose a plan first** and wait for approval.
3. Run the tests or commands you wrote and show the real output. Never claim something works without running it. Say clearly what you could not test (for example hardware you do not have).
4. **Never invent numbers, metrics, benchmark results, or facts about the rulebook.** If a value is unknown, write TBD or an `# ASSUMPTION` comment and add it to Section 5.
5. Do not add cloud calls, telemetry, or downloads at runtime. Weights and datasets are fetched once during setup.
6. Respect the Python 3.6 rule for `deploy/jetson/`.
7. Never commit secrets, datasets, weights, or personal information.
8. Keep README claims aligned with reality: [DONE] only if it exists and ran, otherwise [PLANNED].
9. If you discover a contradiction between this file and the repo, tell me and fix the file, not the other way round, unless I say otherwise.
10. At the end of a task, update Section 6 (Status) if the status changed.