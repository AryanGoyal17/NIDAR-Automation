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
- pytest, 2 Nov 2026 (`main` at `c3c52ed`, `.venv` Python 3.12.10, `scripts/check.ps1`): 271 passed and 18 skipped by default; the 18 are the heavy scene checks (`test_label_scenes.py` 5, `test_trajectory_scenes.py` 11, `test_generator_scenes.py` 2), run with `check.ps1 -Full` (`NIDAR_SCENE_TESTS=1`): 289 passed, 0 skipped (one earlier `-Full` run the same day had 1 failed, 288 passed; the failing test's name was not captured). The 4 live-output tests pass on this laptop only because `outputs/` holds a `detections.jsonl` (see open defects).

Evidence for the items below: local files `outputs/scratch/scan_report.md` (30 Oct) and `outputs/scratch/fix_status_report.md` (1 Nov), gitignored, only on the owner's machine.

**Fixed on `main`:**
- Light direction: `arena.yaml` and the randomiser now put the light above the floor (untextured floor grey 112, was 39). The specular coefficient now reaches the renderer.
- Lint: ruff check 0 errors, ruff format 0 unformatted (41 tracked `.py` files, measured 2 Nov 2026 on `main` at `c3c52ed`). Enforced by the pre-commit hook (ruff 0.16.10, run `pre-commit install` once per clone) and `scripts/check.ps1`.
- Textured surfaces darkened by the body colour: `apply_to_sim` now resets the floor and wall `rgbaColor` to white, so the texture alone sets the albedo (it was multiplied by 0.45 on the floor, 0.80 on walls). Normal-light floor grey over 200 randomised scenes: mean 40.5 → 91.0 (branch `fix/floor-brightness`, measured 1 Nov 2026). Tested in `TestTexturedSurfaceBrightness`.
- Dark floor texture ranges: `randomization.yaml` solid `floor_color` (was 0.20–0.48) and noise `floor_base` (was 0.22–0.45) widened to 0.30–0.75 and 0.30–0.70 (ASSUMPTION, arena floor colour unknown). With the reset, normal-light floor grey over 200 scenes: mean 91.0 → 109.5, share below 60: 18.3 % → 0.6 %, above 200: 0 %.
- Build-specific render pin: `TestRenderPinned` compared sha256 hashes recorded with this laptop's pybullet build. It now compares each pinned pose with a render made from an independently computed view matrix in the same run (per-pose BGR mean absolute difference < 0.05; pins the view/axis maths, not lighting or projection). Fails on a roll sign flip, an up-vector swap and a 0.5° yaw error (branch `chore/check-full`, 1 Nov 2026).
- Collection errors without pybullet: test modules that import pybullet (directly or through `sim.camera` / `sim.maze`) use `pytest.importorskip("pybullet")`, and `test_config.py` uses `pytest.importorskip("yaml")`. Global Python 3.12 without pybullet or yaml: 37 passed, 8 skipped, 0 collection errors (was 8 collection errors; branch `chore/check-full`, 1 Nov 2026).
- Slow scene checks easy to forget: `scripts/check.ps1` prints the Python it uses and the pytest skip count, and the default run ends with "scene tests skipped: run scripts/check.ps1 -Full before merging". `-Full` sets `NIDAR_SCENE_TESTS=1`, runs pytest with `-rs`, fails with "pybullet missing: activate .venv and run again" when pybullet is absent, and fails on any skip whose reason is not in `$FullSkipAllowlist` (empty).

**Fixed on `main`, merged from `fix/labels` (C4, issue 2, issue 8, N7; commits `99536cb` to `ab22c56`, fast-forward 1 Nov 2026; the branch no longer exists locally and has no `origin` ref). Evidence: `outputs/scratch/labelfix/report.md` (1 Nov 2026, gitignored):**
- Measured truncation and occlusion. `SilhouetteRenderer` copies the survivor parts generically from PyBullet shape data (sphere, box, cylinder, capsule; any other shape raises `UnsupportedShapeError` naming the type) into a second DIRECT client and renders each survivor alone, on a canvas extended where its part boxes overhang the frame (capped at `arena.yaml labels.max_canvas_area_factor`, 4.0, ASSUMPTION). Truncation = alone pixels outside the frame / all alone pixels; occlusion = share of the in-frame alone pixels hidden in the real render. A survivor beyond the cap or behind the near plane gets `truncation_exact=False`. `compute_survivor_labels(silhouettes=...)` applies `max_truncation` 0.85 and `max_occlusion` 0.90 (ASSUMPTION, old defaults) to the measured values. Tested in `test_label_renders.py` (copy 0 px different for each shape type, plate truncation within the derived pixel tolerance, wall and survivor occlusion) and, with `NIDAR_SCENE_TESTS=1`, in `test_label_scenes.py` (alone masks equal a reference rendered in the main scene with all other bodies moved away, 6 layouts at 640x360).
- Thresholds at training resolution (issue 8): `arena.yaml labels` gives `min_box_height_px` 10, `min_box_width_px` 6 and `presence_min_area_px` 1.6 (px² at `imgsz` 416), all ASSUMPTION. `LabelConfig.thresholds(width, height)` converts them with scale = imgsz / max(width, height) (min box height 30.8 px at 1280x720, 15.4 px at 640x360; tested at both sizes). `label_frame(...)` returns `FrameLabels(labels, dropped)`, where each dropped label carries its measured values and every reason (`box_too_short`, `box_too_narrow`, `truncated`, `truncation_unbounded`, `occluded`, `low_contrast`). Survivors under the presence floor are absent, not dropped. The old render-pixel defaults (6 px box, 15 px) remain only in the low-level `compute_box_for_body_ids` / `compute_survivor_labels`.
- Visibility filter (N7): `survivor_contrast` measures the largest per-channel |mean(visible survivor pixels) − mean(ring around them)| on the frame after augmentation, ring width `labels.contrast_ring_px` 3 px at imgsz (ASSUMPTION, not tuned). `min_contrast` 3.0 drops essentially-invisible survivors; `hard_contrast` 8.0 tags a kept label `hard` (kept for training, flagged for weighting), both ASSUMPTION, chosen by eye from crops sorted by contrast (`outputs/scratch/labelfix/crop_sheet.png` and `lowlight_below5.png`, gitignored, 1 Nov 2026): on a 256-frame protocol batch plus a forced low-light set, 26 of 27 size-passing low-light crops with contrast under 3.0 were invisible by eye (1 faint), the lowest clearly-visible normal-light crop was contrast 7.4. On the 300-frame `sample_poses` set (10 layouts × 3 seeds × 10 poses, 40/300 low-light): contrast normal light min 4.5 p50 59.3, low light min 0.4 p50 3.8; hard 4/119 kept (3.4%) at 1280×720, 5/119 (4.2%) at 640×360.
- (g) issue 2 re-measured, same 256-frame protocol as the first scan (1280×720, 8 layouts × 2 seeds, 3 survivors, randomisation + augmentation), power plan Balanced/plugged in: before (old `compute_survivor_labels`, no silhouettes) 124 labels, 66 touching the border, truncation>0 **0**, occlusion>0 **0** (first scan: 133/319, 0, 0). After (`label_frame`, kept labels only) 77 kept of 124 present: touching border 33, truncation>0 **34**, occlusion>0 **73**; over all present survivors (kept+dropped) truncation>0 69/124, occlusion>0 99/124. Ambiguous frames 41/256 (16.0% of all, 47.7% of the 86 frames with a survivor present): `box_too_short` 15, `box_too_narrow` 3, `truncated` 21, `truncation_unbounded` 0, `occluded` 2, `low_contrast` 6 (a frame can have >1 reason). Timing (1280×720, same protocol; power plan Balanced, plugged in, at start and end): render 1907 ms/frame with a survivor in view, silhouettes+label_frame add 734 ms (39%); at 640×360, render 445 ms, extra 157 ms (35%); canvas area factor per frame with a survivor in view: mean 1.59 (1280×720) / 1.60 (640×360), p95 ≈3.05.
- Contact sheet (h): `outputs/scratch/labelfix/contact_sheet.png` (gitignored), 24 tiles from the 300-frame `sample_poses` set — green boxes are kept labels (contrast shown), orange is kept-but-`hard`, magenta boxes are dropped labels with their reasons, magenta tile border + red header marks an ambiguous frame.

**Only on the unmerged branch `fix/trajectory-clearance` (commit `0483528`), not on `main`; replaced by `fix/trajectory-v2` below:**
- Survivor 3D clearance in `TrajectorySampler` (applies only when the caller passes the survivors), and survivors never placed on the start cell.
- Known regression on that branch: with survivors passed, 15 of 160 test sequences collapse to 1 frame, and repeated positions rise from 18 to 1356. A survivor plus 0.6 m clearance blocks a 1.5 m corridor.

**Merged into `main` from `fix/trajectory-v2` (redo of `0483528`; commits `43061f9` to `d53b98d`; the branch is deleted locally and on `origin`):**
- Survivors never placed on the start cell (`SurvivorPlacer.place`; 22 of 160 test scenes on `main`, 0 on the branch). Tested in `TestPlacement`.
- `SurvivorInfo.aabb`: each survivor's real 3D box (union of its parts' PyBullet AABBs, any posture). Tested in `TestAABB`.
- `sample_poses` (the source of training frames) takes the survivor boxes (`survivor_boxes=[s.aabb ...]`) and enforces `arena.yaml trajectory.survivor_clearance_m` (0.35 m, ASSUMPTION) and `max_survivor_box_fraction` (0.5, ASSUMPTION). It returns exactly n unique poses or raises `PoseSamplingError` with rejection counts by reason; the counts are logged at INFO on every call. Tested in `test_trajectory.py` and over 8 layouts x 20 seeds in `test_trajectory_scenes.py`.
- `generate_sequence` is for replay and tracking tests only: a self-avoiding walk over the corridor graph (no cell re-entered). At a survivor it stops at the last accepted pose and, by default, adds up to 12 in-place yaw frames of 10 degrees. It ends when there is no unvisited neighbour, and it returns at least `trajectory.min_sequence_frames` (20) frames or raises `SequenceSkipped` with a logged reason. Seeded start-cell retries are deterministic. All four values are in `arena.yaml` as ASSUMPTION. Measured 1 Nov 2026, 8 layouts x 20 seeds, 3 survivors, 60 frames: 0 skipped, 13/160 shorter than 60 (min 39), repeated (x, y, yaw) 0, 0 poses within 0.35 m of a survivor box (`0483528`: 15 one-frame sequences, 1356 repeated positions).
- `camera.py`: axis maths shared through `camera_axes` (renders byte-identical before and after, pinned by `TestRenderPinned`), and `project_aabb` / `projected_box_fraction` (box to image rectangle, within 2 px of the rendered mask box in `TestBoxProjection`).

**Merged into `main` from `feat/generator` (Task 3; commits `7dc1a6e` to `c3c52ed`, linear on `main`; the branch no longer exists locally and has no `origin` ref):**
- `configs/generator.yaml` (every value ASSUMPTION): render 640x360 with the arena FOV, label thresholds at imgsz 416, JPEG quality 95. Layouts: the 6 hand-made ones or a seeded random maze (probability 0.5, 7–15 rows and columns). 0/1/2/3 survivors with weights 0.10/0.30/0.30/0.30. Frame modes aimed 0.55, free 0.20, negative 0.25. Aimed camera 0.6–3.5 m from the survivor's box centre, bearing within 0.6 × half the horizontal FOV. Up to 5 retries per frame.
- `sim/generator.py`, scenes and frames in memory:
  - `derive_seed` is 63 bits of sha256 over (master seed, scene, stream, frame, attempt), with separate streams for layout, maze, randomisation, survivors, count, mode, pose and augmentation. A scene is a pure function of (master seed, scene index, config). The same master seed gives different pose draws in different scenes and layouts (tested; the old shared-seed behaviour is pinned in a test).
  - `plan_scene` chooses the layout and survivor count.
  - `Scene` builds a new PyBullet DIRECT client per scene (textures, lighting, survivors, sampler with survivor boxes, camera, one `SilhouetteRenderer`) and disconnects it on close.
  - Modes:
    - aimed: distance, bearing, `pose_rejection`, survivor centre inside the frame, and a `p.rayTest` line of sight.
    - free: `sample_poses`.
    - negative: `sample_poses` candidates with no survivor box in view along a clear ray; survivors behind walls give hard negatives.
  - `classify`: ambiguous if any present survivor is dropped, positive with kept labels, negative with nothing present. `generate_frame` retries ambiguous frames and mode mismatches. After 5 retries the slot is skipped, so an ambiguous frame is never accepted (tested with an always-ambiguous labeller).
  - Tests: `tests/test_sim/test_generator.py`.
- `sim/generator.py`, writing and running (`python -m sim.generator --out data/sim_raw/<run> --scenes N --frames-per-scene K --seed S --workers W --config configs/generator.yaml`):
  - Files: `images/` (JPEG), `labels/` (YOLO, class 0; empty file only for a negative), and per scene `scenes/<id>.jsonl` (meta), `<id>.attempts.jsonl` (every attempt with its kept and dropped labels) and `<id>.done.json`, written last. `meta.jsonl` and `attempts.jsonl` are merged in scene order. `run.json` holds the arguments and config hash; `manifest.json` holds git commit, config hash, arguments, library versions, counts, skip and retry reasons, timing, CPU per worker, and the power plan at start and end.
  - Meta line: scene/frame ids (scene id = group id for the split), layout and grid, all seeds, mode, retries and reasons, rejected attempts with dropped labels, pose, intrinsics (fx = fy from the FOV, cx = W/2, cy = H/2), near/far clip, light, low-light flag, survivors (posture, world position, yaw, scale, 3D box), labels (truncation, occlusion, contrast, hard) and timing.
  - Every file is written to a temporary name, then renamed. A re-run skips scenes with a done.json; another seed, scene count or config is refused (exit 2).
  - A scene that raises is logged with its traceback, its partial files are removed, `<id>.failed.json` is written and the run continues. The manifest lists it in `failed_scenes`, the CLI exits 1, and a re-run retries it.
  - Workers: a spawn pool with one scene per task. Output is identical for 1 and 2 workers apart from timing (fast test).
  - `replay_frame(meta)` re-renders a frame from its meta line alone. 5 random frames were byte-identical to the written JPEG and labels; no meta field had to be added (heavy test).
- Memory, one worker, 320x180, 1 frame per scene (1 Nov 2026, power plan Balanced, on battery):
  - RSS after 0/10/20/30 scenes: 73.1/74.2/76.7/76.9 MB (heavy test run).
  - From scene 10 to 30, a client per scene grew 0.8 and 1.3 MB (scratch runs). The old pattern (one client, textures re-applied) grew 18.4 and 18.2 MB, about 0.9 MB per scene.
  - The test allows 8 MB (ASSUMPTION, between the two).
- `data_tools/`:
  - `check_generated.py` reads only the written files. It checks:
    - every image has a label file and a meta line, and the other way round;
    - label lines: 5 fields, class 0, values in [0, 1], box inside the image, at least 1 px each side;
    - label files equal the meta labels;
    - an empty label file only for a negative, and no dropped labels in written frames;
    - image size, and no duplicate ids, image bytes or poses within a scene;
    - each scene summary matches its meta lines.
  - `check_generated.py` prints counts per mode and per category. `--stats` adds acceptance, retries, box size at imgsz, truncation/occlusion/contrast histograms, yields per mode and posture, far-speck attempts and phase timing.
  - `draw_boxes.py` draws the label-file boxes on the written JPEGs (seeded tiles, modes in turn); meta.jsonl supplies only the captions.
  - Tests: `tests/test_data_tools/test_check_generated.py`.
- Measured run, 2 Nov 2026, at `88894a0`. 40 scenes × 12 frames, seed 2026, default `generator.yaml` (640x360). All timings: power plan Balanced, plugged in (AC), at start and end. Full report: `outputs/scratch/generator/report.md` (gitignored).
  - **Workers:** 1/2/4 workers give identical output (byte-identical files; meta, attempts and summaries equal without timing). Speed 1.11/2.38/4.27 frames/s (wall 429/200/111 s); CPU per worker 95/95/87 % of its busy time. One run each.
  - **Phase cost:** mean ms per render attempt, 1 worker: render 521, silhouettes 102, augment 85, label_frame 9.
  - **Yield:**
    - 480 slots, 559 renders, 476 frames written. Acceptance (written / renders) 85.2 %; retries per written frame mean 0.17.
    - 4 slots skipped: 2 `ambiguous_after_retries`, 2 `no_valid_pose`.
    - `check_generated`: valid, 0 errors.
  - **Positives:** 252 positive and 224 negative frames (188 in survivor scenes, 36 in the 3 survivor-free scenes). 280 kept labels, 11 hard (3.9 %); 80 low-light frames (16.8 %).
  - **Modes:** realised slot shares in survivor scenes are aimed 53.6 %, free 20.7 %, negative 25.7 % (weights 0.55/0.20/0.25). Aimed gives 98.3 % positives per slot, free 19.6 %.
  - **Box size at 416:** height min 10.4, p10 30.6, median 77.7, p90 134.0.
  - **Ambiguity:**
    - 78 of 335 rendered attempts with a survivor present were ambiguous (23.3 %), all retried and none written.
    - 58 of the 78 were in low-light scenes. Aimed renders were ambiguous in 51 of 90 in low light and 14 of 209 in normal light, mostly `low_contrast`.
    - 17 ambiguous attempts had one survivor kept and another dropped. Only 7 of those dropped boxes are under 10 px tall or 6 px wide at 416.
  - **Truncation:** 13 kept labels have truncation in [0.75, 0.85); the limit 0.85 is unchanged.
  - **JPEG check:** labelling the JPEG-decoded frame (quality 95) instead of the augmented frame flips 0 of 389 present-survivor decisions over all 559 attempts (scratch script, generator unchanged).
  - **Metric depth later:** the meta has what it needs (pose, intrinsics, near/far clip, survivor world positions and 3D boxes). The depth buffer can be re-rendered with `replay_frame`'s render call.
  - **Contact sheet:** `outputs/scratch/generator/contact_sheet_gen40_w1.png` (gitignored).
- Tests: `tests/test_sim/test_generator.py` (fast, 160x90: same args identical, workers 1 = 2, written labels = in-memory labels, empty label file only for negatives, always-ambiguous labeller writes nothing, resume, refused arguments, failing scene) and `tests/test_sim/test_generator_scenes.py` (`NIDAR_SCENE_TESTS=1`: replay, memory).

**On branch `feat/coco-assets` (real-photo assets from COCO val2017, not merged). COCO sanity set, not arena: COCO photos are a regression and sanity check, never an arena score. Rules and reasons: `docs/decisions/007_real_data_sources.md`.**
- `configs/coco.yaml` (every value ASSUMPTION unless marked as the owner's choice) and `data_tools/coco_common.py` (config, annotation index, `ImageZip`, split hash, xywh to YOLO, size limits read from `arena.yaml labels`, sha256, 64-bit dHash, disk guard). Disk guard in decimal GB: every large step stops if less than `disk.min_free_gb` (1.0) would remain free, and a set estimated above `disk.confirm_above_gb` (0.5) is not written without `--yes`.
- `data_tools/get_coco.py`: the only setup-time download (val2017.zip, annotations_trainval2017.zip into `data/coco/`). Streams in 1 MiB chunks, hashes while streaming, resumes a `.part` file when the server answers 206 and restarts on 200, checks the byte count, records url/bytes/sha256/UTC time in `data/coco/DOWNLOAD.json` (sha256 recorded, not verified: no official checksum known, TBD), every attempt (start/end byte, outcome, failed ones included) with `attempts_count` / `resumes_count`, and the `zipfile.testzip` result of each zip. Skips existing files (re-hash; a sha256 different from the recorded one is an error). Extracts only `instances_val2017.json` and `captions_val2017.json` (zip-slip check) and checks each against the CRC32 and size in the zip directory, also on re-runs (mismatch = error; a mismatching existing file is extracted again); records CRC32, sizes and the sha256 of the extracted file. `val2017.zip` stays zipped. A blocked download exits 3 with no substitute source. Fetching only the two members by HTTP range (about 7.3 MB of 252.9 MB) is a possible fallback, NOT implemented (`data/README.md`). Tests: `test_get_coco.py` (fake `urlopen`: fresh, resume, restart, short body, blocked, unsafe member, CRC mismatch, tampered file extracted again, attempts/resumes/testzip recorded and kept on re-run).
- `data/README.md` (committed, the only file under `data/`): licence terms (annotations CC BY 4.0, images from Flickr under their own licences, internal research use only, never committed or redistributed).
- `data_tools/coco_split.py`: every val2017 image id goes to `cutout_source` / `anchor` / `sanity` (0.40/0.40/0.20, ASSUMPTION) by u = first 64 bits of sha256("<seed>:<id>") / 2^64; depends only on seed and id, so stable when images are added or removed. Writes `data/coco/split.json` and prints images, images with persons, person boxes and person crowd regions per group. Tests: `test_coco_split.py` (deterministic, disjoint and complete, shares within 0.015 on 20000 ids, stable on a subset, seed changes it).
- `data_tools/coco_person_sets.py`: YOLO single-class sets `data/coco_person/{anchor,sanity}/{images,labels}` plus `anchor.yaml` and `sanity.yaml` (names {0: person}; sanity.yaml points only at sanity images, header says evaluation only). Images are copies of the zip members' original JPEG bytes (sha256 unchanged, tested with a synthetic zip). All-or-nothing: one non-crowd person under 10 px tall or 6 px wide after letterboxing to 416 (read from `arena.yaml labels`), or any crowd region in the person category, excludes the whole image; persons stored as RLE are kept (boxes only). Negatives (300 anchor / 100 sanity, ASSUMPTION): person-free images in seeded order, dropped if a caption has a person word (list in `coco.yaml`, ASSUMPTION) or if pretrained `yolo11n.pt` finds a person at conf 0.5 (owner's choice); the detector only excludes, never labels. Size estimated from the zip before copying. `manifest.json` per set: counts, exclusions by reason, negative-filter table, box sizes at 416, share under 40 px. `data_tools/coco_sheets.py`: contact sheets for cutouts and YOLO sets. Tests: `test_coco_person_sets.py` with the synthetic fixture in `conftest.py` / `coco_synth.py` (conversion, letterbox limits, all-or-nothing, crowd, other-category crowd ignored, RLE kept, negative filters, bytes unchanged, labels, data yaml, end-to-end, overwrite refused, size confirmation).
- `data_tools/make_person_cutouts.py`: RGBA PNG cutouts in `data/coco_cutouts/images/` only from `cutout_source`: non-crowd polygon persons (RLE skipped and counted; `cv2.fillPoly`, no pycocotools), box not within 2 px of the border, source box height >= 150 px, no other person box (crowd included) covering > 0.10 of its box, mask >= 0.30 of the box (ASSUMPTION). Crop tight to the mask, binary alpha, colour zeroed outside. `manifest.json`: source id, COCO box, cutout box, size, aspect, mask fraction, sha256, rejection counts, distributions, wide share (w/h > 1.2). Size estimated from a 50-cutout PNG sample before writing. Tests: `test_person_cutouts.py` (alpha nonzero and tight, colour zeroed, each rule, overlap with a crowd box, end-to-end).
- `data_tools/leakage.py`: training sources (`--train` dirs, and the COCO source photos of `--cutouts` dirs read from the zip) against protected sets (default: COCO sanity images, and `data/test_proxy`, `data/test_real` if they exist). Exit 1 on the same sha256 or the same COCO id on both sides, a `cutout_source` id used as a full image anywhere, or a cutout whose source is not in `cutout_source`. Near matches (64-bit dHash, Hamming <= 4) are listed as warnings only. Tests: `test_leakage.py` (planted byte duplicate, same id with other bytes, resized copy = warning, cutout_source in anchor, cutout manifest sources).
- Measured run, 2 Nov 2026, at `c52e74c`. All timings: power plan Balanced, plugged in (AC), at start and end. Full report: `outputs/scratch/coco/report.md`; sheets `outputs/scratch/coco/cutouts.png` and `anchor.png` (all gitignored).
  - **Download:** val2017.zip 815,585,330 bytes in 10 attempts (9 resumes); annotations_trainval2017.zip 252,907,541 bytes in 6 attempts (5 resumes). The server reset or timed out repeatedly, and one attempt failed on DNS. No other source was used. `testzip`: no bad member in either zip; both extracted JSONs match the CRC32 and size in the zip directory; sha256 recorded, not verified. The attempt history before `c52e74c` is reconstructed from logs (said in `DOWNLOAD.json`). Download runs 3063.6 s in total.
  - **Split** (1.1 s), images / with persons / person boxes / person crowd regions: cutout_source 1987/1080/4472/104, anchor 1982/1056/4140/85, sanity 1031/557/2165/38 (total 5000/2693/10777/227).
  - **Anchor** (build 34.2 s, estimate 0.274 GB for both sets): 1158 images (858 with persons, 300 negatives), 2289 boxes; box height at 416 min 10.0, p10 24.4, median 95.9, p90 269.6; 21.7 % under 40 px. Person images excluded 198/1056 (too short 121, too narrow 120, person crowd 85; an image can have several).
  - **Sanity** (COCO sanity set, not arena): 560 images (460 with persons, 100 negatives), 1282 boxes; height min 10.1, p10 25.6, median 101.6, p90 271.1; **20.1 % under 40 px**. Excluded 97/557 (62, 60, 38).
  - **Negatives:** candidates 926 / 474. Caption filter removed 128 / 66, detector (yolo11n, conf 0.5) 11 / 2, both 7 / 1; selected 300 / 100. The top caption match is "couple" (49 / 26), probably mostly "a couple of ...", so the filter is stricter than needed (word list unchanged, ASSUMPTION).
  - **Top 5 exclusion reasons, both sets:** negative_over_quota 801, negative_caption 194, person_too_short 183, person_too_narrow 180, person_crowd 123. RLE persons kept: 0.
  - **Cutouts** (11 s): 455 of 4472 persons. First failing rule: too_short 3059, touches_border 500, overlap 432, mask_fraction 26; RLE 0. Height p10/median/p90 171/273/441 px; aspect w/h 0.33/0.55/0.95; wide (> 1.2) 16 (3.5 %).
  - **Seen by eye on the sheets:** some cutouts are partial people (head and shoulders) or carry a background halo (coarse COCO polygons); COCO labels statues and people on screens as persons.
  - **Leakage** (47.1 s): 1554 training sources (1158 anchor + 396 cutout source photos) vs 560 sanity images: 0 errors, 0 near-match warnings. Sanity passed as `--train`: 560 errors, exit 1.
  - **Disk:** data/coco 1.09 GB, data/coco_person 0.28 GB, data/coco_cutouts 0.03 GB. Free on D: 3.44 GB at the start of the session, 2.03 GB at the end (decimal GB; never near the 1 GB floor).

**Known open defects on `main`:**
- Broken postures: sitting, supine and prone body parts are misaligned or detached. Lateral pokes into walls. The config posture list is ignored.
- Survivor parts inside walls (`labelfix/report.md` §6; the 16 scenes of the 256-frame protocol, re-rendered at 640x360): 30 of 48 survivors have a body-part box overlapping a wall box (lying_supine 15/15, lying_prone 15/16, standing 0/5, sitting 0/10, lying_lateral 0/2). Checked by box overlap with boxes shrunk by 3 cm to skip PyBullet AABB margins; this is an upper bound, because a rotated capsule's AABB is larger than the capsule. This inflates occlusion values until the posture geometry (C2) is rebuilt.
- High ambiguous-frame share: 37.1 % of frames with a survivor present are ambiguous in the 300-frame `sample_poses` set at 1280x720 (43/116; 36.2 %, 42/116 at 640x360) and 47.7 % in the 256-frame protocol (41/86), mainly `box_too_short` (26 frames in set 300 at 1280x720, 15 in the protocol) and `truncated` (16 and 21). Skipping ambiguous frames would discard a third to a half of the frames that contain a survivor. With `sim/generator.py`, aimed poses and per-frame resampling bring the written-frame yield to 476 of 480 slots. Ambiguous attempts are retried, never written: 23.3 % of rendered attempts with a survivor present, mostly low-light `low_contrast`.
- Synthetic survivors are mostly high-contrast: normal-light contrast median 59.3 at 1280x720 (53.9 at 640x360; set 300, n=157); kept labels tagged `hard`: 4/119 (3.4 %) and 5/119 (4.2 %) in set 300 at 1280x720 and 640x360, 4/77 (5.2 %) in the 256-frame protocol. How this compares with real footage is TBD (no real test set yet).
- Low-light survivors under contrast 3.0 (dropped as `low_contrast`), share of present low-light survivors: set 300 at 1280x720 7/15; set 300 at 640x360 5/15; 256-frame protocol (1280x720) 6/16; phase-A low-light pool (the protocol's low-light survivors plus a forced low-light set of 96 frames, 1280x720) 34/100. So about 33–47 % of low-light survivors fall under contrast 3.0, depending on the sample; the natural samples are small (15 and 16).
- Saturated floor colours: texture colours are sampled per RGB channel independently, so the wider floor ranges also give strongly tinted floors (for example purple, green).
- Walls darker than floors under the overhead light: lighting gain about 0.40 on walls vs 0.97 on the floor (default light, measured 1 Nov 2026). This comes from the light geometry; the light direction is unchanged.
- Raw non-metric depth: `camera.py` returns the raw depth buffer, not metres, although documented as linear.
- Texture memory growth: `randomize.py` loads new textures every sequence and never frees them (about 0.9 MB per scene when one client is reused). `sim/generator.py` avoids it with a new client per scene; `randomize.py` itself is unchanged.
- `check.ps1 -Full` fails on a fresh clone: the 4 live-output tests in `test_detection_output.py` skip ("No live detections.jsonl found") until the baseline detector has written its output under the gitignored `outputs/` (seen in a `git archive` copy, 1 Nov 2026).

**Removed:** `gcs/` (ground-station stubs, out of scope). `AGENTS.md` (this file is the only agent context).

**Empty placeholders:** `cv/utils/`, `interfaces/schemas/`, `tests/test_interfaces/` (`.gitkeep` only).

**README:** `README.md` exists but is out of date: it still describes the Jetson, RT-DETR and an RGB-only camera. A rewrite is planned.

**Planned (not built yet):**
- **Task 3:** the frame/dataset generator is merged (`sim/generator.py`). Clutter (`sim/`) is not built yet.
- `data_tools/` compositor and dataset builder, and the real test set (`data/test_real/`). Person cutouts from COCO are on branch `feat/coco-assets`.
- `cv/train/`, `cv/eval/`, `cv/tracking/`, the depth module, the detection API.
- NCNN export and benchmark scripts, `deploy/pi/`, the handoff note.

Mark anything not run as **[PLANNED]**. Never write a result into the README unless it was measured by us.

## 7. Repo layout and commands

```
configs/            [exists] default.yaml, arena.yaml, randomization.yaml, generator.yaml, coco.yaml, maze_layouts/. [PLANNED] composite.yaml, train.yaml, tracking.yaml, depth settings
cv/                 [exists] config.py, logger.py, detectors/, utils/ (empty). [PLANNED] eval/, tracking/, train/, depth, detector_api.py
sim/                [exists] maze, layout_gen, camera, trajectory, survivor, randomize, labels, generator (dataset runs), layouts/. [PLANNED] clutter
data_tools/         [exists] draw_boxes (contact sheets), check_generated (run validation + stats), coco_common + get_coco (COCO val2017 setup download), coco_split, coco_person_sets (anchor/sanity YOLO sets), make_person_cutouts, coco_sheets, leakage (train vs protected sets). [PLANNED] compositor, dataset builder
deploy/             [PLANNED] pi/
interfaces/schemas/ [exists, only .gitkeep] the detection JSON schema is PLANNED
docs/               [exists] conventions.md, arena_questions.md, decisions/ (ADRs 001–006), diary/, setup/
scripts/            [exists] check.ps1 (ruff check + ruff format --check + pytest; -Full adds the scene checks), check_env.py, verify_environment.py, download_weights.py, check_sim.py, sim_preview.py, setup scripts. [PLANNED] export, benchmark
requirements/       [exists] requirements-base.txt, requirements-cv.txt, requirements-sim.txt (pybullet), requirements-pi.txt (Pi runtime, TBD items), requirements-dev.txt (requirements.txt + sim + ruff, pytest, pre-commit)
requirements.txt    [exists] base + cv
.pre-commit-config.yaml  [exists] ruff-check and ruff-format hooks (ruff-pre-commit v0.16.10)
notebooks/          [exists, empty]
tests/              [exists] test_cv/, test_sim/, test_data_tools/, test_interfaces/ (empty)
data/  models/  outputs/   [exist, gitignored, never committed; only data/README.md is committed]. data/coco/ holds the COCO download; data/test_real/ is PLANNED; models/ holds local weights.
```

```bash
pip install -r requirements/requirements-dev.txt              # dev laptop setup (install torch first, see README §4)
pre-commit install                                            # once per clone: ruff hook on every commit
python scripts/check_env.py                                   # environment health
python -m sim.generator --out data/sim_raw/run1 --scenes 40 --frames-per-scene 12 --seed 2026 --workers 4   # synthetic dataset (re-run resumes)
python -m data_tools.check_generated --run data/sim_raw/run1 --stats         # validate a run (exit 1 if invalid) + statistics
python -m data_tools.draw_boxes --run data/sim_raw/run1 --out sheet.png --tiles 24   # contact sheet from the written files
python -m data_tools.get_coco                                 # setup only: COCO val2017 into data/coco/ (the only download)
python -m data_tools.coco_split                               # data/coco/split.json: cutout_source / anchor / sanity
python -m data_tools.coco_person_sets                         # data/coco_person/{anchor,sanity} + anchor.yaml, sanity.yaml (--plan: estimate only)
python -m data_tools.make_person_cutouts                      # data/coco_cutouts/ RGBA PNGs + manifest.json
python -m data_tools.leakage --train data/coco_person/anchor/images --cutouts data/coco_cutouts   # exit 1 on exact overlap with protected sets
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
