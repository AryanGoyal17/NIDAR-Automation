# Engineering Diary — Day 2: Decisions & Model Strategy

---

## 1. Goals
- [x] Download baseline weights (`yolov8n.pt` and `yolo11n.pt`) with checksum and size verification.
- [x] Verify simulation environment and headless rendering with PyBullet.
- [x] Remove out-of-scope ground-station (GCS) stubs and references.
- [x] Align detection output conventions (bounding box format, timestamp format, class identification) across code, documentation, and automated tests.
- [x] Formalise architecture decisions for model selection (ADR 004) and data strategy (ADR 005).

---

## 2. What Was Done
- **Model Weights Downloader:**
  - Implemented `scripts/download_weights.py` to fetch pretrained model weights (`yolov8n.pt` and `yolo11n.pt`) into `models/`.
  - Added SHA-256 checksum and file size validation, with automatic skipping of existing files.
- **Simulation Environment Smoke Test:**
  - Implemented `scripts/check_sim.py` to test PyBullet headless direct connection (`p.DIRECT`), load a box, render a camera image (`getCameraImage`), and save output to `outputs/sim_check.png`.
- **Subsystem Scope Clean-up:**
  - Removed out-of-scope GCS stubs (`gcs/` directory, `requirements/requirements-gcs.txt`) and references in documentation and configuration.
- **Conventions Alignment & Testing:**
  - Verified and aligned detection output specification: sub-pixel float bounding boxes (`xyxy` format), ISO 8601 UTC timestamps with millisecond precision, and explicit `class_id` (`0` for `person`).
  - Added unit test suite in `tests/test_cv/test_detection_output.py` to validate record schema, coordinates, center calculations, and output files.
- **Decision Records:**
  - Documented ADR 004 (`docs/decisions/004_model_choice.md`) and ADR 005 (`docs/decisions/005_data_strategy.md`).

---

## 3. Key Decisions Made (from CLAUDE.md)
- **Model Choice:**
  - Primary candidate: **YOLO11n** (lightweight nano model, ~2.6M parameters, single class `person`, trained at input resolution 416×416).
  - Safe fallback & baseline: **YOLOv8n**, trained on identical data and parameters.
  - Dropped candidates:
    - **RT-DETR**: Dropped because target boards cannot sustain required frame rates with transformer encoder/decoder overhead.
    - **YOLOv8s / YOLOv8m**: Rejected due to edge compute constraints when sharing the board with other subsystems.
    - **Custom YOLO + ViT hybrid**: Rejected. YOLO11 features a lightweight partial spatial-attention block (C2PSA) in its deepest backbone stage; it is **not** a Vision Transformer (ViT), and differences between YOLO11n and YOLOv8n cannot be credited to attention alone.
- **Training Data Strategy:**
  - No real labelled arena data exists yet.
  - Four-pillar training set composition:
    1. Domain-randomised PyBullet 3D arena renders.
    2. Real-person cutouts composited onto empty PyBullet arena frames.
    3. Small share of real aerial images (SARD, from NIDARX Kaggle dataset).
    4. Negative frames (empty corridors, walls, obstacles) to suppress false alarms.
  - Splitting rule: Split strictly by **sequence/group, never randomly by frame** (avoids data leakage from visually identical consecutive frames).
  - Dedicated real test set (`data/test_real/`): Kept completely separate from training data; simulation images are never included in the real test set.
- **Improvement Ladder:**
  - Baseline (pretrained YOLOv8) → Fine-tuned (simulation data) → Tracking + multi-frame confirmation → YOLO11 → Final selection based on target board measurements.
- **Confirmation Logic:**
  - A track is confirmed when matched in at least N of the last M frames, with N and M chosen from measured per-frame rates.
- **Evaluation Priority:**
  - Recall is prioritized over precision (a missed survivor is worse than a false alarm).
  - Metrics: Precision, recall, F1, mAP@0.5, survivor recall, false survivors per run, double counts per run, and latency in milliseconds per frame on the target board.

---

## 4. Hardware Targets & Constraints
- **Development Environment:** Windows, Python 3.12, PyTorch 2.6 with CUDA 12.4, Ultralytics 8.4, NVIDIA RTX 3050 Laptop GPU.
- **Candidate Onboard Computers:**
  - **Raspberry Pi 5 (8 or 16 GB, CPU only):** Uses Ultralytics with NCNN export.
  - **Original Jetson Nano (4 GB, 128-core Maxwell GPU):** JetPack 4.6.x, Ubuntu 18.04, Python 3.6 (Ultralytics cannot be installed onboard). Uses ONNX export (opset 12, static shapes) converted to TensorRT FP16 engine on the Nano itself, driven by a standalone Python 3.6 inference script.
- **Runtime Constraints:** Fully offline during missions; no cloud calls or external network dependencies; detection must never run inside the flight-control loop.

---

## 5. Next Steps
- Commit new detection-focused README.
- Build synthetic arena simulation and data generation tools in `sim/`.
- Implement cutout compositing and dataset builder in `data_tools/`.
- Implement fine-tuning pipeline in `cv/train/` and evaluation protocol in `cv/eval/`.
- Implement multi-object tracking and M-of-N survivor confirmation in `cv/tracking/`.
- Prepare model export pipelines for NCNN (Raspberry Pi 5) and TensorRT (Jetson Nano Python 3.6).
