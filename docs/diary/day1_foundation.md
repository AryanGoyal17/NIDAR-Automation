# Engineering Diary — Day 1: Environment & Foundations

- **Date:** 2026-10-03
- **Author / Lead:** Aryan, AI/CV + GCS Lead
- **Sprint / Milestone:** Sprint 1 — Foundation & Baseline Detector (Day 1 of 5)
- **Target Subsystem(s):** CV, Infrastructure, Repository Architecture, Development Environment

---

## 1. Goals for Today
- [x] Establish core tooling and architecture decisions ([`001_tooling.md`](../decisions/001_tooling.md)).
- [x] Verify local Python 3.12 environment and isolate system paths ([`python_verification.md`](../setup/python_verification.md)).
- [x] Configure Git authentication and branching protocol for the 10-person team ([`002_git_workflow.md`](../decisions/002_git_workflow.md)).
- [x] Design and scaffold the full modular repository folder structure.
- [x] Establish a zero-large-file policy for Git with model/dataset separation ([`003_large_files.md`](../decisions/003_large_files.md)).
- [x] Build isolated Python `.venv` and install CUDA-accelerated PyTorch, torchvision, Ultralytics, and OpenCV.
- [x] Write and pass automated hardware and environment diagnostics (`scripts/check_env.py`).
- [x] Implement YAML configuration parser (`cv/config.py`) and dual console/file UTC logger (`cv/logger.py`).
- [x] Author comprehensive repository documentation (`README.md`, `TEMPLATE.md`, and Day 1 log).
- [x] Configure code quality tooling (`pyproject.toml`, Ruff linter/formatter, Pytest smoke test suite).
- [x] Deploy and verify baseline survivor detector (`cv/detectors/detect_survivors.py`) on sample flight imagery.

---

## 2. What I Did

Today marked the foundational kickoff of the CV and GCS software stacks for project NIDAR:

1. **Tooling & Architecture Groundwork (Step 1.1):**
   - Assessed competition requirements (AirMouse 2026–27: GPS-denied indoor drone navigation, offline mission constraints).
   - Selected Python 3.12 as the core language for CV, PyTorch 2.6 with CUDA 12.4 for accelerated tensor operations, Ultralytics YOLOv8 as the primary target detector, and OpenCV 4.11 for real-time video pipeline operations.
   - Identified open team questions: Companion computer hardware selection (Jetson Orin Nano vs Raspberry Pi 5) and ROS 2 distribution (Humble vs Jazzy). Captured these in [`docs/decisions/001_tooling.md`](../decisions/001_tooling.md).

2. **Python & Git Environment Isolation (Steps 1.2 – 1.4):**
   - Verified local Python 3.12.10 on Windows 11 host.
   - Initialized the Git repository and connected remote origin to `https://github.com/AryanGoyal17/NIDAR-Automation.git`.
   - Designed a lightweight, collaborative branching workflow suited for multidisciplinary autonomy, SLAM, and simulation teams (`feature/`, `fix/`, PR review rules).

3. **Repository Scaffolding & Large File Isolation (Steps 1.5 – 1.6):**
   - Built automation scripts (`scripts/setup_repo_structure.ps1` and `.sh`) that generated clean directory hierarchies for `cv/`, `gcs/`, `interfaces/`, `configs/`, `docs/`, `scripts/`, `tests/`, `models/`, and `datasets/`.
   - Created a comprehensive `.gitignore` explicitly blocking model weights (`*.pt`, `*.onnx`, `*.engine`), media datasets, outputs, virtual environments, and IDE artifacts.
   - Authored [`docs/decisions/003_large_files.md`](../decisions/003_large_files.md) detailing our offline shared-drive storage convention with SHA256 checksum tracking.

4. **Dependency Engineering & CUDA Setup (Step 1.7):**
   - Created clean virtual environment `.venv` using Python 3.12.
   - Segmented requirements into `requirements-base.txt`, `requirements-cv.txt`, and `requirements-gcs.txt` unified under `requirements.txt`.
   - Installed PyTorch 2.6.0 with CUDA 12.4 acceleration wheels directly from the PyTorch index.
   - Created `scripts/verify_environment.py` to validate package versions automatically.

5. **Hardware Diagnostic Script (Step 1.8):**
   - Developed `scripts/check_env.py` to probe host OS, Python build, CUDA runtime availability, GPU VRAM, PyTorch device capabilities, and verify OpenCV headless rendering.

6. **Configuration, Logging & Engineering Conventions (Step 1.9):**
   - Authored `configs/default.yaml` defining model parameters (weights, confidence threshold, image size), camera stream settings, and logging defaults.
   - Created `cv/config.py` with dynamic dot-notation attribute access and type safety.
   - Implemented `cv/logger.py` featuring thread-safe multi-handler logging (colored console output + persistent file stream in `outputs/nidar_cv.log`) with microsecond UTC ISO 8601 timestamps.
   - Defined project-wide standards in `docs/conventions.md`: top-left pixel coordinates, unnormalized integer `xyxy` bounding boxes, FRD body frame, and UTC ISO 8601 timestamps.

7. **Documentation & Diary System (Step 1.10):**
   - Authored the top-level `README.md` covering repo architecture, zero-to-hero installation, detector usage, and contribution guidelines.
   - Standardized the engineering diary template (`docs/diary/TEMPLATE.md`) to enforce continuous logging throughout the sprint.

8. **Code Quality & Testing Setup (Step 1.11):**
   - Configured `pyproject.toml` with `ruff` for all-in-one linting and formatting, replacing legacy Black/Flake8/isort with a single high-performance tool.
   - Configured `pytest` and authored automated unit tests in `tests/test_cv/test_config.py` verifying config loading, default values, error handling, and `ConfigDict` dictionary serialization.
   - Ran `ruff check --fix .` and `ruff format .`, standardizing the entire repository.

9. **Baseline Survivor Detector Integration (Step 1.12):**
   - Placed the baseline detector in `cv/detectors/detect_survivors.py`.
   - Wired the detector to read default parameters (`model.name`, `confidence_threshold`, `image_size`, `device`, `results_dir`) dynamically from `configs/default.yaml`.
   - Integrated the centralized logger (`cv.logger`) for structured console and file outputs.
   - Enforced ISO 8601 UTC timestamping (`YYYY-MM-DDTHH:MM:SS.mmmZ`) in accordance with `docs/conventions.md`.
   - Placed pre-downloaded offline YOLOv8n weights into `models/yolov8n.pt`.
   - Successfully executed inference on sample corridor imagery (`data/samples/survivor_hallway.jpg`), emitting structured `detections.jsonl` (Detection Spec v0.1), tabular `detections.csv`, and visual annotated media.

---

## 3. What I Measured

Quantitative metrics recorded during Day 1 setup:

| Component / Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Host Operating System** | Windows 11 Home (64-bit) | Workstation dev environment |
| **Host Python Interpreter** | Python 3.12.10 | Selected for library stability |
| **GPU Hardware** | NVIDIA GeForce RTX 3050 Laptop GPU | 6.0 GB Dedicated VRAM |
| **CUDA Driver Version** | Driver 592.82 / CUDA 12.4 | Hardware compute capability 8.6 |
| **cuDNN Version** | 90100 (cuDNN 9.1) | Accelerated convolution kernels |
| **PyTorch Version** | 2.6.0+cu124 | GPU-enabled wheel |
| **Torchvision Version** | 0.21.0+cu124 | GPU-enabled vision transforms |
| **OpenCV Version** | 4.11.0 | Core image processing |
| **Ultralytics Version** | 8.4.172 | YOLO detection suite |
| **NumPy Version** | 1.26.4 | Pinned `<2.0.0` for C-API binary compatibility |
| **PyTorch Wheel Download** | 2.53 GB | Download duration: ~28m over Wi-Fi (~1.65 MB/s) |
| **Diagnostic Test Image** | 640x480 px, 3 channels | Generated and saved in `< 15 ms` |
| **Pytest Execution Time** | 0.07 s | 5 test cases passed |
| **YOLOv8n Inference Latency** | 12.3 ms / frame (~81.3 FPS) | CUDA on RTX 3050 Laptop GPU |
| **Survivor Detection Confidence** | 0.7127 (71.3%) | Single person detected in hallway image |

---

## 4. What Failed & Root Cause Analysis

### Incident 1: Terminal Crash from Unicode Box Characters
- **Symptom:** When running `scripts/check_env.py`, the script crashed immediately with:
  `UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-31: character maps to <undefined>`.
- **Root Cause:** Windows PowerShell and CMD default to the `cp1252` legacy code page. When Python attempts to write UTF-8 box-drawing characters (`═`, `─`, `✓`, `✗`, `⚠`, `—`) to `sys.stdout`, the Windows console buffer rejects them.
- **Resolution:** Replaced all decorative box-drawing glyphs with universal ASCII characters (`=`, `-`, `[PASS]`, `[FAIL]`, `[WARN]`, `---`). Added an explicit UTF-8 safety check. The script now runs reliably across all Windows, Linux, and Jetson shells.

### Incident 2: Python Version Ambiguity with Windows Launcher
- **Symptom:** Invoking `py` invoked Python 3.14 pre-release rather than the stable Python 3.12 required by PyTorch and OpenCV.
- **Root Cause:** The Windows Python Launcher (`py.exe`) defaults to the highest numbered version installed on the system unless specified.
- **Resolution:** Explicitly instantiated the virtual environment using `python -m venv .venv` targeting the exact Python 3.12 executable on PATH. Once inside the active `.venv`, running `python` consistently maps to Python 3.12.10.

### Incident 3: NumPy 2.x Incompatibility Threat
- **Symptom:** Installing modern machine learning libraries can pull NumPy 2.x, which breaks pre-compiled C-extensions in certain versions of OpenCV and PyTorch.
- **Root Cause:** NumPy 2.0 introduced ABI breaking changes for C-extension modules compiled against NumPy 1.x.
- **Resolution:** Proactively pinned `numpy>=1.24.0,<2.0.0` in `requirements/requirements-base.txt`. The installed environment settled cleanly on `numpy==1.26.4`.

### Incident 4: GitHub Releases Download Connection Reset
- **Symptom:** Automatic downloading of `yolov8n.pt` through Ultralytics crashed with `ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host`.
- **Root Cause:** GitHub release redirect domains (`objects.githubusercontent.com`) were intermittently reset by network socket handling.
- **Resolution:** Downloaded the official weights directly from Hugging Face (`https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt`) and saved them locally to `models/yolov8n.pt` (6.23 MB). Updated `detect_survivors.py` and `configs/default.yaml` to prefer local weights in `models/` first.

### Incident 5: Package Import Collision with `-m cv.detectors.detect_survivors`
- **Symptom:** `RuntimeWarning: 'cv.detectors.detect_survivors' found in sys.modules after import of package 'cv.detectors'`.
- **Root Cause:** `cv/detectors/__init__.py` eagerly imported `main as run_detector` from `detect_survivors.py` while Python's `runpy` module was preparing to run it as a `__main__` entry point.
- **Resolution:** Cleaned `cv/detectors/__init__.py` to avoid circular eager imports.

---

## 5. Key Decisions Made

- **ADR 001 — Core Subsystem Tooling:** Python 3.12 + PyTorch (CUDA 12.4) + Ultralytics YOLOv8 for CV; React + Vite + TypeScript for GCS.
- **ADR 002 — Lightweight Git PR Workflow:** Single `main` branch with short-lived feature branches (`feature/`, `fix/`). Squash & merge to maintain a bisectable linear history.
- **ADR 003 — Strict Exclusion of Large Files:** Zero tolerance for weights or datasets in Git history. All binary assets reside in `models/` or `datasets/` backed by a shared offline drive and SHA256 hashes.
- **Standardized Coordinate System:** Bounding boxes must use `[x_min, y_min, x_max, y_max]` (`xyxy`) pixel coordinates with origin at the top-left corner `(0, 0)`.
- **Strict UTC ISO 8601 Timestamps:** All log entries, message envelopes, and telemetry packets must use UTC with a trailing `Z` and microsecond precision.

---

## 6. Evidence & Artifacts

### 1. Diagnostic Output from `scripts/check_env.py`:
```text
============================================================
              NIDAR SYSTEM ENVIRONMENT CHECK                
============================================================
Host System: Windows-11-10.0.26100-SP0 (AMD64)
Timestamp  : 2026-10-03T16:35:12.418291Z

[1] Python Environment:
  [PASS] Python Version: 3.12.10 (Target: >= 3.10)
  [PASS] Executable: d:\NIDAR\.venv\Scripts\python.exe

[2] Core Dependencies:
  [PASS] numpy: 1.26.4
  [PASS] cv2 (OpenCV): 4.11.0
  [PASS] torch (PyTorch): 2.6.0+cu124
  [PASS] torchvision: 0.21.0+cu124
  [PASS] ultralytics: 8.4.172

[3] Hardware Acceleration & GPU Diagnostics:
  [PASS] CUDA Available: True
  [PASS] CUDA Device Count: 1
  [PASS] Current Device Index: 0
  [PASS] Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
  [PASS] Device Compute Capability: 8.6
  [PASS] Total Dedicated VRAM: 6.00 GB
  [PASS] PyTorch Backend: CUDA 12.4
  [PASS] cuDNN Enabled: True (v90100)

[4] Functional Tests:
  [PASS] OpenCV synthetic test image created: outputs/test_opencv_image.png

============================================================
[PASS] ALL CHECKS PASSED. Ready for NIDAR CV development!
============================================================
```

### 2. Live Detection Output on Hallway Test Image (`outputs/runs_nidar/`):
- **Terminal Execution Log:**
```text
2026-10-03T21:02:33.357Z | INFO     | nidar.detector | Loading YOLO model: models\yolov8n.pt on device: cuda:0...
2026-10-03T21:02:33.430Z | INFO     | nidar.detector | Processing input source: data/samples/survivor_hallway.jpg (confidence threshold: 0.5)
2026-10-03T21:02:39.125Z | INFO     | nidar.detector | frame    0 | survivor_hallway.jpg | survivors: 1 | inference: 12.3 ms
2026-10-03T21:02:39.126Z | INFO     | nidar.detector | Detection run complete.
2026-10-03T21:02:39.126Z | INFO     | nidar.detector | Frames processed : 1
2026-10-03T21:02:39.126Z | INFO     | nidar.detector | Total survivors  : 1
2026-10-03T21:02:39.127Z | INFO     | nidar.detector | Detections JSONL : D:\NIDAR\outputs\runs_nidar\detections.jsonl
2026-10-03T21:02:39.127Z | INFO     | nidar.detector | Detections CSV   : D:\NIDAR\outputs\runs_nidar\detections.csv
2026-10-03T21:02:39.127Z | INFO     | nidar.detector | Annotated media  : D:\NIDAR\outputs\runs_nidar\annotated
```
- **Generated Detection JSONL (Spec v0.1):**
```json
{"frame_id": 0, "source": "survivor_hallway.jpg", "video_time_s": null, "timestamp_utc": "2026-10-03T21:02:39.067Z", "frame_width": 1200, "frame_height": 896, "inference_ms": 12.3, "detections": [{"det_id": 0, "class_name": "person", "confidence": 0.7127, "bbox_xyxy": [557.7, 464.3, 837.3, 704.9], "bbox_center": [697.5, 584.6]}]}
```

### 3. Verified Files Created Today:
- Core docs: [`README.md`](../../README.md), [`docs/conventions.md`](../conventions.md)
- Decisions: [`001_tooling.md`](../decisions/001_tooling.md), [`002_git_workflow.md`](../decisions/002_git_workflow.md), [`003_large_files.md`](../decisions/003_large_files.md)
- Setups: [`python_verification.md`](../setup/python_verification.md), [`git_verification.md`](../setup/git_verification.md), [`environment_verification.md`](../setup/environment_verification.md)
- Scripts: `scripts/check_env.py`, `scripts/verify_environment.py`, `scripts/setup_repo_structure.ps1`
- Configuration & Code: `configs/default.yaml`, `pyproject.toml`, `cv/config.py`, `cv/logger.py`, `cv/detectors/detect_survivors.py`
- Test Suite: `tests/test_cv/test_config.py` (5 passing smoke tests)
- Diary System: `docs/diary/TEMPLATE.md`, `docs/diary/day1_foundation.md`

---

## 7. Next Steps & Tomorrow's Plan (Day 2)

Tomorrow initiates **Phase 2: Object Detection Fundamentals**:
1. Deep-dive into bounding box geometry, Intersection over Union (IoU), and non-maximum suppression (NMS) from scratch.
2. Study YOLO architectural principles: anchor boxes, single-pass feature extraction, detection heads, and confidence score thresholds.
3. Build standalone visual helper tools in `cv/utils/` to draw and verify bounding box transformations.
4. Prepare test frames and ground truth annotations for the baseline model evaluation.
