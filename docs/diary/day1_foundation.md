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
- [x] Formulate project-wide conventions for coordinates, timestamps, frames, and code style ([`conventions.md`](../conventions.md)).
- [x] Author comprehensive repository documentation (`README.md`, `TEMPLATE.md`, and Day 1 log).

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

### 2. Verified Files Created Today:
- Core docs: [`README.md`](../../README.md), [`docs/conventions.md`](../conventions.md)
- Decisions: [`001_tooling.md`](../decisions/001_tooling.md), [`002_git_workflow.md`](../decisions/002_git_workflow.md), [`003_large_files.md`](../decisions/003_large_files.md)
- Setups: [`python_verification.md`](../setup/python_verification.md), [`git_verification.md`](../setup/git_verification.md), [`environment_verification.md`](../setup/environment_verification.md)
- Scripts: `scripts/check_env.py`, `scripts/verify_environment.py`, `scripts/setup_repo_structure.ps1`
- Configuration & Code: `configs/default.yaml`, `cv/config.py`, `cv/logger.py`
- Diary System: `docs/diary/TEMPLATE.md`, `docs/diary/day1_foundation.md`

---

## 7. Next Steps & Tomorrow's Plan (Day 2)

Tomorrow initiates **Phase 2: Object Detection Fundamentals**:
1. Deep-dive into bounding box geometry, Intersection over Union (IoU), and non-maximum suppression (NMS) from scratch.
2. Study YOLO architectural principles: anchor boxes, single-pass feature extraction, detection heads, and confidence score thresholds.
3. Build standalone visual helper tools in `cv/utils/` to draw and verify bounding box transformations.
4. Prepare test frames and ground truth annotations for the baseline model evaluation.
