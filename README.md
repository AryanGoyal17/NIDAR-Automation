# NIDAR — Autonomous Drone CV & Ground Control Station

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20v8.4-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%28Jetson%29-gray)]()

Software repository for the **AI/Computer Vision (CV)** and **Ground Control Station (GCS)** subsystems of project **NIDAR**, competing in the **AirMouse 2026–27** student robotics challenge.

---

## 1. Project Overview

NIDAR is a fully autonomous unmanned aerial vehicle (UAV) engineered to navigate and inspect complex, GPS-denied indoor maze environments. Operating without external satellite signals or active internet connectivity during flight missions, the drone relies on onboard sensor fusion (2D LiDAR SLAM, optical flow, forward-facing camera) and an onboard companion computer (NVIDIA Jetson / high-performance SBC).

This repository houses two primary pillars:
1. **Computer Vision (CV) Pipeline (`cv/`):** Real-time onboard visual object detection, bounding box tracking, target classification, and spatial estimation of maze targets, gates, obstacles, and markers.
2. **Ground Control Station (GCS) (`gcs/`):** A high-reliability mission telemetry dashboard, mission state visualizer, and manual fail-safe command console communicating over telemetry radio/local Wi-Fi.

> **Offline-First Team Rule:** All mission-critical models, weights, schemas, and configurations must function strictly offline in isolated flight environments without relying on live cloud APIs or downloads.

---

## 2. Repository Folder Map

```text
NIDAR/
├── .github/              # CI workflows, PR and issue templates
├── configs/              # Centralized configuration files (YAML)
│   └── default.yaml      # Default detector, camera, logging, and I/O settings
├── cv/                   # Computer Vision package
│   ├── config.py         # Config parser with dot-notation attribute access
│   ├── logger.py         # Thread-safe dual console/file UTC ISO 8601 logger
│   ├── detectors/        # Object detection models and inference wrappers
│   └── utils/            # Image processing, bounding box, and geometry helpers
├── gcs/                  # Ground Control Station application & telemetry backend
├── interfaces/           # Shared IPC schemas, ROS 2 msg specs, JSON schemas
│   └── schemas/          # Versioned communication contracts
├── docs/                 # Continuous engineering documentation
│   ├── conventions.md    # Coordinate frames, units, naming, and style rules
│   ├── decisions/        # Architecture Decision Records (ADR 001, 002, 003)
│   ├── diary/            # Continuous daily engineering logs and template
│   └── setup/            # Hardware and software verification records
├── scripts/              # Environment bootstrap and diagnostic utilities
│   ├── check_env.py      # Hardware, CUDA, PyTorch, and OpenCV diagnostic
│   └── verify_environment.py # Automated dependency and version validator
├── requirements/         # Modular Python requirements specifications
│   ├── requirements-base.txt # Shared utilities, PyYAML, NumPy
│   ├── requirements-cv.txt   # OpenCV, PyTorch CUDA, Ultralytics YOLO
│   ├── requirements-gcs.txt  # GCS backend servers and networking
│   └── requirements.txt      # Umbrella requirements bundle
├── tests/                # Unit and integration test suites
│   ├── unit/             # Isolated algorithmic and component tests
│   └── integration/      # Pipeline and IPC loopback tests
├── datasets/             # Local test media (gitignored, kept offline)
├── models/               # Model checkpoints (.pt, .engine, gitignored)
├── outputs/              # Log files, annotated visual runs, detection outputs
└── notebooks/            # Exploration and exploratory data analysis notebooks
```

---

## 3. Environment Setup from Zero

Follow these steps to set up the development environment on Windows 11 or Linux.

### Prerequisites
- **Python:** 3.12.x (64-bit recommended)
- **Git:** 2.40+
- **NVIDIA GPU (Optional but recommended for dev workstation):** CUDA 12.4 compatible driver

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryanGoyal17/NIDAR-Automation.git
   cd NIDAR-Automation
   ```

2. **Create and activate a virtual environment:**
   *Windows (PowerShell):*
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *Linux / macOS:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install PyTorch with CUDA 12.4 Acceleration (GPU Workstations):**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   *(For CPU-only environments, run standard `pip install torch torchvision`).*

4. **Install Remaining Subsystem Dependencies:**
   ```bash
   pip install -r requirements/requirements.txt
   ```

5. **Run the Diagnostic Health Check:**
   ```bash
   python scripts/check_env.py
   ```
   Verify that:
   - Python is `>= 3.10` (Target: `3.12.10`)
   - OpenCV and Ultralytics are successfully imported
   - CUDA device is detected (e.g., `NVIDIA GeForce RTX 3050 Laptop GPU`)
   - The test image `outputs/test_opencv_image.png` is generated without errors.

---

## 4. Running the Vision Pipeline

The vision system is designed around modular, config-driven components.

### 1. Verify Configuration
Inspect and customize [`configs/default.yaml`](configs/default.yaml) for your run:
```yaml
model:
  weights: "models/yolov8n.pt"
  confidence_threshold: 0.25
  image_size: 640

camera:
  source: 0                 # 0 for webcam, or path to "datasets/sample.mp4"
  fps: 30

logging:
  level: "INFO"
  save_dir: "outputs"
```

### 2. Run the Diagnostic / Baseline Tests
```bash
# Verify environment and OpenCV drawing
python scripts/check_env.py

# Verify dependency versions and imports
python scripts/verify_environment.py
```

### 3. Running Detection (Phase 3+)
Once the baseline detection pipeline is loaded:
```bash
python -m cv.detectors.run_detector --config configs/default.yaml
```
Output logs will stream to stdout and append to [`outputs/nidar_cv.log`](outputs/nidar_cv.log). All annotated detection frames will save to [`outputs/`](outputs/).

---

## 5. Engineering Standards & Conventions

All contributors must adhere to the standardized project conventions defined in [`docs/conventions.md`](docs/conventions.md):
- **Pixel Coordinate System:** Origin `(0, 0)` is at the top-left corner. Horizontal X increases right; vertical Y increases downward.
- **Bounding Box Format:** Standardized as `[x_min, y_min, x_max, y_max]` (`xyxy`) in unnormalized integer pixels.
- **Timestamps:** Standardized as UTC ISO 8601 strings (`YYYY-MM-DDTHH:MM:SS.ffffffZ`) across all telemetry packets, detection metadata, and logs.
- **World Frame:** Drone body frame is `Forward-Right-Down` (FRD); global coordinate frame is `East-North-Up` (ENU) or `North-East-Down` (NED) pending flight controller alignment.
- **Logging Rule:** Never use raw `print()` statements in production code. Use the centralized logger from [`cv/logger.py`](cv/logger.py):
  ```python
  from cv.logger import get_logger
  logger = get_logger(__name__)
  logger.info("Detector initialized successfully")
  ```

---

## 6. How to Contribute

We follow a structured pull-request workflow tailored for a multi-subsystem engineering team:

1. **Check Existing Decisions:** Review [`docs/decisions/`](docs/decisions/) before proposing architectural changes:
   - [`001_tooling.md`](docs/decisions/001_tooling.md): Core language and framework selections.
   - [`002_git_workflow.md`](docs/decisions/002_git_workflow.md): Branching, commit conventions, and review rules.
   - [`003_large_files.md`](docs/decisions/003_large_files.md): Policy on model weights and datasets.
2. **Branch Naming:** Create a feature branch off `main`:
   ```bash
   git checkout -b feature/cv-yolo-detector
   # or: fix/camera-stream-buffer, docs/update-readme
   ```
3. **Keep Commits Atomic & Descriptive:**
   - `feat(cv): add yolo detection wrapper with conf threshold`
   - `fix(logger): handle missing output directory creation`
4. **Never Commit Large Files:** Never commit `.pt`, `.onnx`, `.engine`, `.mp4`, or large `.csv` datasets to git. Store weights in `models/` (gitignored) and link downloads via our shared storage drive.
5. **Continuous Documentation:** Update [`docs/diary/`](docs/diary/) for any major sprint milestone or new subsystem design.

---

## 7. Engineering Diary

In accordance with our core team principle: **"Documentation happens continuously, not at the end."**

Every working session is recorded in [`docs/diary/`](docs/diary/) using [`docs/diary/TEMPLATE.md`](docs/diary/TEMPLATE.md):
- [Day 1: Environment & Foundations](docs/diary/day1_foundation.md) — Tooling, virtual environments, PyTorch CUDA verification, config engine, and logging conventions.

---

## 8. License & Team

Developed by the **NIDAR Robotics Team** for the AirMouse 2026–27 Autonomous Drone Challenge.  
Internal repository — unauthorized redistribution prohibited.
