# Decision 003 — Handling Datasets, Model Weights, and Large Artifacts

| Field       | Value                                      |
|------------|--------------------------------------------|
| **Date**   | 2026-10-03                                 |
| **Author** | Aryan (AI/CV + GCS Lead)                   |
| **Status** | Accepted                                   |
| **Scope**  | Team-wide data & weights management policy |

## 1. Context & The Problem with Big Files in Git

Git is designed for text and source code diffs. When binary files (weights like `yolov8n.pt`, MP4 flight recordings, or zipped datasets) are committed:
1. **Permanent History Bloat:** Git stores a complete copy of every binary revision. Even if you "delete" the file in a later commit, the binary remains in the `.git` folder forever.
2. **Slow Clones & Builds:** Every teammate, CI runner, or companion computer that runs `git clone` downloads every megabyte of historical weights.
3. **GitHub Limits:** GitHub rejects files over 100 MB and throttles repositories exceeding 2 GB.
4. **Competition Rulebook Violation Risk:** In NIDAR, the field environment is strictly **offline (no internet/cloud)**. Relying on an online download during mission launch will lead to mission failure.

---

## 2. Large Artifact Classification

| Category | Typical Sizes | Examples | Storage Location |
|---|---|---|---|
| **Code & Config** | < 1 MB | `.py`, `.json`, `.yaml`, `.md` | In Git repo |
| **Model Weights** | 6 MB – 150 MB | `yolov8n.pt`, `survivor_v1.onnx`, `model.engine` | External storage + local `models/` |
| **Test Datasets** | 100 MB – 5 GB | Low-light maze images, bounding-box annotations | Team Shared Drive + local `data/samples/` |
| **Flight Videos** | 500 MB – 10 GB | Recorded test flights, simulation ROS bags | Team Drive / External SSD + local `data/test_videos/` |
| **Run Outputs** | Variable | Inference logs, annotated video dumps | Local `outputs/` (ephemeral) |

---

## 3. Team Policy: How We Store & Share Large Files

### A. Development Phase (Online Collaboration)
1. **GitHub Releases for Verified Model Milestones:**
   - When a model passes benchmark testing (e.g. `survivor_detector_v1_baseline`), upload the `.pt` and `.onnx` files to a **GitHub Release** attached to the git tag.
   - Scripts can fetch this via a single pre-flight download command.
2. **Team Shared Cloud Storage (Google Drive / OneDrive):**
   - Maintained by the CV Research Engineer and System Lead.
   - Clear folder taxonomy:
     ```text
     [NIDAR_Storage]/
     ├── datasets/
     │   └── maze_survivors_v1_2026-10-03.zip
     ├── models/
     │   ├── yolov8n_baseline.pt
     │   └── checksums.txt
     └── flight_recordings/
         └── flight_run_01_corridor.mp4
     ```
3. **Strict Naming & Checksum Policy:**
   - Every released model or dataset zip file MUST include a SHA-256 hash in a `checksums.txt` file to guarantee integrity and prevent corrupt files from reaching the drone.

### B. Competition & Field Deployment Phase (100% Offline)
- **Local Cache on Drone Companion Computer:** Model files (`.engine` / `.onnx`) and camera calibration profiles must be baked into the drone companion computer's filesystem (`/opt/nidar/models/` or inside the project directory) **before** traveling to the arena.
- **Physical "Gold Copy" USB Drive:** The team must carry a physical USB 3.0 flash drive containing:
  - All verified model weights (`.pt`, `.onnx`, `.engine`)
  - Test calibration images
  - Full Python dependency wheels (for emergency offline reinstall)

---

## 4. Why Not Git LFS?

While Git LFS (Large File Storage) replaces large files with text pointers, student teams frequently exhaust GitHub's free LFS bandwidth/storage quota (1 GB storage, 2 GB monthly bandwidth). Shared Drive + GitHub Releases avoids unexpected billing lockout while remaining simple and reliable.
