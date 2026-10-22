# NIDAR — Human (Survivor) Detection

Python 3.12 (dev machine) | PyTorch 2.6 (CUDA 12.4) | Ultralytics 8.4 | Target: Jetson (variant to be confirmed)

Software for the **human (survivor) detection** subsystem of the NIDAR team's entry to the **AirMouse 2026–27** student robotics challenge. This repository contains only the detection work: models, training, evaluation, tracking and onboard deployment.

> **How to read this README.** Everything marked **[DONE]** exists in this repository and has been run. Everything marked **[PLANNED]** is a design that has not been implemented or measured yet. Numbers taken from papers are labelled as external and are not our results. Our own results go in Section 9 only after they have been measured.

---

## 0. Status at a glance

| Component | Status | Where |
|---|---|---|
| Repository structure, config loader, logger, environment checks | [DONE] | `configs/`, `cv/config.py`, `cv/logger.py`, `scripts/` |
| Conventions, decision records (001–005), engineering diary | [DONE] | `docs/` |
| Baseline detector (pretrained YOLOv8n, image / folder / video / webcam) | [DONE] | `cv/detectors/detect_survivors.py` |
| Model-size and hardware speed test (YOLOv8 n/s/m) | [PLANNED] | `scripts/` |
| Real test set + evaluation script (precision, recall, mAP, false survivors) | [PLANNED] | `data/`, `cv/eval/` |
| Fine-tuned YOLO (simulation data first, then real data) | [PLANNED] | `cv/train/` |
| Tracking + multi-frame confirmation | [PLANNED] | `cv/tracking/` |
| YOLO11 / YOLO26 comparison | [PLANNED] | `cv/detectors/` |
| RT-DETR detector | [PLANNED] | `cv/detectors/rtdetr_detector.py` |
| Onboard (Jetson) inference script | [PLANNED] | `deploy/jetson/` |

---

## 1. Overview and scope

This repository holds one component of the NIDAR team's AirMouse 2026–27 entry: **human (survivor) detection** from the drone's camera video.

**Context.** AirMouse requires a fully autonomous drone to work inside a GPS-denied indoor maze, find up to six survivors, and report them on a real-time 2D map. Navigation, mapping and tagging, and obstacle avoidance are built by other team members and are outside this repository.

**This repository covers:**

1. Detecting people (survivors) in camera frames, with confidence scores.
2. Tracking detections across frames and confirming each survivor only once.
3. Measuring detector accuracy and speed on the target hardware, and exporting the chosen model for onboard use.
4. Defining the detection output that other subsystems consume (Section 8).

It does **not** contain path planning, mapping, obstacle avoidance, flight control or a ground-station application.

Development follows the rule: **baseline → test → measure → find limitations → improve → test again.**

### Design constraints

- **Offline only.** No cloud APIs, downloads or external networks during a mission. Models, weights and schemas must run from local files.
- **Compute is limited.** Detection shares the onboard computer with other subsystems.
- **A missed survivor is worse than a false alarm,** so recall is the main metric, with false alarms controlled by tracking and confirmation.

---

## 2. Repository folder map

```
NIDAR-Automation/
├── configs/              # Configuration (YAML)
│   └── default.yaml      # Detector, camera, logging, and I/O settings
├── cv/                   # Computer-vision package
│   ├── config.py         # Config parser with dot-notation access
│   ├── logger.py         # Thread-safe console/file logger (UTC ISO 8601)
│   ├── detectors/        # Detector wrappers (YOLO now; RT-DETR planned)
│   ├── eval/             # [PLANNED] Evaluation scripts and metrics
│   ├── tracking/         # [PLANNED] Tracker + confirmation logic
│   ├── train/            # [PLANNED] Fine-tuning scripts
│   └── utils/            # Image, bounding-box and geometry helpers
├── deploy/
│   └── jetson/           # [PLANNED] Onboard inference (TensorRT), old-Python compatible
├── interfaces/schemas/   # Versioned detection-message schema
├── docs/                 # Documentation
│   ├── conventions.md    # Frames, units, naming, style
│   ├── decisions/        # Architecture decision records (001, 002, 003, ...)
│   ├── diary/            # Daily engineering log and template
│   └── setup/            # Hardware and software verification records
├── scripts/              # Environment checks and utilities
├── requirements/         # Modular requirements (base, cv)
├── tests/                # test_cv/ and test_interfaces/
├── data/                 # Local test media and datasets (gitignored, offline)
├── models/               # Weights and engines (.pt, .onnx, .engine; gitignored)
├── outputs/              # Logs, annotated runs, detection outputs
└── notebooks/            # Exploration notebooks
```

---

## 3. Hardware requirements

### 3.1 Platforms

| Platform | OS / software | Python | Role |
|---|---|---|---|
| Development laptop (NVIDIA RTX 3050 Laptop GPU) | Windows 11 or Linux, CUDA 12.4 | 3.12 | Development, small-model fine-tuning, evaluation |
| Kaggle / Colab GPU (e.g. 16 GB T4) | Hosted | Hosted default | Larger training runs (e.g. RT-DETR) |
| Jetson Orin Nano (4 GB or 8 GB) | JetPack 6.x (Ubuntu 22.04) | System default (about 3.10; verify on board) | Onboard inference with TensorRT |
| Original Jetson Nano (2 GB or 4 GB) | JetPack 4.6.x (Ubuntu 18.04) | 3.6 | Onboard inference with TensorRT only; no Ultralytics |

**Open item: which Jetson Nano the team has is not yet confirmed.** Check with:

```bash
cat /proc/device-tree/model
head -n 1 /etc/nv_tegra_release     # R32 = JetPack 4 (original Nano), R36 = JetPack 6 (Orin Nano)
free -h                             # about 4 GB or 8 GB of memory
```

### 3.2 What each model needs

| Model | Training | Onboard inference |
|---|---|---|
| YOLOv8n / YOLO11n | A 4 GB GPU is likely enough at a small batch (verify) | Original Nano (TensorRT FP16, reduced input size) and Orin Nano |
| YOLOv8s / YOLO11s | Same, with a smaller batch | Orin Nano; original Nano is likely too slow (measure) |
| RT-DETR-L / X (Ultralytics) | 16 GB GPU recommended (Kaggle/Colab); a 4 GB laptop GPU is likely insufficient (verify) | Orin Nano 8 GB only; not realistic on the original Nano |
| RT-DETR-R18 (official repository) | 16 GB GPU recommended | Orin Nano 4 GB or 8 GB (measure) |

External reference (not our measurement): on a Jetson Orin Nano with TensorRT at FP16, one published benchmark reports about 81 FPS for RT-DETR R18 and about 40 FPS for R50, model-only (no pre-processing, tracking or other tasks). The memory size of that board was not stated.

All onboard numbers must be re-measured on the team's actual board, with the full pipeline running.

### 3.3 Other requirements

- **Camera:** RGB, at least 640×480 at 15–30 FPS (proposal; final camera model and resolution to be confirmed).
- **Thermal sensor:** optional, model not yet decided. Useful only if survivors give off heat.
- **Power:** Jetson Orin Nano modules offer power modes from 7 W up to 25 W. The electrical design must supply the chosen mode.
- **Software offline:** all weights and dependencies are installed before flight.

---

## 4. Environment setup (development laptop)

Prerequisites: Python 3.12.x, Git 2.40+, and optionally an NVIDIA GPU with a CUDA 12.4-compatible driver.

1. **Clone the repository**

```bash
git clone https://github.com/AryanGoyal17/NIDAR-Automation.git
cd NIDAR-Automation
```

2. **Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install PyTorch (GPU workstation)**

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

CPU only: `pip install torch torchvision`.

4. **Install the remaining dependencies**

```bash
pip install -r requirements.txt
```

5. **Download pretrained model weights (offline setup)**

```bash
python scripts/download_weights.py
```

6. **Run the health checks**

```bash
python scripts/check_env.py
python scripts/verify_environment.py
python scripts/check_sim.py
```

Expected: Python 3.10 or newer, OpenCV and Ultralytics import, CUDA device is detected, `outputs/env_check_test.png` is created, and PyBullet renders `outputs/sim_check.png`.

> The setup above is for the development laptop. The onboard computer uses different software; see Section 3.1 and `deploy/jetson/` (planned).

---

## 5. Running the detector

### 5.1 Configuration

`configs/default.yaml`:

```yaml
model:
  name: "models/yolov8n.pt"
  confidence_threshold: 0.50
  image_size: 640
  device: "cuda:0"
  target_classes: [0]

input:
  video_path: "data/test_videos/sample_flight.mp4"
  images_dir: "data/samples/"

output:
  results_dir: "outputs/"
  models_dir: "models/"
  detection_log: "outputs/detections.jsonl"

logging:
  level: "INFO"
  file: "outputs/nidar_cv.log"
```

### 5.2 Baseline detector [DONE]

```bash
# Single image (defaults from the config)
python -m cv.detectors.detect_survivors --source data/samples/survivor_hallway.jpg

# Folder of images with overrides
python -m cv.detectors.detect_survivors --source data/samples/ --conf 0.40 --device cuda:0

# Live webcam with a preview window
python -m cv.detectors.detect_survivors --source 0 --show
```

Output is appended to `outputs/runs_nidar/detections.jsonl` and `outputs/runs_nidar/detections.csv`. Annotated frames are saved in `outputs/runs_nidar/annotated/`.

### 5.3 Planned commands [PLANNED]

```bash
# Select the detector family in the config instead of changing code
python -m cv.detectors.detect_survivors --source clip.mp4 --detector rtdetr

# Evaluate a detector on the labelled real test set
python -m cv.eval.evaluate --detector yolo --weights models/best.pt --data data/test_real

# Fine-tune
python -m cv.train.finetune --detector yolo --data data/train.yaml
```

These are design targets. They do not exist until their status changes to [DONE].

---

## 6. Algorithm specification

This section defines, precisely, what the detection pipeline computes. Notation: an image has height H and width W; a box is **b = (x1, y1, x2, y2)** in pixels with the origin at the top-left, x to the right and y downward.

### 6.1 Problem definition

**Input:** a colour frame I ∈ {0,…,255}^(H×W×3) with capture time t.

**Output:** a finite set of detections D(I) = {(b_i, s_i)}, where b_i is a box and s_i ∈ [0, 1] is a confidence score for the class "person". Frames are then linked into tracks, and each track is either a *candidate* or a *confirmed survivor* (Section 6.6).

### 6.2 Pre-processing (letterbox)

For a model input size S (default 640):

- scale r = min(S / H, S / W)
- resized size (⌊rW⌋, ⌊rH⌋), padded symmetrically to S×S with a constant grey value
- pixel values scaled to [0, 1]: x = I / 255

Boxes predicted in the S×S frame are mapped back to the original image by undoing the padding and dividing by r.

### 6.3 Baseline: YOLO (CNN, NMS-based)

A convolutional network predicts, at three scales (strides 8, 16, 32), candidate boxes with class scores. Candidates with score below the threshold τ are discarded and the remainder are reduced by **non-maximum suppression (NMS)**:

1. Sort candidates by score.
2. Keep the highest-scoring box B.
3. Remove every other box B' with IoU(B, B') > ν.
4. Repeat until none remain.

where

IoU(A, B) = |A ∩ B| / |A ∪ B|.

### 6.4 RT-DETR (hybrid CNN–transformer, NMS-free) [PLANNED]

RT-DETR (Zhao et al., CVPR 2024) combines a CNN backbone with a transformer encoder and decoder:

1. **Backbone.** A CNN produces feature maps S3, S4, S5 at strides 8, 16, 32.
2. **Intra-scale interaction (AIFI).** One transformer encoder layer is applied *only* to the deepest map S5. At S = 640 this map has (640/32)² = 400 tokens, so self-attention is cheap there. Self-attention is

   Attn(Q, K, V) = softmax(Q Kᵀ / √d_k) V,

   whose cost grows with the square of the token count. This is why attention is applied at the deepest level only.
3. **Cross-scale fusion.** A CNN-based module fuses S3, S4 and the refined S5 into multi-scale encoder features.
4. **Query selection.** The top-K encoder tokens (K = 300 in the standard configuration), ranked by their maximum class score, initialise the decoder queries.
5. **Decoder.** The queries are refined over L decoder layers; each query i outputs class logits z_i and a box b̂_i.
6. **Scores.** Per-class scores use a sigmoid: p_{i,c} = σ(z_{i,c}), and the detection score is s_i = max_c p_{i,c}. For a single "person" class, s_i = p_i.

**Inference rule (no NMS):**

D = { (b̂_i, s_i) : s_i ≥ τ }.

Because each ground-truth object is matched to exactly one query during training, duplicates are suppressed by the training objective, and no NMS step is needed.

### 6.5 Training objective (RT-DETR and fine-tuning)

Let the ground-truth set be {y_j = (c_j, b_j)} padded with "no object" to K entries.

**Matching.** Find the one-to-one assignment σ̂ that minimises the total matching cost (Hungarian algorithm):

σ̂ = argmin over σ of Σ_i C(y_i, ŷ_σ(i)), with

C = λ_cls · C_cls + λ_L1 · ‖b_i − b̂_σ(i)‖₁ + λ_giou · (1 − GIoU(b_i, b̂_σ(i))).

**Loss.**

L = λ_cls · L_cls + λ_L1 · Σ ‖b_i − b̂_σ̂(i)‖₁ + λ_giou · Σ (1 − GIoU(b_i, b̂_σ̂(i))).

**Generalised IoU.**

GIoU(A, B) = IoU(A, B) − |C_AB \ (A ∪ B)| / |C_AB|,

where C_AB is the smallest box enclosing both A and B. L_cls is an IoU-aware classification loss. The values of λ_cls, λ_L1 and λ_giou, and the exact classification loss, must be copied from the training code that is actually used (in the Ultralytics implementation, see its RT-DETR loss module; the path depends on the version) and from the [RT-DETR paper](https://arxiv.org/abs/2304.08069). Do not quote values here until verified.

**Fine-tuning protocol [PLANNED].** Start from COCO-pretrained weights. Stage 1: simulation images (domain-randomised textures, lighting and backgrounds). Stage 2: a small set of real arena-style images. Hold out a real test set that is **never** used for training, and split by clip, not by frame, so near-identical frames never fall on both sides.

### 6.6 Tracking and survivor confirmation [PLANNED]

A single detection is not a survivor.

**Tracking.** Each track keeps a state x = [c_x, c_y, w, h, ċ_x, ċ_y, ẇ, ḣ] estimated by a constant-velocity Kalman filter. Each frame, detections are assigned to tracks by minimising a cost 1 − IoU(track prediction, detection) with the Hungarian algorithm (ByteTrack-style association, which also uses low-score detections in a second pass).

**Confirmation rule.** Let d_k = 1 if the track was matched to a detection in frame k. A track is **confirmed** at frame t when

Σ_{k = t−M+1}^{t} d_k ≥ N.

**Choosing N and M.** If a false alarm occurs independently in each frame with probability q, the chance that a false track is confirmed is

P_fa = Σ_{j=N}^{M} C(M, j) q^j (1 − q)^(M−j).

If a real survivor is detected independently in each frame with probability r, the chance it is confirmed is

P_conf = Σ_{j=N}^{M} C(M, j) r^j (1 − r)^(M−j).

Worked example: N = 3, M = 5, q = 0.05, r = 0.8 gives P_fa ≈ 0.0012 and P_conf ≈ 0.942. These assume independent frames, which is optimistic for real video, so choose N and M from measured values of q and r on the real test set.

**One survivor, one tag.** Track IDs can switch (for example when a survivor is briefly hidden), so the track ID alone cannot guarantee that a survivor is counted once. The detection output therefore exposes the track ID and timestamps, and the mapping subsystem, which knows positions, merges confirmed tracks that fall within a radius ρ of an existing survivor. Within this repository, track-switch behaviour is measured as the *double-count* metric (Section 6.8).

### 6.7 What other subsystems need from detection

Placing a survivor on the map is done by the mapping and tagging subsystem, not here. For every confirmed survivor, detection provides the box (including its bottom-centre pixel (u, v)), the confidence, the track ID and the **capture timestamp** of the frame. The mapping subsystem combines these with the drone's pose at that timestamp and the camera calibration. For this to work, this repository must supply accurate capture timestamps and record the camera intrinsics (f_x, f_y, c_x, c_y) from calibration in `configs/`. The calibration procedure is still to be defined.

### 6.8 Metrics

A prediction is a **true positive (TP)** if IoU with an unmatched ground-truth box is at least 0.5, matching in order of decreasing score; unmatched predictions are **false positives (FP)** and unmatched ground-truth boxes are **false negatives (FN)**.

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 · Precision · Recall / (Precision + Recall)
- AP = ∫₀¹ p(r) dr (area under the precision–recall curve); mAP@0.5 uses IoU 0.5.

**Mission-level (survivor) metrics**, measured over complete test runs:

- Survivor recall = (ground-truth survivors matched by a confirmed track) / (ground-truth survivors)
- False survivors per run = confirmed tracks that match no ground-truth survivor
- Double counts per run = ground-truth survivors matched by more than one confirmed track
- Detection latency = time from first appearance in view to confirmation

Speed: milliseconds per frame for detection alone, and for the whole pipeline, measured **on the target board**.

---

## 7. Model ladder and evaluation protocol

Each model is a candidate. The final system is the best row in the comparison table, plus tracking and confirmation, exported for the target board.

| Step | Model | Status |
|---|---|---|
| Baseline | COCO-pretrained YOLOv8, sizes n/s/m compared on speed, one size chosen | Planned (n runs today) |
| 1 | Fine-tuned baseline (simulation first, then real data) | Planned |
| 2 | Tracking + multi-frame confirmation added | Planned |
| 3 | YOLO11 and YOLO26, fine-tuned, with tracking | Planned |
| 4 | RT-DETR, fine-tuned, only if the hardware allows the required frame rate | Planned |

**Fairness rules for every comparison:**

- Same training data, same size class (n or s), same image size and epochs.
- Same tracker and confirmation logic.
- Same **real** test set; simulation images are never in it.
- Speed measured on the same target board, with the same pipeline.

**Required frame rate:** to be agreed with the navigation and flight-control owners (TBD).

---

## 8. Detection output interface (v0.1 draft)

Each processed frame produces one JSON object (one line in `detections.jsonl`). The authoritative conventions are in `docs/conventions.md` (pixel origin top-left, `xyxy` boxes, UTC ISO 8601 timestamps).

```json
{
  "frame_id": 12,
  "source": "corridor.mp4",
  "video_time_s": 0.48,
  "timestamp_utc": "2026-10-20T10:15:00.123Z",
  "frame_width": 1280,
  "frame_height": 720,
  "inference_ms": 18.4,
  "detections": [
    {
      "det_id": 0,
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox_xyxy": [412.0, 180.0, 530.0, 470.0],
      "bbox_center": [471.0, 325.0]
    }
  ]
}
```

Planned additions, to be agreed with the subsystems that consume this output: a **capture timestamp** (when the camera captured the frame, as opposed to when processing finished), a track ID, and a per-survivor status (candidate or confirmed, with first-seen and last-seen time and confidence). Converting a survivor to a map position is done by the mapping subsystem.

---

## 9. Results

> **Not measured yet.** The tables below stay empty until real measurements exist. No numbers from papers belong here.

### 9.1 Frame-level (real test set)

| Model | Precision | Recall | mAP@0.5 | ms/frame (target board) |
|---|---|---|---|---|
| YOLOv8n (pretrained) | TBD | TBD | TBD | TBD |
| YOLOv8s (pretrained) | TBD | TBD | TBD | TBD |
| YOLOv8m (pretrained) | TBD | TBD | TBD | TBD |
| Fine-tuned YOLO | TBD | TBD | TBD | TBD |
| YOLO11 / YOLO26 (fine-tuned) | TBD | TBD | TBD | TBD |
| RT-DETR (fine-tuned) | TBD | TBD | TBD | TBD |

### 9.2 Survivor-level (with tracking and confirmation)

| Model | Survivor recall | False survivors / run | Double counts / run | Latency to confirm |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

### 9.3 Failure-case log

Documented in `docs/failure_cases.md` (conditions, failure type, suspected cause, fix tried).

---

## 10. Safety behaviour and limitations

**Detection-related fallbacks [PLANNED]:**

| Condition | Behaviour |
|---|---|
| Detection uncertain (score between thresholds, or camera and thermal disagree) | Report as *candidate* only and request more frames; the navigation subsystem decides whether to slow down or hold |
| Detector slow or stalled | Publish a stale or degraded status with a timestamp; the flight controller must keep flying safely without it |
| Camera frames missing, corrupt, or very dark or blurred | Publish a degraded-image status and lower the confidence of any detections; never publish old results as current |

**The detection software is never part of the flight-control loop.** The flight controller must remain stable and execute failsafes even if the detector crashes or lags.

**Known limitations and open questions:**

- Survivor type (real people or dummies) is not confirmed. A COCO-pretrained "person" detector may miss dummies, and a thermal sensor sees nothing unless they are heated.
- Public datasets (SARD, VisDrone and others) are outdoor, high-altitude images. They help with posture variety, not with the indoor maze. Real arena-style footage is the most valuable data.
- Simulation-trained models carry a sim-to-real gap; real test data is required before any claim.
- Frames cut from the same video are nearly identical; always split data by clip.
- The camera model, onboard computer variant and required frame rate are still to be confirmed.

---

## 11. Engineering standards and conventions

Defined in `docs/conventions.md`:

- **Pixel coordinates:** origin (0, 0) top-left; x increases right, y increases downward.
- **Bounding boxes:** `[x_min, y_min, x_max, y_max]` (`xyxy`) in pixels.
- **Timestamps:** UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SS.mmmZ`, millisecond precision) in all packets, detections and logs.
- **Logging:** no raw `print()` in production code; use `cv/logger.py`:

```python
from cv.logger import get_logger

logger = get_logger(__name__)
logger.info("Detector initialised")
```

---

## 12. How to contribute

1. Read the decision records in `docs/decisions/` before proposing architectural changes: `001_tooling.md`, `002_git_workflow.md`, `003_large_files.md`, `004_model_choice.md`, `005_data_strategy.md`.
2. Branch from `main` (for example `feature/cv-rtdetr-detector`, `fix/camera-stream-buffer`, `docs/update-readme`).
3. Keep commits atomic, for example `feat(cv): add rtdetr detector wrapper` or `fix(logger): create missing output directory`.
4. **Never commit large files** (`.pt`, `.onnx`, `.engine`, `.mp4`, big CSVs or datasets). Keep them in `models/` and `data/` (gitignored) and share them through the team's storage drive.
5. Update `docs/diary/` after every working session.

---

## 13. Engineering diary

Documentation happens continuously, not at the end. Entries live in `docs/diary/`, using `docs/diary/TEMPLATE.md`.

- Day 1: Environment and foundations: tooling, virtual environments, PyTorch CUDA verification, config engine, logging conventions.

---

## 14. Licences and repository status

- **Ultralytics** (YOLO, RT-DETR wrappers) is licensed AGPL-3.0. Check the licence terms before any distribution of this code.
- **Datasets** (SARD, VisDrone and others) each have their own terms; check them before use in a submission.
- **Repository visibility:** the repository is currently public. The team must decide whether it stays public, and then set the licence text here accordingly. (TBD)

---

## 15. References

- Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), CVPR 2024: https://arxiv.org/abs/2304.08069
- Carion et al., *End-to-End Object Detection with Transformers* (DETR), 2020: https://arxiv.org/abs/2005.12872
- Rezatofighi et al., *Generalized Intersection over Union*, CVPR 2019: https://arxiv.org/abs/1902.09630
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, 2022: https://arxiv.org/abs/2110.06864
- Sambolek and Ivasic-Kos, *Person Detection and Geolocation Estimation in UAV Aerial Images: An Experimental Approach*, ICPRAM 2024: https://www.scitepress.org/Papers/2024/124116/124116.pdf
- Sambolek and Ivasic-Kos, *Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors*, IEEE Access, 2021: https://doi.org/10.1109/ACCESS.2021.3063681

---

Developed by Aryan for the NIDAR team's AirMouse 2026–27 entry.