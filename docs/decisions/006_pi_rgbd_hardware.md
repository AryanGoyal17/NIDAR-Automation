# Decision 006 — Onboard Hardware: Raspberry Pi (CPU) and RGB-D Camera

| Field        | Value                                                              |
|-------------|--------------------------------------------------------------------|
| **Date**    | 2026-10-30                                                         |
| **Author**  | Aryan (AI/CV Lead)                                                 |
| **Status**  | Accepted                                                           |
| **Scope**   | Onboard computer, camera, export format, use of depth             |
| **Supersedes** | ADR 004 §1 (onboard candidates) and §5 (Jetson export path); the Jetson / TensorRT deployment notes in ADR 001 |

---

## 1. Context

- ADR 004 kept two onboard candidates open: a Raspberry Pi 5 (CPU only, NCNN) and the original Jetson Nano (JetPack 4.6, Python 3.6, ONNX → TensorRT). ADR 001 assumed a Jetson-style TensorRT deployment.
- The team has now settled the hardware. The onboard computer is a Raspberry Pi. The drone carries an RGB-D (colour + depth) camera.
- None of the camera or Pi specifications have been confirmed yet. No speed or accuracy has been measured on the Pi.

---

## 2. Decision

1. **Onboard computer: Raspberry Pi, CPU only.** Inference runs through Ultralytics with an **NCNN** export of the chosen model.
2. **The Jetson Nano is dropped.** There is no ONNX-to-TensorRT path, no `deploy/jetson/` and no Python 3.6 code constraint. Onboard code goes in `deploy/pi/` (planned).
3. **Camera: RGB-D.** Colour and depth must be aligned to each other and share a capture timestamp.
4. **The detector uses only the colour image.** Model choice is unchanged from ADR 004: YOLO11n primary, YOLOv8n fallback, single class `person`, input size 416, dropping to 320 if the Pi is too slow.
5. **Depth is used after detection.** For each box:
   - estimate the survivor's distance from the aligned depth inside the box;
   - compute a 3D point in the camera frame from the camera intrinsics;
   - reject boxes whose real-world size is implausible for a person.
6. **Handoff interface:** `SurvivorDetector(...).detect(frame_bgr, depth=None)` returns the box, confidence and class, plus `distance_m`, `point_cam_xyz_m` and `depth_valid`. Each result carries the frame's capture timestamp. Backends are `pytorch` (dev laptop) and `ncnn` (Pi). Details are in `CLAUDE.md` Section 8.

### 2.1 Specifications (none confirmed)

| Item | Value | Status |
|---|---|---|
| Pi model | Raspberry Pi 5 | ASSUMPTION |
| Pi memory | 8 GB | ASSUMPTION |
| Cooling | Active cooling | ASSUMPTION |
| Pi OS and Python version | TBD | TBD |
| Inference speed on the Pi (ms/frame) | Not measured | TBD |
| Required detection frame rate | Set with the controls team | TBD |
| RGB-D camera model | TBD | TBD |
| Colour resolution and frame rate | TBD (sim currently assumes 1280×720 at 30 FPS in `configs/arena.yaml`) | ASSUMPTION |
| Depth resolution and frame rate | TBD | TBD |
| Depth range (min, max) | TBD | TBD |
| Field of view | TBD (sim currently assumes 85° × 54°) | ASSUMPTION |
| Mounting height and tilt | TBD (sim currently assumes 0.8 m, 30° down) | ASSUMPTION |
| Camera intrinsics (fx, fy, cx, cy) | From calibration | TBD |
| Colour–depth alignment | Required; method depends on the camera | TBD |
| Person size limits for the plausibility check | TBD | TBD |

---

## 3. Consequences

- **Docs:** `CLAUDE.md` is updated with this decision. `AGENTS.md` and its Python 3.6 rule are removed. The README hardware section, the README interface section and `docs/conventions.md` still describe the old setup; their rewrites are planned.
- **Deployment:** `deploy/pi/`, an NCNN export script and a Pi benchmark script are planned. All speed numbers must be measured on the Pi with the full pipeline running.
- **Simulation:**
  - The simulator must provide metric depth and camera intrinsics, so the depth module can be tested against ground truth. Today `sim/camera.py` returns the raw, non-metric depth buffer (a known defect).
  - Depth noise and missing-pixel behaviour depend on the camera and are TBD.
- **Configuration:**
  - Depth camera settings, intrinsics, and depth post-processing settings (sampling inside the box, size limits) need new config keys. All of them are ASSUMPTION or TBD until the camera is known.
  - `configs/default.yaml` (`device: cuda:0`, imgsz 640) stays the baseline default. A separate CPU/NCNN profile is planned.
- **Dependencies:** a Pi requirements file is planned. The packages needed for NCNN inference on the Pi must be verified on the board.
- **Evaluation:** distance and 3D-point error can be measured on simulation ground truth. Real-world depth accuracy needs real RGB-D test data (TBD).
- **Risks (unquantified until measured):**
  - CPU-only inference speed on the Pi while it is shared with mapping, planning and obstacle avoidance.
  - Depth quality on survivors lying on the floor, at close range, or near the depth-range limits.

---

## 4. Supersedes

- **ADR 004 §1** "Candidate onboard computers": replaced by §2 above. The Jetson Nano option no longer applies.
- **ADR 004 §5** "Consequences & Export Paths": the Jetson ONNX / TensorRT / Python 3.6 path no longer applies. The NCNN path for the Pi remains. The fairness rules in that section still apply.
- **ADR 001:** the Jetson-related deployment notes (TensorRT, Jetson PyTorch wheels, Jetson OpenCV, the open item "Jetson or Raspberry Pi") no longer apply. The rest of ADR 001 stands.
- ADR 004's model choice (YOLO11n, YOLOv8n, dropped candidates, the C2PSA note) and the improvement ladder are **not** changed.
