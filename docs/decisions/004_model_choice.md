# Decision 004 — Model Choice (YOLO11n, YOLOv8n, and Dropped Candidates)

| Field        | Value                                                          |
|-------------|----------------------------------------------------------------|
| **Date**    | 2026-10-22                                                     |
| **Author**  | Aryan (AI/CV Lead)                                             |
| **Status**  | Accepted                                                       |
| **Scope**   | Detection model architecture and onboard deployment candidates |

---

## 1. Context

The NIDAR AirMouse 2026–27 challenge requires detecting up to six human survivors inside a GPS-denied indoor maze from an autonomous drone. The compute budget is strictly constrained:
- **Development environment:** Windows 11 laptop, Python 3.12, PyTorch 2.6 (CUDA 12.4), Ultralytics 8.4, NVIDIA RTX 3050 Laptop GPU.
- **Candidate onboard computers (not finalised):**
  1. **Raspberry Pi 5 (8 or 16 GB):** CPU-only execution using NCNN export via Ultralytics.
  2. **Original Jetson Nano (4 GB):** 128-core Maxwell GPU, JetPack 4.6.x (Ubuntu 18.04, **Python 3.6**). Ultralytics cannot be installed onboard; deployment requires an ONNX export (opset 12, static shapes) compiled to a TensorRT FP16 engine on the Nano itself, driven by a standalone Python 3.6 inference script.
- **System constraint:** Detection is one subsystem sharing the board with mapping, planning, and obstacle avoidance. Detection must remain lightweight and must **never** run inside the flight-control loop. No cloud APIs or network connectivity exist during missions.

---

## 2. Decision Summary

| Candidate / Variant | Role / Status | Rationale |
|---|---|---|
| **YOLO11n** | **Primary candidate** | Lightweight nano architecture (~2.6M params). Suitable for real-time edge execution at input size 416×416. |
| **YOLOv8n** | **Baseline & safe fallback** | Proven baseline (~3.2M params); fully verified on dev machine; reliable fallback if YOLO11 export or latency underperforms on the board. |
| **YOLOv8s / YOLOv8m** | **Rejected** | Exceed onboard compute budget when shared with SLAM, navigation, and obstacle avoidance. |
| **RT-DETR** | **Dropped** | Transformer encoder/decoder computation is too heavy for CPU-only Pi 5 or 128-core Maxwell Jetson Nano to sustain required frame rates. |
| **Custom YOLO + ViT hybrid** | **Rejected** | High complexity and overhead without justification on edge compute. |

---

## 3. Architecture Details & Clarification on YOLO11 Attention

- **Input Resolution & Single Class:** Both YOLO11n and YOLOv8n will be trained on identical data, at `imgsz: 416` (revisitable if board latency allows), predicting a single class: `person` (class index `0`).
- **YOLO11 C2PSA Block vs. Vision Transformers:**
  - YOLO11 incorporates a lightweight **C2PSA (Cross Stage Partial with Pointwise Spatial Attention)** block in its deepest backbone stage.
  - **This is not a Vision Transformer (ViT).** It uses lightweight spatial attention over high-level feature maps, not multi-head patch-based token attention or transformer encoder-decoder layers.
  - It must **never** be documented or described as a "ViT hybrid".
  - Empirical differences between YOLO11n and YOLOv8n stem from multiple architectural adjustments (kernel arrangements, neck design, C2PSA) and **cannot be credited to attention alone**.

---

## 4. Improvement Ladder

The model development ladder progresses strictly by empirical comparison:
1. **Baseline:** Pretrained YOLOv8n (runs today for pipeline sanity).
2. **Fine-tuned Baseline:** YOLOv8n fine-tuned on arena simulation and composited training data.
3. **Tracking & Multi-Frame Confirmation:** Add multi-object tracking and M-of-N confirmation logic.
4. **YOLO11n:** Train on identical data and parameters, benchmark against YOLOv8n under identical tracker and confirmation logic.
5. **Final Selection:** The best measured candidate on the target board (considering recall, false survivors, double counts, and latency) is selected and exported for onboard deployment.

---

## 5. Consequences & Export Paths

- **If Raspberry Pi 5 is chosen:** Model exported to **NCNN** format via Ultralytics for CPU inference.
- **If Jetson Nano is chosen:** Model exported to **ONNX** (opset 12, static shapes), converted to a **TensorRT FP16 engine** on the physical Nano, executed via standalone Python 3.6 script in `deploy/jetson/`.
- **Fairness in evaluation:** Both models are evaluated on the exact same separate real test set (`data/test_real/`) and benchmarked on identical hardware.
