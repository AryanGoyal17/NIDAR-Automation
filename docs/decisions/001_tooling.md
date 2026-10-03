# Decision 001 — CV + GCS Tooling

| Field        | Value                        |
|-------------|------------------------------|
| **Date**    | 2026-10-03                   |
| **Author**  | Aryan (AI/CV + GCS Lead)     |
| **Status**  | Accepted                     |
| **Scope**   | CV pipeline, GCS software    |

## Context

The NIDAR AirMouse project requires a fully offline CV pipeline (survivor detection)
and a Ground Control Station. Code developed on Windows 11 now must eventually deploy
on a Jetson companion computer running ROS 2 (Linux). Choices must be
beginner-friendly, offline-capable, and integration-safe.

## Decision Table

| Category | Choice | Version / Note | Why | Jetson / ROS 2 Risk | Mitigation |
|---|---|---|---|---|---|
| **Python** | CPython | 3.12.10 | Mature; supported by PyTorch, OpenCV, Ultralytics. Not bleeding-edge like 3.14. | Jetson + ROS 2 Humble uses 3.10; Jazzy uses 3.12. | CV code uses standard Python — no 3.12-only features. Runs on 3.10 unchanged. Ask system lead to confirm ROS 2 distro. |
| **Virtual env** | `venv` (built-in) | — | Simple, no extra install, no conflicts with ROS 2 env sourcing. | Conda can fight ROS 2's `setup.bash` sourcing. | Avoid conda for this project. |
| **Version control** | Git + GitHub | — | History, collaboration, Design Review visibility. | None. | — |
| **Editor** | Google Antigravity | — | Already in use; AI-assisted pair programming. | None. | — |
| **Array library** | NumPy | (pip resolves) | Foundation for all numerical/image data. Every other library depends on it. | None. | — |
| **CV toolkit** | OpenCV (`opencv-python`) | 4.x | Image/video I/O, drawing, resize, colour conversion. Standard API. | Jetson uses a CUDA-built OpenCV; pip version differs. | Python API is identical. Code doesn't change. |
| **Deep-learning framework** | PyTorch (CUDA build) | 2.x + CUDA | Runs neural-network inference on GPU (10-30× faster than CPU). Backend for Ultralytics. | Jetson needs special NVIDIA-provided PyTorch wheels. | For deployment, export model to TensorRT/ONNX — PyTorch not needed on Jetson at inference time. |
| **Detection library** | Ultralytics (YOLO) | 8.x | Beginner-friendly API; pre-trained "person" detector; nano/small models fit edge devices; export to TensorRT/ONNX for Jetson. | None if using export path. | Export to TensorRT before Jetson deployment. |
| **GCS frontend** | React + Vite | — | Aryan already knows React/Vite. GCS runs on ground laptop (not resource-constrained). | None — GCS is ground-side, not on drone. | — |
| **GCS backend** | Python (FastAPI or similar) + WebSocket | — | Bridges ROS 2 topics to the web frontend in real-time. | Must integrate with `rclpy` (ROS 2 Python client). | Design the backend with a clean ROS 2 adapter layer so the bridge is a thin wrapper. |
| **Model format (deployment)** | ONNX → TensorRT | — | ONNX is a portable neural-network format; TensorRT is NVIDIA's optimised inference runtime for Jetson. | None — this is the recommended Jetson path. | — |

## Packages NOT chosen (and why)

| Package | Why not |
|---|---|
| TensorFlow / Keras | Heavier install, weaker Jetson export story vs PyTorch + TensorRT. |
| Conda | Conflicts with ROS 2 environment sourcing. |
| Python 3.14 | Too new; PyTorch and other libraries lack compatible wheels. |
| Electron (for GCS) | Heavy runtime; React + Vite in a browser is lighter and sufficient. |
| Detectron2 | Powerful but complex API; overkill for a beginner starting out. |

## Open Questions

> **None of these block the 5-day foundation sprint.** All sprint work runs on the
> Windows dev laptop against recorded images/video. These decisions matter at
> integration and deployment time (weeks from now).

- [ ] Confirm with system lead: which ROS 2 distro (Humble vs Jazzy)?
- [ ] Confirm with sensors/system lead: companion computer — Jetson (which module?) or Raspberry Pi (+ accelerator?)? This determines the model export format (TensorRT vs ONNX/CPU).
- [ ] Confirm with system lead: GCS ↔ drone communication protocol (MAVLink? ROS 2 topics over DDS?).

## References

- [Ultralytics export docs](https://docs.ultralytics.com/modes/export/)
- [NVIDIA Jetson PyTorch install](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- [ROS 2 Humble supported platforms](https://docs.ros.org/en/humble/Releases.html)
