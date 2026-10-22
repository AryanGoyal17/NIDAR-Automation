# Decision 001 — CV Tooling

| Field        | Value                    |
|-------------|--------------------------|
| **Date**    | 2026-10-03               |
| **Author**  | Aryan (AI/CV Lead)       |
| **Status**  | Accepted                 |
| **Scope**   | CV pipeline              |

## Context

The NIDAR AirMouse project requires a fully offline CV pipeline (survivor detection).
Code developed on Windows 11 now must eventually deploy on an onboard companion
computer (Jetson / Raspberry Pi). Choices must be beginner-friendly, offline-capable,
and integration-safe.

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
| **Model format (deployment)** | ONNX → TensorRT | — | ONNX is a portable neural-network format; TensorRT is NVIDIA's optimised inference runtime for Jetson. | None — this is the recommended Jetson path. | — |

## Packages NOT chosen (and why)

| Package | Why not |
|---|---|
| TensorFlow / Keras | Heavier install, weaker Jetson export story vs PyTorch + TensorRT. |
| Conda | Conflicts with ROS 2 environment sourcing. |
| Python 3.14 | Too new; PyTorch and other libraries lack compatible wheels. |
| Detectron2 | Powerful but complex API; overkill for a beginner starting out. |

## Open Questions

> **None of these block the foundation work.** All initial work runs on the
> Windows dev laptop against recorded images/video. These decisions matter at
> integration and deployment time.

- [ ] Confirm with system lead: which ROS 2 distro (Humble vs Jazzy)?
- [ ] Confirm with sensors/system lead: companion computer — Jetson (which module?) or Raspberry Pi (+ accelerator?)? This determines the model export format (TensorRT vs ONNX/CPU).

## References

- [Ultralytics export docs](https://docs.ultralytics.com/modes/export/)
- [NVIDIA Jetson PyTorch install](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- [ROS 2 Humble supported platforms](https://docs.ros.org/en/humble/Releases.html)

