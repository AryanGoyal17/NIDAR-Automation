# CV Environment Verification Report

**Date:** 2026-10-03  
**Machine:** Aryan's Windows 11 Laptop  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  

## Verification Output

```text
============================================================
       NIDAR AIRMOUSE: CV ENVIRONMENT VERIFICATION       
============================================================
[1] Python Executable : D:\NIDAR\.venv\Scripts\python.exe
    Python Version    : 3.12.10
    In Virtual Env    : YES (Isolated)
[2] NumPy Version     : 1.26.4
[3] OpenCV Version    : 4.11.0
    OpenCV Operations : Working (test matrix shape (480, 640, 3))
[4] PyTorch Version   : 2.6.0+cu124
    CUDA Available    : True
    Active GPU Device : NVIDIA GeForce RTX 3050 6GB Laptop GPU
    GPU Total Memory  : 6144 MB
    GPU Tensor Test   : SUCCESS (cuda:0)
[5] Ultralytics YOLO  : Version 8.4.172
============================================================
Environment Verification Complete.
============================================================
```

## Installed Package Summary

| Package | Version | Purpose |
|---|---|---|
| **Python** | 3.12.10 | Core runtime |
| **PyTorch** | 2.6.0+cu124 | Neural net execution with CUDA 12.4 acceleration |
| **Torchvision** | 0.21.0+cu124 | Vision data transforms and operators |
| **Ultralytics** | 8.4.172 | YOLOv8 survivor detection models |
| **OpenCV** | 4.11.0.86 | Image matrix manipulation & video streaming |
| **NumPy** | 1.26.4 | Array and matrix maths |
| **Pydantic** | 2.13.5 | Strict schema validation for detector output specs |
| **PyYAML** | 6.0.3 | Configuration loader |
| **Pytest** | 9.1.1 | Unit testing framework |
