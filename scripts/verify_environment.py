"""
verify_environment.py
Verifies Python version, virtual environment isolation, OpenCV, NumPy,
and PyTorch CUDA GPU acceleration for NIDAR CV pipeline.
"""

import sys


def check_environment():
    print("=" * 60)
    print("       NIDAR AIRMOUSE: CV ENVIRONMENT VERIFICATION       ")
    print("=" * 60)

    # 1. Python & Venv check
    in_venv = sys.prefix != sys.base_prefix
    print(f"[1] Python Executable : {sys.executable}")
    print(f"    Python Version    : {sys.version.split()[0]}")
    print(f"    In Virtual Env    : {'YES (Isolated)' if in_venv else 'NO (Global - Warning!)'}")
    assert in_venv, "Verification failed: not running inside .venv!"

    # 2. NumPy check
    try:
        import numpy as np

        print(f"[2] NumPy Version     : {np.__version__}")
    except ImportError as e:
        print(f"[2] NumPy Error       : {e}")

    # 3. OpenCV check
    try:
        import cv2

        print(f"[3] OpenCV Version    : {cv2.__version__}")
        # Test basic image matrix creation
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "NIDAR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"    OpenCV Operations : Working (test matrix shape {dummy_frame.shape})")
    except ImportError as e:
        print(f"[3] OpenCV Error      : {e}")

    # 4. PyTorch & CUDA GPU check
    try:
        import torch

        print(f"[4] PyTorch Version   : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"    CUDA Available    : {cuda_available}")
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"    Active GPU Device : {gpu_name}")
            print(f"    GPU Total Memory  : {gpu_mem:.0f} MB")
            # Quick tensor allocation on GPU
            tensor = torch.zeros((1, 3, 640, 640), device="cuda")
            print(f"    GPU Tensor Test   : SUCCESS ({tensor.device})")
        else:
            print("    WARNING: CUDA not detected. PyTorch will run on slow CPU!")
    except ImportError as e:
        print(f"[4] PyTorch Error     : {e}")

    # 5. Ultralytics YOLO check
    try:
        import ultralytics

        print(f"[5] Ultralytics YOLO  : Version {ultralytics.__version__}")
    except ImportError as e:
        print(f"[5] Ultralytics Error : {e}")

    print("=" * 60)
    print("Environment Verification Complete.")
    print("=" * 60)


if __name__ == "__main__":
    check_environment()
