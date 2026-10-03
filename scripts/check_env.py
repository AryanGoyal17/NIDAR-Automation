"""
check_env.py — NIDAR AirMouse Environment Diagnostics

Prints system info, library versions, GPU status, and runs a small
OpenCV visual test (creates a test image with drawn elements and saves it).

Usage:
    .venv/Scripts/python.exe scripts/check_env.py

Exit codes:
    0 — All checks passed
    1 — One or more checks failed (see output for details)
"""

import platform
import sys
from pathlib import Path

# -- Constants ----------------------------------------------------------------
HEADER = "=" * 64
TEST_IMAGE_PATH = Path("outputs/env_check_test.png")
EXPECTED_LIBS = {
    "numpy": "numpy",
    "cv2": "OpenCV (opencv-python)",
    "torch": "PyTorch",
    "torchvision": "Torchvision",
    "ultralytics": "Ultralytics YOLO",
    "pydantic": "Pydantic",
    "yaml": "PyYAML",
    "matplotlib": "Matplotlib",
    "PIL": "Pillow",
    "pytest": "Pytest",
}

passed_all = True


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n---- {title} {'-' * (56 - len(title))}")


def check(label: str, value: str, ok: bool = True) -> None:
    """Print a single check result with a pass/fail indicator."""
    global passed_all
    status = "OK" if ok else "FAIL"
    if not ok:
        passed_all = False
    print(f"  [{status}] {label:<28s} : {value}")


# ==============================================================================
# SECTION 1: System Information
# ==============================================================================
print(HEADER)
print("       NIDAR AIRMOUSE - ENVIRONMENT DIAGNOSTICS")
print(HEADER)

section("System")
check("Operating System", f"{platform.system()} {platform.release()} ({platform.machine()})")
check("OS Version", platform.version())
check("Python Version", platform.python_version())
check("Python Executable", sys.executable)

in_venv = sys.prefix != sys.base_prefix
check(
    "Virtual Environment", "YES (Isolated)" if in_venv else "NO — WARNING: not in .venv!", in_venv
)

# ==============================================================================
# SECTION 2: Library Versions
# ==============================================================================
section("Library Versions")
for module_name, display_name in EXPECTED_LIBS.items():
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "installed (no __version__)")
        check(display_name, version)
    except ImportError:
        check(display_name, "NOT INSTALLED", ok=False)

# ==============================================================================
# SECTION 3: GPU / CUDA Status
# ==============================================================================
section("GPU / CUDA Acceleration")
try:
    import torch

    cuda_available = torch.cuda.is_available()
    check("CUDA Available", str(cuda_available), ok=cuda_available)

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        check("GPU Count", str(gpu_count))
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem_mb = torch.cuda.get_device_properties(i).total_memory / (1024**2)
            check(f"  GPU {i}", f"{name} ({mem_mb:.0f} MB)")

        # Quick GPU tensor round-trip test
        t = torch.randn(1, 3, 640, 640, device="cuda")
        result = t.sum().item()
        check("GPU Tensor Test", f"PASS (sum={result:.2f}, device={t.device})")
    else:
        print()
        print("  WARNING: GPU NOT DETECTED. Possible causes and fixes:")
        print("     1. NVIDIA drivers not installed or outdated.")
        print("        → Run: nvidia-smi   (should show your GPU)")
        print("        → Download latest drivers from nvidia.com/drivers")
        print("     2. PyTorch installed without CUDA support.")
        print("        → Reinstall with:")
        print(
            "           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )
        print("     3. No NVIDIA GPU in this machine (e.g. integrated Intel graphics only).")
        print("        → Detection will run on CPU (slower but functional).")
        print()

    check("PyTorch CUDA Version", str(getattr(torch.version, "cuda", None) or "N/A"))
    check(
        "cuDNN Version",
        str(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"),
    )
    check("cuDNN Enabled", str(torch.backends.cudnn.enabled))

except ImportError:
    check("PyTorch", "NOT INSTALLED — cannot check GPU", ok=False)

# ==============================================================================
# SECTION 4: OpenCV Visual Test
# ==============================================================================
section("OpenCV Visual Test")
try:
    import cv2
    import numpy as np

    # Create a 640x480 dark frame (simulating a drone camera feed)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a dark grey "corridor" background
    cv2.rectangle(frame, (50, 50), (590, 430), (40, 40, 40), -1)

    # Draw a simulated survivor bounding box (green rectangle)
    cv2.rectangle(frame, (200, 150), (400, 380), (0, 255, 0), 2)

    # Add label text above the bounding box
    cv2.putText(frame, "survivor 0.92", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add frame metadata (like a real detection pipeline would)
    cv2.putText(
        frame,
        "NIDAR AirMouse | Frame: 001 | FPS: --",
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    # Draw a small "drone position" crosshair
    cx, cy = 320, 240
    cv2.drawMarker(frame, (cx, cy), (0, 200, 255), cv2.MARKER_CROSS, 20, 2)

    # Ensure output directory exists
    TEST_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEST_IMAGE_PATH), frame)

    check("Create test frame", f"{frame.shape} (height, width, channels)")
    check("Draw bounding box", "Green rectangle at (200,150)-(400,380)")
    check("Draw text overlay", "'survivor 0.92' label")
    check("Draw crosshair", "Drone position marker at (320, 240)")
    check("Save test image", str(TEST_IMAGE_PATH))

    # Verify the saved file is readable
    reloaded = cv2.imread(str(TEST_IMAGE_PATH))
    if reloaded is not None and reloaded.shape == frame.shape:
        check("Reload saved image", f"PASS (shape matches: {reloaded.shape})")
    else:
        check("Reload saved image", "FAIL — saved image could not be reloaded", ok=False)

except Exception as e:
    check("OpenCV Test", f"FAILED: {e}", ok=False)

# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print(HEADER)
if passed_all:
    print("  ALL CHECKS PASSED — Environment is ready for NIDAR CV work.")
    print(f"  Test image saved to: {TEST_IMAGE_PATH.resolve()}")
else:
    print("  WARNING: SOME CHECKS FAILED - Review the output above.")
print(HEADER)

sys.exit(0 if passed_all else 1)
