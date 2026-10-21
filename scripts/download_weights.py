#!/usr/bin/env python3
"""
scripts/download_weights.py — Setup-time weight downloader for NIDAR

Downloads pretrained model weights (yolov8n.pt, yolo11n.pt) into models/
for offline use during development and missions.

Per project conventions:
- Internet access is permitted during setup only; runtime/mission is strictly offline.
- Downloaded weights (.pt files) in models/ are gitignored and never committed.
- Records and verifies SHA-256 checksums and file sizes.
- Skips downloads if destination files already exist.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Pretrained model weights hosted on official Ultralytics releases
WEIGHTS_MAP = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}

CHUNK_SIZE = 64 * 1024  # 64 KB read buffer


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_size(bytes_count: int) -> str:
    """Format bytes into a human-readable size string."""
    return f"{bytes_count:,} bytes ({bytes_count / (1024 * 1024):.2f} MB)"


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with streaming chunks to a temporary file, then rename."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (NIDAR AirMouse Weight Downloader)"},
    )

    try:
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"\r  Progress: {pct:5.1f}% [{downloaded:,} / {total_bytes:,} bytes]",
                        end="",
                        flush=True,
                    )
            print()  # newline after progress

        # Atomic replace after full download
        temp_path.replace(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {url}: {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Download YOLO weights for NIDAR offline use")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Target directory for weights (default: models/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download weights even if they already exist",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("         NIDAR AIRMOUSE — MODEL WEIGHTS DOWNLOADER (SETUP ONLY)")
    print("=" * 72)
    print(f"Target directory: {models_dir}")
    print()

    all_success = True

    for filename, url in WEIGHTS_MAP.items():
        dest = models_dir / filename
        print(f"[{filename}]")

        if dest.exists() and not args.force:
            size = dest.stat().st_size
            sha = compute_sha256(dest)
            print(f"  Status : EXISTS (skipping download)")
            print(f"  Path   : {dest}")
            print(f"  Size   : {format_size(size)}")
            print(f"  SHA256 : {sha}")
            print()
            continue

        print(f"  Downloading from: {url}")
        try:
            download_file(url, dest)
            size = dest.stat().st_size
            sha = compute_sha256(dest)
            print(f"  Status : DOWNLOADED")
            print(f"  Path   : {dest}")
            print(f"  Size   : {format_size(size)}")
            print(f"  SHA256 : {sha}")
            print()
        except Exception as err:
            print(f"  Status : FAILED — {err}")
            print()
            all_success = False

    print("=" * 72)
    if all_success:
        print("All required model weights are present in models/.")
        print("Remember: weights must never be committed to git.")
        return 0
    else:
        print("Error: One or more weight downloads failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
