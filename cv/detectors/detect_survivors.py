#!/usr/bin/env python3
"""
NIDAR AirMouse - Baseline survivor (person) detector
=====================================================

Runs a pretrained YOLO model on an image, a folder of images, a video file,
or a webcam, and writes the detections in a fixed format that the rest of the
system (survivor localisation, GCS) can consume.

By default, loads tuneable settings from configs/default.yaml. Command-line
arguments override config values.

INSTALL (once):
    pip install ultralytics opencv-python

RUN (examples):
    python -m cv.detectors.detect_survivors --source data/samples/survivor_hallway.jpg
    python -m cv.detectors.detect_survivors --source data/samples/
    python -m cv.detectors.detect_survivors --source corridor.mp4 --conf 0.35
    python -m cv.detectors.detect_survivors --source 0 --show          # webcam, press q to quit

OUTPUTS (inside --out, default "outputs/runs_nidar"):
    detections.jsonl   one JSON object per frame (the official detection spec)
    detections.csv     one row per detected person (easy to open in Excel)
    annotated/         annotated images, or annotated_video.mp4 for videos

DETECTION-OUTPUT SPEC v0.1 (one line of detections.jsonl):
{
  "frame_id": 12,                        # running frame counter, starts at 0
  "source": "corridor.mp4",              # file name (or "webcam0")
  "video_time_s": 0.48,                  # seconds into video (null for still images/webcam)
  "timestamp_utc": "2026-10-04T02:15:00.123Z", # UTC ISO 8601 timestamp
  "frame_width": 1280,
  "frame_height": 720,
  "inference_ms": 18.4,                  # model runtime for this frame
  "detections": [
    {
      "det_id": 0,                       # detection index in frame
      "class_name": "person",
      "confidence": 0.87,                # 0.0 - 1.0
      "bbox_xyxy": [x1, y1, x2, y2],     # pixels, top-left and bottom-right corners
      "bbox_center": [cx, cy]            # pixels, center coordinate
    }
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from cv.config import load_config
from cv.logger import get_logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PERSON_CLASS_ID = 0  # In COCO dataset, 0 = person


# --------------------------------------------------------------------------- #
# Frame Generator
# --------------------------------------------------------------------------- #
def iter_frames(
    source: str,
) -> Generator[tuple[int, float | None, cv2.typing.MatLike, str, float], None, None]:
    """
    Yield tuples: (frame_id, video_time_s, frame_bgr, source_name, fps)
    video_time_s is None for still images and webcams.
    """
    # Webcam: the source is an integer digit like "0" or "1"
    if source.isdigit():
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Could not open webcam index {source}")
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_id, None, frame, f"webcam{source}", 30.0
                frame_id += 1
        finally:
            cap.release()
        return

    path = Path(source)
    if not path.exists():
        sys.exit(f"[ERROR] Source path does not exist: {path.resolve()}")

    # Directory of still images
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"[ERROR] No supported images found in directory: {path.resolve()}")
        for frame_id, img_path in enumerate(files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            yield frame_id, None, frame, img_path.name, 1.0
        return

    # Single still image
    if path.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(path))
        if frame is None:
            sys.exit(f"[ERROR] Could not read image: {path.resolve()}")
        yield 0, None, frame, path.name, 1.0
        return

    # Video stream file
    if path.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            sys.exit(f"[ERROR] Could not open video file: {path.resolve()}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame_id, frame_id / fps, frame, path.name, fps
                frame_id += 1
        finally:
            cap.release()
        return

    sys.exit(f"[ERROR] Unsupported media source: {source}")


# --------------------------------------------------------------------------- #
# Visual Annotation
# --------------------------------------------------------------------------- #
def draw_detections(frame: cv2.typing.MatLike, detections: list[dict]) -> cv2.typing.MatLike:
    """Draw bounding boxes and class/confidence labels onto a copy of the frame."""
    annotated = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["bbox_xyxy"])
        # Green bounding box (thickness 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label banner
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 6, top + th + 8), (0, 255, 0), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # Center marker point
        cx, cy = (int(v) for v in d["bbox_center"])
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
    return annotated


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def main():
    # Pre-parse --config so we can populate default argument values
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Load project configuration
    config_path = Path(pre_args.config)
    cfg = load_config(config_path if config_path.exists() else None)

    # Initialize NIDAR centralized logger
    log = get_logger("detector", level=getattr(cfg.logging, "level", "INFO"))

    parser = argparse.ArgumentParser(
        description="NIDAR baseline survivor detector",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input image, folder of images, video file, or webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--model",
        default=getattr(cfg.model, "name", "yolov8n.pt"),
        help="YOLO weights file path (default from config: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=getattr(cfg.model, "confidence_threshold", 0.40),
        help="Minimum confidence threshold (0-1) (default from config: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=getattr(cfg.model, "image_size", 640),
        help="Inference image resolution (default from config: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=getattr(cfg.model, "device", "cuda:0"),
        help="Compute device (e.g. 'cuda:0' or 'cpu') (default from config: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(getattr(cfg.output, "results_dir", "outputs")) / "runs_nidar"),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live window preview (press 'q' to quit)",
    )
    args = parser.parse_args(remaining_argv)

    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists() and (Path("models") / args.model).exists():
        model_path = Path("models") / args.model

    log.info(f"Loading YOLO model: {model_path} on device: {args.device}...")
    model = YOLO(str(model_path))

    jsonl_path = out_dir / "detections.jsonl"
    csv_path = out_dir / "detections.csv"

    jsonl_file = open(jsonl_path, "w", encoding="utf-8")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame_id",
            "source",
            "video_time_s",
            "timestamp_utc",
            "det_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
        ]
    )

    video_writer = None
    total_frames = 0
    total_people = 0

    log.info(f"Processing input source: {args.source} (confidence threshold: {args.conf})")

    try:
        for frame_id, video_time_s, frame, source_name, fps in iter_frames(args.source):
            h, w = frame.shape[:2]

            # Inference execution via YOLO
            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )
            result = results[0]

            detections = []
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                detections.append(
                    {
                        "det_id": i,
                        "class_name": "person",
                        "confidence": round(float(confs[i]), 4),
                        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "bbox_center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                    }
                )

            # Microsecond UTC ISO 8601 timestamp per docs/conventions.md
            timestamp_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            record = {
                "frame_id": frame_id,
                "source": source_name,
                "video_time_s": None if video_time_s is None else round(video_time_s, 3),
                "timestamp_utc": timestamp_utc,
                "frame_width": w,
                "frame_height": h,
                "inference_ms": round(float(result.speed.get("inference", 0.0)), 1),
                "detections": detections,
            }
            jsonl_file.write(json.dumps(record) + "\n")

            for d in detections:
                csv_writer.writerow(
                    [
                        frame_id,
                        source_name,
                        record["video_time_s"],
                        timestamp_utc,
                        d["det_id"],
                        d["class_name"],
                        d["confidence"],
                        *d["bbox_xyxy"],
                        *d["bbox_center"],
                    ]
                )

            # Visual output generation
            annotated = draw_detections(frame, detections)
            is_video = video_time_s is not None or source_name.startswith("webcam")
            if is_video:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(out_dir / "annotated_video.mp4"), fourcc, fps, (w, h)
                    )
                video_writer.write(annotated)
            else:
                annotated_path = out_dir / "annotated" / f"{Path(source_name).stem}_det.jpg"
                cv2.imwrite(str(annotated_path), annotated)

            total_frames += 1
            total_people += len(detections)
            log.info(
                f"frame {frame_id:4d} | {source_name} | survivors: {len(detections)} "
                f"| inference: {record['inference_ms']} ms"
            )

            if args.show:
                cv2.imshow("NIDAR Survivor Detection (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        jsonl_file.close()
        csv_file.close()
        if video_writer is not None:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    log.info("Detection run complete.")
    log.info(f"Frames processed : {total_frames}")
    log.info(f"Total survivors  : {total_people}")
    log.info(f"Detections JSONL : {jsonl_path.resolve()}")
    log.info(f"Detections CSV   : {csv_path.resolve()}")
    log.info(f"Annotated media  : {(out_dir / 'annotated').resolve()}")


if __name__ == "__main__":
    main()
