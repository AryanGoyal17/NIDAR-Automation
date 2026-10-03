# NIDAR AirMouse - Project Conventions

| Field       | Value                    |
|------------|--------------------------|
| **Date**   | 2026-10-03               |
| **Author** | Aryan (AI/CV + GCS Lead) |
| **Status** | Living document          |

> **Why conventions?** Multiple sub-teams (CV, Autonomy, SLAM, Simulation,
> Flight Control) produce data that must interoperate. If the CV pipeline
> tags a survivor at pixel `(320, 240)` and the SLAM system maps it to
> world coordinates `(2.1, 3.4)`, everyone must agree on axis directions,
> units, and timestamp formats. Defining these on Day 1 prevents painful
> debugging during integration.

---

## 1. Coordinate Systems

### 1a. Pixel / Image Coordinates

```text
  (0, 0) ────────────────────── x (width, cols)
    │
    │     Image frame (OpenCV convention)
    │     - Origin: top-left corner
    │     - x-axis: rightward (column index)
    │     - y-axis: downward  (row index)
    │     - Units: pixels (integers)
    │
    y (height, rows)
```

| Property | Value | Reason |
|---|---|---|
| Origin | Top-left corner of the image | OpenCV, NumPy, and YOLO all use this convention. |
| x-axis | Rightward (increasing column index) | Standard image convention. |
| y-axis | Downward (increasing row index) | Standard image convention. Differs from maths convention (y-up). |
| Units | Pixels (integers for pixel coords, floats for sub-pixel) | Bounding boxes are in pixels. |

### 1b. Bounding Box Format

All bounding boxes in this project use **`xyxy` format** (top-left, bottom-right):

```text
  (x1, y1) ─────────────┐
    │                    │
    │   Detected Object  │
    │                    │
    └───────────── (x2, y2)
```

| Field | Type | Description |
|---|---|---|
| `x1` | float | Left edge x-coordinate (pixels) |
| `y1` | float | Top edge y-coordinate (pixels) |
| `x2` | float | Right edge x-coordinate (pixels) |
| `y2` | float | Bottom edge y-coordinate (pixels) |

**Why xyxy and not xywh?** Ultralytics YOLO returns results in xyxy format
by default. Using xyxy avoids an extra conversion step and makes it trivial
to draw rectangles with OpenCV's `cv2.rectangle((x1,y1), (x2,y2), ...)`.

### 1c. World / Map Coordinates (for integration with SLAM)

```text
        y (North / forward)
        ^
        │
        │     Maze map (2D, top-down view)
        │     - Origin: agreed start/entry point
        │     - Units: metres (float)
        │
        └──────────> x (East / rightward)
```

| Property | Value | Reason |
|---|---|---|
| Origin | Maze entry/start point (set by SLAM team) | All survivor positions are relative to the known entry. |
| x-axis | Rightward (East) | Matches ROS 2 REP 103 (ENU: East-North-Up). |
| y-axis | Forward (North) | Matches ROS 2 REP 103. |
| Units | Metres (float) | SI standard. All world distances in metres. |
| z-axis (if needed) | Upward | Matches ROS 2 REP 103. Relevant for drone altitude. |

> **Note:** Confirm with your SLAM/Autonomy lead that they follow ROS 2 REP 103
> (East-North-Up). If they use NED (North-East-Down, common in aviation), the
> axes differ and a transform must be documented here.

---

## 2. Timestamps

| Property | Value | Reason |
|---|---|---|
| Format | ISO 8601 with UTC timezone | `2026-10-03T14:32:01.123Z` |
| Timezone | Always UTC (suffix `Z`) | The drone (Linux) and ground station (Windows) may have different local clocks. UTC removes ambiguity. |
| Precision | Milliseconds (3 decimal places) | Sufficient for correlating detection events with pose/SLAM data at typical frame rates (10-60 FPS). |

**In Python:**
```python
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") \
          + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
```

---

## 3. File Naming

### 3a. Source Code

| Rule | Convention | Example |
|---|---|---|
| Python files | `snake_case.py` | `yolo_detector.py`, `frame_utils.py` |
| Python classes | `PascalCase` | `YoloDetector`, `DetectionResult` |
| Functions & variables | `snake_case` | `run_inference()`, `confidence_threshold` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_CONFIG_PATH`, `MAX_DETECTIONS` |
| Config files | `snake_case.yaml` | `default.yaml`, `competition_run.yaml` |

### 3b. Data Files and Outputs

| File Type | Naming Pattern | Example |
|---|---|---|
| Test images | `{scene}_{sequence:04d}.png` | `corridor_0001.png` |
| Test videos | `{scene}_{date}.mp4` | `maze_test_2026-10-05.mp4` |
| Model weights | `{architecture}_{variant}_{version}.pt` | `yolov8n_survivor_v1.pt` |
| Detection logs | `detections.jsonl` (JSON lines) | One JSON object per detection |
| Annotated frames | `frame_{number:06d}.png` | `frame_000042.png` |

### 3c. Frame Numbering

| Property | Value | Reason |
|---|---|---|
| Indexing | 0-based integers | Matches OpenCV's `cv2.VideoCapture` frame counter. |
| Format in filenames | Zero-padded, 6 digits | `frame_000042.png` sorts correctly in file explorers. |
| Format in code/logs | Plain integer | `frame_id: 42` in JSON logs. |

---

## 4. Detection Output Format

Every detection record (whether logged to a file, sent over WebSocket to the
GCS, or published as a ROS 2 message) must contain these fields:

```json
{
  "frame_id": 42,
  "timestamp": "2026-10-03T14:32:01.123Z",
  "bbox_xyxy": [200.0, 150.0, 400.0, 380.0],
  "confidence": 0.92,
  "class_id": 0,
  "class_name": "person"
}
```

> This schema will be formalised with Pydantic in Phase 4 (STEP 4.x).

---

## 5. Python Style

| Rule | Convention | Reason |
|---|---|---|
| Formatter | Project default (PEP 8) | Consistency across all contributors. |
| Type hints | Required on all function signatures | Catches bugs early; documents intent. |
| Docstrings | Required on all public functions/classes | Explains *why*, not just *what*. |
| Max line length | 100 characters | Comfortable on a laptop screen with a side panel. |
| Imports | Standard library, then third-party, then local. Separated by blank lines. | PEP 8 convention; easy to scan. |

---

## 6. Units Summary Table

| Quantity | Unit | Type | Context |
|---|---|---|---|
| Pixel coordinates | pixels | int / float | Image frame |
| Bounding box coords | pixels (float) | xyxy format | Detection output |
| Confidence score | 0.0 to 1.0 | float | Detection output |
| World position | metres (float) | (x, y) or (x, y, z) | Map / SLAM |
| Drone altitude | metres (float) | z-axis up | Flight controller |
| Time | ISO 8601 UTC | string | Everywhere |
| Inference latency | milliseconds | float | Performance logs |
| Frame rate | frames per second (FPS) | float | Performance logs |
