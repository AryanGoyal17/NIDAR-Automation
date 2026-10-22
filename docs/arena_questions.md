# Arena & Sensor Parameter Verification Checklist

| Field       | Value                                                              |
|------------|--------------------------------------------------------------------|
| **Status** | Living team questionnaire                                          |
| **Scope**  | Parameters in `configs/arena.yaml` to confirm with project leads    |
| **Target Teams** | System Lead, Controls/Autonomy, Hardware/Sensors, Competition Rulebook |

---

## Overview

The computer vision synthetic data pipeline and PyBullet simulation rely on [`configs/arena.yaml`](file:///D:/NIDAR/configs/arena.yaml) as the single source of truth for all spatial, optical, and environmental parameters.

Currently, several values are working engineering estimates marked with `# ASSUMPTION`. This checklist defines the exact questions to confirm with the respective sub-teams and competition organizers before the final simulation-to-real transfer and training runs.

---

## 1. Survivor (Target) Specifications

*Primary Contact: Competition Rulebook / System Lead*

- [ ] **1.1 Survivor Physical Form:**
  - Are survivors **real volunteers** or **mannequins / CPR dummies**?
  - *Impact:* A standard COCO-pretrained model is trained on living people. Rigid mannequins, plastic skin, or featureless dummy faces require tailored synthetic textures or fine-tuning cutouts.
- [ ] **1.2 Thermal Signature:**
  - If dummies are used, are they heated? (Thermal sensors see zero contrast on unheated dummies).
- [ ] **1.3 Clothing & Appearance:**
  - Are survivors dressed in random civilian clothing, or do they wear high-visibility vests / standard overalls?
  - Are there any restrictions on clothing color (e.g. camouflage against floor/walls)?
- [ ] **1.4 Postures & Poses:**
  - What postures are permitted in the arena?
  - ( ) Flat on back (supine)
  - ( ) Flat on stomach (prone)
  - ( ) Lateral recovery position
  - ( ) Slumped sitting against walls
  - ( ) Trapped or partially occluded under fallen simulated debris
- [ ] **1.5 Target Distribution:**
  - Can multiple survivors be located in the same room or corridor, or is there a minimum separation distance?
  - (Rulebook specifies: up to six survivors total).

---

## 2. Arena Geometry & Dimensions

*Primary Contact: Autonomy / SLAM Lead & Competition Organizers*

- [ ] **2.1 Corridor Clear Width:**
  - What is the minimum and nominal corridor width? (Current assumption: `1.5 m`).
- [ ] **2.2 Wall Height:**
  - What is the partition wall height? (Current assumption: `2.0 m`).
  - Does the drone fly below the wall top, or is over-wall flight possible/permitted?
- [ ] **2.3 Enclosure & Ceiling:**
  - Is the maze ceiling **fully enclosed** or **open top**? (Current assumption: enclosed, `3.0 m` height).
  - Are there overhead obstacles (trusses, lighting bars, safety netting, dangling cables)?
- [ ] **2.4 Room Sizes & Layout Topology:**
  - What are the dimensions of rooms along the maze? (Current assumption: `2.0 m` to `4.5 m`).
  - Are there narrow doorways, T-junctions, dead ends, and 90-degree blind turns?
- [ ] **2.5 Obstacles & Clutter:**
  - Will there be non-human clutter in the corridors (boxes, barrels, furniture, fake rubble)?
  - *Impact:* Used to generate realistic negative frames in simulation to prevent false positives.

---

## 3. Materials, Colors & Textures

*Primary Contact: Hardware / Field Setup Team*

- [ ] **3.1 Wall Materials & Finish:**
  - What material are walls made of (bare plywood, painted drywall, tarpaulin, cardboard)?
  - What is the wall color and reflectivity? (Current assumption: light grey/off-white matte, `[0.80, 0.80, 0.78]`).
  - Are there posters, signs, or sponsor banners on walls?
- [ ] **3.2 Floor Material & Finish:**
  - What is the flooring surface (unfinished concrete, linoleum tile, low-pile carpet, glossy gymnasium floor)?
  - *Impact:* Glossy floors produce strong specular reflections of survivors and drone lights, creating potential phantom detections. (Current assumption: matte concrete, `[0.45, 0.45, 0.47]`).
  - Are there floor markings, lane lines, or hazard tape?

---

## 4. Lighting & Environmental Conditions

*Primary Contact: System Lead / Flight Operations*

- [ ] **4.1 Ambient Light Level:**
  - What is the illumination level inside the arena?
  - ( ) Bright indoor office / facility lighting (300–500 lux)
  - ( ) Dim emergency lighting (50–100 lux)
  - ( ) Pitch dark / blackout arena requiring active drone illumination
- [ ] **4.2 Uniformity & Shadows:**
  - Is lighting diffuse and uniform, or are there sharp spotlight cones and deep shadow pockets?
- [ ] **4.3 Onboard Illumination:**
  - Does the drone carry an onboard LED floodlight?
  - If yes, what is its beam angle, color temperature, and lumen output?

---

## 5. Drone Camera & Optical Sensor

*Primary Contact: Hardware & Sensors Lead*

- [ ] **5.1 Sensor Model & Interface:**
  - Exact camera model (e.g. Raspberry Pi Camera Module 3, USB UVC webcam, Intel RealSense D435 RGB stream).
- [ ] **5.2 Native Resolution & Stream Format:**
  - What resolution will the camera stream deliver to the detector? (Current assumption: `1280x720` at `30 FPS`).
  - Pixel format delivered to OpenCV (e.g. BGR24, NV12).
- [ ] **5.3 Lens Optical Characteristics:**
  - Horizontal Field of View (H-FOV): (Current assumption: `85.0 deg`).
  - Vertical Field of View (V-FOV): (Current assumption: `54.0 deg`).
  - Optical distortion: Wide-angle fisheye (requiring undistortion) or rectilinear lens?
  - Calibrated intrinsics ($f_x, f_y, c_x, c_y$) available?
- [ ] **5.4 Physical Mounting Geometry on Drone:**
  - Fixed downward tilt angle relative to forward flight axis: (Current assumption: `30.0 deg` downward pitch).
  - Nominal cruise altitude above floor: (Current assumption: `0.8 m`).
  - Camera offset from drone center of mass (lever arm $[x, y, z]$).
- [ ] **5.5 Shutter Mechanism:**
  - Global shutter or rolling shutter? (Rolling shutter requires evaluating motion blur during rapid yaw/rotation).

---

## 6. Flight Dynamics & Subsystem Handoff

*Primary Contact: Flight Control & Mapping Leads*

- [ ] **6.1 Cruise Speed & Maneuvers:**
  - Nominal forward exploration speed (m/s).
  - Maximum allowable yaw rotation rate (deg/s) while searching for survivors.
- [ ] **6.2 Timestamping Synchronization:**
  - Does the camera driver tag frames with a hardware monotonic capture timestamp, or system UTC clock?
  - (Mapping subsystem requires capture timestamp to match drone pose).
- [ ] **6.3 Target Frame Rate for Detection:**
  - Minimum acceptable detector FPS to feed the tracking and mapping pipeline.
