# Decision 007 — Real-Photo Data Sources: COCO val2017 Anchor, Cutouts and Sanity Set

| Field        | Value                                                              |
|-------------|--------------------------------------------------------------------|
| **Date**    | 2026-11-02                                                         |
| **Author**  | Aryan (AI/CV Lead)                                                 |
| **Status**  | Accepted                                                           |
| **Scope**   | Real photos used in training, person cutouts for the compositor, and a held-out real-photo sanity set |
| **Extends** | ADR 005 (data strategy): adds COCO val2017 as a real-photo source next to SARD |

---

## 1. Context

- There is no arena footage, and the team cannot capture photos yet. The real test set (`data/test_real/`) does not exist.
- A detector fine-tuned only on simulation can forget what real people look like. Its scores on simulated images say little about real images.
- COCO val2017 has 5,000 real photos with person boxes and polygon masks, under licences that allow internal research use (annotations CC BY 4.0; images from Flickr under their own licences; see `data/README.md`).

**COCO sanity set, not arena.** COCO photos are everyday scenes (streets, kitchens, sports). Results on them are a regression and sanity check: does a model trained on our data still find real people? They are not a measure of performance in the AirMouse maze. Every report and README line that quotes a COCO number must say "COCO sanity set, not arena".

---

## 2. Decision

### 2.1 Three disjoint groups, split by image id

Every val2017 image id goes into exactly one group by a seeded hash of the id: `u = first 64 bits of sha256("<seed>:<image_id>") / 2^64`, cumulative fractions (`configs/coco.yaml split`).

| Group | Share | Used for |
|---|---|---|
| `cutout_source` | 0.40 (ASSUMPTION) | RGBA person cutouts for the compositor (planned). Never in anchor or sanity |
| `anchor` | 0.40 (ASSUMPTION) | Real photos mixed into training (A) |
| `sanity` | 0.20 (ASSUMPTION) | Held-out real-photo sanity set (C). **Never trained on** |

- **Stable:** an id's group depends only on the seed and the id, so adding or removing images never moves another image.
- **Split by image id:** COCO val2017 has one id per photo, so two crops or two cutouts of the same photo cannot end up in different groups.
- **Known limitation:** near-duplicate photos (for example a burst of shots from the same Flickr user) have different ids and may land in different groups. The split does not detect them. The leakage check's dHash warnings (2.5) find only near-identical images, not two different shots of the same scene.

### 2.2 Anchor and sanity sets (YOLO, single class `person`)

- **Labels from COCO boxes only:** xywh is converted to normalised YOLO and clipped to the image. Class 0 = person. The pretrained model's class names are ignored.
- **All-or-nothing per image:** an image is used only if every non-crowd person in it is kept. One person whose box, letterboxed to 416 (scale = 416 / max(w, h)), is under 10 px tall or 6 px wide excludes the whole image. The limits are read from `arena.yaml labels` (`imgsz`, `min_box_height_px`, `min_box_width_px`), the same rule the simulator's labels use. Any crowd region in the **person** category also excludes the image. Crowd regions of other categories do not, because they do not hide unlabelled people. Reason: a training image with a visible, unlabelled person teaches the model to ignore people.
- **Persons stored as RLE** (`iscrowd = 0` with an RLE mask) are kept, because only the box is needed. They are counted.
- **Negatives** (300 anchor, 100 sanity, ASSUMPTION): images with no person annotation at all, taken in seeded order. A candidate is dropped if any of its COCO captions contains a person word (list in `coco.yaml`, ASSUMPTION), or if the pretrained `yolo11n.pt` finds a person at conf ≥ 0.5 (owner's choice). The detector only **excludes** suspicious negatives and never creates labels. Reason: COCO sometimes misses people, and a negative with a person in it teaches the model to miss people.
- **Files:** the images are copies of the original JPEG bytes from `val2017.zip`, so their sha256 equals the zip member's. Copies were chosen over symlinks, which need developer mode on Windows. Each set has `manifest.json` (decisions, exclusion reasons, negative-filter table, box sizes at 416). `anchor.yaml` and `sanity.yaml` set `names: {0: person}`. `sanity.yaml` points only at sanity images and is marked evaluation-only.

### 2.3 Person cutouts (for the compositor)

Only from `cutout_source`. A non-crowd person becomes an RGBA cutout if:

- its segmentation is a polygon (RLE persons are skipped and counted);
- its box is not within 2 px of the image border, so it is not cut off;
- its source box is ≥ 150 px tall;
- no other person box (crowd regions included) covers more than 10 % of its box (ASSUMPTION);
- its mask covers ≥ 30 % of its box (ASSUMPTION).

The polygon is drawn with `cv2.fillPoly` (no pycocotools needed). The crop is tight to the mask, alpha is binary 0/255, and colour is zeroed outside the mask. Softening the edge is the compositor's job. Wide cutouts (w/h > 1.2) are counted as possible lying people.

### 2.4 Download

`data_tools/get_coco.py` is the only script in the repo that uses the internet, and only at setup time. It streams in chunks, hashes while streaming and resumes with HTTP Range when the server allows it. It records size and sha256 in `data/coco/DOWNLOAD.json`; COCO publishes no checksum we know of, so these are recorded, not verified (TBD). A blocked download stops the script. No other dataset is substituted.

### 2.5 Leakage check

`data_tools/leakage.py` compares training sources with protected sets. Training sources are the training directories plus the COCO source photos of the cutouts. The protected sets are the sanity set, plus `data/test_proxy` and `data/test_real` when they exist.

- **Errors (exit 1):**
  - the same sha256 on both sides;
  - the same COCO id on both sides;
  - a `cutout_source` id used as a full image;
  - a cutout whose source is not in `cutout_source`.
- **Warnings only:** near matches by 64-bit dHash, Hamming distance ≤ 4.

### 2.6 Disk guard

Every large step stops if less than 1 GB (decimal) would remain free. A set estimated above 0.5 GB is not written without `--yes`, after the owner agrees.

---

## 3. Consequences

- Training can mix real photos (anchor) with simulation, and the compositor (planned) has real people to paste onto empty PyBullet frames.
- The sanity set gives a cheap regression check on real people. It does not replace a real arena test set, which is still TBD. Reports keep simulation, COCO sanity and (later) real-arena results separate.
- Nothing under `data/` is committed or redistributed. Only `data/README.md` (licence terms) is in git.
- Open: whether the 40/40/20 shares, the negative counts and the cutout thresholds suit training. They are ASSUMPTIONs until training results exist.

## 4. Measured run (2 Nov 2026; COCO sanity set, not arena)

- **Download:** both zips complete after several resumed attempts (the server reset or timed out repeatedly):
  - val2017.zip: 10 attempts.
  - annotations_trainval2017.zip: 6 attempts.
  - `testzip` finds no bad member in either.
  - Both extracted JSONs match the CRC32 and size in the zip directory.
  - sha256 values are recorded, not verified against an official checksum.
- **Split** (5000 images):
  - cutout_source: 1987 images, 4472 person boxes.
  - anchor: 1982 images, 4140 person boxes.
  - sanity: 1031 images, 2165 person boxes.
- **Anchor:** 1158 images (858 with persons, 300 negatives), 2289 boxes. 198 of the 1056 person images were excluded.
- **Sanity:** 560 images (460 with persons, 100 negatives), 1282 boxes, 20.1 % of them under 40 px tall at 416. 97 of the 557 person images were excluded.
- **Most common exclusions, both sets combined:** negatives over quota 801, caption filter 194, person too short 183, too narrow 180, person crowd 123. The detector removed 13 negatives.
- **Cutouts:** 455 of 4472 persons. Median height 273 px, median aspect 0.55, 16 wide (3.5 %).
- **Leakage check:** 0 errors, 0 near-match warnings. A planted leak fails with exit 1.
- **Disk:** 1.09 GB downloads, 0.28 GB sets, 0.03 GB cutouts.

Seen by eye on the contact sheets:
- some cutouts are partial people (head and shoulders) or carry a background halo from the coarse COCO polygons;
- COCO labels statues and people on screens as persons.

The full report (`outputs/scratch/coco/report.md`) is gitignored and stays on the owner's machine.

**Download fallback (not implemented):** only `instances_val2017.json` and `captions_val2017.json` are needed (about 7.3 MB compressed of the 252.9 MB annotations zip). If a fresh setup stalls, fetching just those two members by HTTP range request is a possible fallback.
