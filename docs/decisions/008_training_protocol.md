# Decision 008 — Training Protocol: E1 (Sim Only) vs E2 (Sim + COCO Anchor)

| Field        | Value                                                              |
|-------------|--------------------------------------------------------------------|
| **Date**    | 2026-11-02                                                         |
| **Author**  | Aryan (AI/CV Lead)                                                 |
| **Status**  | Accepted (dry runs only; no full training yet)                     |
| **Scope**   | Training data variants, validation and model selection, evaluation operating point |
| **Extends** | ADR 004 (model choice), ADR 005 (data strategy), ADR 007 (real data sources) |

---

## 1. Context

- There is no arena footage, no proxy test set and no real test set (`data/test_proxy/`, `data/test_real/` do not exist). Every number available before them is either **simulation validation, not arena** or **COCO sanity set, not arena**.
- The simulator (`sim/generator.py`) gives labelled frames of procedural survivors. Their boxes are large: median label box height at 416 was 77.7 px in the 40-scene run of 2 Nov 2026 (p10 30.6). The real maze will show smaller, farther people. The train and val generator configs widen the aimed distance to 0.6–6.0 m, sampled log-uniformly (ASSUMPTION).
- COCO-pretrained YOLOv8n and YOLO11n find almost none of the procedural survivors (pretrained baseline on the dry sim_val: recall 0/88 and 1/88 at the operating threshold; see CLAUDE.md section 6). So fine-tuning on simulation is needed, and it risks forgetting real people.
- The COCO anchor set (1158 real photos, ADR 007) can be mixed into training to keep real people in view.

---

## 2. Decision

### 2.1 Two training variants on identical sim data

| Variant | Train list | Validation list (chooses the best epoch, early stopping) |
|---|---|---|
| **E1** | sim_train | sim_val |
| **E2** | sim_train + every anchor_train image once | sim_val + anchor_val |

- **sim_train / sim_val:** separate generator runs (`configs/generator_train.yaml`, seed 1000; `configs/generator_val.yaml`, seed 2000). Val uses the hand-made layouts `loop` and `t_junction`, which train never sees; train uses `dead_end`, `l_turn`, `straight_corridor`, `two_rooms`. Random mazes come from different master seeds. `data_tools/build_dataset.py` asserts that train and val share no group (master seed, scene index), master seed, maze seed, layout grid or hand-made layout.
- **anchor_train / anchor_val:** the COCO anchor set split 85 / 15 % by a salted hash of the COCO id (`sha256("<seed>:anchor_val:<id>")`). The salt is needed: ADR 007 chose the anchor group with the unsalted hash and the same seed, so the unsalted value is always in [0.40, 0.80) and would never fall below 0.15.
- Both variants train YOLO11n and YOLOv8n with the same settings (`configs/train.yaml`: imgsz 416, 40 epochs, patience 10, seed 0, single class). Augmentation stays at the Ultralytics defaults; each run records the values in effect.
- No image is copied. The builder writes Ultralytics list files (sorted, absolute paths, so the manifest hash is stable) and refuses to finish if `data_tools/leakage.py` finds any overlap with the protected sets.

### 2.2 The sanity set never selects anything

- The COCO sanity set (560 images) is never in a training list or a validation list, so it never chooses the best epoch or triggers early stopping.
- It is also not used to choose the confidence threshold. `cv/eval/evaluate.py` chooses each model's operating point on **anchor_val**: the lowest confidence with at most 0.2 false positives per image (ASSUMPTION; recall is the priority). The same threshold is then applied unchanged to sim_val, anchor_val, the sanity set and any future test set. The script refuses the sanity set as the operating set.
- **Why:** anything used to make a choice is no longer an unbiased estimate. The sanity set has to stay a clean check of "does the model still find real people?". If it picked an epoch or a threshold, its score would be optimistic.

### 2.3 E2 proportions band

The builder stops before writing if anchor images are under 10 % or over 50 % of E2's training list (owner's rule), unless `--allow-anchor-share` is given. With the planned run sizes (160 sim scenes × 32 frames, about 5120 frames; 992 anchor_train images), the expected share is about 16 % of images. Box share depends on the sim box rate (about 0.5 per frame in the dry run), roughly 43 % of boxes. Both are estimates until the full generation exists.

---

## 3. Known limits

1. **E2 trains on more images per epoch.** With the same 40 epochs, E2 makes more optimiser steps than E1, by a factor of (sim_train + anchor_train) / sim_train. A difference between E1 and E2 can therefore come from extra steps as well as from the real photos. The epoch count is not changed to equalise this (owner's decision). The images and iterations per epoch of each variant are reported with the results. At batch 64 (accumulate 1, so optimiser steps = iterations): dry lists E1 320 images → 5 iterations, E2 1312 → 21; planned full lists (estimate) E1 5120 → 80, E2 6112 → 96, so E2 makes about 20 % more steps over 40 epochs.
2. **Validation differs between variants.** E2's best epoch is chosen on sim_val + anchor_val, E1's on sim_val only. That is intended (E2 should not lose real people), but it means the two variants select epochs by different criteria.
3. **Mosaic mixes sources.** Ultralytics mosaic (on by default, closed for the last 10 epochs) puts four training images into one. In E2 a mosaic tile can join sim frames and COCO photos, a combination that never occurs at test time. This is accepted, not tuned.
4. **Vertical flips are off** by default (`flipud` 0.0); horizontal flips are on (0.5). Lying survivors seen from above could justify vertical flips. This is not changed without a measured reason.
5. **HSV jitter** (hsv_h 0.015, hsv_s 0.7, hsv_v 0.4) overlaps with the simulator's own lighting and colour randomisation. The two together may be stronger than either alone. Not measured.
6. **No arena data.** Simulation and COCO results do not predict performance in the maze. A real test set is needed before any choice between E1 and E2 is final.
7. **Small sim validation.** A 30-scene val run (planned) covers only two hand-made layouts and the random mazes of 30 scenes. Its numbers have wide uncertainty.
8. **Dataloader workers 0.** On the development laptop, workers 2 and 4 crashed (segfault, and once WinError 1455 "paging file is too small"); training runs with workers 0, so mosaic and decoding run in the main process. Measured 2 Nov 2026, see CLAUDE.md.
9. **Dry runs are not the full runs.** The 1-epoch dry runs use a 300-image subset, and with `close_mosaic` 10 Ultralytics turns mosaic off in a 1-epoch run. Time estimates from them are rough.

---

## 4. Consequences

- Training is `python -m cv.train.finetune` after `python -m data_tools.build_dataset`. Evaluation is `python -m cv.eval.evaluate` with the threshold chosen on anchor_val.
- Results tables always state which set they come from: "simulation validation, not arena" or "COCO sanity set, not arena".
- Revisit this ADR when a proxy or real test set exists, or when composited data (cutouts on sim frames) is added.
