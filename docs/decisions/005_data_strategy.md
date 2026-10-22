# Decision 005 — Training Data Strategy and Evaluation Split Protocol

| Field        | Value                                                              |
|-------------|--------------------------------------------------------------------|
| **Date**    | 2026-10-22                                                         |
| **Author**  | Aryan (AI/CV Lead)                                                 |
| **Status**  | Accepted                                                           |
| **Scope**   | Dataset generation, compositing pipeline, and train/val/test splits|

---

## 1. Context

The NIDAR AirMouse challenge takes place inside an indoor maze, where an autonomous drone must locate human survivors. However, the team currently faces several data constraints:
- **No real labelled arena data exists yet.** Arena construction details (corridor width, wall colors/height, ceiling, lighting) and survivor specifications (whether survivors are real people or dummies) remain open questions.
- **Public datasets are inadequate on their own:** Existing aerial search-and-rescue datasets (e.g. SARD, VisDrone) consist predominantly of outdoor, high-altitude footage. While helpful for varied human poses, they do not reflect indoor corridor geometry, lighting conditions, or drone camera perspectives.
- **Sim-to-real gap:** Pure synthetic rendering fails to capture real fabric folds, skin reflections, motion blur, and sensor noise.

To train high-recall models without access to an arena, a rigorous synthetic-plus-augmented data strategy is required.

---

## 2. Decision: Four-Pillar Training Data Composition

The training dataset will be synthesised and blended from four complementary sources:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING SET COMPOSITION                     │
├───────────────────────────────┬─────────────────────────────────┤
│ 1. PyBullet Arena Renders     │ 2. Real-Person Composites       │
│    Domain-randomised 3D maze, │    Real human cutouts blended   │
│    camera poses & lighting.   │    onto empty PyBullet frames.  │
├───────────────────────────────┼─────────────────────────────────┤
│ 3. Real Aerial Share (SARD)   │ 4. Negative Frames              │
│    Small share from NIDARX    │    Empty corridors, walls,      │
│    Kaggle dataset.            │    corners (no false alarms).   │
└───────────────────────────────┴─────────────────────────────────┘
```

1. **PyBullet Synthetic Renders (Domain-Randomised):**
   - Headless PyBullet simulation generates 3D maze geometry with varying corridor widths, wall colors/textures, ceiling presence, lighting conditions, and camera angles.
2. **Real-Person Composites:**
   - Real-world human silhouette cutouts composited onto empty PyBullet arena frames at realistic scales and floor-plane perspectives.
   - Solves the synthetic appearance gap by providing genuine human clothing textures, poses, and limbs in the context of the indoor arena.
3. **Small Share of Real Aerial Imagery (SARD):**
   - A curated, small subset of real aerial search-and-rescue imagery (from the SARD / NIDARX Kaggle collection) to teach robust generalisation to human features from elevated viewpoints.
4. **Negative Frames:**
   - Empty corridors, blank walls, doorways, and obstacle boxes without people, training the model to suppress background false positives.

---

## 3. Data Splitting Rules

### Rule 1: Split Strictly by Sequence, Never by Frame
- Video frames captured milliseconds apart in the same flight clip are visually near-identical.
- **Random frame-level splitting is strictly prohibited.** Shuffling frames across train and validation sets causes severe data leakage, resulting in meaningless, artificially inflated validation scores.
- Splits must partition entire video sequences, synthetic flight runs, or scene groups.

### Rule 2: Dedicated, Uncontaminated Real Test Set (`data/test_real/`)
- A dedicated real test set consisting exclusively of real recorded imagery/video (`data/test_real/`) is set aside.
- **Simulation images are strictly prohibited in the real test set.**
- Training and validation never touch `data/test_real/`.
- All final model comparisons and ladder rankings must report real test set metrics (precision, recall, mAP@0.5, survivor recall, false survivors, double counts) completely separate from simulation metrics.

---

## 4. Reproducibility and Provenance

- **Deterministic Generation:** Every synthetic render, compositing run, and dataset build must record:
  - Random seed.
  - Configuration file path (e.g. `configs/randomization.yaml`, `configs/composite.yaml`).
  - Git commit SHA of the codebase at generation time.
- **Storage:** Large generated datasets and raw footage are stored on the team shared drive and locally in gitignored `data/`, never committed to the git repository (per ADR 003). Checksums (`sha256`) and build manifests are tracked instead.
