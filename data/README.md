# data/

Everything under `data/` is gitignored except this file. Datasets, downloads,
generated images, cutouts and labels are **never committed and never
redistributed**. Record sha256 checksums instead (CLAUDE.md section 9).

## COCO val2017 (`data/coco/`, `data/coco_person/`, `data/coco_cutouts/`)

**COCO sanity set, not arena.** These are real everyday photos (people in
streets, kitchens, sports fields), not footage of the AirMouse maze. Scores on
them are a regression and sanity check (does a model fine-tuned on simulation
still find real people?). They say nothing about performance in the arena.
Every report and README line that quotes a COCO number must say
"COCO sanity set, not arena".

### Licence terms

- **Annotations** (boxes, polygons, captions): COCO Consortium, licensed under
  [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  Source: https://cocodataset.org/#termsofuse
- **Images**: taken from Flickr. Each image keeps its own Flickr licence (the
  `licenses` list and each image's `license` field in
  `instances_val2017.json`). The COCO Consortium does not own the copyright of
  the images.
- **Our use**: internal research use only (training and sanity-checking the
  NIDAR survivor detector). Images, cutouts and derived label sets are not
  committed to git and not redistributed in any form (no uploads, no shared
  datasets, no images in the README or reports that leave the team).

### How it is fetched and built

The only internet access is at setup time:

```powershell
python -m data_tools.get_coco                 # downloads into data/coco/, records sha256 in data/coco/DOWNLOAD.json
python -m data_tools.coco_split               # data/coco/split.json: cutout_source / anchor / sanity
python -m data_tools.coco_person_sets         # data/coco_person/{anchor,sanity}, anchor.yaml, sanity.yaml
python -m data_tools.make_person_cutouts      # data/coco_cutouts/ (RGBA PNGs from cutout_source only)
python -m data_tools.leakage --train data/coco_person/anchor/images --cutouts data/coco_cutouts
```

Rules and reasons: `docs/decisions/007_real_data_sources.md`. Settings:
`configs/coco.yaml`.

| Folder | Content | Used for |
|---|---|---|
| `data/coco/` | `val2017.zip` (kept zipped), `annotations_trainval2017.zip`, the extracted `instances_val2017.json` and `captions_val2017.json`, `DOWNLOAD.json`, `split.json` | source |
| `data/coco_person/anchor/` | YOLO person set (copies of the original JPEGs) | training anchor |
| `data/coco_person/sanity/` | YOLO person set, held out | **never trained on**; COCO sanity set, not arena |
| `data/coco_cutouts/` | RGBA person cutouts + `manifest.json` | later compositor |

## Other folders

- `data/sim_raw/<run>/`: synthetic runs from `python -m sim.generator`.
- `data/samples/`, `data/test_videos/`: local sample media.
- `data/test_real/`: [PLANNED] small real arena test set, never used for training.
