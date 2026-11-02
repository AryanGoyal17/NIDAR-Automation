"""
tests/test_sim/test_generator_scenes.py — Heavy generator checks
================================================================

Slow, so skipped by default. Run with::

    $env:NIDAR_SCENE_TESTS = "1"; python -m pytest -q tests/test_sim/test_generator_scenes.py -s

- 5 random written frames re-rendered from their meta line alone
  (:func:`sim.generator.replay_frame`) are byte-identical to the written JPEG;
- process memory stays bounded over 30 scenes in one worker (measured in a
  fresh interpreter, so other tests' imports do not change it).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import replace

import pytest

p = pytest.importorskip("pybullet")

import sim.generator as gen  # noqa: E402
from sim.generator import (  # noqa: E402
    GeneratorConfig,
    generate,
    replay_frame,
    yolo_text,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NIDAR_SCENE_TESTS") != "1",
    reason="heavy generator checks: set NIDAR_SCENE_TESTS=1 to run",
)

MASTER = 2026
SCENES, FRAMES = 4, 3

# Allowed process memory (RSS) growth from scene 10 to scene 30 in one worker.
# ASSUMPTION, set from a measurement on 1 Nov 2026 (320x180, 1 frame per scene,
# power plan Balanced, on battery): a new client per scene grew 0.8 and 1.3 MB
# over those 20 scenes (two runs); the old pattern (one client, textures
# re-applied per scene) grew 18.4 and 18.2 MB. 8 MB lies between the two:
# about 6x the larger measured growth and under half the old leak.
MAX_RSS_GROWTH_MB = 8.0


@pytest.fixture(scope="module")
def cfg() -> GeneratorConfig:
    return replace(GeneratorConfig.from_yaml(), width=320, height=180)


def test_replay_byte_identical(cfg, tmp_path) -> None:
    run = tmp_path / "run"
    generate(run, SCENES, FRAMES, MASTER, workers=1, cfg=cfg)
    metas = [json.loads(x) for x in (run / "meta.jsonl").read_text().splitlines()]
    chosen = random.Random(0).sample(metas, min(5, len(metas)))
    assert len(chosen) == 5
    for meta in chosen:
        jpeg, labels = replay_frame(meta, cfg)
        assert jpeg == (run / meta["image"]).read_bytes(), meta["frame_id"]
        written = (run / meta["label"]).read_text(encoding="utf-8")
        assert yolo_text(labels.labels) == written, meta["frame_id"]


MEMORY_SCRIPT = """
import gc, json, sys
from dataclasses import replace
import psutil
from sim.generator import GeneratorConfig, SceneTask, config_hash, run_scene_task
cfg = replace(GeneratorConfig.from_yaml(), width=320, height=180)
chash = config_hash(cfg)
proc = psutil.Process()
def rss_mb():
    gc.collect()
    return proc.memory_info().rss / 2**20
marks = {0: rss_mb()}
for i in range(30):
    result = run_scene_task(SceneTask(cfg, sys.argv[1], %d, i, 1, chash))
    assert result["status"] == "done", result
    if (i + 1) %% 10 == 0:
        marks[i + 1] = rss_mb()
print(json.dumps(marks))
"""


def test_memory_bounded_over_30_scenes(tmp_path) -> None:
    """
    Measured in a fresh interpreter: in the full suite other tests load torch
    and Ultralytics into the pytest process (RSS about 446 MB instead of about
    61 MB), and that process's RSS noise alone exceeded the limit (2 Nov 2026).
    """
    pytest.importorskip("psutil")
    out = subprocess.run(
        [sys.executable, "-c", MEMORY_SCRIPT % MASTER, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    marks = {int(k): v for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}
    power = gen.power_state()
    print(f"\nRSS MB after N scenes: {marks} ({power})")
    assert marks[30] - marks[10] < MAX_RSS_GROWTH_MB, marks
