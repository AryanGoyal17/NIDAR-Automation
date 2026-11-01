"""
tests/test_sim/test_generator.py — Fast checks of sim/generator.py
==================================================================

Tiny scenes at 160x90 so the default suite stays quick (workers=1 vs 2 included:
about 7 s). Replay and memory checks are in test_generator_scenes.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

p = pytest.importorskip("pybullet")

import sim.generator as gen  # noqa: E402
from sim.generator import (  # noqa: E402
    AMBIGUOUS,
    NEGATIVE,
    POSITIVE,
    GeneratorConfig,
    RunMismatchError,
    Scene,
    classify,
    config_hash,
    derive_seed,
    generate,
    plan_scene,
    write_scene,
)
from sim.labels import DroppedLabel, FrameLabels, SurvivorLabel, load_yolo_labels  # noqa: E402

MASTER = 7


@pytest.fixture(scope="module")
def cfg() -> GeneratorConfig:
    return replace(GeneratorConfig.from_yaml(), width=160, height=90)


@pytest.fixture
def no_power(monkeypatch) -> None:
    """Skip the powercfg/PowerShell calls (about 1 s each) in fast run tests."""
    monkeypatch.setattr(gen, "power_state", lambda: "not read (test)")


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r.pop("timing", None)
    return rows


def snapshot(out: Path) -> dict:
    """Everything a run wrote, minus timing fields (which differ between runs)."""
    files = {
        f"{d}/{f.name}": f.read_bytes()
        for d in ("images", "labels")
        for f in sorted((out / d).glob("*"))
    }
    done = {f.name: _rows(f) for f in sorted((out / "scenes").glob("*.done.json"))}
    return {
        "meta": _rows(out / "meta.jsonl"),
        "attempts": _rows(out / "attempts.jsonl"),
        "done": done,
        "files": files,
    }


def _scene_with_survivors(cfg: GeneratorConfig, start: int = 0) -> int:
    for idx in range(start, start + 50):
        if plan_scene(cfg, MASTER, idx).n_survivors > 0:
            return idx
    raise AssertionError("no scene with survivors in 50 indices")


class TestSeeds:
    def test_pure(self) -> None:
        assert derive_seed(1, 2, "pose", 3, 0) == derive_seed(1, 2, "pose", 3, 0)

    def test_distinct_streams_scenes_and_frames(self) -> None:
        seeds = {
            derive_seed(1, 0, "pose", 0, 0),
            derive_seed(1, 0, "augment", 0, 0),
            derive_seed(1, 1, "pose", 0, 0),
            derive_seed(1, 0, "pose", 1, 0),
            derive_seed(1, 0, "pose", 0, 1),
            derive_seed(2, 0, "pose", 0, 0),
        }
        assert len(seeds) == 6

    def test_range(self) -> None:
        for i in range(100):
            assert 0 <= derive_seed(0, i, "layout") < 2**63


class TestPlan:
    def test_deterministic(self, cfg: GeneratorConfig) -> None:
        for idx in range(10):
            assert plan_scene(cfg, MASTER, idx) == plan_scene(cfg, MASTER, idx)

    def test_layout_pool_and_counts(self, cfg: GeneratorConfig) -> None:
        plans = [plan_scene(cfg, MASTER, i) for i in range(120)]
        sources = {pl.layout_source for pl in plans}
        assert sources == {"handmade", "random"}
        assert {pl.n_survivors for pl in plans} == {0, 1, 2, 3}
        handmade = {pl.layout_name for pl in plans if pl.layout_source == "handmade"}
        assert handmade == {name for name, _ in gen.handmade_layouts(cfg)}
        for pl in plans:
            if pl.layout_source == "random":
                assert cfg.random_rows[0] <= len(pl.grid) <= cfg.random_rows[1] + 1

    def test_all_random_layouts(self, cfg: GeneratorConfig) -> None:
        c = replace(cfg, random_layout_prob=1.0)
        assert all(plan_scene(c, MASTER, i).layout_source == "random" for i in range(20))


class TestClassify:
    def _label(self) -> SurvivorLabel:
        return SurvivorLabel(survivor_id=0, bbox_xyxy=(1, 1, 5, 9), yolo_bbox=(0.1, 0.1, 0.1, 0.1))

    def test_outcomes(self) -> None:
        kept, dropped = self._label(), DroppedLabel(self._label(), ("truncated",))
        assert classify(FrameLabels([], [])) == NEGATIVE
        assert classify(FrameLabels([kept], [])) == POSITIVE
        assert classify(FrameLabels([], [dropped])) == AMBIGUOUS
        assert classify(FrameLabels([kept], [dropped])) == AMBIGUOUS


class TestFrames:
    def test_ambiguous_never_accepted(self, cfg: GeneratorConfig, monkeypatch) -> None:
        """Every render ambiguous: no frame is accepted, as negative or otherwise."""
        always = FrameLabels(
            labels=[],
            dropped=[DroppedLabel(SurvivorLabel(survivor_id=0), ("box_too_short",))],
        )
        monkeypatch.setattr(gen, "label_frame", lambda *a, **k: always)
        plan = plan_scene(cfg, MASTER, 0)
        with Scene(plan, cfg) as scene:
            for f in range(3):
                result = scene.generate_frame(f)
                assert not result.accepted
                assert result.image is None and result.labels is None
                assert result.skip_reason == "ambiguous_after_retries"
                rendered = [a for a in result.attempts if a.get("outcome") is not None]
                assert rendered and all(a["outcome"] == AMBIGUOUS for a in rendered)
                assert len(result.attempts) == cfg.max_retries + 1

    def test_same_plan_same_frames(self, cfg: GeneratorConfig) -> None:
        plan = plan_scene(cfg, MASTER, _scene_with_survivors(cfg))
        runs = []
        for _ in range(2):
            with Scene(plan, cfg) as scene:
                runs.append([scene.generate_frame(f) for f in range(3)])
        for a, b in zip(*runs, strict=True):
            assert (a.mode, a.accepted, a.outcome, a.pose) == (
                b.mode,
                b.accepted,
                b.outcome,
                b.pose,
            )
            if a.accepted:
                assert np.array_equal(a.image, b.image)
                assert [lb.yolo_bbox for lb in a.labels.labels] == [
                    lb.yolo_bbox for lb in b.labels.labels
                ]

    def test_accepted_outcome_matches_mode(self, cfg: GeneratorConfig) -> None:
        plan = plan_scene(cfg, MASTER, _scene_with_survivors(cfg))
        with Scene(plan, cfg) as scene:
            for f in range(4):
                r = scene.generate_frame(f)
                if r.accepted:
                    assert r.outcome in gen.ACCEPTED_OUTCOMES[r.mode]
                    assert classify(r.labels) == r.outcome

    def test_survivor_free_scene_is_all_negative(self, cfg: GeneratorConfig) -> None:
        idx = next(i for i in range(200) if plan_scene(cfg, MASTER, i).n_survivors == 0)
        with Scene(plan_scene(cfg, MASTER, idx), cfg) as scene:
            assert all(scene.frame_mode(f) == "negative" for f in range(10))

    def test_client_disconnected_on_close(self, cfg: GeneratorConfig) -> None:
        scene = Scene(plan_scene(cfg, MASTER, 0), cfg)
        cid = scene.cid
        assert p.getConnectionInfo(cid)["isConnected"]
        scene.close()
        assert not p.getConnectionInfo(cid)["isConnected"]
        scene.close()  # idempotent


class TestPoses:
    def test_aimed_candidate_rules(self, cfg: GeneratorConfig) -> None:
        plan = plan_scene(cfg, MASTER, _scene_with_survivors(cfg))
        with Scene(plan, cfg) as scene:
            found = 0
            for k in range(10):
                pose, target, _ = scene.aimed_candidate(derive_seed(0, 0, "t", k))
                if pose is None:
                    continue
                found += 1
                lo, hi = scene.survivor_boxes[target]
                cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
                dist = math.hypot(cx - pose.x, cy - pose.y)
                assert cfg.aimed_distance_m[0] - 1e-9 <= dist <= cfg.aimed_distance_m[1] + 1e-9
                bearing = math.degrees(math.atan2(cy - pose.y, cx - pose.x))
                off = (pose.yaw - bearing + 180.0) % 360.0 - 180.0
                assert abs(off) <= cfg.aimed_bearing_fraction * scene.half_hfov_deg + 1e-9
                assert scene.sampler.pose_rejection(pose) is None
                assert scene.line_of_sight(pose, target)
            assert found > 0

    def test_pose_offsets_differ_between_layouts(self, cfg: GeneratorConfig) -> None:
        """Same master seed, two scenes with different layouts: different pose draws."""
        a = plan_scene(cfg, MASTER, 0)
        b_idx = next(i for i in range(1, 50) if plan_scene(cfg, MASTER, i).grid != a.grid)
        b = plan_scene(cfg, MASTER, b_idx)

        def first_free(plan) -> tuple[float, ...]:
            with Scene(plan, cfg) as scene:
                seed = derive_seed(MASTER, plan.scene_index, "pose", 0, 0)
                pose, _ = scene.free_candidate(seed)
                cs = scene.sampler.cell_size
                return (pose.x % cs, pose.y % cs, pose.z, pose.pitch, pose.roll)

        oa, ob = first_free(a), first_free(b)
        # z, pitch and roll are drawn after the cell choice, independent of the
        # layout: one shared seed would make them identical.
        assert oa[2:] != ob[2:]
        assert oa != ob

    def test_shared_seed_would_repeat_offsets(self, cfg: GeneratorConfig) -> None:
        """The old behaviour this guards against: one seed, two layouts, same z/pitch/roll."""
        from sim.trajectory import TrajectorySampler

        grids = [plan_scene(cfg, MASTER, i).grid for i in range(10)]
        g1 = grids[0]
        g2 = next(g for g in grids if g != g1)
        s1 = TrajectorySampler(list(g1), cfg.arena_config).sample_poses(1, seed=42)[0]
        s2 = TrajectorySampler(list(g2), cfg.arena_config).sample_poses(1, seed=42)[0]
        assert (s1.z, s1.pitch, s1.roll) == (s2.z, s2.pitch, s2.roll)


class TestRun:
    SCENES, FRAMES = 2, 2

    def _run(self, out: Path, cfg: GeneratorConfig, seed: int = MASTER) -> dict:
        return generate(out, self.SCENES, self.FRAMES, seed, workers=1, cfg=cfg)

    def test_same_args_identical(self, cfg, tmp_path, no_power) -> None:
        self._run(tmp_path / "a", cfg)
        self._run(tmp_path / "b", cfg)
        a, b = snapshot(tmp_path / "a"), snapshot(tmp_path / "b")
        assert a["files"] and a == b

    def test_workers_1_and_2_identical(self, cfg, tmp_path, no_power) -> None:
        self._run(tmp_path / "w1", cfg)
        m = generate(tmp_path / "w2", self.SCENES, self.FRAMES, MASTER, workers=2, cfg=cfg)
        assert len(m["timing"]["per_worker"]) == 2
        a, b = snapshot(tmp_path / "w1"), snapshot(tmp_path / "w2")
        assert a["files"] and a == b

    def test_layout(self, cfg, tmp_path, no_power) -> None:
        m = self._run(tmp_path, cfg)
        metas = _rows(tmp_path / "meta.jsonl")
        assert m["counts"]["frames_written"] == len(metas) > 0
        images = sorted(f.stem for f in (tmp_path / "images").glob("*.jpg"))
        labels = sorted(f.stem for f in (tmp_path / "labels").glob("*.txt"))
        assert images == labels == sorted(r["frame_id"] for r in metas)
        assert not list(tmp_path.rglob("*.tmp*"))
        for r in metas:
            assert r["group_id"] == r["scene_id"]
            assert r["dropped"] == []
            assert set(r["camera"]) >= {"fx", "fy", "cx", "cy", "near_clip_m", "far_clip_m"}

    def test_written_labels_match_memory(self, cfg, tmp_path) -> None:
        plan = plan_scene(cfg, MASTER, _scene_with_survivors(cfg))
        seen = []
        write_scene(plan, cfg, tmp_path, 4, config_hash(cfg), lambda r, m: seen.append((r, m)))
        assert seen
        for result, meta in seen:
            written = load_yolo_labels(tmp_path / meta["label"])
            memory = result.labels.labels
            assert len(written) == len(memory)
            for (cls, *box), label in zip(written, memory, strict=True):
                assert cls == 0
                assert np.allclose(box, label.yolo_bbox, atol=1e-6)
            assert [m["yolo_bbox"] for m in meta["labels"]] == [
                [round(v, 6) for v in lb.yolo_bbox] for lb in memory
            ]

    def test_empty_label_file_only_for_negative(self, cfg, tmp_path, no_power) -> None:
        generate(tmp_path, 3, 3, MASTER, workers=1, cfg=cfg)
        for r in _rows(tmp_path / "meta.jsonl"):
            empty = (tmp_path / r["label"]).read_text(encoding="utf-8") == ""
            assert empty == (r["outcome"] == NEGATIVE)
            assert r["dropped"] == []

    def test_ambiguous_never_written(self, cfg, tmp_path, no_power, monkeypatch) -> None:
        always = FrameLabels(
            labels=[], dropped=[DroppedLabel(SurvivorLabel(survivor_id=0), ("truncated",))]
        )
        monkeypatch.setattr(gen, "label_frame", lambda *a, **k: always)
        m = self._run(tmp_path, replace(cfg, max_retries=1))
        assert m["counts"]["frames_written"] == 0
        assert m["counts"]["skipped_by_reason"] == {
            "ambiguous_after_retries": self.SCENES * self.FRAMES
        }
        assert not list(tmp_path.glob("images/*")) and not list(tmp_path.glob("labels/*"))
        assert (tmp_path / "meta.jsonl").read_text(encoding="utf-8") == ""

    def test_resume_skips_completed_scenes(self, cfg, tmp_path, no_power) -> None:
        self._run(tmp_path, cfg)
        before = snapshot(tmp_path)
        done0 = tmp_path / "scenes" / "s00000.done.json"
        done1 = tmp_path / "scenes" / "s00001.done.json"
        mtime0 = done0.stat().st_mtime_ns
        done1.unlink()
        m = self._run(tmp_path, cfg)
        assert m["counts"]["scenes_resumed"] == 1 and m["counts"]["scenes_run_now"] == 1
        assert done0.stat().st_mtime_ns == mtime0
        assert snapshot(tmp_path) == before
        m = self._run(tmp_path, cfg)
        assert m["counts"]["scenes_resumed"] == 2 and m["counts"]["scenes_run_now"] == 0

    def test_other_args_refused(self, cfg, tmp_path, no_power) -> None:
        self._run(tmp_path, cfg)
        with pytest.raises(RunMismatchError):
            self._run(tmp_path, cfg, seed=MASTER + 1)
        with pytest.raises(RunMismatchError):
            self._run(tmp_path, replace(cfg, max_retries=2))


def _tiny_config_file(tmp_path: Path) -> Path:
    text = Path("configs/generator.yaml").read_text(encoding="utf-8")
    text = text.replace("width: 640", "width: 160").replace("height: 360", "height: 90")
    path = tmp_path / "generator_tiny.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestSceneFailure:
    def test_failed_scene_recorded_and_retried(self, tmp_path, no_power, monkeypatch) -> None:
        config = _tiny_config_file(tmp_path)
        out = tmp_path / "run"
        args = ["--out", str(out), "--scenes", "3", "--frames-per-scene", "2", "--seed", "7"]
        args += ["--config", str(config)]
        real_scene = gen.Scene

        def flaky(plan, cfg):
            if plan.scene_index == 1:
                raise RuntimeError("boom in scene 1")
            return real_scene(plan, cfg)

        monkeypatch.setattr(gen, "Scene", flaky)
        assert gen.main(args) == 1
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["failed_scenes"] == [
            {"scene_id": "s00001", "error": "RuntimeError: boom in scene 1"}
        ]
        assert manifest["counts"]["scenes_done"] == 2
        failed = json.loads((out / "scenes" / "s00001.failed.json").read_text(encoding="utf-8"))
        assert "boom in scene 1" in failed["traceback"]
        assert not (out / "scenes" / "s00001.done.json").exists()
        assert not list((out / "images").glob("s00001_*"))
        assert {r["scene_id"] for r in _rows(out / "meta.jsonl")} <= {"s00000", "s00002"}

        monkeypatch.setattr(gen, "Scene", real_scene)
        assert gen.main(args) == 0
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["failed_scenes"] == []
        assert manifest["counts"]["scenes_resumed"] == 2
        assert manifest["counts"]["scenes_run_now"] == 1
        assert (out / "scenes" / "s00001.done.json").exists()
        assert not (out / "scenes" / "s00001.failed.json").exists()

    def test_cli_mismatch_exit_code(self, tmp_path, no_power) -> None:
        config = _tiny_config_file(tmp_path)
        base = ["--out", str(tmp_path / "r"), "--scenes", "1", "--frames-per-scene", "1"]
        base += ["--config", str(config)]
        assert gen.main([*base, "--seed", "1"]) == 0
        assert gen.main([*base, "--seed", "2"]) == 2
