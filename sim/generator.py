"""
sim/generator.py — Synthetic training dataset generator
=======================================================

Turns the simulator into training frames: for each scene, a maze (hand-made or
seeded random), randomised textures and lighting, 0-3 procedural survivors,
and K frames, each rendered, augmented and labelled with
:func:`sim.labels.label_frame`.

Frame modes (weights in ``configs/generator.yaml``):

- ``aimed``: the camera is placed at a distance from a chosen survivor drawn
  uniformly or log-uniformly from ``aimed.distance_m`` (0.6-3.5 m by default), with a
  clear line of sight (ray cast) and the survivor's bearing within a fraction
  of half the horizontal FOV. Accepted only with at least one kept label.
- ``free``: :meth:`TrajectorySampler.sample_poses`, as before. Accepted as a
  positive or a negative.
- ``negative``: a pose where no survivor is present in the render, in scenes
  with survivors (hard negatives) and in survivor-free scenes.

A frame that is ambiguous (a survivor present but its label dropped) or does
not match its mode is re-drawn with a new pose, up to ``retries.max_retries``
times; then the frame slot is skipped. An ambiguous frame is never accepted,
so it can never be written as a negative.

Determinism: everything for a scene is a pure function of (master seed,
scene index, config). Every random stream has its own seed from
:func:`derive_seed`, so the same master seed gives different poses in
different scenes and layouts. Each scene uses its own PyBullet DIRECT client,
disconnected at the end (this also frees the scene's textures). Scenes run
one per worker task, so the output is identical for any worker count (only
the ``timing`` fields differ).

Run layout (``--out``)::

    images/<scene>_f<frame>.jpg    labels/<scene>_f<frame>.txt (YOLO, class 0)
    scenes/<scene>.jsonl           meta lines of the scene's written frames
    scenes/<scene>.attempts.jsonl  every attempt (rendered or not)
    scenes/<scene>.done.json       scene summary, written last: scene complete
    scenes/<scene>.failed.json     traceback of a failed scene (no done.json)
    meta.jsonl, attempts.jsonl     merged in scene order at the end of a run
    run.json, manifest.json        run arguments; counts, timing, versions

Every file is written to a temporary name and renamed. Re-running the same
command skips scenes with a ``done.json`` and retries failed ones.
:func:`replay_frame` re-renders a written frame from its meta line alone.

Usage::

    python -m sim.generator --out data/sim_raw/run1 --scenes 40 --frames-per-scene 12 \\
        --seed 2026 --workers 4 --config configs/generator.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pybullet as p

from cv.config import load_config
from cv.logger import get_logger
from sim.camera import CameraPose, PinholeCamera, camera_axes
from sim.labels import FrameLabels, LabelConfig, SilhouetteRenderer, SurvivorLabel, label_frame
from sim.layout_gen import MazeGenerator
from sim.maze import MazeBuilder, MazeResult, load_layout
from sim.randomize import FrameAugmentor, SequenceParams, SequenceRandomizer
from sim.survivor import SurvivorInfo, SurvivorPlacer
from sim.trajectory import PoseSamplingError, TrajectorySampler, pose_key

DEFAULT_GENERATOR_CONFIG = Path("configs/generator.yaml")

log = get_logger("generator")

MODES = ("aimed", "free", "negative")

# classify() outcomes
POSITIVE = "positive"
NEGATIVE = "negative"
AMBIGUOUS = "ambiguous"

# Outcomes a mode accepts; anything else is retried
ACCEPTED_OUTCOMES = {
    "aimed": (POSITIVE,),
    "free": (POSITIVE, NEGATIVE),
    "negative": (NEGATIVE,),
}

# Attempt failure reasons
RETRY_AMBIGUOUS = "ambiguous"
RETRY_AIMED_NO_LABEL = "aimed_no_label"
RETRY_NEGATIVE_HAS_SURVIVOR = "negative_has_survivor"
RETRY_NO_VALID_POSE = "no_valid_pose"

# Skip reason of a frame slot, from the reason of its last attempt
SKIP_REASONS = {
    RETRY_AMBIGUOUS: "ambiguous_after_retries",
    RETRY_AIMED_NO_LABEL: "mode_mismatch_after_retries",
    RETRY_NEGATIVE_HAS_SURVIVOR: "mode_mismatch_after_retries",
    RETRY_NO_VALID_POSE: "no_valid_pose",
}

# Candidate pose rejections (no render)
CAND_DUPLICATE = "duplicate"
CAND_TARGET_OFF_FRAME = "target_off_frame"
CAND_LINE_OF_SIGHT = "line_of_sight"
CAND_SURVIVOR_VISIBLE = "survivor_visible"

SCENE_STREAMS = ("layout", "maze", "randomization", "survivors", "count")


# --------------------------------------------------------------------------- #
# Config and seeds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorConfig:
    """Settings from ``configs/generator.yaml`` (every value an ASSUMPTION there)."""

    arena_config: str = "configs/arena.yaml"
    randomization_config: str = "configs/randomization.yaml"
    width: int = 640
    height: int = 360
    imgsz: int = 416
    jpeg_quality: int = 95
    handmade_dir: str = "sim/layouts"
    handmade_include: tuple[str, ...] | None = None
    """Hand-made layout names (file stems) to draw from; None means all."""
    random_layout_prob: float = 0.5
    random_rows: tuple[int, int] = (7, 15)
    random_cols: tuple[int, int] = (7, 15)
    survivor_count_weights: tuple[float, ...] = (0.10, 0.30, 0.30, 0.30)
    mode_weights: tuple[float, float, float] = (0.55, 0.20, 0.25)
    """Weights of :data:`MODES` (aimed, free, negative)."""
    height_range: tuple[float, float] = (0.6, 1.0)
    pitch_range: tuple[float, float] = (-35.0, -25.0)
    roll_jitter_deg: float = 3.0
    yaw_jitter_deg: float = 12.0
    aimed_distance_m: tuple[float, float] = (0.6, 3.5)
    aimed_distance_sampling: str = "uniform"
    """``uniform`` or ``log_uniform`` (one rng draw either way)."""
    aimed_bearing_fraction: float = 0.6
    aimed_candidates: int = 200
    negative_candidates: int = 50
    max_retries: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_GENERATOR_CONFIG) -> GeneratorConfig:
        """Load the generator config file."""
        c = load_config(path)

        def pair(v: Any) -> tuple[float, float]:
            return (float(v[0]), float(v[1]))

        return cls(
            arena_config=str(c.arena_config),
            randomization_config=str(c.randomization_config),
            width=int(c.render.width),
            height=int(c.render.height),
            imgsz=int(c.imgsz),
            jpeg_quality=int(c.jpeg_quality),
            handmade_dir=str(c.layouts.handmade_dir),
            handmade_include=_include(getattr(c.layouts, "handmade_include", None)),
            random_layout_prob=float(c.layouts.random_prob),
            random_rows=(int(c.layouts.random_rows[0]), int(c.layouts.random_rows[1])),
            random_cols=(int(c.layouts.random_cols[0]), int(c.layouts.random_cols[1])),
            survivor_count_weights=tuple(float(w) for w in c.survivors.count_weights),
            mode_weights=tuple(float(getattr(c.modes, m)) for m in MODES),
            height_range=pair(c.camera.height_range),
            pitch_range=pair(c.camera.pitch_range),
            roll_jitter_deg=float(c.camera.roll_jitter_deg),
            yaw_jitter_deg=float(c.camera.yaw_jitter_deg),
            aimed_distance_m=pair(c.aimed.distance_m),
            aimed_distance_sampling=str(getattr(c.aimed, "distance_sampling", "uniform")),
            aimed_bearing_fraction=float(c.aimed.bearing_fraction_of_half_hfov),
            aimed_candidates=int(c.aimed.candidates_per_attempt),
            negative_candidates=int(c.negative.candidates_per_attempt),
            max_retries=int(c.retries.max_retries),
        )


def _include(v: Any) -> tuple[str, ...] | None:
    return None if v is None else tuple(str(name) for name in v)


def derive_seed(master_seed: int, scene_index: int, stream: str, *extra: int) -> int:
    """
    Seed of one random stream: 63 bits of sha256 over (master, scene, stream, extra).

    Distinct streams (layout, randomisation, survivors, poses, augmentation,
    ...) and distinct scenes get unrelated seeds, so no two share draws.
    """
    key = "|".join(str(v) for v in ("nidar-gen", master_seed, scene_index, stream, *extra))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 1


def handmade_layouts(cfg: GeneratorConfig) -> list[tuple[str, Path]]:
    """Hand-made layout files as (name, path), sorted by name."""
    paths = sorted(Path(cfg.handmade_dir).glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no layouts in {cfg.handmade_dir}")
    layouts = [(path.stem, path) for path in paths]
    if cfg.handmade_include is None:
        return layouts
    known = {name for name, _ in layouts}
    unknown = sorted(set(cfg.handmade_include) - known)
    if unknown or not cfg.handmade_include:
        raise ValueError(
            f"layouts.handmade_include {list(cfg.handmade_include)}: unknown {unknown} "
            f"(known: {sorted(known)})"
        )
    return [(name, path) for name, path in layouts if name in cfg.handmade_include]


def aimed_distance(cfg: GeneratorConfig, rng: random.Random) -> float:
    """One aimed distance draw: uniform or log-uniform over ``aimed_distance_m``."""
    lo, hi = cfg.aimed_distance_m
    if cfg.aimed_distance_sampling == "uniform":
        return rng.uniform(lo, hi)
    if cfg.aimed_distance_sampling == "log_uniform":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    raise ValueError(f"aimed.distance_sampling {cfg.aimed_distance_sampling!r}")


@dataclass(frozen=True)
class ScenePlan:
    """What a scene is made of, decided before any PyBullet call."""

    master_seed: int
    scene_index: int
    layout_name: str
    layout_source: str
    """``handmade`` or ``random``."""
    grid: tuple[str, ...]
    layout_seed: int | None
    """MazeGenerator seed for a random layout, else None."""
    n_survivors: int
    """Requested survivor count (SurvivorPlacer caps it at the free cells)."""
    seeds: dict[str, int] = field(default_factory=dict)
    """Scene stream seeds (:data:`SCENE_STREAMS`)."""

    @property
    def scene_id(self) -> str:
        """Scene id, also the group id for the train/val split."""
        return scene_id(self.scene_index)


def scene_id(scene_index: int) -> str:
    """``s00000``-style id of a scene index."""
    return f"s{scene_index:05d}"


def plan_scene(cfg: GeneratorConfig, master_seed: int, scene_index: int) -> ScenePlan:
    """Layout and survivor count of a scene: a pure function of its arguments."""
    seeds = {s: derive_seed(master_seed, scene_index, s) for s in SCENE_STREAMS}
    rng = random.Random(seeds["layout"])
    if rng.random() < cfg.random_layout_prob:
        rows = rng.randint(*cfg.random_rows)
        cols = rng.randint(*cfg.random_cols)
        gen = MazeGenerator(cfg.arena_config).generate(rows, cols, seed=seeds["maze"])
        grid = list(gen.grid)
        name, source, layout_seed = f"random_{gen.num_rows}x{gen.num_cols}", "random", seeds["maze"]
    else:
        layouts = handmade_layouts(cfg)
        name, path = layouts[rng.randrange(len(layouts))]
        grid, source, layout_seed = load_layout(path), "handmade", None
    weights = cfg.survivor_count_weights
    n = random.Random(seeds["count"]).choices(range(len(weights)), weights=weights)[0]
    return ScenePlan(
        master_seed=master_seed,
        scene_index=scene_index,
        layout_name=name,
        layout_source=source,
        grid=tuple(grid),
        layout_seed=layout_seed,
        n_survivors=n,
        seeds=seeds,
    )


# --------------------------------------------------------------------------- #
# Frame classification
# --------------------------------------------------------------------------- #
def classify(frame_labels: FrameLabels) -> str:
    """
    ``ambiguous`` if any present survivor was dropped, else ``positive`` with
    kept labels, else ``negative`` (no survivor present).
    """
    if frame_labels.dropped:
        return AMBIGUOUS
    if frame_labels.labels:
        return POSITIVE
    return NEGATIVE


def label_summary(label: SurvivorLabel, scale_416: float) -> dict[str, Any]:
    """A label's metadata, with its box size at training resolution."""
    x1, y1, x2, y2 = label.bbox_xyxy
    d = label.to_dict()
    d["box_w_at_imgsz"] = round((x2 - x1) * scale_416, 2)
    d["box_h_at_imgsz"] = round((y2 - y1) * scale_416, 2)
    return d


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
@dataclass
class RenderedAttempt:
    """One render of a candidate pose."""

    image: np.ndarray
    """Augmented BGR frame (what is trained on)."""
    labels: FrameLabels
    timing_ms: dict[str, float]


@dataclass
class FrameResult:
    """Outcome of one frame slot (:meth:`Scene.generate_frame`)."""

    frame_index: int
    mode: str
    accepted: bool
    attempts: list[dict[str, Any]]
    """Every attempt: seeds, pose, outcome, kept and dropped labels, reason, timing."""
    outcome: str | None = None
    """``positive`` or ``negative`` for an accepted frame."""
    pose: CameraPose | None = None
    image: np.ndarray | None = None
    labels: FrameLabels | None = None
    target: int | None = None
    """Aimed survivor index (aimed mode)."""
    skip_reason: str | None = None


class Scene:
    """
    A built scene: maze, textures and lighting, survivors, camera, labeller.

    Owns a new PyBullet DIRECT client and a :class:`SilhouetteRenderer`; both
    are closed by :meth:`close` (or the context manager).
    """

    def __init__(self, plan: ScenePlan, cfg: GeneratorConfig) -> None:
        # sample_poses logs at INFO on every call; the generator calls it per
        # attempt and records its own counts instead.
        logging.getLogger("nidar.trajectory").setLevel(logging.WARNING)
        self.plan = plan
        self.cfg = cfg
        self.maze: MazeResult = MazeBuilder(cfg.arena_config).build_from_grid(list(plan.grid))
        self.cid = self.maze.physics_client
        self.silhouettes: SilhouetteRenderer | None = None
        try:
            randomizer = SequenceRandomizer(cfg.randomization_config)
            self.params: SequenceParams = randomizer.sample_sequence_params(
                seed=plan.seeds["randomization"]
            )
            randomizer.apply_to_sim(self.maze, self.params, physics_client=self.cid)
            self.survivors: list[SurvivorInfo] = SurvivorPlacer(cfg.arena_config).place(
                self.maze, n=plan.n_survivors, seed=plan.seeds["survivors"]
            )
            self.survivor_boxes = [s.aabb for s in self.survivors]
            self.sampler = TrajectorySampler(
                list(plan.grid), cfg.arena_config, survivor_boxes=self.survivor_boxes
            )
            self.camera = PinholeCamera(cfg.arena_config, width=cfg.width, height=cfg.height)
            self.silhouettes = SilhouetteRenderer(
                self.camera, self.survivors, self.cid, arena_config_path=cfg.arena_config
            )
            self.augmentor = FrameAugmentor(cfg.randomization_config)
            self.label_config = replace(
                LabelConfig.from_arena_config(cfg.arena_config), imgsz=cfg.imgsz
            )
        except BaseException:
            self.close()
            raise
        self.scale_416 = cfg.imgsz / max(cfg.width, cfg.height)
        self.half_hfov_deg = math.degrees(
            math.atan((cfg.width / 2.0) / self.camera.focal_length_px)
        )
        self.used_keys: set[tuple[float, float, int]] = set()

    def close(self) -> None:
        """Close the silhouette renderer and disconnect the scene's client (idempotent)."""
        if self.silhouettes is not None:
            self.silhouettes.close()
            self.silhouettes = None
        if self.cid is not None:
            p.disconnect(self.cid)
            self.cid = None

    def __enter__(self) -> Scene:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Camera --

    def intrinsics(self) -> dict[str, float]:
        """Pinhole intrinsics of the render (square pixels, principal point at the centre)."""
        f = self.camera.focal_length_px
        return {
            "fx": f,
            "fy": f,
            "cx": self.cfg.width / 2.0,
            "cy": self.cfg.height / 2.0,
        }

    def project_point(self, pose: CameraPose, point: tuple[float, float, float]) -> tuple | None:
        """Pixel (u, v) of a world point, or None if it is not in front of the camera."""
        eye, fwd, up = camera_axes(pose)
        right = np.cross(fwd, up)
        v = np.asarray(point, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        depth = float(v @ fwd)
        if depth <= self.camera.near_clip:
            return None
        f = self.camera.focal_length_px
        u = self.cfg.width / 2.0 + f * float(v @ right) / depth
        vv = self.cfg.height / 2.0 - f * float(v @ up) / depth
        return (u, vv)

    def line_of_sight(self, pose: CameraPose, survivor_index: int) -> bool:
        """True if a ray from the eye to the survivor's box centre hits only that survivor."""
        lo, hi = self.survivor_boxes[survivor_index]
        centre = tuple((a + b) / 2.0 for a, b in zip(lo, hi, strict=True))
        hit = p.rayTest((pose.x, pose.y, pose.z), centre, physicsClientId=self.cid)[0]
        return hit[0] == -1 or hit[0] in self.survivors[survivor_index].body_ids

    # -- Candidate poses (no render) --

    def frame_mode(self, frame_index: int) -> str:
        """Mode of a frame slot; every frame of a survivor-free scene is negative."""
        if not self.survivors:
            return "negative"
        seed = derive_seed(self.plan.master_seed, self.plan.scene_index, "mode", frame_index)
        return random.Random(seed).choices(MODES, weights=self.cfg.mode_weights)[0]

    def aimed_candidate(self, seed: int) -> tuple[CameraPose | None, int, Counter[str]]:
        """A pose aimed at a seeded choice of survivor, or None within the candidate budget."""
        cfg = self.cfg
        rng = random.Random(seed)
        target = rng.randrange(len(self.survivors))
        lo, hi = self.survivor_boxes[target]
        cx, cy, cz = ((a + b) / 2.0 for a, b in zip(lo, hi, strict=True))
        max_offset = cfg.aimed_bearing_fraction * self.half_hfov_deg
        rejected: Counter[str] = Counter()
        for _ in range(cfg.aimed_candidates):
            dist = aimed_distance(cfg, rng)
            az = rng.uniform(0.0, 2.0 * math.pi)
            x, y = cx + dist * math.cos(az), cy + dist * math.sin(az)
            bearing = math.degrees(math.atan2(cy - y, cx - x))
            pose = CameraPose(
                x=x,
                y=y,
                z=rng.uniform(*cfg.height_range),
                yaw=(bearing + rng.uniform(-max_offset, max_offset)) % 360.0,
                pitch=rng.uniform(*cfg.pitch_range),
                roll=rng.uniform(-cfg.roll_jitter_deg, cfg.roll_jitter_deg),
            )
            reason = self.aimed_rejection(pose, target, (cx, cy, cz))
            if reason is None:
                return pose, target, rejected
            rejected[reason] += 1
        return None, target, rejected

    def aimed_rejection(
        self, pose: CameraPose, target: int, centre: tuple[float, float, float]
    ) -> str | None:
        """Why an aimed candidate is rejected, or None."""
        if pose_key(pose) in self.used_keys:
            return CAND_DUPLICATE
        reason = self.sampler.pose_rejection(pose)
        if reason is not None:
            return reason
        uv = self.project_point(pose, centre)
        if uv is None or not (0 <= uv[0] < self.cfg.width and 0 <= uv[1] < self.cfg.height):
            return CAND_TARGET_OFF_FRAME
        if not self.line_of_sight(pose, target):
            return CAND_LINE_OF_SIGHT
        return None

    def _sample(self, n: int, seed: int) -> list[CameraPose]:
        cfg = self.cfg
        try:
            return self.sampler.sample_poses(
                n,
                seed=seed,
                height_range=cfg.height_range,
                pitch_range=cfg.pitch_range,
                yaw_jitter_deg=cfg.yaw_jitter_deg,
                roll_jitter_deg=cfg.roll_jitter_deg,
            )
        except PoseSamplingError:
            return []

    def free_candidate(self, seed: int) -> tuple[CameraPose | None, Counter[str]]:
        """One :meth:`TrajectorySampler.sample_poses` pose, or None if it is a duplicate."""
        rejected: Counter[str] = Counter()
        for pose in self._sample(1, seed):
            if pose_key(pose) in self.used_keys:
                rejected[CAND_DUPLICATE] += 1
                continue
            return pose, rejected
        return None, rejected

    def negative_candidate(self, seed: int) -> tuple[CameraPose | None, Counter[str]]:
        """
        A sampled pose with no survivor likely visible: none whose 3D box
        projects into the frame with a clear ray to its centre. Survivors in
        view behind walls are allowed (hard negatives); the render decides.
        """
        rejected: Counter[str] = Counter()
        for pose in self._sample(self.cfg.negative_candidates, seed):
            if pose_key(pose) in self.used_keys:
                rejected[CAND_DUPLICATE] += 1
                continue
            if any(
                self.camera.project_aabb(pose, box) is not None and self.line_of_sight(pose, i)
                for i, box in enumerate(self.survivor_boxes)
            ):
                rejected[CAND_SURVIVOR_VISIBLE] += 1
                continue
            return pose, rejected
        return None, rejected

    def candidate(self, mode: str, seed: int) -> tuple[CameraPose | None, int | None, Counter]:
        """Candidate pose for a mode: (pose or None, aimed target or None, rejections)."""
        if mode == "aimed":
            return self.aimed_candidate(seed)
        if mode == "free":
            pose, rejected = self.free_candidate(seed)
        else:
            pose, rejected = self.negative_candidate(seed)
        return pose, None, rejected

    # -- Rendering and labelling --

    def render_attempt(self, pose: CameraPose, augment_seed: int) -> RenderedAttempt:
        """Render, augment, silhouettes and :func:`label_frame` for one pose."""
        prm = self.params
        t0 = time.perf_counter()
        out = self.camera.render(
            pose,
            physics_client=self.cid,
            light_direction=prm.light_direction,
            light_color=prm.light_color,
            light_distance=prm.light_distance,
            ambient_coeff=prm.ambient_coeff,
            diffuse_coeff=prm.diffuse_coeff,
            specular_coeff=prm.specular_coeff,
            shadow=prm.shadow,
        )
        t1 = time.perf_counter()
        image = self.augmentor.augment(out.bgr, is_low_light=prm.is_low_light, seed=augment_seed)
        t2 = time.perf_counter()
        assert self.silhouettes is not None
        sils = self.silhouettes.render(pose)
        t3 = time.perf_counter()
        labels = label_frame(out.seg_mask, self.survivors, sils, image, self.label_config)
        t4 = time.perf_counter()
        return RenderedAttempt(
            image=image,
            labels=labels,
            timing_ms={
                "render_ms": (t1 - t0) * 1000.0,
                "augment_ms": (t2 - t1) * 1000.0,
                "silhouette_ms": (t3 - t2) * 1000.0,
                "label_ms": (t4 - t3) * 1000.0,
            },
        )

    def generate_frame(self, frame_index: int) -> FrameResult:
        """
        Draw, render and label frame slot ``frame_index`` with up to
        ``max_retries`` retries. Only a ``positive`` or ``negative`` outcome
        accepted by the mode is returned as accepted; an ambiguous frame never is.
        """
        plan, cfg = self.plan, self.cfg
        mode = self.frame_mode(frame_index)
        attempts: list[dict[str, Any]] = []
        for attempt in range(cfg.max_retries + 1):
            pose_seed = derive_seed(
                plan.master_seed, plan.scene_index, "pose", frame_index, attempt
            )
            augment_seed = derive_seed(
                plan.master_seed, plan.scene_index, "augment", frame_index, attempt
            )
            pose, target, rejected = self.candidate(mode, pose_seed)
            record: dict[str, Any] = {
                "attempt": attempt,
                "pose_seed": pose_seed,
                "augment_seed": augment_seed,
                "target": target,
                "candidate_rejections": dict(sorted(rejected.items())),
            }
            attempts.append(record)
            if pose is None:
                record.update(outcome=None, reason=RETRY_NO_VALID_POSE)
                log.info(f"{plan.scene_id} f{frame_index} attempt {attempt}: no valid {mode} pose")
                continue
            self.used_keys.add(pose_key(pose))
            rendered = self.render_attempt(pose, augment_seed)
            outcome = classify(rendered.labels)
            if outcome == AMBIGUOUS:
                reason = RETRY_AMBIGUOUS
            elif outcome not in ACCEPTED_OUTCOMES[mode]:
                reason = RETRY_AIMED_NO_LABEL if mode == "aimed" else RETRY_NEGATIVE_HAS_SURVIVOR
            else:
                reason = None
            record.update(
                pose=pose_dict(pose),
                outcome=outcome,
                reason=reason,
                kept=[label_summary(lb, self.scale_416) for lb in rendered.labels.labels],
                dropped=[
                    {**label_summary(d.label, self.scale_416), "reasons": list(d.reasons)}
                    for d in rendered.labels.dropped
                ],
                timing=rendered.timing_ms,
            )
            if reason is None:
                return FrameResult(
                    frame_index=frame_index,
                    mode=mode,
                    accepted=True,
                    attempts=attempts,
                    outcome=outcome,
                    pose=pose,
                    image=rendered.image,
                    labels=rendered.labels,
                    target=target,
                )
            detail = ",".join(rendered.labels.ambiguous_reasons) if outcome == AMBIGUOUS else ""
            log.info(
                f"{plan.scene_id} f{frame_index} attempt {attempt} ({mode}): retry, "
                f"reason={reason} {detail}".rstrip()
            )
        skip = SKIP_REASONS[attempts[-1]["reason"]]
        log.info(
            f"{plan.scene_id} f{frame_index} ({mode}): skipped after {len(attempts)} attempts, "
            f"reason={skip}"
        )
        return FrameResult(
            frame_index=frame_index,
            mode=mode,
            accepted=False,
            attempts=attempts,
            skip_reason=skip,
        )


def pose_dict(pose: CameraPose) -> dict[str, float]:
    """Pose as a dict with full float precision (for replay)."""
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "yaw": pose.yaw,
        "pitch": pose.pitch,
        "roll": pose.roll,
    }


# --------------------------------------------------------------------------- #
# Writing a scene
# --------------------------------------------------------------------------- #
RUN_FILE = "run.json"
MANIFEST_FILE = "manifest.json"
META_FILE = "meta.jsonl"
ATTEMPTS_FILE = "attempts.jsonl"

PIXEL_CONVENTION = (
    "origin top-left, x right, y down; continuous pixel edges: pixel (i, j) spans "
    "[i, i+1] x [j, j+1], so a box over pixels xs.min()..xs.max() is "
    "[xs.min(), xs.max()+1]; principal point (cx, cy) = (W/2, H/2)"
)
POSE_CONVENTION = (
    "world x east, y north, z up (metres); yaw deg from +x towards +y; pitch deg, "
    "negative = down; roll deg about the optical axis; axes as sim.camera.camera_axes"
)


class RunMismatchError(RuntimeError):
    """The output directory holds a run made with other arguments or config."""


def frame_id(scene_index: int, frame_index: int) -> str:
    """``s00000_f000``-style id of a frame slot."""
    return f"{scene_id(scene_index)}_f{frame_index:03d}"


def _normalised_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def config_hash(cfg: GeneratorConfig) -> str:
    """sha256 of the effective generator config and the arena and randomization files."""
    h = hashlib.sha256()
    h.update(json.dumps(asdict(cfg), sort_keys=True).encode("utf-8"))
    h.update(_normalised_text(cfg.arena_config).encode("utf-8"))
    h.update(_normalised_text(cfg.randomization_config).encode("utf-8"))
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """JPEG bytes of a BGR image (OpenCV)."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def yolo_text(labels: list[SurvivorLabel]) -> str:
    """YOLO label file content (same format as :func:`sim.labels.save_yolo_labels`)."""
    return "".join(label.to_yolo_line() + "\n" for label in labels)


def _scene_files(out_dir: Path, sid: str) -> list[Path]:
    files = list((out_dir / "images").glob(f"{sid}_*"))
    files += list((out_dir / "labels").glob(f"{sid}_*"))
    files += list((out_dir / "scenes").glob(f"{sid}.*"))
    return files


def remove_scene_files(out_dir: Path, sid: str) -> None:
    """Remove every file of a scene (stale output of an interrupted or failed run)."""
    for path in _scene_files(out_dir, sid):
        path.unlink(missing_ok=True)


def scene_info(scene: Scene) -> dict[str, Any]:
    """Scene part of a meta line: layout, seeds, camera, light, survivors."""
    plan, cfg, prm, cam = scene.plan, scene.cfg, scene.params, scene.camera
    return {
        "scene_id": plan.scene_id,
        "group_id": plan.scene_id,
        "scene_index": plan.scene_index,
        "master_seed": plan.master_seed,
        "layout": {
            "name": plan.layout_name,
            "source": plan.layout_source,
            "seed": plan.layout_seed,
            "rows": len(plan.grid),
            "cols": len(plan.grid[0]),
            "cell_size_m": scene.maze.cell_size,
            "grid": list(plan.grid),
        },
        "scene_seeds": dict(plan.seeds),
        "camera": {
            "width": cfg.width,
            "height": cfg.height,
            "vfov_deg": cam.vertical_fov_deg,
            "hfov_deg": 2.0 * scene.half_hfov_deg,
            **scene.intrinsics(),
            "near_clip_m": cam.near_clip,
            "far_clip_m": cam.far_clip,
            "pixel_convention": PIXEL_CONVENTION,
            "pose_convention": POSE_CONVENTION,
        },
        "light": {
            "direction": prm.light_direction,
            "color": prm.light_color,
            "distance": prm.light_distance,
            "ambient_coeff": prm.ambient_coeff,
            "diffuse_coeff": prm.diffuse_coeff,
            "specular_coeff": prm.specular_coeff,
            "shadow": prm.shadow,
            "wall_texture": prm.wall_texture_type,
            "floor_texture": prm.floor_texture_type,
        },
        "low_light": prm.is_low_light,
        "survivors_requested": plan.n_survivors,
        "survivors": [
            {
                "index": i,
                "posture": s.posture,
                "cell": list(s.cell),
                "world_pos_m": list(s.world_pos),
                "yaw_deg": s.yaw_deg,
                "scale": s.scale,
                "aabb_m": [list(s.aabb[0]), list(s.aabb[1])],
            }
            for i, s in enumerate(scene.survivors)
        ],
    }


def frame_meta(scene: Scene, result: FrameResult, base: dict[str, Any]) -> dict[str, Any]:
    """Meta line of an accepted frame."""
    plan = scene.plan
    fid = frame_id(plan.scene_index, result.frame_index)
    final = result.attempts[-1]
    assert result.labels is not None and result.pose is not None
    return {
        **base,
        "frame_id": fid,
        "frame_index": result.frame_index,
        "image": f"images/{fid}.jpg",
        "label": f"labels/{fid}.txt",
        "jpeg_quality": scene.cfg.jpeg_quality,
        "frame_seeds": {
            "mode": derive_seed(plan.master_seed, plan.scene_index, "mode", result.frame_index),
            "pose": final["pose_seed"],
            "augment": final["augment_seed"],
        },
        "mode": result.mode,
        "outcome": result.outcome,
        "target": result.target,
        "retries": len(result.attempts) - 1,
        "retry_reasons": [a["reason"] for a in result.attempts[:-1]],
        "rejected_attempts": [
            {k: a.get(k) for k in ("attempt", "reason", "outcome", "pose", "kept", "dropped")}
            for a in result.attempts[:-1]
        ],
        "pose": pose_dict(result.pose),
        "labels": final["kept"],
        "dropped": final["dropped"],
        "timing": {
            **final["timing"],
            "all_attempts_ms": sum(sum(a.get("timing", {}).values()) for a in result.attempts),
        },
    }


def write_scene(
    plan: ScenePlan,
    cfg: GeneratorConfig,
    out_dir: Path,
    frames_per_scene: int,
    chash: str,
    frame_callback: Any = None,
) -> dict[str, Any]:
    """
    Generate and write one scene; returns its summary (also written as done.json).

    ``frame_callback(result, meta)`` is called for every accepted frame
    before it is written (tests compare in-memory and written labels with it).
    """
    sid = plan.scene_id
    remove_scene_files(out_dir, sid)
    t0, c0 = time.perf_counter(), time.process_time()
    metas: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    with Scene(plan, cfg) as scene:
        base = {**scene_info(scene), "config_hash": chash}
        for f in range(frames_per_scene):
            result = scene.generate_frame(f)
            for a in result.attempts:
                attempts.append(
                    {
                        "scene_id": sid,
                        "frame_index": f,
                        "mode": result.mode,
                        "frame_written": result.accepted,
                        "skip_reason": result.skip_reason,
                        "low_light": scene.params.is_low_light,
                        "n_survivors": len(scene.survivors),
                        **a,
                    }
                )
            if not result.accepted:
                skipped[result.skip_reason or "unknown"] += 1
                continue
            meta = frame_meta(scene, result, base)
            if frame_callback is not None:
                frame_callback(result, meta)
            assert result.image is not None and result.labels is not None
            atomic_write_bytes(out_dir / meta["image"], encode_jpeg(result.image, cfg.jpeg_quality))
            atomic_write_text(out_dir / meta["label"], yolo_text(result.labels.labels))
            metas.append(meta)
        n_placed = len(scene.survivors)
        low_light = scene.params.is_low_light

    phase: Counter[str] = Counter()
    for a in attempts:
        phase.update(a.get("timing", {}))
    summary = {
        "scene_id": sid,
        "scene_index": plan.scene_index,
        "status": "done",
        "layout": plan.layout_name,
        "survivors": n_placed,
        "low_light": low_light,
        "frames_written": len(metas),
        "positives": sum(m["outcome"] == POSITIVE for m in metas),
        "negatives": sum(m["outcome"] == NEGATIVE for m in metas),
        "labels": sum(len(m["labels"]) for m in metas),
        "written_by_mode": dict(sorted(Counter(m["mode"] for m in metas).items())),
        "skipped": dict(sorted(skipped.items())),
        "attempts": len(attempts),
        "render_attempts": sum(a.get("outcome") is not None for a in attempts),
        "retry_reasons": dict(
            sorted(Counter(a["reason"] for a in attempts if a["reason"] is not None).items())
        ),
        "timing": {
            "wall_s": time.perf_counter() - t0,
            "cpu_s": time.process_time() - c0,
            "pid": os.getpid(),
            "phase_ms_total": dict(phase),
        },
    }
    scenes_dir = out_dir / "scenes"
    atomic_write_text(scenes_dir / f"{sid}.jsonl", _jsonl(metas))
    atomic_write_text(scenes_dir / f"{sid}.attempts.jsonl", _jsonl(attempts))
    atomic_write_text(scenes_dir / f"{sid}.done.json", json.dumps(summary, sort_keys=True))
    log.info(
        f"{sid} done: {plan.layout_name}, {n_placed} survivors, {len(metas)}/{frames_per_scene} "
        f"frames, {summary['render_attempts']} renders, {summary['timing']['wall_s']:.1f} s"
    )
    return summary


@dataclass(frozen=True)
class SceneTask:
    """Everything a worker needs to write one scene."""

    cfg: GeneratorConfig
    out_dir: str
    master_seed: int
    scene_index: int
    frames_per_scene: int
    chash: str


def run_scene_task(task: SceneTask) -> dict[str, Any]:
    """
    Worker entry: write one scene, or record its failure.

    Any exception is caught: the traceback is logged and written to
    ``scenes/<scene>.failed.json``, the scene's partial files are removed and
    no done.json is written, so a re-run retries the scene.
    """
    out_dir = Path(task.out_dir)
    sid = scene_id(task.scene_index)
    try:
        plan = plan_scene(task.cfg, task.master_seed, task.scene_index)
        return write_scene(plan, task.cfg, out_dir, task.frames_per_scene, task.chash)
    except Exception as exc:  # noqa: BLE001 - one bad scene must not stop the run
        tb = traceback.format_exc()
        log.error(f"{sid} failed: {exc!r}\n{tb}")
        remove_scene_files(out_dir, sid)
        failure = {
            "scene_id": sid,
            "scene_index": task.scene_index,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }
        atomic_write_text(out_dir / "scenes" / f"{sid}.failed.json", json.dumps(failure))
        return failure


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def power_state() -> str:
    """Active power plan and AC state (Windows), e.g. ``power plan Balanced, plugged in (AC)``."""
    if platform.system() != "Windows":
        return "TBD (power plan only read on Windows)"
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        plan = out.split("(")[-1].rstrip(")") if "(" in out else out
        status = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Battery).BatteryStatus",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"TBD (could not read power state: {exc})"
    ac = {
        "": "no battery reported",
        "1": "on battery",
        "2": "plugged in (AC)",
        "3": "plugged in (AC, fully charged)",
        "6": "plugged in (AC, charging)",
    }.get(status, f"BatteryStatus={status}")
    return f"power plan {plan}, {ac}"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "TBD"


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "opencv-python", "pybullet", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_record(
    cfg: GeneratorConfig, scenes: int, frames_per_scene: int, seed: int, config_path: str
) -> dict[str, Any]:
    """What makes a run: compared on resume (the worker count is not part of it)."""
    return {
        "seed": seed,
        "scenes": scenes,
        "frames_per_scene": frames_per_scene,
        "config_path": str(config_path),
        "config_hash": config_hash(cfg),
        "generator_config": json.loads(json.dumps(asdict(cfg))),
    }


def _done_path(out: Path, index: int) -> Path:
    return out / "scenes" / f"{scene_id(index)}.done.json"


def _merge(out: Path, done: list[int], suffix: str, target: str) -> None:
    parts = [(out / "scenes" / f"{scene_id(i)}{suffix}").read_text(encoding="utf-8") for i in done]
    atomic_write_text(out / target, "".join(parts))


def _counter_sum(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        c.update(r[key])
    return c


def generate(
    out_dir: str | Path,
    scenes: int,
    frames_per_scene: int,
    seed: int,
    workers: int = 1,
    cfg: GeneratorConfig | None = None,
    config_path: str | Path = DEFAULT_GENERATOR_CONFIG,
) -> dict[str, Any]:
    """
    Generate (or resume) a run and return its manifest.

    Scenes with a done.json are skipped; the others (new, interrupted or
    failed earlier) are generated. ``workers`` > 1 uses a spawn process pool,
    one scene per task. Raises :class:`RunMismatchError` if ``out_dir`` holds
    a run with other arguments or config.
    """
    cfg = cfg if cfg is not None else GeneratorConfig.from_yaml(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = run_record(cfg, scenes, frames_per_scene, seed, str(config_path))
    run_path = out / RUN_FILE
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous != record:
            diff = sorted(k for k in record if previous.get(k) != record[k])
            raise RunMismatchError(
                f"{out} holds a run with other arguments or config (differs in: {diff}); "
                "use another --out"
            )
    else:
        atomic_write_text(run_path, json.dumps(record, indent=2, sort_keys=True))

    power_start = power_state()
    log.info(
        f"generator start: {out}, {scenes} scenes x {frames_per_scene} frames, seed {seed}, "
        f"workers {workers}; {power_start}"
    )
    pending = [i for i in range(scenes) if not _done_path(out, i).exists()]
    tasks = [
        SceneTask(cfg, str(out), seed, i, frames_per_scene, record["config_hash"]) for i in pending
    ]

    t0 = time.perf_counter()
    if workers <= 1:
        results = [run_scene_task(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(run_scene_task, tasks))
    wall_s = time.perf_counter() - t0
    power_end = power_state()

    done = [i for i in range(scenes) if _done_path(out, i).exists()]
    _merge(out, done, ".jsonl", META_FILE)
    _merge(out, done, ".attempts.jsonl", ATTEMPTS_FILE)
    summaries = [json.loads(_done_path(out, i).read_text(encoding="utf-8")) for i in done]
    failed = sorted(
        (
            {"scene_id": r["scene_id"], "error": r["error"]}
            for r in results
            if r["status"] == "failed"
        ),
        key=lambda r: r["scene_id"],
    )
    ran = [r for r in results if r["status"] == "done"]

    phase = _counter_sum([r["timing"] for r in ran], "phase_ms_total")
    renders_now = sum(r["render_attempts"] for r in ran)
    frames_now = sum(r["frames_written"] for r in ran)
    per_worker: dict[str, dict[str, float]] = {}
    for r in ran:
        w = per_worker.setdefault(
            str(r["timing"]["pid"]), {"scenes": 0, "cpu_s": 0.0, "busy_wall_s": 0.0}
        )
        w["scenes"] += 1
        w["cpu_s"] += r["timing"]["cpu_s"]
        w["busy_wall_s"] += r["timing"]["wall_s"]
    for w in per_worker.values():
        w["cpu_s_per_run_wall_s"] = w["cpu_s"] / wall_s if wall_s > 0 else 0.0

    manifest = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": record["config_hash"],
        "arguments": {
            "out": str(out),
            "scenes": scenes,
            "frames_per_scene": frames_per_scene,
            "seed": seed,
            "workers": workers,
            "config": str(config_path),
        },
        "generator_config": record["generator_config"],
        "library_versions": library_versions(),
        "counts": {
            "scenes_requested": scenes,
            "scenes_done": len(done),
            "scenes_resumed": scenes - len(pending),
            "scenes_run_now": len(ran),
            "failed_scenes": len(failed),
            "frame_slots": len(done) * frames_per_scene,
            "frames_written": sum(s["frames_written"] for s in summaries),
            "positives": sum(s["positives"] for s in summaries),
            "negatives": sum(s["negatives"] for s in summaries),
            "labels": sum(s["labels"] for s in summaries),
            "attempts": sum(s["attempts"] for s in summaries),
            "render_attempts": sum(s["render_attempts"] for s in summaries),
            "written_by_mode": dict(sorted(_counter_sum(summaries, "written_by_mode").items())),
            "skipped_by_reason": dict(sorted(_counter_sum(summaries, "skipped").items())),
            "retries_by_reason": dict(sorted(_counter_sum(summaries, "retry_reasons").items())),
        },
        "failed_scenes": failed,
        "timing": {
            "note": "this invocation only (resumed scenes excluded)",
            "power_start": power_start,
            "power_end": power_end,
            "workers": workers,
            "cpu_count": os.cpu_count(),
            "wall_s": wall_s,
            "frames_written": frames_now,
            "render_attempts": renders_now,
            "frames_per_s": frames_now / wall_s if wall_s > 0 else 0.0,
            "render_attempts_per_s": renders_now / wall_s if wall_s > 0 else 0.0,
            "phase_ms_mean_per_render_attempt": (
                {k: v / renders_now for k, v in sorted(phase.items())} if renders_now else {}
            ),
            "per_worker": per_worker,
        },
    }
    atomic_write_text(out / MANIFEST_FILE, json.dumps(manifest, indent=2, sort_keys=True))
    log.info(
        f"generator end: {len(done)}/{scenes} scenes done, {len(failed)} failed, "
        f"{frames_now} frames in {wall_s:.1f} s "
        f"({manifest['timing']['frames_per_s']:.2f} frames/s); {power_end}"
    )
    return manifest


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def plan_from_meta(meta: dict[str, Any]) -> ScenePlan:
    """The scene plan recorded in a meta line."""
    layout = meta["layout"]
    return ScenePlan(
        master_seed=int(meta["master_seed"]),
        scene_index=int(meta["scene_index"]),
        layout_name=layout["name"],
        layout_source=layout["source"],
        grid=tuple(layout["grid"]),
        layout_seed=layout["seed"],
        n_survivors=int(meta["survivors_requested"]),
        seeds={k: int(v) for k, v in meta["scene_seeds"].items()},
    )


def replay_frame(meta: dict[str, Any], cfg: GeneratorConfig) -> tuple[bytes, FrameLabels]:
    """
    Re-render a written frame from its meta line alone: JPEG bytes and labels.

    Uses the recorded layout grid, scene seeds, pose and augmentation seed;
    ``cfg`` must have the recorded config hash.
    """
    if config_hash(cfg) != meta["config_hash"]:
        raise RunMismatchError("config differs from the one recorded in the meta line")
    pose = CameraPose(**meta["pose"])
    with Scene(plan_from_meta(meta), cfg) as scene:
        rendered = scene.render_attempt(pose, int(meta["frame_seeds"]["augment"]))
    return encode_jpeg(rendered.image, int(meta["jpeg_quality"])), rendered.labels


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NIDAR synthetic dataset generator")
    parser.add_argument("--out", required=True, help="Run directory, e.g. data/sim_raw/run1")
    parser.add_argument("--scenes", type=int, required=True, help="Number of scenes")
    parser.add_argument("--frames-per-scene", type=int, required=True, help="Frame slots per scene")
    parser.add_argument("--seed", type=int, required=True, help="Master seed")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    parser.add_argument("--config", default=str(DEFAULT_GENERATOR_CONFIG), help="Generator config")
    args = parser.parse_args(argv)

    try:
        manifest = generate(
            args.out,
            scenes=args.scenes,
            frames_per_scene=args.frames_per_scene,
            seed=args.seed,
            workers=args.workers,
            config_path=args.config,
        )
    except RunMismatchError as exc:
        log.error(str(exc))
        return 2
    for failure in manifest["failed_scenes"]:
        log.error(f"failed scene {failure['scene_id']}: {failure['error']}")
    return 1 if manifest["failed_scenes"] else 0


if __name__ == "__main__":
    sys.exit(main())
