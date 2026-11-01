"""
sim/generator.py — Synthetic training dataset generator
=======================================================

Turns the simulator into training frames: for each scene, a maze (hand-made or
seeded random), randomised textures and lighting, 0-3 procedural survivors,
and K frames, each rendered, augmented and labelled with
:func:`sim.labels.label_frame`.

Frame modes (weights in ``configs/generator.yaml``):

- ``aimed``: the camera is placed 0.6-3.5 m from a chosen survivor, with a
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
disconnected at the end (this also frees the scene's textures).

Usage::

    from sim.generator import GeneratorConfig, Scene, plan_scene
    cfg = GeneratorConfig.from_yaml()
    with Scene(plan_scene(cfg, master_seed=7, scene_index=0), cfg) as scene:
        frame = scene.generate_frame(0)
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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
            aimed_bearing_fraction=float(c.aimed.bearing_fraction_of_half_hfov),
            aimed_candidates=int(c.aimed.candidates_per_attempt),
            negative_candidates=int(c.negative.candidates_per_attempt),
            max_retries=int(c.retries.max_retries),
        )


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
    return [(path.stem, path) for path in paths]


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
            dist = rng.uniform(*cfg.aimed_distance_m)
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
