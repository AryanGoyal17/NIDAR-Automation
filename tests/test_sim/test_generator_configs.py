"""
tests/test_sim/test_generator_configs.py — train/val generator configs (Task 4b-1)

The train and val runs must draw from disjoint hand-made layouts that together
cover every layout in sim/layouts, and differ from generator.yaml only where
intended (layouts, aimed distance range and sampling).
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

pytest.importorskip("yaml")
pytest.importorskip("pybullet")  # sim.generator imports sim.maze

from sim.generator import GeneratorConfig  # noqa: E402

INTENDED = {"handmade_include", "aimed_distance_m", "aimed_distance_sampling"}


@pytest.fixture(scope="module")
def configs() -> dict[str, GeneratorConfig]:
    return {
        k: GeneratorConfig.from_yaml(f"configs/generator{s}.yaml")
        for k, s in (("base", ""), ("train", "_train"), ("val", "_val"))
    }


def test_layouts_disjoint_and_complete(configs) -> None:
    from sim.generator import handmade_layouts

    train = {n for n, _ in handmade_layouts(configs["train"])}
    val = {n for n, _ in handmade_layouts(configs["val"])}
    every = {n for n, _ in handmade_layouts(configs["base"])}
    assert train.isdisjoint(val)
    assert train | val == every
    assert val == {"loop", "t_junction"}


@pytest.mark.parametrize("kind", ["train", "val"])
def test_only_intended_differences(configs, kind: str) -> None:
    base, other = asdict(configs["base"]), asdict(configs[kind])
    changed = {k for k in base if base[k] != other[k]}
    assert changed == INTENDED
    assert configs[kind].aimed_distance_m == (0.6, 6.0)
    assert configs[kind].aimed_distance_sampling == "log_uniform"
