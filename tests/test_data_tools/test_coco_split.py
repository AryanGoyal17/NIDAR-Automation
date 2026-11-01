"""
tests/test_data_tools/test_coco_split.py — Seeded COCO image split
==================================================================

Determinism, disjointness, stability when ids are added or removed. Synthetic ids only.
"""

from __future__ import annotations

import pytest

from data_tools.coco_common import GROUPS, split_group
from data_tools.coco_split import make_split

FRACTIONS = {"cutout_source": 0.4, "anchor": 0.4, "sanity": 0.2}


class TestSplit:
    def test_deterministic_and_disjoint(self):
        ids = list(range(1, 5001))
        a = make_split(ids, 2026, FRACTIONS)
        b = make_split(list(reversed(ids)), 2026, FRACTIONS)
        assert a == b
        sets = [set(a[g]) for g in GROUPS]
        assert sum(len(s) for s in sets) == len(ids)
        assert set().union(*sets) == set(ids)
        for i in range(3):
            for j in range(i + 1, 3):
                assert not sets[i] & sets[j]

    def test_shares_near_fractions(self):
        groups = make_split(list(range(1, 20001)), 2026, FRACTIONS)
        for g, f in FRACTIONS.items():
            assert abs(len(groups[g]) / 20000 - f) < 0.015

    def test_stable_when_ids_added_or_removed(self):
        full = make_split(list(range(1, 3001)), 7, FRACTIONS)
        part = make_split(list(range(500, 2001)), 7, FRACTIONS)
        where = {i: g for g, ids in full.items() for i in ids}
        assert all(where[i] == g for g, ids in part.items() for i in ids)

    def test_seed_changes_split(self):
        a = make_split(list(range(1, 1001)), 1, FRACTIONS)
        b = make_split(list(range(1, 1001)), 2, FRACTIONS)
        assert a != b

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError):
            split_group(1, 0, {"cutout_source": 0.5, "anchor": 0.5, "sanity": 0.2})
