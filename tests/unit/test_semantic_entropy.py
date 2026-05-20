"""Unit tests for SE clustering and entropy math, using a fake NLI scorer."""

import math

import numpy as np
import pytest

from seh_handler.custom_types.domain_types import Sample, SampleSet
from seh_handler.services.semantic_entropy import (
    cluster_by_bidirectional_entailment,
    compute_semantic_entropy,
)


class FakeNLI:
    """Bidirectional entailment iff the two strings, lowercased and stripped, are equal."""

    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


def _ss(answers: list[str]) -> SampleSet:
    return SampleSet(prompt="x", samples=[Sample(text=a) for a in answers])


def test_all_identical_answers_yield_one_cluster_and_zero_entropy():
    ss = _ss(["Paris"] * 10)
    res = compute_semantic_entropy(ss, scorer=FakeNLI())
    assert len(res.clusters) == 1
    assert res.semantic_entropy == pytest.approx(0.0, abs=1e-6)


def test_all_distinct_answers_yield_max_entropy():
    ss = _ss([f"answer-{i}" for i in range(8)])
    res = compute_semantic_entropy(ss, scorer=FakeNLI())
    assert len(res.clusters) == 8
    # Uniform over 8 clusters -> ln 8
    assert res.semantic_entropy == pytest.approx(math.log(8), abs=1e-4)


def test_mixed_clusters_match_shannon_formula():
    # 6 "Paris", 3 "London", 1 "Berlin"  → probs 0.6, 0.3, 0.1
    answers = ["Paris"] * 6 + ["London"] * 3 + ["Berlin"]
    ss = _ss(answers)
    res = compute_semantic_entropy(ss, scorer=FakeNLI())
    probs = np.array([0.6, 0.3, 0.1])
    expected = float(-np.sum(probs * np.log(probs)))
    assert res.semantic_entropy == pytest.approx(expected, abs=1e-4)
    assert {c.size for c in res.clusters} == {6, 3, 1}


def test_empty_sample_set_raises():
    with pytest.raises(ValueError):
        compute_semantic_entropy(SampleSet(prompt="x", samples=[]), scorer=FakeNLI())


def test_clustering_preserves_member_indices():
    answers = ["a", "b", "a", "c", "a", "b"]
    clusters = cluster_by_bidirectional_entailment(answers, scorer=FakeNLI())
    # Find the "a" cluster
    a_cluster = next(c for c in clusters if c.representative_text == "a")
    assert sorted(a_cluster.member_indices) == [0, 2, 4]
