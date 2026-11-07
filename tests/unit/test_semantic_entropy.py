"""Unit tests for SE clustering and entropy math, using a fake NLI scorer.

Two estimators share one clustering:
  - likelihood-weighted SE (the default `semantic_entropy`, Farquhar et al.'s
    main estimator): p(c) is the normalized sum of its samples'
    length-normalized sequence likelihoods.
  - discrete SE (`discrete_semantic_entropy`): p(c) = |c| / N, needs no
    logprobs.
"""

import math

import numpy as np
import pytest

from hallucinations_math_lab.constants import DetectorName, SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet, TokenLogprob
from hallucinations_math_lab.services.detector_protocol import DetectionContext
from hallucinations_math_lab.services.semantic_entropy import (
    DiscreteSemanticEntropyDetector,
    SemanticEntropyDetector,
    cluster_by_bidirectional_entailment,
    compute_semantic_entropies,
    compute_semantic_entropy,
)

DISCRETE = SemanticEntropyEstimator.DISCRETE


class FakeNLI:
    """Bidirectional entailment iff the two strings, lowercased and stripped, are equal."""

    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


def _ss(answers: list[str]) -> SampleSet:
    return SampleSet(prompt="x", samples=[Sample(text=a) for a in answers])


def test_all_identical_answers_yield_one_cluster_and_zero_entropy():
    ss = _ss(["Paris"] * 10)
    res = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE)
    assert len(res.clusters) == 1
    assert res.semantic_entropy == pytest.approx(0.0, abs=1e-6)


def test_all_distinct_answers_yield_max_entropy():
    ss = _ss([f"answer-{i}" for i in range(8)])
    res = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE)
    assert len(res.clusters) == 8
    # Uniform over 8 clusters -> ln 8
    assert res.semantic_entropy == pytest.approx(math.log(8), abs=1e-4)


def test_mixed_clusters_match_shannon_formula():
    # 6 "Paris", 3 "London", 1 "Berlin"  → probs 0.6, 0.3, 0.1
    answers = ["Paris"] * 6 + ["London"] * 3 + ["Berlin"]
    ss = _ss(answers)
    res = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE)
    probs = np.array([0.6, 0.3, 0.1])
    expected = float(-np.sum(probs * np.log(probs)))
    assert res.semantic_entropy == pytest.approx(expected, abs=1e-4)
    assert {c.size for c in res.clusters} == {6, 3, 1}


def test_empty_sample_set_raises():
    with pytest.raises(ValueError):
        compute_semantic_entropy(
            SampleSet(prompt="x", samples=[]), scorer=FakeNLI(), estimator=DISCRETE
        )


def test_no_estimators_raises_before_any_nli_call():
    calls: list[tuple[str, str]] = []

    class CountingNLI(FakeNLI):
        def bidirectional_entails(self, a: str, b: str) -> bool:
            calls.append((a, b))
            return super().bidirectional_entails(a, b)

    with pytest.raises(ValueError, match="no estimators"):
        compute_semantic_entropies(_ss(["Paris", "Lyon"]), (), scorer=CountingNLI())
    assert calls == []


def test_clustering_preserves_member_indices():
    answers = ["a", "b", "a", "c", "a", "b"]
    clusters = cluster_by_bidirectional_entailment(answers, scorer=FakeNLI())
    # Find the "a" cluster
    a_cluster = next(c for c in clusters if c.representative_text == "a")
    assert sorted(a_cluster.member_indices) == [0, 2, 4]


class RecordingNLI(FakeNLI):
    """FakeNLI that records every (a, b) pair it is asked to judge."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.calls.append((a, b))
        return super().bidirectional_entails(a, b)


def test_nli_inputs_are_conditioned_on_the_question():
    # Farquhar et al.'s reference code prepends the question to every answer
    # before DeBERTa NLI, so short answers are judged in context.
    question = "What is the capital of France?"
    nli = RecordingNLI()
    ss = SampleSet(prompt=question, samples=[Sample(text="Paris"), Sample(text="Lyon")])
    compute_semantic_entropy(ss, scorer=nli, estimator=DISCRETE)
    assert nli.calls == [(f"{question} Paris", f"{question} Lyon")]


def test_cluster_representative_text_stays_the_bare_answer():
    question = "What is the capital of France?"
    ss = SampleSet(prompt=question, samples=[Sample(text="Paris"), Sample(text="Paris")])
    res = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE)
    assert [c.representative_text for c in res.clusters] == ["Paris"]


# --- likelihood-weighted SE ------------------------------------------------


def _weighted(answers_and_logprobs: list[tuple[str, list[float]]]) -> SampleSet:
    return SampleSet(
        prompt="q",
        samples=[
            Sample(text=a, token_logprobs=[TokenLogprob(token="t", logprob=lp) for lp in lps])
            for a, lps in answers_and_logprobs
        ],
    )


def test_single_cluster_scores_exactly_zero_for_both_estimators():
    ss = _weighted([("Paris", [-0.3])] * 4)
    for estimator in SemanticEntropyEstimator:
        res = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=estimator)
        assert res.semantic_entropy == 0.0


def test_equal_likelihoods_reduce_weighted_se_to_discrete_se():
    ss = _weighted([("Paris", [-0.5])] * 6 + [("London", [-0.5])] * 3 + [("Berlin", [-0.5])])
    weighted = compute_semantic_entropy(ss, scorer=FakeNLI()).semantic_entropy
    discrete = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE).semantic_entropy
    assert weighted == pytest.approx(discrete, abs=1e-9)


def test_weighted_se_matches_a_hand_computed_value():
    # Two "Paris" samples at mean logprob -0.1 and one "Lyon" at -2.0.
    ss = _weighted([("Paris", [-0.1]), ("Paris", [-0.1]), ("Lyon", [-2.0])])
    w = np.array([2 * math.exp(-0.1), math.exp(-2.0)])
    p = w / w.sum()
    expected = float(-np.sum(p * np.log(p)))
    res = compute_semantic_entropy(ss, scorer=FakeNLI())
    assert res.semantic_entropy == pytest.approx(expected, abs=1e-9)
    # The unlikely answer counts for less than its 1/3 share of samples.
    discrete = compute_semantic_entropy(ss, scorer=FakeNLI(), estimator=DISCRETE)
    assert res.semantic_entropy < discrete.semantic_entropy


def test_sequence_likelihood_is_length_normalized():
    short = _weighted([("Paris", [-1.0]), ("Lyon", [-1.0])])
    long = _weighted([("Paris", [-1.0, -1.0, -1.0]), ("Lyon", [-1.0])])
    a = compute_semantic_entropy(short, scorer=FakeNLI()).semantic_entropy
    b = compute_semantic_entropy(long, scorer=FakeNLI()).semantic_entropy
    assert a == pytest.approx(b) == pytest.approx(math.log(2))


def test_weighted_se_fails_fast_on_a_sample_without_logprobs():
    ss = SampleSet(
        prompt="q",
        samples=[Sample(text="Paris", token_logprobs=[TokenLogprob(token="t", logprob=-0.1)])]
        + [Sample(text="Lyon")],
    )
    with pytest.raises(ValueError, match="sample 1"):
        compute_semantic_entropy(ss, scorer=FakeNLI())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_weighted_se_fails_fast_on_non_finite_logprobs(bad):
    ss = _weighted([("Paris", [-0.1]), ("Lyon", [bad])])
    with pytest.raises(ValueError, match="non-finite"):
        compute_semantic_entropy(ss, scorer=FakeNLI())


def test_detector_names_identify_the_estimator():
    ss = _weighted([("Paris", [-0.1]), ("Lyon", [-2.0])])
    ctx = DetectionContext(prompt="q", samples=ss, pooled_hidden_state=None)
    weighted = SemanticEntropyDetector(scorer=FakeNLI()).score(ctx)
    discrete = DiscreteSemanticEntropyDetector(scorer=FakeNLI()).score(ctx)
    assert weighted.name is DetectorName.SEMANTIC_ENTROPY
    assert discrete.name is DetectorName.DISCRETE_SEMANTIC_ENTROPY
    assert discrete.score == pytest.approx(math.log(2))
    assert weighted.score < discrete.score
