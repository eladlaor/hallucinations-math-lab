"""Unit tests for length-normalized predictive entropy."""

import math

import pytest

from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet, TokenLogprob
from hallucinations_math_lab.services.predictive_entropy import length_normalized_predictive_entropy


def _mk_sample(logprobs: list[float]) -> Sample:
    return Sample(
        text="ans",
        token_logprobs=[TokenLogprob(token=f"t{i}", logprob=lp) for i, lp in enumerate(logprobs)],
    )


def test_identical_high_confidence_samples_give_low_entropy():
    samples = [_mk_sample([-0.01, -0.01, -0.01]) for _ in range(5)]
    ss = SampleSet(prompt="x", samples=samples)
    pe = length_normalized_predictive_entropy(ss)
    assert pe == pytest.approx(0.01, abs=1e-6)


def test_low_confidence_samples_give_higher_entropy_than_high_confidence():
    high = SampleSet(prompt="x", samples=[_mk_sample([-0.01] * 3) for _ in range(5)])
    low = SampleSet(prompt="x", samples=[_mk_sample([-2.0] * 3) for _ in range(5)])
    assert length_normalized_predictive_entropy(low) > length_normalized_predictive_entropy(high)


def test_empty_sample_set_raises():
    ss = SampleSet(prompt="x", samples=[])
    with pytest.raises(ValueError):
        length_normalized_predictive_entropy(ss)


def test_samples_without_logprobs_raise():
    ss = SampleSet(prompt="x", samples=[Sample(text="a"), Sample(text="b")])
    with pytest.raises(ValueError):
        length_normalized_predictive_entropy(ss)


def test_math_matches_expected_formula():
    # Two samples; one with mean logprob -1.0, one with -3.0.
    # PE = mean(-mean_logprob) = mean(1.0, 3.0) = 2.0
    s1 = _mk_sample([-1.0, -1.0])
    s2 = _mk_sample([-3.0, -3.0])
    ss = SampleSet(prompt="x", samples=[s1, s2])
    assert length_normalized_predictive_entropy(ss) == pytest.approx(2.0)
    # Sanity: not equal to log of something we didn't ask for
    assert not math.isclose(length_normalized_predictive_entropy(ss), 0.0)
