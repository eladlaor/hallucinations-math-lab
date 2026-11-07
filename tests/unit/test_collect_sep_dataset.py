"""SEP dataset collection: per-question SE labels.

Contract pinned here (SEP_CLASSIFIER.md, "Which SE goes into the label?"):
  - The samples are clustered once; discrete and likelihood-weighted SE are both
    computed from that one clustering, so no extra NLI calls are made.
  - The row carries both values, the per-sample cluster ids and the cluster count.
"""

import math

import pytest

from eval.collect_sep_dataset import cluster_ids_per_sample, label_sample_set
from hallucinations_math_lab.custom_types.domain_types import (
    Cluster,
    Sample,
    SampleSet,
    TokenLogprob,
)
from hallucinations_math_lab.services.semantic_entropy import (
    cluster_by_bidirectional_entailment,
    compute_semantic_entropy,
)


class CountingNLI:
    """Bidirectional entailment iff the two strings are equal; counts every call."""

    def __init__(self) -> None:
        self.calls = 0

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.calls += 1
        return a == b


def _sample(text: str, logprob: float) -> Sample:
    return Sample(text=text, token_logprobs=[TokenLogprob(token="t", logprob=logprob)])


# 6 "Paris", 3 "London", 1 "Berlin", with unequal likelihoods so the two estimators differ.
_SAMPLE_SET = SampleSet(
    prompt="What is the capital of France?",
    samples=[
        _sample("Paris", -0.1),
        _sample("London", -2.0),
        _sample("Paris", -0.2),
        _sample("Berlin", -3.0),
        _sample("Paris", -0.1),
        _sample("London", -1.5),
        _sample("Paris", -0.3),
        _sample("Paris", -0.1),
        _sample("London", -2.5),
        _sample("Paris", -0.2),
    ],
)


def test_collector_row_has_both_estimators_from_one_clustering() -> None:
    one_clustering = CountingNLI()
    cluster_by_bidirectional_entailment(
        [s.text for s in _SAMPLE_SET.samples], scorer=one_clustering, question=_SAMPLE_SET.prompt
    )

    nli = CountingNLI()
    labels = label_sample_set(_SAMPLE_SET, scorer=nli)

    assert nli.calls == one_clustering.calls
    assert labels.se_discrete == pytest.approx(
        -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
    )
    weighted = compute_semantic_entropy(_SAMPLE_SET, scorer=CountingNLI()).semantic_entropy
    assert labels.se_likelihood_weighted == pytest.approx(weighted)
    assert labels.se_likelihood_weighted != pytest.approx(labels.se_discrete)
    assert labels.cluster_ids == [0, 1, 0, 2, 0, 1, 0, 0, 1, 0]
    assert labels.n_clusters == 3


def test_collector_validates_logprobs_before_any_nli_call() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="Paris"), Sample(text="Lyon")])
    nli = CountingNLI()
    with pytest.raises(ValueError, match="token logprobs"):
        label_sample_set(sample_set, scorer=nli)
    assert nli.calls == 0


def test_cluster_ids_reject_a_sample_in_no_cluster() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="a"), Sample(text="b")])
    clusters = [Cluster(member_indices=[0], representative_text="a")]
    with pytest.raises(ValueError, match=r"Samples \[1\] are in no cluster"):
        cluster_ids_per_sample(clusters, sample_set)


def test_cluster_ids_reject_a_sample_in_two_clusters() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="a"), Sample(text="b")])
    clusters = [
        Cluster(member_indices=[0, 1], representative_text="a"),
        Cluster(member_indices=[1], representative_text="b"),
    ]
    with pytest.raises(ValueError, match="Sample 1 is in clusters 0 and 1"):
        cluster_ids_per_sample(clusters, sample_set)
