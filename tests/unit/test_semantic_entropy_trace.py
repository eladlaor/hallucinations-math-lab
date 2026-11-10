"""The opt-in SE trace: the real clustering run, recorded one NLI comparison at a time.

The trace is what the playground replays. It must record exactly the
comparisons the clustering ran, in order, and must never change the clusters
or, when off, the NLI calls.
"""

import math

import pytest

from hallucinations_math_lab.constants import DetectorName, NLILabel, SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import (
    Cluster,
    ClusteringTrace,
    Sample,
    SampleSet,
    TokenLogprob,
)
from hallucinations_math_lab.evaluator import evaluate_responses
from hallucinations_math_lab.services.detector_protocol import DetectionContext, Detector
from hallucinations_math_lab.services.nli import DirectionalEntailmentScorer, NLIScorer
from hallucinations_math_lab.services.semantic_entropy import (
    DiscreteSemanticEntropyDetector,
    SemanticEntropyDetector,
    cluster_by_bidirectional_entailment,
    cluster_with_trace,
    compute_semantic_entropy,
)

ENTAILMENT = NLILabel.ENTAILMENT
NEUTRAL = NLILabel.NEUTRAL


class DirectionalNLI:
    """Entailment, in either direction, iff the texts are equal after lowercasing.

    Counts bidirectional and per-direction calls separately, so a test can
    tell which path the clustering took.
    """

    def __init__(self) -> None:
        self.bidirectional_calls = 0
        self.directional_calls: list[tuple[str, str]] = []

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        self.directional_calls.append((premise, hypothesis))
        return ENTAILMENT if premise.strip().lower() == hypothesis.strip().lower() else NEUTRAL

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.bidirectional_calls += 1
        return a.strip().lower() == b.strip().lower()


class OneWayNLI(DirectionalNLI):
    """A short answer entails its longer form, but not the reverse ("Paris" vs "Paris, France")."""

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        self.directional_calls.append((premise, hypothesis))
        return ENTAILMENT if premise in hypothesis else NEUTRAL

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.bidirectional_calls += 1
        return a in b and b in a


class BidirectionalOnlyNLI:
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a == b


def replay(trace: ClusteringTrace, answers: list[str]) -> list[Cluster]:
    """Rebuild the clusters from the trace's placements alone, as the UI does."""
    clusters: list[Cluster] = []
    for placement in trace.placements:
        if placement.opened_new_cluster:
            assert placement.cluster_index == len(clusters)
            clusters.append(
                Cluster(
                    member_indices=[placement.answer_index],
                    representative_text=answers[placement.answer_index],
                )
            )
        else:
            clusters[placement.cluster_index].member_indices.append(placement.answer_index)
    return clusters


ANSWERS = ["Paris", "Lyon", "paris", "Nice", "Lyon", "Paris"]


def test_the_trace_replays_to_exactly_the_returned_clusters() -> None:
    clusters, trace = cluster_with_trace(ANSWERS, scorer=DirectionalNLI())

    assert replay(trace, ANSWERS) == clusters


def test_tracing_never_changes_the_clusters() -> None:
    traced, _ = cluster_with_trace(ANSWERS, scorer=DirectionalNLI(), question="Q?")

    assert traced == cluster_by_bidirectional_entailment(
        ANSWERS, scorer=DirectionalNLI(), question="Q?"
    )


def test_the_trace_records_every_comparison_that_ran_in_order() -> None:
    _, trace = cluster_with_trace(["Paris", "Lyon", "paris"], scorer=DirectionalNLI())

    rows = [
        (c.answer_index, c.cluster_index, c.representative_index, c.forward, c.backward, c.joined)
        for c in trace.comparisons
    ]
    # Answer 1 fails against cluster 0 and opens cluster 1; answer 2 joins
    # cluster 0 and is never compared with cluster 1 (the search stops there).
    assert rows == [
        (1, 0, 0, NEUTRAL, None, False),
        (2, 0, 0, ENTAILMENT, ENTAILMENT, True),
    ]
    assert [(p.answer_index, p.cluster_index, p.opened_new_cluster) for p in trace.placements] == [
        (0, 0, True),
        (1, 1, True),
        (2, 0, False),
    ]


def test_one_way_entailment_does_not_join() -> None:
    """Strict: both directions must be entailment, exactly as bidirectional_entails."""
    nli = OneWayNLI()
    clusters, trace = cluster_with_trace(["Paris", "Paris, France"], scorer=nli)

    assert len(clusters) == 2
    assert trace.comparisons[0].forward == ENTAILMENT
    assert trace.comparisons[0].backward == NEUTRAL
    assert trace.comparisons[0].joined is False
    assert clusters == cluster_by_bidirectional_entailment(
        ["Paris", "Paris, France"], scorer=OneWayNLI()
    )


def test_the_trace_judges_question_prefixed_answers() -> None:
    nli = DirectionalNLI()
    cluster_with_trace(["Paris", "Lyon"], scorer=nli, question="Capital?")

    assert nli.directional_calls == [("Capital? Paris", "Capital? Lyon")]


def test_untraced_clustering_makes_only_bidirectional_calls() -> None:
    nli = DirectionalNLI()
    cluster_by_bidirectional_entailment(ANSWERS, scorer=nli)

    # Answers 1..5 against the clusters existing so far, stopping at a join:
    # 1 + 1 + 2 + 2 + 1.
    assert nli.bidirectional_calls == 7
    assert nli.directional_calls == []


def test_tracing_needs_a_per_direction_scorer() -> None:
    with pytest.raises(TypeError, match="classify"):
        cluster_with_trace(["a", "b"], scorer=BidirectionalOnlyNLI())


def test_a_single_answer_opens_one_cluster_with_no_comparisons() -> None:
    nli = DirectionalNLI()
    clusters, trace = cluster_with_trace(["x"], scorer=nli)

    assert [c.member_indices for c in clusters] == [[0]]
    assert trace.comparisons == []
    assert [(p.answer_index, p.cluster_index, p.opened_new_cluster) for p in trace.placements] == [
        (0, 0, True)
    ]
    assert nli.directional_calls == []


NLI_FAILURE = "NLI model crashed"


class FailingNLI(DirectionalNLI):
    """The per-direction verdict fails, as a crashed or unloadable NLI model does."""

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        raise RuntimeError(NLI_FAILURE)


def test_an_nli_error_mid_trace_aborts_the_clustering() -> None:
    with pytest.raises(RuntimeError, match=NLI_FAILURE):
        cluster_with_trace(["Paris", "Lyon"], scorer=FailingNLI())


def test_tracing_an_empty_answer_list_raises() -> None:
    with pytest.raises(ValueError):
        cluster_with_trace([], scorer=DirectionalNLI())


def _sample_set(answers_and_logprob: list[tuple[str, float | None]]) -> SampleSet:
    return SampleSet(
        prompt="q",
        samples=[
            Sample(
                text=a,
                token_logprobs=[] if lp is None else [TokenLogprob(token="t", logprob=lp)],
            )
            for a, lp in answers_and_logprob
        ],
    )


def test_weighted_se_trace_carries_the_likelihoods_and_cluster_shares() -> None:
    ss = _sample_set([("Paris", -0.1), ("Lyon", -3.0), ("Paris", -0.2)])

    result = compute_semantic_entropy(ss, scorer=DirectionalNLI(), trace=True)

    assert result.trace is not None
    assert result.trace.estimator == SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED
    likelihoods = [math.exp(-0.1), math.exp(-3.0), math.exp(-0.2)]
    assert result.trace.answer_likelihoods == pytest.approx(likelihoods)
    total = sum(likelihoods)
    p = [(likelihoods[0] + likelihoods[2]) / total, likelihoods[1] / total]
    assert result.trace.cluster_probabilities == pytest.approx(p)
    assert result.semantic_entropy == pytest.approx(-sum(x * math.log(x) for x in p))
    assert replay(result.trace.clustering, ["Paris", "Lyon", "Paris"]) == result.clusters


def test_discrete_se_trace_uses_cluster_sizes_and_no_likelihoods() -> None:
    ss = _sample_set([("Paris", None), ("Lyon", None), ("Paris", None)])

    result = compute_semantic_entropy(
        ss, scorer=DirectionalNLI(), estimator=SemanticEntropyEstimator.DISCRETE, trace=True
    )

    assert result.trace is not None
    assert result.trace.estimator == SemanticEntropyEstimator.DISCRETE
    assert result.trace.answer_likelihoods is None
    assert result.trace.cluster_probabilities == pytest.approx([2 / 3, 1 / 3])


def test_se_without_trace_returns_no_trace() -> None:
    ss = _sample_set([("Paris", -0.1), ("Lyon", -3.0)])
    nli = DirectionalNLI()

    result = compute_semantic_entropy(ss, scorer=nli)

    assert result.trace is None
    assert nli.directional_calls == []


@pytest.mark.parametrize("detector_cls", [SemanticEntropyDetector, DiscreteSemanticEntropyDetector])
@pytest.mark.parametrize("trace", [True, False])
def test_detectors_trace_only_when_the_context_asks(
    detector_cls: type[SemanticEntropyDetector], trace: bool
) -> None:
    ss = _sample_set([("Paris", -0.1), ("Lyon", -3.0)])
    ctx = DetectionContext(
        prompt="q", samples=ss, pooled_hidden_state=None, trace_semantic_entropy=trace
    )

    score = detector_cls(scorer=DirectionalNLI()).score(ctx)

    assert (score.se_trace is not None) is trace
    if score.se_trace is not None:
        assert score.se_trace.estimator == detector_cls.estimator


@pytest.mark.parametrize(
    ("escalation_detector", "expected_estimator"),
    [
        (DetectorName.SEMANTIC_ENTROPY, SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED),
        (DetectorName.DISCRETE_SEMANTIC_ENTROPY, SemanticEntropyEstimator.DISCRETE),
    ],
)
def test_with_both_se_estimators_the_trace_follows_the_escalation_detector(
    escalation_detector: DetectorName, expected_estimator: SemanticEntropyEstimator
) -> None:
    """The replay must end on the SE value that drove escalation, whatever the detector order."""
    ss = _sample_set([("Paris", -0.1), ("Lyon", -3.0)])
    orders: tuple[list[Detector], ...] = (
        [
            SemanticEntropyDetector(scorer=DirectionalNLI()),
            DiscreteSemanticEntropyDetector(scorer=DirectionalNLI()),
        ],
        [
            DiscreteSemanticEntropyDetector(scorer=DirectionalNLI()),
            SemanticEntropyDetector(scorer=DirectionalNLI()),
        ],
    )
    for detectors in orders:
        result = evaluate_responses(
            "q",
            ss,
            detectors=detectors,
            escalation_detector=escalation_detector,
            escalation_threshold=1.0,
            greedy_answer="Paris",
            trace_semantic_entropy=True,
        )

        assert result.se_trace is not None
        assert result.se_trace.estimator == expected_estimator


def test_the_deberta_scorer_can_be_traced() -> None:
    """NLIScorer already labels one direction at a time (constructing it loads nothing)."""
    scorer = NLIScorer(model_id="some/nli-model", revision="0" * 40)

    assert isinstance(scorer, DirectionalEntailmentScorer)
