"""Escalation policy unit tests.

Pin the behavior of the pluggable escalation policies independently of any
detector math, plus the precedence + validation contract of how
`evaluate_responses` consumes a supplied policy:
  - SingleDetector reproduces the legacy `score > threshold` decision.
  - AnyAbove is OR-semantics; AllAbove is AND-semantics.
  - Empty threshold mappings fail fast (ValueError).
  - describe() is deterministic.
  - A supplied policy takes precedence over the scalar escalation pair.
  - A policy referencing an unscored detector fails fast.
"""

from dataclasses import dataclass

import pytest

from taste import (
    AllAbove,
    AnyAbove,
    DetectorName,
    Sample,
    SampleSet,
    SingleDetector,
    TokenLogprob,
    evaluate_responses,
)
from taste.constants import DetectorCategory
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorScore,
)

_TEST_METADATA = DetectorMetadata(
    signal="test double",
    assumption="for test isolation only",
    failure_modes=("never run in production",),
    category=DetectorCategory.UNCERTAINTY,
)


@dataclass
class _ConstScoreDetector(Detector):
    """Test double — declares prerequisites and returns a fixed score."""

    name: DetectorName
    requires_samples: bool
    requires_hidden_state: bool
    fixed_score: float = 0.0
    metadata: DetectorMetadata = _TEST_METADATA

    def score(self, ctx: DetectionContext) -> DetectorScore:
        return DetectorScore(
            name=self.name,
            score=self.fixed_score,
            cost_usd=0.0,
            latency_ms=0.0,
        )


# --- Policy logic ---


@pytest.mark.parametrize("score, expected", [(0.4, False), (0.5, False), (0.6, True)])
def test_single_detector_matches_legacy_behavior(score, expected):
    policy = SingleDetector(DetectorName.SEMANTIC_ENTROPY, 0.5)
    scores = {DetectorName.SEMANTIC_ENTROPY: score}
    assert policy.should_escalate(scores) is (score > 0.5)
    assert policy.should_escalate(scores) is expected


def test_any_above_or_semantics():
    policy = AnyAbove(
        {DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0}
    )
    assert policy.should_escalate(
        {DetectorName.SEMANTIC_ENTROPY: 0.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
    ) is False
    # only one crosses -> escalate
    assert policy.should_escalate(
        {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
    ) is True
    assert policy.should_escalate(
        {DetectorName.SEMANTIC_ENTROPY: 0.5, DetectorName.PREDICTIVE_ENTROPY: 2.5}
    ) is True
    assert policy.detectors_referenced == (
        DetectorName.SEMANTIC_ENTROPY,
        DetectorName.PREDICTIVE_ENTROPY,
    )


def test_all_above_and_semantics():
    policy = AllAbove(
        {DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0}
    )
    # only one crosses -> no escalate
    assert policy.should_escalate(
        {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
    ) is False
    # both cross -> escalate
    assert policy.should_escalate(
        {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 2.5}
    ) is True


def test_empty_thresholds_raises():
    with pytest.raises(ValueError, match="AnyAbove requires at least one"):
        AnyAbove({})
    with pytest.raises(ValueError, match="AllAbove requires at least one"):
        AllAbove({})


def test_describe_strings_are_deterministic():
    assert SingleDetector(DetectorName.SEMANTIC_ENTROPY, 1.0).describe() == "semantic_entropy > 1.0"
    any_policy = AnyAbove(
        {DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0}
    )
    assert any_policy.describe() == "ANY(semantic_entropy > 1.0, predictive_entropy > 2.0)"
    assert any_policy.describe() == any_policy.describe()
    all_policy = AllAbove(
        {DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0}
    )
    assert all_policy.describe() == "ALL(semantic_entropy > 1.0, predictive_entropy > 2.0)"


# --- Integration with evaluate_responses ---


def _samples() -> SampleSet:
    return SampleSet(
        prompt="q",
        samples=[
            Sample(text="x", token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])
        ],
    )


def test_evaluate_responses_policy_takes_precedence():
    """With both a policy and the scalar pair supplied, the policy decides.
    Here PE=5.0 crosses the policy's 1.0 trigger even though SE (the scalar
    escalation detector) is well below its 99.0 threshold."""
    se = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY,
        requires_samples=True,
        requires_hidden_state=False,
        fixed_score=0.0,
    )
    pe = _ConstScoreDetector(
        name=DetectorName.PREDICTIVE_ENTROPY,
        requires_samples=True,
        requires_hidden_state=False,
        fixed_score=5.0,
    )
    policy = AnyAbove(
        {DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 1.0}
    )
    result = evaluate_responses(
        prompt="q",
        samples=_samples(),
        detectors=[se, pe],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=99.0,  # would never fire on its own
        greedy_answer="x",
        escalation_policy=policy,
    )
    assert result.escalated is True
    assert result.escalation_policy_desc == policy.describe()
    # scalar baseline still recorded
    assert result.escalation_detector == DetectorName.SEMANTIC_ENTROPY
    assert result.escalation_threshold == 99.0


def test_evaluate_responses_missing_referenced_detector_raises():
    se = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY,
        requires_samples=True,
        requires_hidden_state=False,
        fixed_score=0.0,
    )
    # Policy references PE, which is not in the detector list / scores.
    policy = AnyAbove({DetectorName.PREDICTIVE_ENTROPY: 1.0})
    with pytest.raises(ValueError, match="predictive_entropy"):
        evaluate_responses(
            prompt="q",
            samples=_samples(),
            detectors=[se],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=0.5,
            greedy_answer="x",
            escalation_policy=policy,
        )


def test_legacy_path_leaves_policy_desc_none():
    """When no escalation_policy is supplied, escalation_policy_desc is None
    (back-compat single-detector path)."""
    se = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY,
        requires_samples=True,
        requires_hidden_state=False,
        fixed_score=0.0,
    )
    result = evaluate_responses(
        prompt="q",
        samples=_samples(),
        detectors=[se],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=0.5,
        greedy_answer="x",
    )
    assert result.escalation_policy_desc is None
    assert result.escalated is False
