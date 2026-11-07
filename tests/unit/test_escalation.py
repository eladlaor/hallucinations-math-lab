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
  - Settings routes each escalation detector to the threshold on its own
    score scale (SEP -> SEP_THRESHOLD; SE, discrete SE, PE -> SE_THRESHOLD).
"""

from dataclasses import dataclass

import pytest

from hallucinations_math_lab import (
    AllAbove,
    AnyAbove,
    DetectorName,
    Sample,
    SampleSet,
    SingleDetector,
    TokenLogprob,
    evaluate_responses,
)
from hallucinations_math_lab.config import (
    Settings,
    escalation_threshold_env_var,
    escalation_threshold_field,
)
from hallucinations_math_lab.constants import (
    ESCALATION_THRESHOLDED_DETECTORS,
    DetectorCategory,
)
from hallucinations_math_lab.services.detector_protocol import (
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
    policy = AnyAbove({DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0})
    assert (
        policy.should_escalate(
            {DetectorName.SEMANTIC_ENTROPY: 0.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
        )
        is False
    )
    # only one crosses -> escalate
    assert (
        policy.should_escalate(
            {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
        )
        is True
    )
    assert (
        policy.should_escalate(
            {DetectorName.SEMANTIC_ENTROPY: 0.5, DetectorName.PREDICTIVE_ENTROPY: 2.5}
        )
        is True
    )
    assert policy.detectors_referenced == (
        DetectorName.SEMANTIC_ENTROPY,
        DetectorName.PREDICTIVE_ENTROPY,
    )


def test_all_above_and_semantics():
    policy = AllAbove({DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 2.0})
    # only one crosses -> no escalate
    assert (
        policy.should_escalate(
            {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 0.5}
        )
        is False
    )
    # both cross -> escalate
    assert (
        policy.should_escalate(
            {DetectorName.SEMANTIC_ENTROPY: 1.5, DetectorName.PREDICTIVE_ENTROPY: 2.5}
        )
        is True
    )


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
        samples=[Sample(text="x", token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])],
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
    policy = AnyAbove({DetectorName.SEMANTIC_ENTROPY: 1.0, DetectorName.PREDICTIVE_ENTROPY: 1.0})
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


# --- Settings: per-detector threshold routing ---


def test_escalation_threshold_for_routes_per_detector():
    """SEP outputs p(high SE), not nats, so it has its own threshold. SE,
    discrete SE and PE stay on SE_THRESHOLD. Anything else has no threshold."""
    settings = Settings(se_threshold=1.3, sep_threshold=0.25)

    assert settings.escalation_threshold_for(DetectorName.SEMANTIC_ENTROPY_PROBE) == 0.25
    for nats_detector in (
        DetectorName.SEMANTIC_ENTROPY,
        DetectorName.DISCRETE_SEMANTIC_ENTROPY,
        DetectorName.PREDICTIVE_ENTROPY,
    ):
        assert settings.escalation_threshold_for(nats_detector) == 1.3
    for unroutable in (DetectorName.LLAMA_JUDGE_QA, DetectorName.LLAMA_JUDGE_QA_SAMPLES):
        with pytest.raises(ValueError, match=str(unroutable)):
            settings.escalation_threshold_for(unroutable)


def test_escalation_threshold_env_var_sets_that_detectors_threshold(monkeypatch, tmp_path):
    """Setting the env var a detector's threshold is reported under changes that
    detector's threshold: a real env round-trip, not a re-derivation of the name."""
    monkeypatch.chdir(tmp_path)  # no .env file in reach
    assert escalation_threshold_env_var(DetectorName.SEMANTIC_ENTROPY_PROBE) == "SEP_THRESHOLD"
    for detector in ESCALATION_THRESHOLDED_DETECTORS:
        with monkeypatch.context() as patched:
            patched.setenv(escalation_threshold_env_var(detector), "0.37")
            settings = Settings()
            assert settings.escalation_threshold_for(detector) == 0.37


@pytest.mark.parametrize("detector", list(DetectorName))
def test_threshold_routing_covers_exactly_the_thresholded_detectors(detector):
    """A detector routes to a threshold field iff ESCALATION_THRESHOLDED_DETECTORS has it."""
    if detector in ESCALATION_THRESHOLDED_DETECTORS:
        assert escalation_threshold_field(detector) in Settings.model_fields
    else:
        with pytest.raises(ValueError, match=str(detector)):
            escalation_threshold_field(detector)


def test_sep_threshold_defaults_to_half():
    assert Settings.model_fields["sep_threshold"].default == 0.5


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_sep_threshold_bounds(bad: float):
    """SEP's score is a probability, so its threshold must lie in [0, 1]."""
    with pytest.raises(ValueError, match="sep_threshold"):
        Settings(sep_threshold=bad)


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_sep_threshold_accepts_closed_unit_interval(edge: float):
    assert Settings(sep_threshold=edge).sep_threshold == edge


def test_settings_rejects_sep_escalation_without_sep_enabled():
    with pytest.raises(ValueError, match="semantic_entropy_probe"):
        Settings(
            enabled_detectors=[DetectorName.PREDICTIVE_ENTROPY, DetectorName.SEMANTIC_ENTROPY],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
        )


def test_settings_accepts_sep_escalation_with_sep_enabled():
    settings = Settings(
        enabled_detectors=[DetectorName.SEMANTIC_ENTROPY_PROBE],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    assert settings.escalation_detector == DetectorName.SEMANTIC_ENTROPY_PROBE


@pytest.mark.parametrize(
    "unroutable", [DetectorName.LLAMA_JUDGE_QA, DetectorName.LLAMA_JUDGE_QA_SAMPLES]
)
def test_settings_rejects_escalation_detector_without_threshold(unroutable: DetectorName):
    """A detector with no routable threshold fails at config load, not at the
    first /status call or evaluator build."""
    with pytest.raises(ValueError, match=f"No escalation threshold .* detector {unroutable!s}"):
        Settings(
            enabled_detectors=[DetectorName.SEMANTIC_ENTROPY, unroutable],
            escalation_detector=unroutable,
        )
