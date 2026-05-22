"""DetectorAgent orchestration tests.

These pin the orchestrator's contract, independent of any single detector's
math (which is covered in test_semantic_entropy.py and test_predictive_entropy.py):
  - Iterates the configured detector list and collects all scores.
  - Records the escalation detector + threshold on the result.
  - Flips the escalated flag at the configured threshold.
  - Skips SLM sampling when no enabled detector requires samples.
  - Fails fast when a detector requires hidden states but the SLM provider
    returns None (e.g. vLLM without an embed sibling).
  - Refuses to construct if escalation_detector is not in enabled_detectors.

The detector list is injected directly so the tests don't depend on the
config-driven default registry."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from taste.agents.detector_agent import DetectorAgent
from taste.constants import DetectorName
from taste.custom_types.domain_types import Sample, SampleSet, TokenLogprob
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorScore,
)
from taste.services.predictive_entropy import PredictiveEntropyDetector
from taste.services.semantic_entropy import SemanticEntropyDetector


class FakeSLM:
    """In-memory SLMProvider double. `hidden_state` defaults to None (matches
    a vLLM provider without an embed sibling); pass a tensor to simulate a
    SEP-capable provider."""

    model_id = "fake-slm"

    def __init__(self, answers: list[str], hidden_state=None) -> None:
        self._answers = answers
        self._hidden_state = hidden_state
        self.sample_calls = 0
        self.hidden_state_calls = 0

    def sample(self, prompt: str) -> SampleSet:
        self.sample_calls += 1
        return SampleSet(
            prompt=prompt,
            samples=[
                Sample(text=a, token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])
                for a in self._answers
            ],
        )

    def greedy(self, prompt: str) -> str:
        return self._answers[0]

    def pooled_hidden_state(self, prompt: str):
        self.hidden_state_calls += 1
        return self._hidden_state


class FakeNLI:
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


@dataclass
class _ConstScoreDetector(Detector):
    """Test double — declares its prerequisites and returns a fixed score."""

    name: DetectorName
    requires_samples: bool
    requires_hidden_state: bool
    fixed_score: float = 0.0

    def score(self, ctx: DetectionContext) -> DetectorScore:
        return DetectorScore(name=self.name, score=self.fixed_score)


@pytest.fixture
def patch_nli():
    with patch(
        "taste.services.semantic_entropy.get_nli_scorer",
        return_value=FakeNLI(),
    ):
        yield


def _make_agent(slm, detectors, *, threshold: float, escalation: DetectorName):
    return DetectorAgent(
        slm=slm,
        detectors=detectors,
        escalation_detector=escalation,
        escalation_threshold=threshold,
    )


def test_low_se_does_not_escalate(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    agent = _make_agent(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = agent.detect("What is the capital of France?")
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY] == pytest.approx(0.0, abs=1e-6)
    assert res.escalated is False


def test_high_se_triggers_escalation_flag(patch_nli):
    slm = FakeSLM([f"distinct-{i}" for i in range(10)])
    agent = _make_agent(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = agent.detect("Something the model can't answer.")
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY] > 0.5
    assert res.escalated is True


def test_all_configured_detectors_run_and_record_scores(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    agent = _make_agent(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = agent.detect("q")
    assert set(res.detector_scores.keys()) == {
        DetectorName.PREDICTIVE_ENTROPY,
        DetectorName.SEMANTIC_ENTROPY,
    }


def test_result_records_escalation_policy(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    agent = _make_agent(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = agent.detect("q")
    assert res.escalation_detector == DetectorName.SEMANTIC_ENTROPY
    assert res.escalation_threshold == 0.5


def test_no_sample_skipped_when_no_detector_needs_samples():
    """If every enabled detector has requires_samples=False, the SLM is never sampled."""
    slm = FakeSLM(["unused"])
    detector = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=False,
        fixed_score=0.1,
    )
    agent = _make_agent(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    res = agent.detect("q")
    assert slm.sample_calls == 0
    assert res.samples == []
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.1)


def test_hidden_state_requirement_fails_fast_when_provider_returns_none():
    """When a configured detector requires hidden states but the SLM provider
    returns None (e.g. vLLM without an embed sibling configured), the
    orchestrator must raise rather than dispatch the detector with no
    hidden state in its context."""
    slm = FakeSLM(["x"])  # default hidden_state=None
    detector = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=True,
    )
    agent = _make_agent(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    with pytest.raises(RuntimeError, match="pooled hidden states"):
        agent.detect("q")


def test_hidden_state_detector_receives_tensor_when_provider_supplies_one():
    """Positive case: an SLM provider with hidden states wired up feeds the
    pooled tensor into the detector's context and SEP-style detectors run."""
    import torch as _torch

    fake_tensor = _torch.zeros(8)
    slm = FakeSLM(["x"], hidden_state=fake_tensor)

    captured: dict[str, object] = {}

    @dataclass
    class _CapturingDetector(_ConstScoreDetector):
        def score(self, ctx):
            captured["hidden_state"] = ctx.pooled_hidden_state
            return DetectorScore(name=self.name, score=self.fixed_score)

    detector = _CapturingDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=True,
        fixed_score=0.42,
    )
    agent = _make_agent(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    res = agent.detect("q")
    assert captured["hidden_state"] is fake_tensor
    assert slm.hidden_state_calls == 1
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.42)


def test_escalation_detector_must_be_enabled():
    """Default config sets escalation_detector=SE; constructing the agent with only
    PE enabled must fail fast rather than silently picking another detector."""
    slm = FakeSLM(["x"])
    with pytest.raises(ValueError, match="not in the enabled detector list"):
        DetectorAgent(slm=slm, detectors=[PredictiveEntropyDetector()])
