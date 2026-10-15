"""ResponseEvaluator + evaluate_responses orchestration tests.

These pin the public contract of the library's evaluator, independent of any
single detector's math (which is covered in test_semantic_entropy.py and
test_predictive_entropy.py):
  - Iterates the configured detector list and collects all scores.
  - Records the escalation detector + threshold on the result.
  - Flips the escalated flag at the configured threshold.
  - Skips SLM sampling when no enabled detector requires samples.
  - Fails fast when a detector requires hidden states but the SLM provider
    returns None (e.g. vLLM without an embed sibling).
  - Refuses to construct if escalation_detector is not in enabled_detectors.
  - evaluate_responses (response-shaped API) runs without an SLM provider
    when the caller supplies samples + greedy_answer directly. This is the
    library entry point consumers like langrag use.

The detector list is injected directly so the tests don't depend on the
config-driven default registry."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
)
from hallucinations_math_lab.constants import DetectorCategory
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


class FakeSLM:
    """In-memory SamplingProvider double. `hidden_state` defaults to None
    (matches a vLLM provider without an embed sibling); pass a tensor to
    simulate a SEP-capable provider."""

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
    metadata: DetectorMetadata = _TEST_METADATA

    def score(self, ctx: DetectionContext) -> DetectorScore:
        return DetectorScore(
            name=self.name,
            score=self.fixed_score,
            cost_usd=0.0,
            latency_ms=0.0,
        )


@pytest.fixture
def patch_nli():
    with patch(
        "hallucinations_math_lab.services.semantic_entropy.get_nli_scorer",
        return_value=FakeNLI(),
    ):
        yield


def _make_evaluator(slm, detectors, *, threshold: float, escalation: DetectorName):
    return ResponseEvaluator(
        slm=slm,
        detectors=detectors,
        escalation_detector=escalation,
        escalation_threshold=threshold,
    )


# --- Provider-shaped API (ResponseEvaluator.evaluate) ---


def test_low_se_does_not_escalate(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    evaluator = _make_evaluator(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = evaluator.evaluate("What is the capital of France?")
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY] == pytest.approx(0.0, abs=1e-6)
    assert res.escalated is False


def test_high_se_triggers_escalation_flag(patch_nli):
    slm = FakeSLM([f"distinct-{i}" for i in range(10)])
    evaluator = _make_evaluator(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = evaluator.evaluate("Something the model can't answer.")
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY] > 0.5
    assert res.escalated is True


def test_all_configured_detectors_run_and_record_scores(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    evaluator = _make_evaluator(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = evaluator.evaluate("q")
    assert set(res.detector_scores.keys()) == {
        DetectorName.PREDICTIVE_ENTROPY,
        DetectorName.SEMANTIC_ENTROPY,
    }


def test_result_records_escalation_policy(patch_nli):
    slm = FakeSLM(["Paris"] * 10)
    evaluator = _make_evaluator(
        slm,
        [PredictiveEntropyDetector(), SemanticEntropyDetector()],
        threshold=0.5,
        escalation=DetectorName.SEMANTIC_ENTROPY,
    )
    res = evaluator.evaluate("q")
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
    evaluator = _make_evaluator(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    res = evaluator.evaluate("q")
    assert slm.sample_calls == 0
    assert res.samples == []
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.1)


def test_hidden_state_requirement_fails_fast_when_provider_returns_none():
    """When a configured detector requires hidden states but the SLM provider
    returns None (e.g. vLLM without an embed sibling configured), the
    evaluator must raise rather than dispatch the detector with no
    hidden state in its context."""
    slm = FakeSLM(["x"])  # default hidden_state=None
    detector = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=True,
    )
    evaluator = _make_evaluator(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    with pytest.raises(RuntimeError, match="pooled hidden state"):
        evaluator.evaluate("q")


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
            return DetectorScore(
                name=self.name,
                score=self.fixed_score,
                cost_usd=0.0,
                latency_ms=0.0,
            )

    detector = _CapturingDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=True,
        fixed_score=0.42,
    )
    evaluator = _make_evaluator(
        slm,
        [detector],
        threshold=1.0,
        escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    res = evaluator.evaluate("q")
    assert captured["hidden_state"] is fake_tensor
    assert slm.hidden_state_calls == 1
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.42)


def test_escalation_detector_must_be_enabled():
    """Default config sets escalation_detector=SE; constructing the evaluator with
    only PE enabled must fail fast rather than silently picking another detector."""
    slm = FakeSLM(["x"])
    with pytest.raises(ValueError, match="not in the enabled detector list"):
        ResponseEvaluator(slm=slm, detectors=[PredictiveEntropyDetector()])


# --- Response-shaped API (evaluate_responses) ---


def test_evaluate_responses_runs_without_a_provider(patch_nli):
    """The response-shaped surface: caller supplies samples + greedy_answer
    directly. The library does not touch an LLM. This is the langrag-style
    entry point."""
    samples = SampleSet(
        prompt="What is the capital of France?",
        samples=[
            Sample(text="Paris.", token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])
            for _ in range(10)
        ],
    )
    result = evaluate_responses(
        prompt="What is the capital of France?",
        samples=samples,
        detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector()],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=0.5,
        greedy_answer="Paris.",
    )
    assert result.detector_scores[DetectorName.SEMANTIC_ENTROPY] == pytest.approx(0.0, abs=1e-6)
    assert result.escalated is False
    assert result.final_answer == "Paris."


def test_evaluate_responses_requires_greedy_answer():
    """The library cannot synthesize the small model's greedy answer when it
    isn't driving the LLM. Surface this as a clear ValueError rather than
    a downstream NoneType crash."""
    samples = SampleSet(prompt="q", samples=[Sample(text="x")])
    with pytest.raises(ValueError, match="greedy_answer"):
        evaluate_responses(
            prompt="q",
            samples=samples,
            detectors=[SemanticEntropyDetector()],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=0.5,
            greedy_answer=None,
        )


def test_evaluate_responses_rejects_unknown_escalation_detector(patch_nli):
    """If escalation_detector isn't in the detector list, the response-shaped
    API must reject rather than dispatch and crash later when the dict lookup
    misses."""
    samples = SampleSet(
        prompt="q",
        samples=[Sample(text="x", token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])],
    )
    with pytest.raises(ValueError, match="not in the detector list"):
        evaluate_responses(
            prompt="q",
            samples=samples,
            detectors=[PredictiveEntropyDetector()],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,  # not in the list
            escalation_threshold=0.5,
            greedy_answer="x",
        )


def test_evaluate_responses_invokes_judge_on_escalation(patch_nli):
    """When samples disagree, SE crosses the threshold, the supplied judge
    is called, and the result's final_answer + source reflect the judge."""

    class FakeJudge:
        def __init__(self):
            self.calls = 0

        def answer(self, prompt: str) -> tuple[str, str]:
            self.calls += 1
            return ("judge says X", "because reasons")

    judge = FakeJudge()
    samples = SampleSet(
        prompt="q",
        samples=[
            Sample(text=f"distinct-{i}", token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])
            for i in range(10)
        ],
    )
    from hallucinations_math_lab.constants import AnswerSource

    result = evaluate_responses(
        prompt="q",
        samples=samples,
        detectors=[SemanticEntropyDetector()],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=0.5,
        greedy_answer="distinct-0",
        judge=judge,
    )
    assert result.escalated is True
    assert judge.calls == 1
    assert result.final_answer == "judge says X"
    assert result.final_answer_source == AnswerSource.JUDGE_MODEL
    assert result.judge_reasoning == "because reasons"
