"""ResponseEvaluator + evaluate_responses orchestration tests.

These pin the public contract of the library's evaluator, independent of any
single detector's math (which is covered in test_semantic_entropy.py and
test_predictive_entropy.py):
  - Iterates the configured detector list and collects all scores.
  - Records the escalation detector + threshold on the result.
  - Flips the escalated flag at the configured threshold.
  - Skips SLM sampling when no enabled detector requires samples.
  - Fails fast when a detector requires hidden states but the SLM provider
    returns None (allowed by the protocol for third-party providers).
  - Refuses to construct if escalation_detector is not in enabled_detectors.
  - With a SEP probe: asks the provider for the probe's layers, and refuses
    at construction, before any sampling, a backbone or prompt the probe was
    not trained on and layers the provider cannot serve; warns when
    the SEP escalation threshold in use differs from the probe's recommended
    threshold.
  - evaluate_responses (response-shaped API) runs without an SLM provider
    when the caller supplies samples + greedy_answer directly. This is the
    library entry point consumers like langrag use.

The detector list is injected directly so the tests don't depend on the
config-driven default registry."""

import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hallucinations_math_lab import (
    AnyAbove,
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
)
from hallucinations_math_lab.config import Settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, DetectorCategory
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorScore,
)
from hallucinations_math_lab.services.sampling_provider import UnsupportedHiddenStateRequest
from hallucinations_math_lab.services.semantic_entropy_probe import (
    SemanticEntropyProbeDetector,
    SEPProbeMismatchError,
)
from tests.unit.sep_probe_fixtures import (
    PROBE_HIDDEN_SIZE,
    PROBE_NUM_HIDDEN_LAYERS,
    PROBE_RECOMMENDED_THRESHOLD,
    ProbeBackboneProvider,
    probe_config,
    write_probe_dir,
)

_TEST_METADATA = DetectorMetadata(
    signal="test double",
    assumption="for test isolation only",
    failure_modes=("never run in production",),
    category=DetectorCategory.UNCERTAINTY,
)


class FakeSLM(ProbeBackboneProvider):
    """In-memory SamplingProvider double, on the fixture probes' backbone.
    `hidden_state` defaults to None (a provider that exposes no hidden
    states); pass a tensor to simulate a SEP-capable provider."""

    def __init__(self, answers: list[str], hidden_state=None) -> None:
        self._answers = answers
        self._hidden_state = hidden_state
        self.sample_calls = 0
        self.hidden_state_calls = 0
        self.requested_layers: list[tuple[int, ...]] = []

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

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]):
        self.requested_layers.append(layers)
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
    hidden_state_layers: tuple[int, ...] = (PROBE_NUM_HIDDEN_LAYERS,)

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
    returns None (the protocol allows it for providers that expose none), the
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

    fake_tensor = _torch.zeros(1, 8)
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
    assert slm.requested_layers == [(slm.num_hidden_layers,)]
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.42)


def test_response_evaluator_sep_escalates_above_sep_threshold(monkeypatch):
    """Regression: the evaluator used SE_THRESHOLD (nats) for every detector.
    SEP scores a probability, so with SE_THRESHOLD=1.0 it could never escalate.
    With no explicit threshold, SEP must be compared against SEP_THRESHOLD."""
    settings = Settings(
        se_threshold=1.0,
        sep_threshold=0.5,
        enabled_detectors=[DetectorName.SEMANTIC_ENTROPY_PROBE],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    monkeypatch.setattr("hallucinations_math_lab.evaluator.get_settings", lambda: settings)
    sep = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=False,
        fixed_score=0.8,
    )
    res = ResponseEvaluator(slm=FakeSLM(["x"]), detectors=[sep]).evaluate("q")
    assert res.escalation_threshold == 0.5
    assert res.escalated is True


def test_response_evaluator_explicit_threshold_overrides_routing(monkeypatch):
    settings = Settings(sep_threshold=0.5)
    monkeypatch.setattr("hallucinations_math_lab.evaluator.get_settings", lambda: settings)
    sep = _ConstScoreDetector(
        name=DetectorName.SEMANTIC_ENTROPY_PROBE,
        requires_samples=False,
        requires_hidden_state=False,
        fixed_score=0.8,
    )
    res = _make_evaluator(
        FakeSLM(["x"]), [sep], threshold=0.9, escalation=DetectorName.SEMANTIC_ENTROPY_PROBE
    ).evaluate("q")
    assert res.escalation_threshold == 0.9
    assert res.escalated is False


def test_unroutable_escalation_detector_needs_an_explicit_threshold():
    """Every result records a scalar escalation_threshold for its escalation
    detector, even when a policy decides escalation. A detector with no
    configured threshold has no honest value to record (SE_THRESHOLD is in
    nats, a different scale), so the caller must supply one."""
    judge_qa = _ConstScoreDetector(
        name=DetectorName.LLAMA_JUDGE_QA,
        requires_samples=False,
        requires_hidden_state=False,
        fixed_score=0.8,
    )
    policy = AnyAbove({DetectorName.LLAMA_JUDGE_QA: 0.5})
    with pytest.raises(ValueError, match="pass an explicit escalation_threshold"):
        ResponseEvaluator(
            slm=FakeSLM(["x"]),
            detectors=[judge_qa],
            escalation_detector=DetectorName.LLAMA_JUDGE_QA,
            escalation_policy=policy,
        )
    res = ResponseEvaluator(
        slm=FakeSLM(["x"]),
        detectors=[judge_qa],
        escalation_detector=DetectorName.LLAMA_JUDGE_QA,
        escalation_threshold=0.5,
        escalation_policy=policy,
    ).evaluate("q")
    assert res.escalated is True


def test_escalation_detector_must_be_enabled():
    """Default config sets escalation_detector=SE; constructing the evaluator with
    only PE enabled must fail fast rather than silently picking another detector."""
    slm = FakeSLM(["x"])
    with pytest.raises(ValueError, match="not in the enabled detector list"):
        ResponseEvaluator(slm=slm, detectors=[PredictiveEntropyDetector()])


# --- SEP probe wiring (ResponseEvaluator construction) ---


def _sep_evaluator(slm: FakeSLM, probe_dir: Path) -> ResponseEvaluator:
    return ResponseEvaluator(slm=slm, detectors=[SemanticEntropyProbeDetector(probe_dir)])


@pytest.fixture
def sep_settings(monkeypatch):
    """SEP enabled and escalating on SEP_THRESHOLD; returns a setter for it."""

    def _set(sep_threshold: float = PROBE_RECOMMENDED_THRESHOLD) -> None:
        settings = Settings(
            sep_threshold=sep_threshold,
            enabled_detectors=[DetectorName.SEMANTIC_ENTROPY_PROBE],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
        )
        monkeypatch.setattr("hallucinations_math_lab.evaluator.get_settings", lambda: settings)

    _set()
    return _set


def test_provider_asked_for_probe_layers(tmp_path, sep_settings):
    """A windowed probe reads several layers; the evaluator must request
    exactly those, not only the last layer."""
    layers = (1, 2)
    probe_dir = write_probe_dir(tmp_path / "probe", probe_config(layers=layers))
    slm = FakeSLM(["x"], hidden_state=torch.zeros(len(layers), PROBE_HIDDEN_SIZE))

    res = _sep_evaluator(slm, probe_dir).evaluate("q")

    assert slm.requested_layers == [layers]
    # Zero features and zero bias: sigmoid(0).
    assert res.detector_scores[DetectorName.SEMANTIC_ENTROPY_PROBE] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("attribute", "value", "reported_field"),
    [
        ("model_id", "org/other-model", "model_id"),
        ("model_revision", "f" * 40, "revision"),
        # The probe's layers stay valid at L + 1, so check_compatible is what raises.
        ("num_hidden_layers", PROBE_NUM_HIDDEN_LAYERS + 1, "num_hidden_layers"),
        ("hidden_size", PROBE_HIDDEN_SIZE + 1, "hidden_size"),
        ("chat_template_sha256", "0" * 64, "chat_template_sha256"),
    ],
)
def test_backbone_mismatch_raises_before_any_sample(
    tmp_path, sep_settings, attribute, value, reported_field
):
    probe_dir = write_probe_dir(tmp_path / "probe")
    slm = FakeSLM(["x"], hidden_state=torch.zeros(1, PROBE_HIDDEN_SIZE))
    setattr(slm, attribute, value)

    with pytest.raises(SEPProbeMismatchError, match=f"{reported_field}: probe .*{value!r}"):
        _sep_evaluator(slm, probe_dir)
    assert slm.sample_calls == 0
    assert slm.hidden_state_calls == 0


def test_answer_instruction_mismatch_raises_before_any_sample(tmp_path, monkeypatch):
    settings = Settings(
        answer_instruction="Answer in one word.",
        enabled_detectors=[DetectorName.SEMANTIC_ENTROPY_PROBE],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    monkeypatch.setattr("hallucinations_math_lab.evaluator.get_settings", lambda: settings)
    slm = FakeSLM(["x"])

    with pytest.raises(SEPProbeMismatchError, match="Answer in one word."):
        _sep_evaluator(slm, write_probe_dir(tmp_path / "probe"))
    assert slm.sample_calls == 0


def test_vllm_capability_checked_at_build_time(tmp_path, sep_settings, monkeypatch):
    """The vLLM embed runner pools only the last layer: a windowed probe must
    be refused when the evaluator is built, before any request is served."""
    from demo.model_registry import get_model_entry
    from demo.slm_provider import VLLMOpenAIProvider
    from hallucinations_math_lab.config import get_settings

    model_id = "google/gemma-4-E2B-it"
    entry = get_model_entry(model_id)
    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", model_id)
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", "http://localhost:8081/v1")
    monkeypatch.setenv("VLLM_EMBED_MODEL_ID", "")
    layers = (entry.num_hidden_layers - 1, entry.num_hidden_layers)
    probe_dir = write_probe_dir(
        tmp_path / "probe",
        probe_config(
            model_id=model_id,
            num_hidden_layers=entry.num_hidden_layers,
            hidden_size=entry.hidden_size,
            layers=layers,
        ),
    )
    import transformers

    tokenizer_spy = MagicMock()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_spy)
    try:
        with patch("demo.slm_provider.OpenAI") as openai_cls:
            provider = VLLMOpenAIProvider()
            with pytest.raises(UnsupportedHiddenStateRequest, match="can only pool the last layer"):
                ResponseEvaluator(slm=provider, detectors=[SemanticEntropyProbeDetector(probe_dir)])
        # Both clients are the patched class's return value: nothing was sent,
        # and the check needed no tokenizer download.
        openai_cls.return_value.post.assert_not_called()
        openai_cls.return_value.chat.completions.create.assert_not_called()
        tokenizer_spy.assert_not_called()
    finally:
        get_settings.cache_clear()


def test_warns_when_sep_threshold_differs_from_recommended(tmp_path, sep_settings, caplog):
    """The env value is the threshold; the probe's recommendation is only
    reported, and a difference is flagged."""
    probe_dir = write_probe_dir(tmp_path / "probe")
    sep_settings(PROBE_RECOMMENDED_THRESHOLD + 0.1)

    with caplog.at_level(logging.WARNING, logger="hallucinations_math_lab.evaluator"):
        evaluator = _sep_evaluator(
            FakeSLM(["x"], hidden_state=torch.zeros(1, PROBE_HIDDEN_SIZE)), probe_dir
        )

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "recommended threshold" in record.getMessage()
    assert getattr(record, LOG_EXTRA_KEY) == {
        "probe_dir": str(probe_dir),
        "threshold_in_use": pytest.approx(PROBE_RECOMMENDED_THRESHOLD + 0.1),
        "recommended_threshold": PROBE_RECOMMENDED_THRESHOLD,
    }
    # The env value stays the threshold in use.
    assert evaluator.evaluate("q").escalation_threshold == pytest.approx(
        PROBE_RECOMMENDED_THRESHOLD + 0.1
    )


def test_no_threshold_warning_when_sep_threshold_matches_recommended(
    tmp_path, sep_settings, caplog
):
    """Matching thresholds are still logged, both values, at INFO."""
    probe_dir = write_probe_dir(tmp_path / "probe")
    with caplog.at_level(logging.INFO, logger="hallucinations_math_lab.evaluator"):
        _sep_evaluator(FakeSLM(["x"]), probe_dir)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    (record,) = [r for r in caplog.records if r.getMessage() == "SEP escalation threshold."]
    assert record.levelno == logging.INFO
    assert getattr(record, LOG_EXTRA_KEY) == {
        "probe_dir": str(probe_dir),
        "threshold_in_use": PROBE_RECOMMENDED_THRESHOLD,
        "recommended_threshold": PROBE_RECOMMENDED_THRESHOLD,
    }


def test_no_threshold_report_when_a_policy_decides(tmp_path, sep_settings, caplog):
    """Under an escalation policy SEP_THRESHOLD is not what decides, so it is
    not compared with the probe's recommendation."""
    probe_dir = write_probe_dir(tmp_path / "probe")
    sep_settings(PROBE_RECOMMENDED_THRESHOLD + 0.1)
    with caplog.at_level(logging.INFO, logger="hallucinations_math_lab.evaluator"):
        ResponseEvaluator(
            slm=FakeSLM(["x"]),
            detectors=[SemanticEntropyProbeDetector(probe_dir)],
            escalation_policy=AnyAbove({DetectorName.SEMANTIC_ENTROPY_PROBE: 0.5}),
        )
    assert not [r for r in caplog.records if "threshold" in r.getMessage()]


def test_hidden_state_detector_must_name_its_layers():
    """The provider is asked for the layers the detector reads; a detector
    requiring hidden states without naming them is refused at construction."""

    @dataclass
    class _NoLayers:
        name: DetectorName = DetectorName.SEMANTIC_ENTROPY_PROBE
        metadata: DetectorMetadata = _TEST_METADATA
        requires_samples: bool = False
        requires_hidden_state: bool = True

        def score(self, ctx: DetectionContext) -> DetectorScore:
            raise AssertionError("never scored")

    with pytest.raises(TypeError, match="hidden_state_layers"):
        _make_evaluator(FakeSLM(["x"]), [_NoLayers()], threshold=0.5, escalation=_NoLayers.name)


def test_hidden_state_detectors_must_agree_on_layers():
    """One pooled hidden state feeds every hidden-state detector."""
    detectors: list[Detector] = [
        _ConstScoreDetector(
            name=name,
            requires_samples=False,
            requires_hidden_state=True,
            hidden_state_layers=layers,
        )
        for name, layers in [
            (DetectorName.SEMANTIC_ENTROPY_PROBE, (2,)),
            (DetectorName.PREDICTIVE_ENTROPY, (1, 2)),
        ]
    ]
    with pytest.raises(ValueError, match="different hidden-state layers"):
        _make_evaluator(
            FakeSLM(["x"]),
            detectors,
            threshold=0.5,
            escalation=DetectorName.SEMANTIC_ENTROPY_PROBE,
        )


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
