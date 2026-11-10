"""Two public entry points to the response-evaluation pipeline.

`evaluate_responses(...)` — RESPONSE-SHAPED. The caller already has N
sampled completions from their own LLM and just wants scores back. The
library never drives an LLM in this path. This is the natural fit for
consumers like langrag who own their inference stack and want a pure
scoring function.

`ResponseEvaluator(slm_provider).evaluate(prompt)` — PROVIDER-SHAPED. The
caller plugs in something that implements `SamplingProvider` and the
evaluator drives sampling for them. This is what the demo CLI uses, and
it is implemented in terms of `evaluate_responses` (so the two surfaces
share one code path).

Both return a `DetectionResult` containing per-detector scores, the
escalation decision, and (if a judge was supplied and escalation fired)
the judge's answer + reasoning.

Note on the cluster field: only SE populates `clusters` on the result. If
SE is not in the configured detector list, `clusters` is None.
"""

import logging

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, AnswerSource, DetectorName
from hallucinations_math_lab.custom_types.domain_types import (
    DetectionResult,
    SampleSet,
    SemanticEntropyTrace,
)
from hallucinations_math_lab.escalation import EscalationPolicy, SingleDetector
from hallucinations_math_lab.judges.protocol import Judge
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    HiddenStateDetector,
)
from hallucinations_math_lab.services.sampling_provider import SamplingProvider
from hallucinations_math_lab.services.semantic_entropy_probe import (
    ProbeSource,
    SemanticEntropyProbeDetector,
    resolve_probe_source,
)

logger = logging.getLogger(__name__)


def evaluate_responses(
    prompt: str,
    samples: SampleSet | None,
    *,
    detectors: list[Detector],
    pooled_hidden_state=None,
    escalation_detector: DetectorName,
    escalation_threshold: float,
    judge: Judge | None = None,
    greedy_answer: str | None = None,
    escalation_policy: EscalationPolicy | None = None,
    trace_semantic_entropy: bool = False,
) -> DetectionResult:
    """Score a set of already-sampled responses against the configured detectors.

    Arguments:
      prompt: the original question. Carried through to the result and
        passed to the judge if escalation fires.
      samples: SampleSet containing N completions from the caller's LLM,
        each with token logprobs (needed by PE; not needed by SEP). May
        be None if every configured detector has requires_samples=False.
      detectors: ordered list of Detector instances to run. Every entry
        must have its prerequisites satisfied — fail-fast otherwise.
      pooled_hidden_state: torch.Tensor of shape [len(layers), hidden_size]
        (the layers the hidden-state detector reads, at the last prompt
        token), or None. Required by any detector with
        requires_hidden_state=True (e.g. SEP).
      escalation_detector: which detector's score drives the escalation
        decision. Must appear in the `detectors` list.
      escalation_threshold: the score above which escalation fires, on
        escalation_detector's own scale (nats for SE / PE, a probability for
        SEP). `Settings.escalation_threshold_for` picks the configured one.
      judge: optional Judge implementation. When supplied and escalation
        fires, the judge re-answers the prompt and the result's
        final_answer is replaced.
      greedy_answer: the small model's deterministic answer to surface
        when no escalation fires. Required by the response-shaped API
        because the library is not driving the LLM and so cannot produce
        a greedy decode on its own. ResponseEvaluator supplies this for
        the provider-shaped path.
      escalation_policy: optional multi-detector escalation rule. When
        supplied it takes precedence over escalation_detector /
        escalation_threshold (which are still recorded as the caller-passed
        baseline); the richer human-readable reason lives in the result's
        escalation_policy_desc. When None, behavior is byte-identical to the
        historical single-detector path (escalation_detector > threshold).
      trace_semantic_entropy: when True, SE records its clustering comparison
        by comparison and the result carries it as `se_trace`, for a
        step-by-step replay. Off by default; it never changes the clusters.

    Raises:
      ValueError: if escalation_detector is not in the detectors list, or if
        a supplied escalation_policy references a detector absent from scores.
      DetectorPrerequisiteError: if any detector's declared inputs are
        missing from the assembled DetectionContext.
    """
    if not any(d.name == escalation_detector for d in detectors):
        raise ValueError(
            f"escalation_detector={escalation_detector!s} is not in the detector list "
            f"{[d.name for d in detectors]!s}."
        )
    if greedy_answer is None:
        raise ValueError(
            "evaluate_responses requires a greedy_answer (the small model's deterministic "
            "answer). Provider-shaped callers go through ResponseEvaluator, which fills "
            "this in. Response-shaped callers must supply it explicitly."
        )

    need_samples = any(d.requires_samples for d in detectors)
    if need_samples and samples is None:
        raise ValueError(
            "At least one configured detector requires samples, but `samples` is None. "
            "Pass a SampleSet from your LLM, or remove sample-requiring detectors "
            "(e.g. SemanticEntropyDetector, PredictiveEntropyDetector)."
        )

    need_hidden_state = any(d.requires_hidden_state for d in detectors)
    if need_hidden_state and pooled_hidden_state is None:
        raise ValueError(
            "At least one configured detector requires a pooled hidden state, but "
            "`pooled_hidden_state` is None. Pass your model's hidden states at the final "
            "prompt token, one row per layer the probe reads, or remove SEP from the "
            "detector list."
        )

    ctx = DetectionContext(
        prompt=prompt,
        samples=samples,
        pooled_hidden_state=pooled_hidden_state,
        trace_semantic_entropy=trace_semantic_entropy,
    )

    scores: dict[DetectorName, float] = {}
    clusters = None
    se_traces: dict[DetectorName, SemanticEntropyTrace] = {}
    for detector in detectors:
        result = detector.score(ctx)
        scores[result.name] = result.score
        if result.clusters is not None:
            clusters = result.clusters
        if result.se_trace is not None:
            se_traces[result.name] = result.se_trace
    se_trace = _pick_se_trace(se_traces, escalation_detector)

    policy = (
        escalation_policy
        if escalation_policy is not None
        else SingleDetector(name=escalation_detector, threshold=escalation_threshold)
    )
    for referenced in policy.detectors_referenced:
        if referenced not in scores:
            raise ValueError(
                f"Escalation policy references detector {referenced!s}, which was not "
                f"scored this run. Scored detectors: {[str(k) for k in scores]!s}."
            )

    should_escalate = policy.should_escalate(scores)

    logger.info(
        "Response evaluation complete.",
        extra={
            LOG_EXTRA_KEY: {
                "scores": {str(k): round(v, 4) for k, v in scores.items()},
                "escalation_detector": str(escalation_detector),
                "escalation_threshold": escalation_threshold,
                "escalation_policy": policy.describe(),
                "escalate": should_escalate,
                "n_clusters": len(clusters) if clusters else 0,
                "judge_supplied": judge is not None,
            }
        },
    )

    final_answer = greedy_answer
    final_source = AnswerSource.SMALL_MODEL_GREEDY
    judge_reasoning: str | None = None
    if should_escalate and judge is not None:
        try:
            final_answer, judge_reasoning = judge.answer(prompt)
            final_source = AnswerSource.JUDGE_MODEL
        except Exception as e:
            raise RuntimeError(f"Judge escalation failed for prompt={prompt[:80]!r}: {e!s}") from e

    return DetectionResult(
        prompt=prompt,
        greedy_answer=greedy_answer,
        samples=[s.text for s in samples.samples] if samples else [],
        sample_token_logprobs=[s.token_logprobs for s in samples.samples] if samples else [],
        detector_scores=scores,
        clusters=clusters,
        escalation_detector=escalation_detector,
        escalation_threshold=escalation_threshold,
        escalated=should_escalate,
        final_answer=final_answer,
        final_answer_source=final_source,
        judge_reasoning=judge_reasoning,
        escalation_policy_desc=(
            None if escalation_policy is None else escalation_policy.describe()
        ),
        se_trace=se_trace,
    )


def _pick_se_trace(
    se_traces: dict[DetectorName, SemanticEntropyTrace], escalation_detector: DetectorName
) -> SemanticEntropyTrace | None:
    """The trace of the escalation detector when it recorded one, so the
    replay ends on the SE value that drove escalation; otherwise the last one
    recorded (both SE estimators share the clustering, only p(c) differs)."""
    if escalation_detector in se_traces:
        return se_traces[escalation_detector]
    return next(reversed(se_traces.values()), None)


def _build_default_detectors(enabled: list[DetectorName], model_id: str) -> list[Detector]:
    """Map config-enabled detector names to concrete instances.

    Kept inside the evaluator module so the public surface stays small.
    `model_id` is the model being evaluated; it picks SEP's default probe."""
    # Lazy import — these detectors pull in torch / nli machinery.
    from hallucinations_math_lab.services.predictive_entropy import PredictiveEntropyDetector
    from hallucinations_math_lab.services.semantic_entropy import (
        DiscreteSemanticEntropyDetector,
        SemanticEntropyDetector,
    )

    detectors: list[Detector] = []
    for name in enabled:
        if name == DetectorName.PREDICTIVE_ENTROPY:
            detectors.append(PredictiveEntropyDetector())
        elif name == DetectorName.SEMANTIC_ENTROPY:
            detectors.append(SemanticEntropyDetector())
        elif name == DetectorName.DISCRETE_SEMANTIC_ENTROPY:
            detectors.append(DiscreteSemanticEntropyDetector())
        elif name == DetectorName.SEMANTIC_ENTROPY_PROBE:
            detectors.append(SemanticEntropyProbeDetector(_resolve_sep_probe_source(model_id)))
        else:
            raise ValueError(
                f"Detector {name!s} is in ENABLED_DETECTORS but unknown to the evaluator."
            )
    return detectors


def _resolve_sep_probe_source(model_id: str) -> ProbeSource:
    """The ProbeSource SEP_PROBE_PATH names (a path or an `hf://` repo), else
    the default for `model_id` — the model actually being evaluated, which in a
    multi-model sweep is not VLLM_MODEL_ID. A missing or legacy `.pt` probe
    raises here, so the misconfiguration surfaces before the first request."""
    return resolve_probe_source(get_settings().sep_probe_path, model_id)


def _hidden_state_layers(detectors: list[Detector]) -> tuple[int, ...] | None:
    """The layers to ask the provider for: the ones every hidden-state detector
    reads (one pooled hidden state feeds them all), or None when none needs one."""
    layers_by_detector: dict[str, tuple[int, ...]] = {}
    for detector in detectors:
        if not detector.requires_hidden_state:
            continue
        if not isinstance(detector, HiddenStateDetector):
            raise TypeError(
                f"Detector {detector.name!s} requires a hidden state but has no "
                "hidden_state_layers naming the layers it reads; implement HiddenStateDetector."
            )
        layers_by_detector[str(detector.name)] = detector.hidden_state_layers
    distinct = set(layers_by_detector.values())
    if len(distinct) > 1:
        raise ValueError(
            f"The hidden-state detectors read different hidden-state layers "
            f"{layers_by_detector!s}, but one pooled hidden state feeds them all."
        )
    return distinct.pop() if distinct else None


def _check_sep_probe_fits(
    probe: SemanticEntropyProbeDetector, slm: SamplingProvider, answer_instruction: str
) -> None:
    """Refuse a probe trained on another backbone or prompt than `slm` uses."""
    probe.check_compatible(
        model_id=slm.model_id,
        revision=slm.model_revision,
        num_hidden_layers=slm.num_hidden_layers,
        hidden_size=slm.hidden_size,
        answer_instruction=answer_instruction,
        chat_template_sha256=slm.chat_template_sha256,
    )


def _report_sep_threshold(probe: SemanticEntropyProbeDetector, threshold_in_use: float) -> None:
    """Log the SEP escalation threshold in use next to the probe's recommended
    one, as a warning when they differ. The probe's value is never applied."""
    recommended = probe.config.output.recommended_threshold
    thresholds = {
        "probe_dir": str(probe.probe_dir),
        "threshold_in_use": threshold_in_use,
        "recommended_threshold": recommended,
    }
    if threshold_in_use != recommended:
        logger.warning(
            "The SEP escalation threshold in use differs from the probe's recommended "
            "threshold; the threshold in use is applied.",
            extra={LOG_EXTRA_KEY: thresholds},
        )
    else:
        logger.info("SEP escalation threshold.", extra={LOG_EXTRA_KEY: thresholds})


class ResponseEvaluator:
    """Provider-shaped entry point — wraps a SamplingProvider and exposes a
    one-call `evaluate(prompt)` that drives sampling, scores, and (optionally)
    judges.

    Implemented in terms of `evaluate_responses` so the response-shaped and
    provider-shaped surfaces share a single code path. Consumers that
    already own sampling should prefer `evaluate_responses` directly; the
    demo CLI and end-to-end harnesses use `ResponseEvaluator`."""

    def __init__(
        self,
        slm: SamplingProvider | None = None,
        detectors: list[Detector] | None = None,
        escalation_detector: DetectorName | None = None,
        escalation_threshold: float | None = None,
        judge: Judge | None = None,
        *,
        escalation_policy: EscalationPolicy | None = None,
    ) -> None:
        if slm is None:
            # Lazy import — demo isn't part of the library surface, so we
            # only reach into it when the caller declined to inject one.
            from demo.slm_provider import build_default_provider

            slm = build_default_provider()

        settings = get_settings()
        self._slm: SamplingProvider = slm
        self._detectors: list[Detector] = (
            detectors
            if detectors is not None
            else _build_default_detectors(settings.enabled_detectors, slm.model_id)
        )
        self._escalation_detector = (
            escalation_detector if escalation_detector is not None else settings.escalation_detector
        )
        # Resolved even when a policy decides escalation: the result always
        # records this baseline, and it must be on the detector's own scale.
        # A detector with no configured threshold therefore needs an explicit one.
        self._escalation_threshold = (
            escalation_threshold
            if escalation_threshold is not None
            else settings.escalation_threshold_for(self._escalation_detector)
        )
        self._judge = judge
        self._escalation_policy = escalation_policy

        if not any(d.name == self._escalation_detector for d in self._detectors):
            raise ValueError(
                f"Escalation detector {self._escalation_detector!s} is not in the enabled detector "
                f"list {[d.name for d in self._detectors]!s}."
            )

        if self._escalation_policy is not None:
            enabled_names = [d.name for d in self._detectors]
            for referenced in self._escalation_policy.detectors_referenced:
                if referenced not in enabled_names:
                    raise ValueError(
                        f"Escalation policy references detector {referenced!s}, which is not in the "
                        f"enabled detector list {enabled_names!s}."
                    )

        # Hidden-state preflight, before any request: the provider must be able
        # to serve the layers (checked first: on vLLM it needs no tokenizer
        # download), and every probe must fit the backbone and prompt under test.
        self._hidden_state_layers = _hidden_state_layers(self._detectors)
        if self._hidden_state_layers is not None:
            self._slm.check_hidden_state_request(self._hidden_state_layers)
        sep_decides = (
            self._escalation_policy is None
            and self._escalation_detector == DetectorName.SEMANTIC_ENTROPY_PROBE
        )
        for detector in self._detectors:
            if isinstance(detector, SemanticEntropyProbeDetector):
                _check_sep_probe_fits(detector, self._slm, settings.answer_instruction)
                if sep_decides:
                    _report_sep_threshold(detector, self._escalation_threshold)

    @property
    def slm(self) -> SamplingProvider:
        """The sampling provider this evaluator drives (read-only)."""
        return self._slm

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """The detectors this evaluator runs, in order (read-only)."""
        return tuple(self._detectors)

    @property
    def judge(self) -> Judge | None:
        """The judge escalated prompts go to, or None (read-only)."""
        return self._judge

    def evaluate(self, prompt: str, *, trace_semantic_entropy: bool = False) -> DetectionResult:
        """Sample, score and (optionally) judge one prompt. `trace_semantic_entropy`
        is passed to `evaluate_responses`."""
        try:
            need_samples = any(d.requires_samples for d in self._detectors)

            sample_set = self._slm.sample(prompt) if need_samples else None
            greedy = self._slm.greedy(prompt)
            pooled_hidden_state = (
                self._slm.pooled_hidden_state(prompt, layers=self._hidden_state_layers)
                if self._hidden_state_layers is not None
                else None
            )

            return evaluate_responses(
                prompt=prompt,
                samples=sample_set,
                detectors=self._detectors,
                pooled_hidden_state=pooled_hidden_state,
                escalation_detector=self._escalation_detector,
                escalation_threshold=self._escalation_threshold,
                judge=self._judge,
                greedy_answer=greedy,
                escalation_policy=self._escalation_policy,
                trace_semantic_entropy=trace_semantic_entropy,
            )
        except Exception as e:
            raise RuntimeError(
                f"ResponseEvaluator.evaluate failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e
