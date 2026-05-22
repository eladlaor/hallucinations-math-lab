"""Detector Agent — orchestrator over a configured list of detectors.

The agent owns four responsibilities and nothing more:
  1. Drive the SLM provider to produce a SampleSet and (optionally) a pooled
     hidden state, *only when* at least one configured detector needs each.
  2. Build a DetectionContext once.
  3. Dispatch the context to every configured detector and collect their
     DetectorScores.
  4. Apply the escalation policy: compare the escalation detector's score to
     its threshold.

The detector implementations live in `services/`. The agent is unaware of how
SE, PE, or SEP are computed.
"""

import logging

from taste.config import get_settings
from taste.constants import LOG_EXTRA_KEY, AnswerSource, DetectorName
from taste.custom_types.domain_types import DetectionResult
from taste.services.detector_protocol import (
    Detector,
    DetectionContext,
    DetectorPrerequisiteError,
)
import re
from pathlib import Path

from taste.services.predictive_entropy import PredictiveEntropyDetector
from taste.services.semantic_entropy import SemanticEntropyDetector
from taste.services.semantic_entropy_probe import SemanticEntropyProbeDetector
from demo.slm_provider import SLMProvider, build_default_provider

logger = logging.getLogger(__name__)


def _resolve_sep_probe_path() -> Path:
    """Resolve the SEP probe path from settings, defaulting to a per-model slug
    under data/probes/. Returns the resolved Path even if the file does not yet
    exist — the SEP detector itself raises FileNotFoundError on construction."""
    settings = get_settings()
    if settings.sep_probe_path:
        return Path(settings.sep_probe_path)
    slug = re.sub(r"[^a-z0-9]+", "-", settings.vllm_model_id.lower()).strip("-")
    return Path("data") / "probes" / f"{slug}.pt"


def _build_default_detectors(enabled: list[DetectorName]) -> list[Detector]:
    """Map config-enabled names to concrete Detector instances."""
    detectors: list[Detector] = []
    for name in enabled:
        if name == DetectorName.PREDICTIVE_ENTROPY:
            detectors.append(PredictiveEntropyDetector())
        elif name == DetectorName.SEMANTIC_ENTROPY:
            detectors.append(SemanticEntropyDetector())
        elif name == DetectorName.SEMANTIC_ENTROPY_PROBE:
            detectors.append(SemanticEntropyProbeDetector(probe_path=_resolve_sep_probe_path()))
        else:
            raise ValueError(
                f"Detector {name!s} is in ENABLED_DETECTORS but unknown to the orchestrator."
            )
    return detectors


class DetectorAgent:
    def __init__(
        self,
        slm: SLMProvider | None = None,
        detectors: list[Detector] | None = None,
        escalation_detector: DetectorName | None = None,
        escalation_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._slm: SLMProvider = slm or build_default_provider()
        self._detectors: list[Detector] = (
            detectors if detectors is not None else _build_default_detectors(settings.enabled_detectors)
        )
        self._escalation_detector = (
            escalation_detector if escalation_detector is not None else settings.escalation_detector
        )
        self._escalation_threshold = (
            escalation_threshold if escalation_threshold is not None else settings.se_threshold
        )

        if not any(d.name == self._escalation_detector for d in self._detectors):
            raise ValueError(
                f"Escalation detector {self._escalation_detector!s} is not in the enabled detector "
                f"list {[d.name for d in self._detectors]!s}."
            )

    def detect(self, prompt: str) -> DetectionResult:
        try:
            need_samples = any(d.requires_samples for d in self._detectors)
            need_hidden_state = any(d.requires_hidden_state for d in self._detectors)

            sample_set = self._slm.sample(prompt) if need_samples else None
            greedy = self._slm.greedy(prompt)
            pooled_hidden_state = (
                self._slm.pooled_hidden_state(prompt) if need_hidden_state else None
            )

            if need_hidden_state and pooled_hidden_state is None:
                raise RuntimeError(
                    "A configured detector requires pooled hidden states, but the SLM provider "
                    f"({type(self._slm).__name__}) returned None. For VLLMOpenAIProvider, ensure "
                    "VLLM_EMBED_BASE_URL is set and points at a vLLM instance launched with --task embed."
                )

            ctx = DetectionContext(
                prompt=prompt,
                samples=sample_set,
                pooled_hidden_state=pooled_hidden_state,
            )

            scores: dict[DetectorName, float] = {}
            clusters = None
            for detector in self._detectors:
                try:
                    result = detector.score(ctx)
                except DetectorPrerequisiteError:
                    raise
                scores[result.name] = result.score
                if result.clusters is not None:
                    clusters = result.clusters

            escalation_score = scores[self._escalation_detector]
            should_escalate = escalation_score > self._escalation_threshold

            logger.info(
                "Detector verdict.",
                extra={
                    LOG_EXTRA_KEY: {
                        "scores": {str(k): round(v, 4) for k, v in scores.items()},
                        "escalation_detector": str(self._escalation_detector),
                        "escalation_threshold": self._escalation_threshold,
                        "escalate": should_escalate,
                        "n_clusters": len(clusters) if clusters else 0,
                    }
                },
            )

            return DetectionResult(
                prompt=prompt,
                greedy_answer=greedy,
                samples=[s.text for s in sample_set.samples] if sample_set else [],
                detector_scores=scores,
                clusters=clusters,
                escalation_detector=self._escalation_detector,
                escalation_threshold=self._escalation_threshold,
                escalated=should_escalate,
                final_answer=greedy,
                final_answer_source=AnswerSource.SMALL_MODEL_GREEDY,
            )
        except Exception as e:
            raise RuntimeError(
                f"DetectorAgent.detect failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e
