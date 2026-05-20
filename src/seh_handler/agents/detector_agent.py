"""Detector Agent. Wraps the SE pipeline as a callable unit.

Currently a plain class implementing the detection logic. The ADK app in `main.py`
exposes it as a tool/agent. This separation keeps the SE math testable without
pulling in google-adk at import time.
"""

import logging

from seh_handler.config import get_settings
from seh_handler.constants import LOG_EXTRA_KEY, AnswerSource
from seh_handler.custom_types.domain_types import DetectionResult
from seh_handler.services.predictive_entropy import length_normalized_predictive_entropy
from seh_handler.services.semantic_entropy import compute_semantic_entropy
from seh_handler.services.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class DetectorAgent:
    def __init__(self, vllm: VLLMClient | None = None) -> None:
        self._vllm = vllm or VLLMClient()
        self._threshold = get_settings().se_threshold

    def detect(self, prompt: str) -> DetectionResult:
        try:
            sample_set = self._vllm.sample(prompt)
            greedy = self._vllm.greedy(prompt)
            pe = length_normalized_predictive_entropy(sample_set)
            se_result = compute_semantic_entropy(sample_set)
            should_escalate = se_result.semantic_entropy > self._threshold

            logger.info(
                "Detector verdict.",
                extra={
                    LOG_EXTRA_KEY: {
                        "se": round(se_result.semantic_entropy, 4),
                        "pe": round(pe, 4),
                        "threshold": self._threshold,
                        "escalate": should_escalate,
                        "n_clusters": len(se_result.clusters),
                    }
                },
            )

            return DetectionResult(
                prompt=prompt,
                greedy_answer=greedy,
                samples=[s.text for s in sample_set.samples],
                predictive_entropy=pe,
                semantic_entropy=se_result.semantic_entropy,
                clusters=se_result.clusters,
                escalated=should_escalate,
                final_answer=greedy,
                final_answer_source=AnswerSource.SMALL_MODEL_GREEDY,
            )
        except Exception as e:
            raise RuntimeError(
                f"DetectorAgent.detect failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e
