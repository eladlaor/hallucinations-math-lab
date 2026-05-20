"""Judge Agent. Calls Gemini via Vertex when the detector says SE > threshold."""

import logging

from seh_handler.constants import LOG_EXTRA_KEY, AnswerSource
from seh_handler.custom_types.domain_types import DetectionResult
from seh_handler.services.judge import GeminiJudge

logger = logging.getLogger(__name__)


class JudgeAgent:
    def __init__(self, judge: GeminiJudge | None = None) -> None:
        self._judge = judge or GeminiJudge()

    def maybe_escalate(self, detection: DetectionResult) -> DetectionResult:
        if not detection.escalated:
            return detection

        try:
            answer, reasoning = self._judge.answer(detection.prompt)
        except Exception as e:
            raise RuntimeError(
                f"JudgeAgent escalation failed for prompt={detection.prompt[:80]!r}: {e!s}"
            ) from e

        logger.info(
            "Judge escalation produced answer.",
            extra={
                LOG_EXTRA_KEY: {
                    "se": round(detection.semantic_entropy, 4),
                    "judge_answer_preview": answer[:80],
                }
            },
        )

        return detection.model_copy(
            update={
                "final_answer": answer,
                "final_answer_source": AnswerSource.JUDGE_MODEL,
                "judge_reasoning": reasoning,
            }
        )
