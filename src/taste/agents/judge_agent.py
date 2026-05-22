"""Judge Agent. Calls Gemini via Vertex when the detector says SE > threshold."""

import logging

from taste.constants import LOG_EXTRA_KEY, AnswerSource
from taste.custom_types.domain_types import DetectionResult
from taste.services.judge import GeminiJudge

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

        escalation_score = detection.detector_scores.get(detection.escalation_detector)
        logger.info(
            "Judge escalation produced answer.",
            extra={
                LOG_EXTRA_KEY: {
                    "escalation_detector": str(detection.escalation_detector),
                    "escalation_score": round(escalation_score, 4) if escalation_score is not None else None,
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
