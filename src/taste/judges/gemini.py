"""Gemini judge via Vertex AI Generative SDK. Lazy-imported so this module
loads without GCP credentials in the environment — the dependency on
google-cloud-aiplatform is only paid when GeminiJudge is actually
constructed and invoked."""

import logging

from taste.config import get_settings
from taste.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)


class GeminiJudge:
    """Vertex AI Gemini wrapper implementing the `Judge` protocol.

    Reference implementation only — consumers running outside the demo's
    Vertex deployment are encouraged to supply their own Judge impl
    (any class with `.answer(prompt) -> (str, str)`)."""

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or get_settings().judge_model_id
        self._model = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        settings = get_settings()
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            if not settings.gcp_project_id:
                raise RuntimeError("GCP_PROJECT_ID is empty; cannot init Vertex AI.")
            vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)
            self._model = GenerativeModel(self._model_id)
            logger.info(
                "Initialized Vertex Gemini judge.",
                extra={
                    LOG_EXTRA_KEY: {
                        "model_id": self._model_id,
                        "project": settings.gcp_project_id,
                        "region": settings.gcp_region,
                    }
                },
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to init Gemini judge model={self._model_id!r}: {e!s}"
            ) from e

    def answer(self, prompt: str) -> tuple[str, str]:
        """Return (judge_answer, judge_reasoning). Reasoning is best-effort."""
        self._ensure_loaded()
        framed = (
            "You are a careful expert. Answer the following question concisely and "
            "accurately. If you do not know, say so explicitly. Then on a new line, "
            "after the marker 'REASONING:', briefly justify your answer.\n\n"
            f"Question: {prompt}\n\nAnswer:"
        )
        try:
            resp = self._model.generate_content(framed)  # type: ignore[union-attr]
            text = (resp.text or "").strip()
        except Exception as e:
            raise RuntimeError(
                f"Gemini judge generation failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e

        if "REASONING:" in text:
            answer, _, reasoning = text.partition("REASONING:")
            return answer.strip(), reasoning.strip()
        return text, ""
