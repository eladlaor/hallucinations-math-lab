"""Gemini judge on Vertex AI via the Google Gen AI SDK (`google-genai`).

The SDK is imported lazily, so this module loads without the `judge` extra
or GCP credentials; both are needed only when `GeminiJudge` first answers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY
from hallucinations_math_lab.custom_types.llm_schemas import JudgeVerdict

if TYPE_CHECKING:
    from google.genai import Client

logger = logging.getLogger(__name__)

JUDGE_INSTRUCTION = (
    "You are a careful expert. Answer the question concisely and accurately. "
    "If you do not know the answer, say so explicitly rather than guessing."
)


class GeminiJudge:
    """Vertex AI Gemini implementation of the `Judge` protocol.

    The verdict comes back as JSON validated against `JudgeVerdict`, so the
    answer and its reasoning never depend on parsing free text.

    Reference implementation only: any object with
    `.answer(prompt) -> (answer, reasoning)` can serve as the judge.
    """

    def __init__(self, model_id: str | None = None, client: Client | None = None) -> None:
        """
        Args:
            model_id: Gemini model id; defaults to `JUDGE_MODEL_ID` from settings.
            client: A pre-built `google.genai.Client`. When omitted, one is
                created on first use for Vertex AI with `GCP_PROJECT_ID` /
                `GCP_REGION` and Application Default Credentials.
        """
        settings = get_settings()
        self._model_id = model_id or settings.judge_model_id
        self._temperature = settings.judge_temperature
        self._client = client

    def _ensure_client(self) -> Client:
        if self._client is not None:
            return self._client
        settings = get_settings()
        if not settings.gcp_project_id:
            raise RuntimeError("GCP_PROJECT_ID is empty; cannot create a Vertex AI client.")
        try:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=settings.gcp_project_id, location=settings.gcp_region
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Vertex AI client for judge model={self._model_id!r}: {e!s}"
            ) from e
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
        return self._client

    def answer(self, prompt: str) -> tuple[str, str]:
        """Return `(answer, reasoning)` for `prompt`."""
        from google.genai import types

        client = self._ensure_client()
        try:
            resp = client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_INSTRUCTION,
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=JudgeVerdict,
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Gemini judge generation failed for model={self._model_id!r} "
                f"prompt={prompt[:80]!r}: {e!s}"
            ) from e

        verdict = resp.parsed
        if not isinstance(verdict, JudgeVerdict):
            raise RuntimeError(
                f"Gemini judge returned no valid JudgeVerdict for model={self._model_id!r} "
                f"prompt={prompt[:80]!r}; raw text={(resp.text or '')[:200]!r}"
            )
        return verdict.answer.strip(), verdict.reasoning.strip()
