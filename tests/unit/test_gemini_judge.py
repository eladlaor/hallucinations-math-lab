"""GeminiJudge tests with an injected fake `google.genai` client (no GCP).

Contract pinned here:
  - The request asks for JSON validated against `JudgeVerdict`, at the
    configured judge temperature, with the prompt passed through verbatim.
  - A parsed verdict is returned as `(answer, reasoning)`.
  - A response that does not parse into a verdict fails fast.
  - Without an injected client, an empty GCP_PROJECT_ID fails fast.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hallucinations_math_lab.custom_types.llm_schemas import JudgeVerdict
from hallucinations_math_lab.judges.gemini import GeminiJudge

pytest.importorskip("google.genai")

MODEL_ID = "gemini-test"
PROMPT = "Who was the first person to walk on the Moon?"


def _client_returning(parsed: object, text: str = "") -> MagicMock:
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=parsed, text=text)
    return client


def test_answer_returns_parsed_verdict_fields():
    verdict = JudgeVerdict(answer=" Neil Armstrong ", reasoning=" Apollo 11, 1969. ")
    judge = GeminiJudge(model_id=MODEL_ID, client=_client_returning(verdict))

    assert judge.answer(PROMPT) == ("Neil Armstrong", "Apollo 11, 1969.")


def test_answer_requests_structured_output():
    client = _client_returning(JudgeVerdict(answer="a", reasoning="r"))
    GeminiJudge(model_id=MODEL_ID, client=client).answer(PROMPT)

    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == MODEL_ID
    assert kwargs["contents"] == PROMPT
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema is JudgeVerdict
    assert kwargs["config"].temperature == 0.0


def test_answer_fails_fast_on_unparseable_response():
    judge = GeminiJudge(model_id=MODEL_ID, client=_client_returning(None, text="not json"))

    with pytest.raises(RuntimeError, match="no valid JudgeVerdict"):
        judge.answer(PROMPT)


def test_missing_project_id_fails_fast_without_injected_client(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("GCP_PROJECT_ID", "")
    try:
        with pytest.raises(RuntimeError, match="GCP_PROJECT_ID is empty"):
            GeminiJudge(model_id=MODEL_ID).answer(PROMPT)
    finally:
        get_settings.cache_clear()
