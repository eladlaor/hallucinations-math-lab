"""POST /api/ask — the Ask the model tab (way in 3), mirroring `demo.cli detect` / `chat`."""

import pytest

from hallucinations_math_lab.config import Settings
from tests.unit.web.conftest import (
    FAKE_GREEDY,
    FAKE_JUDGE_ANSWER,
    FAKE_JUDGE_REASONING,
    PROVIDER_ERROR,
    ClientFactory,
    FailingProvider,
    FakeJudge,
    FakeProvider,
)

CONSISTENT = ["Paris"] * 10
DIVERGENT = ["Ride", "Tereshkova", "Whitson", "Koch", "Jemison", "Collins", "Ride", "Ride"]


def _ask(client, mode: str, question: str = "Capital of France?"):
    return client.post("/api/ask", json={"question": question, "mode": mode})


def test_detect_returns_scores_clusters_and_the_greedy_answer(
    make_client: ClientFactory, judge: FakeJudge
) -> None:
    body = _ask(make_client(provider=FakeProvider(CONSISTENT)), "detect").json()

    assert body["greedy_answer"] == FAKE_GREEDY
    assert body["samples"] == CONSISTENT
    assert set(body["detector_scores"]) == {"predictive_entropy", "semantic_entropy"}
    assert body["detector_scores"]["semantic_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert len(body["clusters"]) == 1
    assert body["escalated"] is False
    assert body["final_answer_source"] == "small_model_greedy"
    assert judge.calls == 0


def test_detect_never_calls_the_judge_even_when_it_escalates(
    make_client: ClientFactory, judge: FakeJudge
) -> None:
    body = _ask(make_client(provider=FakeProvider(DIVERGENT)), "detect").json()

    assert body["escalated"] is True
    assert body["final_answer"] == DIVERGENT[0]
    assert body["final_answer_source"] == "small_model_greedy"
    assert judge.calls == 0


def test_chat_hands_an_escalated_question_to_the_judge(
    make_client: ClientFactory, judge: FakeJudge
) -> None:
    body = _ask(make_client(provider=FakeProvider(DIVERGENT)), "chat").json()

    assert body["escalated"] is True
    assert body["final_answer"] == FAKE_JUDGE_ANSWER
    assert body["final_answer_source"] == "judge_model"
    assert body["judge_reasoning"] == FAKE_JUDGE_REASONING
    assert judge.calls == 1


def test_chat_keeps_the_model_answer_when_it_does_not_escalate(
    make_client: ClientFactory, judge: FakeJudge
) -> None:
    body = _ask(make_client(provider=FakeProvider(CONSISTENT)), "chat").json()

    assert body["escalated"] is False
    assert body["final_answer_source"] == "small_model_greedy"
    assert judge.calls == 0


def test_chat_without_judge_credentials_is_refused_with_the_reason(
    make_client: ClientFactory, judge: FakeJudge
) -> None:
    response = _ask(make_client(settings=Settings(gcp_project_id="")), "chat")

    assert response.status_code == 400
    assert "GCP_PROJECT_ID" in response.json()["detail"]
    assert judge.calls == 0


def test_provider_failure_is_surfaced_with_its_message(make_client: ClientFactory) -> None:
    response = _ask(make_client(provider=FailingProvider()), "detect")

    assert response.status_code == 500
    assert PROVIDER_ERROR in response.json()["detail"]


def test_the_provider_is_built_once(
    make_client: ClientFactory, nli, judge, static_dir, results_dir
):
    from fastapi.testclient import TestClient

    from demo.web.app import create_app

    built: list[FakeProvider] = []

    def factory() -> FakeProvider:
        built.append(FakeProvider(CONSISTENT))
        return built[-1]

    client = TestClient(
        create_app(
            nli_scorer=nli,
            provider_factory=factory,
            judge_factory=lambda: judge,
            results_dir=results_dir,
            static_dir=static_dir,
            settings=Settings(),
        )
    )
    _ask(client, "detect")
    _ask(client, "detect")

    assert len(built) == 1


@pytest.mark.parametrize(
    "payload",
    [{"question": "", "mode": "detect"}, {"question": "Q?", "mode": "shout"}, {"question": "Q?"}],
)
def test_invalid_ask_requests_are_422(make_client: ClientFactory, payload: dict) -> None:
    assert make_client().post("/api/ask", json=payload).status_code == 422
