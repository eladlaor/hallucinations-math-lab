"""POST /api/score — the Score answers tab (way in 1, no model generates answers)."""

import math

import pytest

from hallucinations_math_lab.config import Settings
from tests.unit.web.conftest import ClientFactory, KeywordNLI

EXPECTED = {
    # preset: (PE, SE, number of clusters) — the example script's printed table.
    "consistent": (0.050, 0.000, 1),
    "paraphrased": (0.600, 0.000, 1),
    "divergent": (0.400, 1.834, 7),
}


def _preset_request(preset: dict) -> dict:
    return {
        "question": preset["question"],
        "answers": [
            {"text": s["text"], "token_logprobs": s["token_logprobs"]} for s in preset["samples"]
        ],
    }


@pytest.mark.parametrize("name", list(EXPECTED))
def test_presets_score_like_the_example(make_client: ClientFactory, name: str) -> None:
    client = make_client()
    preset = next(p for p in client.get("/api/presets").json() if p["name"] == name)

    body = client.post("/api/score", json=_preset_request(preset)).json()

    pe, se, n_clusters = EXPECTED[name]
    assert body["predictive_entropy"] == pytest.approx(pe, abs=5e-4)
    assert body["semantic_entropy"] == pytest.approx(se, abs=5e-4)
    assert len(body["clusters"]) == n_clusters
    assert body["n_answers"] == len(preset["samples"])


def test_clusters_carry_member_indices_and_a_representative(make_client: ClientFactory) -> None:
    body = (
        make_client()
        .post(
            "/api/score",
            json={
                "question": "Who wrote 1984?",
                "answers": [{"text": "Orwell"}, {"text": "Huxley"}, {"text": "orwell."}],
            },
        )
        .json()
    )

    assert body["clusters"] == [
        {"member_indices": [0, 2], "representative_text": "Orwell"},
        {"member_indices": [1], "representative_text": "Huxley"},
    ]
    p = [2 / 3, 1 / 3]
    assert body["semantic_entropy"] == pytest.approx(-sum(x * math.log(x) for x in p), abs=1e-6)


def test_typed_answers_without_logprobs_score_se_but_not_pe(make_client: ClientFactory) -> None:
    body = (
        make_client()
        .post(
            "/api/score",
            json={
                "question": "Capital of France?",
                "answers": [{"text": "Paris"}, {"text": "Lyon"}],
            },
        )
        .json()
    )

    assert body["predictive_entropy"] is None
    assert body["semantic_entropy"] == pytest.approx(math.log(2), abs=1e-6)


def test_pe_is_null_when_only_some_answers_carry_logprobs(make_client: ClientFactory) -> None:
    """A partial PE would silently average a subset; the API refuses to guess."""
    body = (
        make_client()
        .post(
            "/api/score",
            json={
                "question": "Capital of France?",
                "answers": [
                    {"text": "Paris", "token_logprobs": [{"token": "Paris", "logprob": -0.1}]},
                    {"text": "Lyon"},
                ],
            },
        )
        .json()
    )

    assert body["predictive_entropy"] is None


def test_score_returns_the_configured_threshold(make_client: ClientFactory) -> None:
    client = make_client(settings=Settings(se_threshold=0.42))
    body = client.post("/api/score", json={"question": "Q?", "answers": [{"text": "A"}]}).json()

    assert body["se_threshold"] == 0.42


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Q?", "answers": []},
        {"question": "", "answers": [{"text": "A"}]},
        {"question": "Q?", "answers": [{"text": "   "}]},
        {"question": "Q?"},
    ],
)
def test_invalid_score_requests_are_422(make_client: ClientFactory, payload: dict) -> None:
    assert make_client().post("/api/score", json=payload).status_code == 422


def test_nli_failure_is_surfaced_with_its_message(
    make_client: ClientFactory, nli: KeywordNLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(a: str, b: str) -> bool:
        raise RuntimeError("Failed to load NLI model 'microsoft/deberta-large-mnli'")

    monkeypatch.setattr(nli, "bidirectional_entails", broken)
    response = make_client().post(
        "/api/score", json={"question": "Q?", "answers": [{"text": "A"}, {"text": "B"}]}
    )

    assert response.status_code == 500
    assert "Failed to load NLI model" in response.json()["detail"]


def test_answers_with_logprobs_use_likelihood_weighted_se(make_client: ClientFactory) -> None:
    """Two meanings, the first far likelier: the weighted SE sits well below ln 2."""
    body = (
        make_client()
        .post(
            "/api/score",
            json={
                "question": "Capital of France?",
                "answers": [
                    {"text": "Paris", "token_logprobs": [{"token": "Paris", "logprob": -0.1}]},
                    {"text": "Lyon", "token_logprobs": [{"token": "Lyon", "logprob": -3.0}]},
                ],
            },
        )
        .json()
    )

    w = [math.exp(-0.1), math.exp(-3.0)]
    p = [x / sum(w) for x in w]
    assert body["se_estimator"] == "likelihood_weighted"
    assert body["semantic_entropy"] == pytest.approx(-sum(x * math.log(x) for x in p), abs=1e-6)


@pytest.mark.parametrize(
    "answers",
    [
        [{"text": "Paris"}, {"text": "Lyon"}],
        [
            {"text": "Paris", "token_logprobs": [{"token": "Paris", "logprob": -0.1}]},
            {"text": "Lyon"},
        ],
    ],
    ids=["none", "some"],
)
def test_answers_without_logprobs_on_every_answer_use_discrete_se(
    make_client: ClientFactory, answers: list[dict]
) -> None:
    """Typed answers carry no likelihoods, so SE counts answers per meaning, and says so."""
    body = (
        make_client()
        .post("/api/score", json={"question": "Capital of France?", "answers": answers})
        .json()
    )

    assert body["se_estimator"] == "discrete"
    assert body["semantic_entropy"] == pytest.approx(math.log(2), abs=1e-6)
