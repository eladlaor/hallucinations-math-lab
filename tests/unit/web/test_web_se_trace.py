"""The opt-in SE trace on /api/score and /api/ask: what the clustering replay plays back.

Off by default: without `se_trace: true` the responses and the NLI calls are
exactly what they were before the trace existed.
"""

import math
from typing import Any

import pytest

from hallucinations_math_lab import DetectionResult, DetectorName, NLILabel
from hallucinations_math_lab.config import Settings
from tests.unit.web.conftest import TEST_PROJECT, ClientFactory, FakeProvider, KeywordNLI

SCORE_RESPONSE_KEYS = {
    "semantic_entropy",
    "se_estimator",
    "predictive_entropy",
    "clusters",
    "n_answers",
    "se_threshold",
}
# NLI comparisons the greedy clustering runs on each preset (each answer against
# the clusters existing so far, stopping at the first join).
PRESET_NLI_CALLS = {"paraphrased": 9, "divergent": 25}
PRESET_CLUSTERS = {"paraphrased": 1, "divergent": 7}
DIVERGENT_ANSWERS = ["Ride", "Tereshkova", "Whitson", "Ride", "Koch", "Tereshkova"]
# NLI comparisons one /api/ask detect run makes on DIVERGENT_ANSWERS with the
# default detectors (likelihood-weighted SE clusters them once): 1+2+1+3+2.
ASK_DIVERGENT_NLI_CALLS = 9
# /api/ask without the flag: every DetectionResult field but the trace, nulls included.
ASK_RESPONSE_KEYS = set(DetectionResult.model_fields) - {"se_trace"}
# The paraphrased preset's /api/score body exactly as the API returned it before
# the trace existed (captured from the pre-trace code): pins it byte for byte.
PARAPHRASED_SCORE_BODY = (
    b'{"semantic_entropy":0.0,"se_estimator":"likelihood_weighted",'
    b'"predictive_entropy":0.5999999999999999,'
    b'"clusters":[{"member_indices":[0,1,2,3,4,5,6,7,8,9],"representative_text":"Paris"}],'
    b'"n_answers":10,"se_threshold":1.0}'
)
NLI_FAILURE = "NLI model crashed"


class BidirectionalOnlyNLI:
    """A scorer with no per-direction `classify`: it can cluster but not be traced."""

    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


class FailingClassifyNLI(KeywordNLI):
    """Clusters fine untraced, but the per-direction verdict the trace needs fails."""

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        raise RuntimeError(NLI_FAILURE)


def _preset_request(client: Any, name: str, **extra: Any) -> dict:
    preset = next(p for p in client.get("/api/presets").json() if p["name"] == name)
    return {
        "question": preset["question"],
        "answers": [
            {"text": s["text"], "token_logprobs": s["token_logprobs"]} for s in preset["samples"]
        ],
        **extra,
    }


def _replay(trace: dict) -> list[list[int]]:
    """Member indices per cluster, rebuilt from the placements alone."""
    clusters: list[list[int]] = []
    for placement in trace["placements"]:
        if placement["opened_new_cluster"]:
            assert placement["cluster_index"] == len(clusters)
            clusters.append([placement["answer_index"]])
        else:
            clusters[placement["cluster_index"]].append(placement["answer_index"])
    return clusters


def _assert_comparisons_follow_the_greedy_rule(trace: dict, clusters: list[dict]) -> None:
    """Each answer is compared with clusters 0, 1, ... in order until it joins one,
    always against the cluster's first member, the representative."""
    by_answer: dict[int, list[dict]] = {}
    for comparison in trace["comparisons"]:
        by_answer.setdefault(comparison["answer_index"], []).append(comparison)
    assert list(by_answer) == sorted(by_answer)  # answers in order
    for placement in trace["placements"]:
        rows = by_answer.get(placement["answer_index"], [])
        assert [r["cluster_index"] for r in rows] == list(range(len(rows)))
        for row in rows:
            assert (
                row["representative_index"] == clusters[row["cluster_index"]]["member_indices"][0]
            )
            both = row["forward"] == "entailment" and row["backward"] == "entailment"
            assert row["joined"] is both
        assert all(not r["joined"] for r in rows[:-1])
        if placement["opened_new_cluster"]:
            assert not any(r["joined"] for r in rows)
        else:
            assert rows[-1]["joined"]
            assert rows[-1]["cluster_index"] == placement["cluster_index"]


@pytest.mark.parametrize("name", list(PRESET_CLUSTERS))
def test_score_trace_replays_to_the_returned_clusters(
    make_client: ClientFactory, nli: KeywordNLI, name: str
) -> None:
    client = make_client()
    body = client.post("/api/score", json=_preset_request(client, name, se_trace=True)).json()

    trace = body["se_trace"]
    assert len(body["clusters"]) == PRESET_CLUSTERS[name]
    assert _replay(trace["clustering"]) == [c["member_indices"] for c in body["clusters"]]
    _assert_comparisons_follow_the_greedy_rule(trace["clustering"], body["clusters"])
    assert len(trace["clustering"]["comparisons"]) == PRESET_NLI_CALLS[name]


@pytest.mark.parametrize("name", list(PRESET_CLUSTERS))
def test_score_trace_matches_the_untraced_scores(make_client: ClientFactory, name: str) -> None:
    client = make_client()
    plain = client.post("/api/score", json=_preset_request(client, name)).json()
    traced = client.post("/api/score", json=_preset_request(client, name, se_trace=True)).json()

    assert {k: traced[k] for k in SCORE_RESPONSE_KEYS} == plain


def test_score_trace_shows_the_likelihood_weights_se_used(make_client: ClientFactory) -> None:
    client = make_client()
    body = client.post(
        "/api/score", json=_preset_request(client, "divergent", se_trace=True)
    ).json()

    trace = body["se_trace"]
    assert trace["estimator"] == body["se_estimator"] == "likelihood_weighted"
    # Every divergent answer has mean token logprob -0.4.
    assert trace["answer_likelihoods"] == pytest.approx([math.exp(-0.4)] * 10)
    p = trace["cluster_probabilities"]
    assert sum(p) == pytest.approx(1.0)
    assert -sum(x * math.log(x) for x in p) == pytest.approx(body["semantic_entropy"])


def test_score_trace_of_typed_answers_uses_cluster_sizes(make_client: ClientFactory) -> None:
    body = (
        make_client()
        .post(
            "/api/score",
            json={
                "question": "Who wrote 1984?",
                "answers": [{"text": "Orwell"}, {"text": "Huxley"}, {"text": "orwell."}],
                "se_trace": True,
            },
        )
        .json()
    )

    trace = body["se_trace"]
    assert trace["estimator"] == body["se_estimator"] == "discrete"
    assert trace["answer_likelihoods"] is None
    assert trace["cluster_probabilities"] == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.parametrize("name", list(PRESET_CLUSTERS))
def test_score_without_trace_is_unchanged(
    make_client: ClientFactory, nli: KeywordNLI, name: str
) -> None:
    client = make_client()
    omitted = client.post("/api/score", json=_preset_request(client, name))
    calls_when_omitted = nli.calls
    off = client.post("/api/score", json=_preset_request(client, name, se_trace=False))

    assert set(omitted.json()) == SCORE_RESPONSE_KEYS
    assert off.content == omitted.content
    assert calls_when_omitted == PRESET_NLI_CALLS[name]
    assert nli.calls == 2 * PRESET_NLI_CALLS[name]
    assert nli.directional_calls == 0


def test_score_without_trace_matches_the_pre_trace_body(make_client: ClientFactory) -> None:
    client = make_client()
    response = client.post("/api/score", json=_preset_request(client, "paraphrased"))

    assert response.content == PARAPHRASED_SCORE_BODY


def test_score_trace_surfaces_an_nli_error_as_500(make_client: ClientFactory) -> None:
    client = make_client(nli_scorer=FailingClassifyNLI())
    response = client.post("/api/score", json=_preset_request(client, "divergent", se_trace=True))

    assert response.status_code == 500
    assert NLI_FAILURE in response.json()["detail"]


def _ask(client: Any, **extra: Any) -> Any:
    return client.post(
        "/api/ask", json={"question": "Who walked on Mars?", "mode": "detect", **extra}
    )


def test_ask_trace_replays_to_the_returned_clusters(make_client: ClientFactory) -> None:
    body = _ask(make_client(provider=FakeProvider(DIVERGENT_ANSWERS)), se_trace=True).json()

    trace = body["se_trace"]
    assert trace["estimator"] == "likelihood_weighted"
    assert _replay(trace["clustering"]) == [c["member_indices"] for c in body["clusters"]]
    _assert_comparisons_follow_the_greedy_rule(trace["clustering"], body["clusters"])


def test_ask_without_trace_is_unchanged(make_client: ClientFactory, nli: KeywordNLI) -> None:
    client = make_client(provider=FakeProvider(DIVERGENT_ANSWERS))
    omitted = _ask(client)
    calls_when_omitted = nli.calls
    off = _ask(client, se_trace=False)

    assert set(omitted.json()) == ASK_RESPONSE_KEYS
    assert off.content == omitted.content
    assert calls_when_omitted == ASK_DIVERGENT_NLI_CALLS
    assert nli.calls == 2 * ASK_DIVERGENT_NLI_CALLS
    assert nli.directional_calls == 0


def test_ask_trace_without_semantic_entropy_is_refused(make_client: ClientFactory) -> None:
    settings = Settings(
        gcp_project_id=TEST_PROJECT,
        enabled_detectors=[DetectorName.PREDICTIVE_ENTROPY],
        escalation_detector=DetectorName.PREDICTIVE_ENTROPY,
    )
    response = _ask(make_client(settings=settings), se_trace=True)

    assert response.status_code == 400
    assert "semantic_entropy" in response.json()["detail"]


def test_ask_trace_with_discrete_semantic_entropy(
    make_client: ClientFactory, nli: KeywordNLI
) -> None:
    settings = Settings(
        gcp_project_id=TEST_PROJECT,
        enabled_detectors=[DetectorName.DISCRETE_SEMANTIC_ENTROPY],
        escalation_detector=DetectorName.DISCRETE_SEMANTIC_ENTROPY,
    )
    client = make_client(provider=FakeProvider(DIVERGENT_ANSWERS), settings=settings)
    body = _ask(client, se_trace=True).json()

    trace = body["se_trace"]
    assert trace["estimator"] == "discrete"
    assert trace["answer_likelihoods"] is None
    assert _replay(trace["clustering"]) == [c["member_indices"] for c in body["clusters"]]
    # Discrete SE runs on the app's shared scorer, the injected fake.
    comparisons = trace["clustering"]["comparisons"]
    assert nli.directional_calls == len(comparisons) + sum(
        c["backward"] is not None for c in comparisons
    )


@pytest.mark.parametrize("route", ["/api/score", "/api/ask"])
def test_trace_without_a_directional_scorer_is_refused_up_front(
    make_client: ClientFactory, route: str
) -> None:
    client = make_client(
        provider=FakeProvider(DIVERGENT_ANSWERS), nli_scorer=BidirectionalOnlyNLI()
    )
    body = (
        _preset_request(client, "divergent", se_trace=True)
        if route == "/api/score"
        else {"question": "Who walked on Mars?", "mode": "detect", "se_trace": True}
    )
    refused = client.post(route, json=body)
    plain = client.post(route, json={**body, "se_trace": False})

    assert refused.status_code == 400
    assert "classify" in refused.json()["detail"]
    assert plain.status_code == 200
