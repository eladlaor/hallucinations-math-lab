"""Web playground skeleton: status, presets, the static UI and the committed build."""

import importlib
from pathlib import Path

import pytest

from demo.web.app import create_app
from demo.web.constants import DEFAULT_HOST, DEFAULT_PORT
from hallucinations_math_lab import DetectorName
from hallucinations_math_lab.config import Settings
from tests.unit.web.conftest import TEST_PROJECT, ClientFactory, KeywordNLI

# Imported by name at runtime: mypy maps examples/ as top-level modules (the
# examples import each other as scripts), so a static import would clash.
SCENARIOS = importlib.import_module("examples.score_precollected_samples").SCENARIOS


def test_status_reports_provider_model_judge_and_threshold(make_client: ClientFactory) -> None:
    settings = Settings(
        gcp_project_id=TEST_PROJECT,
        slm_provider="vllm",
        vllm_model_id="google/gemma-4-E4B-it",
        se_threshold=0.7,
        sep_threshold=0.3,
    )
    body = make_client(settings=settings).get("/api/status").json()

    assert body == {
        "provider": "vllm",
        "model_id": "google/gemma-4-E4B-it",
        "judge_configured": True,
        "judge_unavailable_reason": None,
        "se_threshold": 0.7,
        "sep_threshold": 0.3,
        "escalation_threshold": 0.7,
        "nli_model_id": settings.nli_model_id,
    }


def test_status_escalation_threshold_follows_the_escalation_detector(
    make_client: ClientFactory,
) -> None:
    settings = Settings(
        se_threshold=0.7,
        sep_threshold=0.3,
        enabled_detectors=[DetectorName.SEMANTIC_ENTROPY_PROBE],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
    )
    body = make_client(settings=settings).get("/api/status").json()

    assert body["escalation_threshold"] == 0.3


def test_status_explains_why_the_judge_is_unavailable(make_client: ClientFactory) -> None:
    body = make_client(settings=Settings(gcp_project_id="")).get("/api/status").json()

    assert body["judge_configured"] is False
    assert "GCP_PROJECT_ID" in body["judge_unavailable_reason"]


def test_presets_are_the_example_scenarios(make_client: ClientFactory) -> None:
    body = make_client().get("/api/presets").json()

    assert [p["name"] for p in body] == list(SCENARIOS)
    for preset in body:
        question, samples = SCENARIOS[preset["name"]]
        assert preset["question"] == question
        assert preset["samples"] == [s.model_dump() for s in samples]


def test_root_serves_the_ui(make_client: ClientFactory) -> None:
    response = make_client().get("/")

    assert response.status_code == 200
    assert "<title>playground</title>" in response.text


def test_unknown_api_route_is_a_json_404_not_the_ui(make_client: ClientFactory) -> None:
    response = make_client().get("/api/nope")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_create_app_fails_fast_without_a_built_ui(tmp_path: Path, nli: KeywordNLI) -> None:
    with pytest.raises(FileNotFoundError, match="npm run build"):
        create_app(
            nli_scorer=nli,
            provider_factory=lambda: None,  # type: ignore[arg-type, return-value]
            judge_factory=lambda: None,  # type: ignore[arg-type, return-value]
            results_dir=tmp_path,
            static_dir=tmp_path / "missing",
            settings=Settings(),
        )


def test_server_binds_localhost_on_the_documented_port() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8765
