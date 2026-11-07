"""Fakes and fixtures for the web playground API tests.

The fakes stand in for DeBERTa, the model under test and the Gemini judge, so
every API test runs on CPU in milliseconds and never downloads a model.
"""

import json
import re
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from demo.web.app import create_app
from hallucinations_math_lab import Sample, SampleSet, TokenLogprob
from hallucinations_math_lab.config import Settings
from tests.unit.sep_probe_fixtures import ProbeBackboneProvider

PARIS = "paris"
TEST_PROJECT = "test-project"
FAKE_MODEL_ID = "fake/model"
FAKE_GREEDY = "Paris"
FAKE_JUDGE_ANSWER = "Paris (judge)"
FAKE_JUDGE_REASONING = "The capital of France is Paris."
PROVIDER_ERROR = "model weights not found: fake/model"


def _normalize(text: str) -> str:
    return re.sub(rf"[{re.escape(string.punctuation)}\s]+", " ", text.lower()).strip()


class KeywordNLI:
    """Two texts mean the same thing when both mention Paris, or when they are
    equal after lowercasing and dropping punctuation. Enough to reproduce the
    real DeBERTa clusters on the three preset scenarios."""

    def __init__(self) -> None:
        self.calls = 0

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.calls += 1
        na, nb = _normalize(a), _normalize(b)
        return (PARIS in na and PARIS in nb) or na == nb


class FakeProvider(ProbeBackboneProvider):
    """Answers every prompt with the given sample texts; greedy is the first."""

    def __init__(self, texts: list[str]) -> None:
        self.model_id = FAKE_MODEL_ID
        self._texts = texts

    def sample(self, prompt: str) -> SampleSet:
        return SampleSet(
            prompt=prompt,
            samples=[
                Sample(text=t, token_logprobs=[TokenLogprob(token=t, logprob=-0.5)])
                for t in self._texts
            ],
        )

    def greedy(self, prompt: str) -> str:
        return self._texts[0]

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]) -> None:
        return None


class FailingProvider(FakeProvider):
    """Fails the way a missing or unreachable model does: on first use."""

    def __init__(self) -> None:
        super().__init__([FAKE_GREEDY])

    def sample(self, prompt: str) -> SampleSet:
        raise RuntimeError(PROVIDER_ERROR)


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, prompt: str) -> tuple[str, str]:
        self.calls += 1
        return FAKE_JUDGE_ANSWER, FAKE_JUDGE_REASONING


def benchmark_row(
    qid: str, correct: bool, se: float, pe: float, model_id: str = FAKE_MODEL_ID
) -> dict[str, Any]:
    """One run_benchmark.py results row."""
    return {
        "id": qid,
        "question": f"question {qid}?",
        "ground_truth_aliases": ["right"],
        "greedy_answer": "right" if correct else "wrong",
        "detector_scores": {"semantic_entropy": se, "predictive_entropy": pe},
        "model_id": model_id,
        "sample_n": 10,
        "sample_temperature": 1.0,
        "max_tokens": 64,
        "answer_instruction": "Answer the following question as briefly as possible.",
        "sep_probe": None,
    }


def write_run(path: Path, model_id: str = FAKE_MODEL_ID) -> None:
    """A small labeled run in which hallucinated rows tend to score higher."""
    rows = [
        benchmark_row(f"q{i}", correct=i % 2 == 0, se=0.1 * i, pe=0.05 * (i % 3), model_id=model_id)
        for i in range(12)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    d = tmp_path / "static"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>playground</title>")
    return d


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def nli() -> KeywordNLI:
    return KeywordNLI()


@pytest.fixture
def judge() -> FakeJudge:
    return FakeJudge()


ClientFactory = Callable[..., TestClient]


@pytest.fixture
def make_client(
    nli: KeywordNLI, judge: FakeJudge, static_dir: Path, results_dir: Path
) -> ClientFactory:
    """A TestClient over create_app with fakes; override the provider or settings."""

    def _make(
        provider: FakeProvider | None = None,
        settings: Settings | None = None,
    ) -> TestClient:
        chosen = provider or FakeProvider([FAKE_GREEDY] * 10)
        app = create_app(
            nli_scorer=nli,
            provider_factory=lambda: chosen,
            judge_factory=lambda: judge,
            results_dir=results_dir,
            static_dir=static_dir,
            settings=settings or Settings(gcp_project_id=TEST_PROJECT),
        )
        return TestClient(app)

    return _make
