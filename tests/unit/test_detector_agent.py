"""Detector agent logic test using a fake vLLM client. Verifies the escalation flag flips
at the SE threshold without touching the real model."""

from unittest.mock import patch

import pytest

from seh_handler.agents.detector_agent import DetectorAgent
from seh_handler.custom_types.domain_types import Sample, SampleSet, TokenLogprob


class FakeVLLM:
    def __init__(self, answers: list[str]) -> None:
        self._answers = answers

    def sample(self, prompt: str) -> SampleSet:
        return SampleSet(
            prompt=prompt,
            samples=[
                Sample(text=a, token_logprobs=[TokenLogprob(token="t", logprob=-0.5)])
                for a in self._answers
            ],
        )

    def greedy(self, prompt: str) -> str:
        return self._answers[0]


class FakeNLI:
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a.strip().lower() == b.strip().lower()


@pytest.fixture
def patch_nli():
    with patch("seh_handler.services.semantic_entropy.get_nli_scorer", return_value=FakeNLI()):
        yield


def test_low_se_does_not_escalate(patch_nli):
    vllm = FakeVLLM(["Paris"] * 10)
    agent = DetectorAgent(vllm=vllm)
    agent._threshold = 0.5
    res = agent.detect("What is the capital of France?")
    assert res.semantic_entropy == pytest.approx(0.0, abs=1e-6)
    assert res.escalated is False


def test_high_se_triggers_escalation_flag(patch_nli):
    vllm = FakeVLLM([f"distinct-{i}" for i in range(10)])
    agent = DetectorAgent(vllm=vllm)
    agent._threshold = 0.5
    res = agent.detect("Something the model can't answer.")
    assert res.semantic_entropy > 0.5
    assert res.escalated is True
