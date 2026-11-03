"""End-to-end check of the eval loop: run_benchmark.py -> compute_auroc.py.

Uses an in-memory SLM and a string-equality NLI so the whole loop runs on CPU
in milliseconds. The SLM answers consistently for questions it "knows" and
scatters for the ones it doesn't, so SE should rank perfectly (AUROC 1.0).

Also pins the leakage guard: a SEP probe may not be evaluated on questions
listed in its training-id sidecar, and a probe without a sidecar is refused.
"""

import json
from pathlib import Path

import pytest
import torch

from eval.compute_auroc import main as compute_auroc
from eval.run_benchmark import run as run_benchmark
from eval.sep_probe_provenance import write_training_ids
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
)
from hallucinations_math_lab.services.semantic_entropy_probe import (
    SemanticEntropyProbeDetector,
    SEPProbe,
)

KNOWN_ANSWERS = {"Capital of France?": "Paris", "Author of 1984?": "Orwell"}
HIDDEN_DIM = 4


class ScatteringSLM:
    """Repeats the right answer for known questions, invents one per sample otherwise."""

    model_id = "fake-slm"

    def _answers(self, prompt: str) -> list[str]:
        if prompt in KNOWN_ANSWERS:
            return [KNOWN_ANSWERS[prompt]] * 10
        return [f"guess-{i}" for i in range(10)]

    def sample(self, prompt: str) -> SampleSet:
        return SampleSet(
            prompt=prompt,
            samples=[
                Sample(text=a, token_logprobs=[TokenLogprob(token=a, logprob=-0.5)])
                for a in self._answers(prompt)
            ],
        )

    def greedy(self, prompt: str) -> str:
        return self._answers(prompt)[0]

    def pooled_hidden_state(self, prompt: str):
        return None


class HiddenStateSLM(ScatteringSLM):
    def pooled_hidden_state(self, prompt: str):
        return torch.zeros(HIDDEN_DIM)


class EqualityNLI:
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a == b


def _write_questions(path: Path) -> None:
    rows = [
        {"id": "q1", "question": "Capital of France?", "ground_truth_aliases": ["Paris"]},
        {"id": "q2", "question": "Author of 1984?", "ground_truth_aliases": ["Orwell"]},
        {"id": "q3", "question": "Largest moon of Neptune?", "ground_truth_aliases": ["Triton"]},
        {"id": "adv", "question": "Who walked on Mars?", "expected_behavior": "refuse"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_benchmark_then_auroc(tmp_path, capsys):
    questions = tmp_path / "questions.jsonl"
    results = tmp_path / "results.jsonl"
    _write_questions(questions)

    evaluator = ResponseEvaluator(
        slm=ScatteringSLM(),
        detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector(scorer=EqualityNLI())],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
    )
    run_benchmark(questions, results, judge=False, evaluator=evaluator)

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(rows) == 4
    assert set(rows[0]["detector_scores"]) == {"predictive_entropy", "semantic_entropy"}
    assert rows[2]["escalated"] is True
    assert rows[0]["model_id"] == "fake-slm"
    assert {"sample_n", "sample_temperature", "max_tokens", "answer_instruction"} <= set(rows[0])
    assert rows[0]["answer_instruction"]  # the brief-answer prompt is part of the run

    compute_auroc(results)
    report = json.loads(capsys.readouterr().out)
    assert report["n_rows"] == 3  # the adversarial row has no aliases
    assert report["n_hallucinations"] == 1
    assert report["run_config"]["model_id"] == "fake-slm"
    assert report["detectors"]["semantic_entropy"]["auroc"] == 1.0


def _probe_evaluator(probe_path: Path) -> ResponseEvaluator:
    torch.save(SEPProbe(HIDDEN_DIM).state_dict(), probe_path)
    return ResponseEvaluator(
        slm=HiddenStateSLM(),
        detectors=[SemanticEntropyProbeDetector(probe_path=probe_path)],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
        escalation_threshold=1.0,
    )


def test_benchmark_refuses_questions_the_probe_was_trained_on(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_path = tmp_path / "probe.pt"
    evaluator = _probe_evaluator(probe_path)
    write_training_ids(probe_path, ["q2", "unrelated"], "fake-slm", tmp_path / "calib.jsonl")

    with pytest.raises(RuntimeError, match="used to train the SEP probe"):
        run_benchmark(questions, tmp_path / "out.jsonl", judge=False, evaluator=evaluator)
    assert not (tmp_path / "out.jsonl").exists()


def test_benchmark_accepts_a_probe_trained_on_other_questions(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_path = tmp_path / "probe.pt"
    evaluator = _probe_evaluator(probe_path)
    write_training_ids(probe_path, ["other-1", "other-2"], "fake-slm", tmp_path / "calib.jsonl")

    run_benchmark(questions, tmp_path / "out.jsonl", judge=False, evaluator=evaluator, limit=1)
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1


def test_benchmark_refuses_a_probe_without_provenance(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    evaluator = _probe_evaluator(tmp_path / "probe.pt")

    with pytest.raises(RuntimeError, match="no training-id sidecar"):
        run_benchmark(questions, tmp_path / "out.jsonl", judge=False, evaluator=evaluator)
