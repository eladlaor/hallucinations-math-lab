"""Re-scoring a saved benchmark run without the model.

A run_benchmark.py results file keeps every sampled answer with its token
logprobs, so the sample-based detectors (PE, SE, discrete SE) can be re-run on
it offline: same answers, new detectors or threshold, no model call. Pins:

- run_benchmark.py saves the token logprobs it needs;
- re-scoring with the run's own detectors reproduces the run's scores exactly;
- a new threshold changes the escalation decision, and an escalated row takes
  the judge answer the run already recorded;
- the output is a normal results file that compute_auroc.py accepts;
- old files without logprobs, and SEP, are refused rather than scored wrongly.
"""

import json
from pathlib import Path

import pytest
import torch

from eval.compute_auroc import main as compute_auroc
from eval.rescore_benchmark import rescore
from eval.run_benchmark import run as run_benchmark
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    SemanticEntropyDetector,
)
from hallucinations_math_lab.services.semantic_entropy_probe import (
    SemanticEntropyProbeDetector,
    SEPProbe,
)
from tests.unit.test_benchmark_pipeline import (
    HIDDEN_DIM,
    CountingJudge,
    EqualityNLI,
    ScatteringSLM,
    _write_questions,
)


def _detectors():
    return [PredictiveEntropyDetector(), SemanticEntropyDetector(scorer=EqualityNLI())]


def _saved_run(tmp_path: Path, judge: CountingJudge | None = None) -> Path:
    questions = tmp_path / "questions.jsonl"
    results = tmp_path / "results.jsonl"
    _write_questions(questions)
    evaluator = ResponseEvaluator(
        slm=ScatteringSLM(),
        detectors=_detectors(),
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
        judge=judge,
    )
    run_benchmark(questions, results, judge=judge is not None, evaluator=evaluator)
    return results


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_benchmark_saves_each_samples_token_logprobs(tmp_path):
    rows = _rows(_saved_run(tmp_path))
    first = rows[0]
    assert len(first["sample_token_logprobs"]) == len(first["samples"])
    assert first["sample_token_logprobs"][0] == [{"token": "Paris", "logprob": -0.5}]


def test_rescoring_with_the_same_detectors_reproduces_the_run(tmp_path):
    original = _saved_run(tmp_path)
    rescored = tmp_path / "rescored.jsonl"
    rescore(
        original,
        rescored,
        detectors=_detectors(),
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
    )
    before, after = _rows(original), _rows(rescored)
    assert [r["detector_scores"] for r in after] == [r["detector_scores"] for r in before]
    assert [r["escalated"] for r in after] == [r["escalated"] for r in before]
    assert [r["n_clusters"] for r in after] == [r["n_clusters"] for r in before]


def test_a_new_threshold_changes_escalation_and_reuses_the_recorded_judge_answer(tmp_path):
    original = _saved_run(tmp_path, judge=CountingJudge())
    rescored = tmp_path / "rescored.jsonl"
    rescore(
        original,
        rescored,
        detectors=_detectors(),
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=-1.0,  # every row escalates
    )
    rows = _rows(rescored)
    assert all(r["escalated"] for r in rows)
    assert all(r["final_answer"] == "judge says" for r in rows)
    assert all(r["final_answer_source"] == "judge_model" for r in rows)


def test_without_a_recorded_judge_answer_an_escalated_row_keeps_the_greedy_answer(tmp_path):
    original = _saved_run(tmp_path)
    rescored = tmp_path / "rescored.jsonl"
    rescore(
        original,
        rescored,
        detectors=_detectors(),
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=-1.0,
    )
    rows = _rows(rescored)
    assert all(r["escalated"] for r in rows)
    assert [r["final_answer"] for r in rows] == [r["greedy_answer"] for r in rows]


def test_the_rescored_file_is_a_normal_results_file(tmp_path, capsys):
    original = _saved_run(tmp_path)
    rescored = tmp_path / "rescored.jsonl"
    rescore(
        original,
        rescored,
        detectors=_detectors(),
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
    )
    compute_auroc(rescored)
    report = json.loads(capsys.readouterr().out)
    assert report["run_config"]["model_id"] == "fake-slm"
    assert report["detectors"]["semantic_entropy"]["auroc"] == 1.0


def test_a_file_saved_without_token_logprobs_is_refused(tmp_path):
    original = _saved_run(tmp_path)
    rows = _rows(original)
    for row in rows:
        del row["sample_token_logprobs"]
    original.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="sample_token_logprobs"):
        rescore(
            original,
            tmp_path / "rescored.jsonl",
            detectors=_detectors(),
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=1.0,
        )


def test_sep_is_refused_because_it_needs_the_running_model(tmp_path):
    original = _saved_run(tmp_path)
    probe_path = tmp_path / "probe.pt"
    torch.save(SEPProbe(HIDDEN_DIM).state_dict(), probe_path)
    with pytest.raises(ValueError, match="hidden state"):
        rescore(
            original,
            tmp_path / "rescored.jsonl",
            detectors=[*_detectors(), SemanticEntropyProbeDetector(probe_path=probe_path)],
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=1.0,
        )


def test_rescoring_refuses_to_overwrite_its_input(tmp_path):
    original = _saved_run(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        rescore(
            original,
            original,
            detectors=_detectors(),
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=1.0,
        )
