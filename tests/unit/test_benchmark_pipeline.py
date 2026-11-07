"""End-to-end check of the eval loop: run_benchmark.py -> compute_auroc.py.

Uses an in-memory SLM and a string-equality NLI so the whole loop runs on CPU
in milliseconds. The SLM answers consistently for questions it "knows" and
scatters for the ones it doesn't, so SE should rank perfectly (AUROC 1.0).

Also pins the leakage guard: a SEP probe may not be evaluated on questions
listed in its `train_ids.json`, and a probe without one is refused; and the
provenance every row records for the probe in use.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from eval.benchmark_results import run_config
from eval.compute_auroc import main as compute_auroc
from eval.run_benchmark import run as run_benchmark
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
)
from hallucinations_math_lab.constants import (
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_TRAIN_IDS_FILE,
)
from hallucinations_math_lab.services.semantic_entropy_probe import (
    ProbeSource,
    SemanticEntropyProbeDetector,
)
from tests.unit.sep_probe_fixtures import (
    PROBE_HIDDEN_SIZE,
    PROBE_MODEL_ID,
    ProbeBackboneProvider,
    write_probe_dir,
    write_train_ids,
)

KNOWN_ANSWERS = {"Capital of France?": "Paris", "Author of 1984?": "Orwell"}


class ScatteringSLM(ProbeBackboneProvider):
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

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]):
        return None


class HiddenStateSLM(ScatteringSLM):
    """Serves hidden states as the backbone the fixture probes were trained on."""

    model_id = PROBE_MODEL_ID

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]):
        return torch.zeros(len(layers), PROBE_HIDDEN_SIZE)


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
    assert rows[0]["sep_probe"] is None  # no SEP detector, no probe

    compute_auroc(results)
    report = json.loads(capsys.readouterr().out)
    assert report["n_rows"] == 3  # the adversarial row has no aliases
    assert report["n_hallucinations"] == 1
    assert report["run_config"]["model_id"] == "fake-slm"
    assert report["detectors"]["semantic_entropy"]["auroc"] == 1.0


class CountingJudge:
    """Answers every question with a fixed string and counts its calls."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def answer(self, prompt: str) -> tuple[str, str]:
        self.prompts.append(prompt)
        return "judge says", ""


def test_benchmark_records_the_judge_answer_on_every_row(tmp_path):
    """Comparing escalation to random escalation at the same rate needs the
    judge's answer to every question. Escalated rows reuse the answer the
    judge already gave, so each question costs exactly one judge call."""
    questions = tmp_path / "questions.jsonl"
    results = tmp_path / "results.jsonl"
    _write_questions(questions)
    judge = CountingJudge()
    evaluator = ResponseEvaluator(
        slm=ScatteringSLM(),
        detectors=[SemanticEntropyDetector(scorer=EqualityNLI())],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
        judge=judge,
    )
    run_benchmark(questions, results, evaluator=evaluator)

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert [row["judge_answer"] for row in rows] == ["judge says"] * 4
    assert sorted(judge.prompts) == sorted(row["question"] for row in rows)


def test_benchmark_without_a_judge_records_no_judge_answer(tmp_path):
    questions = tmp_path / "questions.jsonl"
    results = tmp_path / "results.jsonl"
    _write_questions(questions)
    evaluator = ResponseEvaluator(
        slm=ScatteringSLM(),
        detectors=[SemanticEntropyDetector(scorer=EqualityNLI())],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
    )
    run_benchmark(questions, results, judge=False, evaluator=evaluator)
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert all(row["judge_answer"] is None for row in rows)


def _probe_evaluator(probe: Path | ProbeSource) -> ResponseEvaluator:
    return ResponseEvaluator(
        slm=HiddenStateSLM(),
        detectors=[SemanticEntropyProbeDetector(probe)],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY_PROBE,
        escalation_threshold=1.0,
    )


def test_benchmark_refuses_questions_the_probe_was_trained_on(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["q2", "unrelated"])

    with pytest.raises(RuntimeError, match="used to train the SEP probe"):
        run_benchmark(
            questions, tmp_path / "out.jsonl", judge=False, evaluator=_probe_evaluator(probe_dir)
        )
    assert not (tmp_path / "out.jsonl").exists()


def test_benchmark_accepts_a_probe_trained_on_other_questions(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["other-1", "other-2"])

    run_benchmark(
        questions,
        tmp_path / "out.jsonl",
        judge=False,
        evaluator=_probe_evaluator(probe_dir),
        limit=1,
    )
    assert len((tmp_path / "out.jsonl").read_text().splitlines()) == 1


def test_benchmark_refuses_a_probe_without_provenance(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    evaluator = _probe_evaluator(write_probe_dir(tmp_path / "probe"))

    with pytest.raises(RuntimeError, match="no training-id file"):
        run_benchmark(questions, tmp_path / "out.jsonl", judge=False, evaluator=evaluator)


def test_train_ids_inside_probe_dir(tmp_path):
    """The guard reads `<probe dir>/train_ids.json`, validated as SEPTrainIds; a
    stale sidecar beside the directory (the old `.pt` layout) does not count."""
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    stale = tmp_path / f"probe.{SEP_TRAIN_IDS_FILE}"
    stale.write_text(json.dumps({"model_id": PROBE_MODEL_ID, "question_ids": ["other"]}))

    with pytest.raises(RuntimeError, match="no training-id file"):
        run_benchmark(
            questions, tmp_path / "out.jsonl", judge=False, evaluator=_probe_evaluator(probe_dir)
        )

    # Keys alone are not enough: the file must be a whole SEPTrainIds, checked at load.
    (probe_dir / SEP_TRAIN_IDS_FILE).write_text(json.dumps({"question_ids": ["other"]}))
    with pytest.raises(ValueError, match="Malformed SEP probe training-id file"):
        _probe_evaluator(probe_dir)


def test_benchmark_checks_the_train_ids_the_evaluator_loaded(tmp_path):
    """A probe trained on the benchmark is refused even if its train_ids.json is
    rewritten after the evaluator was built: the run scores the loaded weights."""
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["q1", "q2", "q3", "adv"])
    evaluator = _probe_evaluator(probe_dir)
    write_train_ids(probe_dir, ["other"])

    with pytest.raises(RuntimeError, match="used to train the SEP probe"):
        run_benchmark(questions, tmp_path / "out.jsonl", judge=False, evaluator=evaluator)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_config_records_probe_ref_revision_and_sha256(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    commit = "c" * 40
    probe_dir = write_probe_dir(tmp_path / "snapshots" / commit)
    write_train_ids(probe_dir, ["other"])
    source = ProbeSource(directory=probe_dir, ref="hf://owner/sep-probe@v1", revision=commit)
    results = tmp_path / "out.jsonl"

    run_benchmark(questions, results, judge=False, evaluator=_probe_evaluator(source))

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    expected = {
        "ref": "hf://owner/sep-probe@v1",
        "revision": commit,
        "weights_sha256": _sha256(probe_dir / SEP_PROBE_WEIGHTS_FILE),
        "config_sha256": _sha256(probe_dir / SEP_PROBE_CONFIG_FILE),
    }
    assert all(row["sep_probe"] == expected for row in rows)
    # Part of the run config, so the scoring scripts carry it and refuse to mix probes.
    assert run_config(rows, str(results))["sep_probe"] == expected


def test_run_config_records_a_local_probe_without_revision(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["other"])
    results = tmp_path / "out.jsonl"

    run_benchmark(questions, results, judge=False, evaluator=_probe_evaluator(probe_dir), limit=1)

    (row,) = [json.loads(line) for line in results.read_text().splitlines()]
    assert row["sep_probe"]["ref"] == str(probe_dir)
    assert row["sep_probe"]["revision"] is None


def test_run_config_hashes_the_probe_the_evaluator_loaded(tmp_path):
    """A directory retrained after the evaluator was built does not change the record."""
    questions = tmp_path / "questions.jsonl"
    _write_questions(questions)
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["other"])
    loaded_weights = _sha256(probe_dir / SEP_PROBE_WEIGHTS_FILE)
    loaded_config = _sha256(probe_dir / SEP_PROBE_CONFIG_FILE)
    evaluator = _probe_evaluator(probe_dir)
    (probe_dir / SEP_PROBE_WEIGHTS_FILE).write_bytes(b"retrained")
    (probe_dir / SEP_PROBE_CONFIG_FILE).write_text("retrained")
    results = tmp_path / "out.jsonl"

    run_benchmark(questions, results, judge=False, evaluator=evaluator, limit=1)

    (row,) = [json.loads(line) for line in results.read_text().splitlines()]
    assert row["sep_probe"]["weights_sha256"] == loaded_weights
    assert row["sep_probe"]["config_sha256"] == loaded_config


def test_sep_run_refuses_a_question_without_an_id(tmp_path):
    """Without an id a question cannot be checked against the probe's training set."""
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"question": "Who walked on Mars?"}) + "\n")
    probe_dir = write_probe_dir(tmp_path / "probe")
    write_train_ids(probe_dir, ["other"])
    results = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match=r"Question 1 \('Who walked on Mars\?'\) has no 'id'"):
        run_benchmark(questions, results, judge=False, evaluator=_probe_evaluator(probe_dir))
    assert not results.exists()
