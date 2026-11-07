"""Escalation-threshold calibration on a dev slice.

Contract pinned here:
  - τ maximizes Youden's J (true-positive rate − false-positive rate) for
    flagging hallucinations, under the evaluator's strict `score > τ` rule.
  - Ties in J go to the higher τ, which escalates fewer questions.
  - The dev and test slices must not share a question id.
"""

import pytest

from eval.calibrate_threshold import build_parser, calibrate_report, youden_threshold
from hallucinations_math_lab.constants import ESCALATION_THRESHOLDED_DETECTORS, DetectorName

SE = str(DetectorName.SEMANTIC_ENTROPY)
SEP = str(DetectorName.SEMANTIC_ENTROPY_PROBE)
RUN = {
    "model_id": "m",
    "sample_n": 10,
    "sample_temperature": 1.0,
    "max_tokens": 64,
    "answer_instruction": "Answer briefly.",
    "sep_probe": None,
}


def test_youden_threshold_separates_perfectly_separable_scores():
    labels = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 1.5, 2.0]
    tau, tpr, fpr = youden_threshold(labels, scores)
    assert 0.3 <= tau < 1.5
    assert (tpr, fpr) == (1.0, 0.0)


def test_youden_threshold_picks_the_higher_tau_on_ties():
    # τ = 0.1 (TPR 1, FPR 0.5) and τ = 0.6 (TPR 0.5, FPR 0) both give J = 0.5;
    # the higher one escalates fewer questions.
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.5, 0.6, 0.9]
    tau, tpr, fpr = youden_threshold(labels, scores)
    assert (tau, tpr, fpr) == (0.6, 0.5, 0.0)


def test_youden_threshold_needs_both_classes():
    with pytest.raises(ValueError, match="both"):
        youden_threshold([1, 1], [0.1, 0.2])


def _row(qid: str, answer: str, se: float, detector: str = SE) -> dict:
    return {
        "id": qid,
        "ground_truth_aliases": ["Paris"],
        **RUN,
        "greedy_answer": answer,
        "detector_scores": {detector: se},
    }


def test_calibrate_report_labels_rows_and_reports_the_operating_point():
    rows = [_row(f"c{i}", "Paris", 0.1 * i) for i in range(8)]
    rows += [_row(f"h{i}", "Lyon", 2.0 + i) for i in range(2)]
    report = calibrate_report(rows, SE, source="dev")
    assert report["detector"] == SE
    assert 0.7 <= report["tau"] < 2.0
    assert report["tpr"] == 1.0 and report["fpr"] == 0.0
    assert report["escalation_rate"] == 0.2
    assert report["n_rows"] == 10


def test_calibrate_report_refuses_overlapping_test_ids():
    rows = [_row("q1", "Paris", 0.1), _row("q2", "Lyon", 2.0)]
    with pytest.raises(RuntimeError, match="q2"):
        calibrate_report(rows, SE, source="dev", test_ids={"q2", "q9"})


def test_calibrate_reports_env_var_for_sep():
    """SEP scores a probability, so its τ belongs in SEP_THRESHOLD, not SE_THRESHOLD."""
    rows = [_row("c1", "Paris", 0.1, SEP), _row("h1", "Lyon", 0.9, SEP)]
    assert calibrate_report(rows, SEP, source="dev")["env_var"] == "SEP_THRESHOLD"


def test_calibrate_reports_env_var_for_nats_detectors():
    rows = [_row("c1", "Paris", 0.1), _row("h1", "Lyon", 2.0)]
    assert calibrate_report(rows, SE, source="dev")["env_var"] == "SE_THRESHOLD"


def test_calibrate_refuses_a_detector_without_an_escalation_threshold():
    judge = str(DetectorName.LLAMA_JUDGE_QA)
    rows = [_row("c1", "Paris", 0.1, judge), _row("h1", "Lyon", 0.9, judge)]
    with pytest.raises(ValueError, match=judge):
        calibrate_report(rows, judge, source="dev")


def test_calibrate_reports_the_exact_tau_its_rates_describe():
    """The printed τ is the value to deploy, so it is not rounded: rounding
    0.12341 to 0.1234 would also flag the correct row scored 0.12341."""
    scores = [0.12341, 0.12344, 0.5, 0.9, 0.95]
    rows = [_row("c1", "Paris", scores[0])]
    rows += [_row(f"h{i}", "Lyon", score) for i, score in enumerate(scores[1:])]
    report = calibrate_report(rows, SE, source="dev")
    assert report["tau"] == 0.12341
    assert (report["tpr"], report["fpr"]) == (1.0, 0.0)


def test_cli_offers_only_detectors_with_an_escalation_threshold():
    parser = build_parser()
    for detector in ESCALATION_THRESHOLDED_DETECTORS:
        assert parser.parse_args(["--dev", "d.jsonl", "--detector", str(detector)])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dev", "d.jsonl", "--detector", str(DetectorName.LLAMA_JUDGE_QA)])
