"""Escalation-threshold calibration on a dev slice.

Contract pinned here:
  - τ maximizes Youden's J (true-positive rate − false-positive rate) for
    flagging hallucinations, under the evaluator's strict `score > τ` rule.
  - Ties in J go to the higher τ, which escalates fewer questions.
  - The dev and test slices must not share a question id.
"""

import pytest

from eval.calibrate_threshold import calibrate_report, youden_threshold
from hallucinations_math_lab.constants import DetectorName

SE = str(DetectorName.SEMANTIC_ENTROPY)
RUN = {
    "model_id": "m",
    "sample_n": 10,
    "sample_temperature": 1.0,
    "max_tokens": 64,
    "answer_instruction": "Answer briefly.",
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


def _row(qid: str, answer: str, se: float) -> dict:
    return {
        "id": qid,
        "ground_truth_aliases": ["Paris"],
        **RUN,
        "greedy_answer": answer,
        "detector_scores": {SE: se},
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
