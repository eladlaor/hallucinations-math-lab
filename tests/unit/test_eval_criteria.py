"""Success-criteria scoring scripts, on tiny hand-built benchmark rows.

Contract pinned here:
  - compute_auroc reports SE AUROC and the SE−PE gap against their targets.
  - escalation_uplift reports greedy vs final accuracy and fails fast when no
    row was escalated or the judge never answered.
  - adversarial_separation runs a one-sided Mann-Whitney U and fails fast on a
    missing detector score.
  - train_sep_probe's held-out Pearson r tracks a learnable signal and rejects
    splits too small to correlate.
"""

import numpy as np
import pytest

from eval.adversarial_separation import separation_report
from eval.benchmark_results import is_correct
from eval.compute_auroc import auroc_report
from eval.escalation_uplift import uplift_report
from eval.train_sep_probe import holdout_pearson_r
from hallucinations_math_lab.constants import AnswerSource, DetectorName

SE = str(DetectorName.SEMANTIC_ENTROPY)
PE = str(DetectorName.PREDICTIVE_ENTROPY)
SLM = str(AnswerSource.SMALL_MODEL_GREEDY)
JUDGE = str(AnswerSource.JUDGE_MODEL)


def _row(
    greedy: str,
    se: float,
    pe: float,
    final: str | None = None,
    escalated: bool = False,
    source: str = SLM,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "ground_truth_aliases": ["Paris"] if aliases is None else aliases,
        "greedy_answer": greedy,
        "final_answer": greedy if final is None else final,
        "final_answer_source": source,
        "escalated": escalated,
        "detector_scores": {SE: se, PE: pe},
    }


def test_is_correct_is_case_insensitive_substring():
    assert is_correct("It is PARIS, France.", ["Paris"])
    assert not is_correct("", ["Paris"])
    assert not is_correct("Lyon", ["Paris"])


def test_auroc_report_checks_se_target_and_gap():
    rows = [
        _row("Paris", se=0.1, pe=0.5),
        _row("Paris", se=0.2, pe=0.9),
        _row("Lyon", se=1.5, pe=0.6),
        _row("Nice", se=1.8, pe=0.4),
    ]
    report = auroc_report(rows, source="fixture")

    assert report["auroc"] == {SE: 1.0, PE: 0.25}
    assert report["criteria"]["se_auroc"]["met"] is True
    assert report["criteria"]["se_minus_pe_auroc"] == {"value": 0.75, "target": 0.05, "met": True}


def test_uplift_report_measures_judge_gain():
    rows = [
        _row("Paris", se=0.1, pe=0.1),
        _row("Lyon", se=2.0, pe=1.0, final="Paris", escalated=True, source=JUDGE),
        _row("Nice", se=2.0, pe=1.0, final="Nice", escalated=True, source=JUDGE),
        _row("Lyon", se=0.3, pe=0.3),
    ]
    report = uplift_report(rows, source="fixture")

    assert report["greedy_accuracy_pct"] == 25.0
    assert report["final_accuracy_pct"] == 50.0
    assert report["criteria"]["escalation_uplift_pp"] == {"value": 25.0, "target": 5.0, "met": True}


def test_uplift_report_fails_fast_without_escalations():
    with pytest.raises(RuntimeError, match="was escalated"):
        uplift_report([_row("Paris", se=0.1, pe=0.1)], source="fixture")


def test_uplift_report_fails_fast_when_judge_never_answered():
    rows = [_row("Lyon", se=2.0, pe=1.0, escalated=True, source=SLM)]
    with pytest.raises(RuntimeError, match="without --no-judge"):
        uplift_report(rows, source="fixture")


def test_separation_report_detects_higher_adversarial_scores():
    adversarial = [_row("x", se=se, pe=0.0, aliases=[]) for se in (1.6, 1.8, 2.0, 2.1, 2.3)]
    reference = [_row("Paris", se=se, pe=0.0) for se in (0.0, 0.1, 0.2, 0.3, 0.4)]
    reference.append(_row("Lyon", se=3.0, pe=0.0))  # wrong answer: excluded

    report = separation_report(adversarial, reference, detector=SE)

    assert report["n_adversarial"] == 5
    assert report["n_correct_reference"] == 5
    assert report["criteria"]["adversarial_p_value"]["met"] is True


def test_separation_report_fails_fast_on_missing_detector():
    rows = [_row("Paris", se=0.1, pe=0.1)]
    with pytest.raises(RuntimeError, match="no 'semantic_entropy_probe' score"):
        separation_report(rows, rows, detector=str(DetectorName.SEMANTIC_ENTROPY_PROBE))


def test_holdout_pearson_r_tracks_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X @ np.array([1.0, -0.5, 0.0, 2.0]) + rng.normal(scale=0.05, size=60)

    assert holdout_pearson_r(X, y, ridge_alpha=0.1, holdout_fraction=0.25, seed=0) > 0.95


def test_holdout_pearson_r_rejects_tiny_split():
    X, y = np.zeros((5, 2)), np.arange(5.0)
    with pytest.raises(ValueError, match="need at least 3"):
        holdout_pearson_r(X, y, ridge_alpha=1.0, holdout_fraction=0.2, seed=0)
