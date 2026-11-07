"""Success-criteria scoring scripts, on tiny hand-built benchmark rows.

Contract pinned here:
  - AUROC (rank formula) matches sklearn; bootstrap CIs are deterministic and
    count degenerate resamples; McNemar's exact p matches the binomial test.
  - compute_auroc reports per-detector AUROC with CIs; the SE−PE criterion
    needs both the point gap and a CI that excludes zero.
  - escalation_uplift needs both the uplift bar and McNemar significance,
    reports answer lengths, and fails fast without escalations or a judge.
  - adversarial_separation needs both p < 0.01 and the AUC effect size, and
    fails fast on a missing detector score.
  - Results files must be one run, and compared runs must match.
  - TriviaQA slices are bounds-checked; probe provenance rejects overlap.
"""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from eval.adversarial_separation import separation_report
from eval.benchmark_results import is_correct, run_config
from eval.compute_auroc import auroc_report
from eval.datasets.load_triviaqa import slice_bounds
from eval.escalation_uplift import (
    escalation_accuracy,
    random_escalation_accuracy,
    uplift_report,
)
from eval.inference_stats import auroc, mcnemar_exact_p, paired_bootstrap
from eval.sep_probe_provenance import assert_disjoint, training_ids_path, write_training_ids
from hallucinations_math_lab.constants import SEP_TRAIN_IDS_FILE, AnswerSource, DetectorName

SE = str(DetectorName.SEMANTIC_ENTROPY)
PE = str(DetectorName.PREDICTIVE_ENTROPY)
SLM = str(AnswerSource.SMALL_MODEL_GREEDY)
JUDGE = str(AnswerSource.JUDGE_MODEL)
RUN = {
    "model_id": "google/gemma-4-E4B-it",
    "sample_n": 10,
    "sample_temperature": 1.0,
    "max_tokens": 64,
    "answer_instruction": "Answer the following question as briefly as possible.",
}


def _row(
    greedy: str,
    se: float,
    pe: float,
    final: str | None = None,
    escalated: bool = False,
    source: str = SLM,
    aliases: list[str] | None = None,
    run: dict | None = None,
    judge: str | None = None,
) -> dict:
    # The judge's answer to this question, recorded on every row. By default
    # it is whatever the row finally returned (so escalated rows agree with
    # their final answer and the rest change nothing).
    final_answer = greedy if final is None else final
    return {
        "ground_truth_aliases": ["Paris"] if aliases is None else aliases,
        **(RUN if run is None else run),
        "greedy_answer": greedy,
        "final_answer": final_answer,
        "judge_answer": final_answer if judge is None else judge,
        "final_answer_source": source,
        "escalated": escalated,
        "detector_scores": {SE: se, PE: pe},
    }


# --- statistics primitives -------------------------------------------------


def test_is_correct_matches_a_normalized_alias_as_whole_tokens():
    assert is_correct("It is PARIS, France.", ["Paris"])
    assert is_correct("New York City", ["new york"])
    assert not is_correct("", ["Paris"])
    assert not is_correct("Lyon", ["Paris"])


def test_is_correct_ignores_articles_and_punctuation_squad_style():
    assert is_correct("The Beatles.", ["Beatles"])
    assert is_correct("Beatles", ["The Beatles"])
    assert is_correct("Dr. Who", ["dr who"])


@pytest.mark.parametrize(
    ("answer", "alias"),
    [
        ("It was 1921.", "2"),  # a digit inside a number
        ("Parisian cafes", "Paris"),  # a prefix of a longer word
        ("Theodore Roosevelt", "Teddy"),  # no shared token
    ],
)
def test_is_correct_rejects_substrings_that_are_not_whole_tokens(answer, alias):
    assert not is_correct(answer, [alias])


def test_is_correct_ignores_aliases_that_normalize_to_nothing():
    assert not is_correct("Anything at all", ["", "The", "!"])


def test_rank_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.integers(0, 5, size=200).astype(float)  # many ties
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


def test_auroc_is_undefined_for_one_class():
    assert auroc(np.ones(5, dtype=int), np.arange(5.0)) is None


def test_bootstrap_is_deterministic_and_counts_degenerate_draws():
    labels = np.array([0, 0, 0, 1])  # a resample often draws no positive
    scores = np.array([0.1, 0.2, 0.3, 0.9])

    def stat(idx: np.ndarray) -> float | None:
        return auroc(labels[idx], scores[idx])

    first, second = paired_bootstrap(4, stat), paired_bootstrap(4, stat)
    assert first == second
    assert first.n_degenerate > 0
    assert first.n_resamples + first.n_degenerate == 2000


def test_mcnemar_uses_only_discordant_pairs():
    before = np.array([True] * 50 + [False] * 10)
    after = np.array([True] * 50 + [True] * 10)  # 10 fixes, 0 regressions
    assert mcnemar_exact_p(before, after) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact_p(before, before) == 1.0


# --- criterion 1 and 2 -----------------------------------------------------


def test_auroc_report_checks_se_target_and_gap():
    rows = [_row("Paris", se=0.1 * i, pe=0.5) for i in range(20)]
    rows += [_row("Lyon", se=5.0 + 0.1 * i, pe=0.5) for i in range(20)]
    report = auroc_report(rows, source="fixture")

    assert report["run_config"] == RUN
    assert report["detectors"][SE]["auroc"] == 1.0
    assert report["detectors"][PE]["auroc"] == 0.5
    gap = report["criteria"]["se_minus_pe_auroc"]
    assert gap["value"] == 0.5
    assert gap["ci"]["low"] > 0
    assert gap["met"] is True
    assert report["criteria"]["se_auroc"]["met"] is True


def test_auroc_report_gives_the_greedy_answer_length_distribution():
    rows = [_row("Paris", se=0.1, pe=0.5)] * 9 + [_row("It is Lyon in France", se=2.0, pe=0.5)]
    words = auroc_report(rows, source="fixture")["greedy_answer_words"]
    assert words == {"median": 1.0, "p90": 1.4, "max": 5}


def test_gap_criterion_fails_when_ci_includes_zero():
    """A gap that clears the bar on four rows is noise, not evidence."""
    rows = [
        _row("Paris", se=0.1, pe=0.5),
        _row("Paris", se=0.2, pe=0.9),
        _row("Lyon", se=1.5, pe=0.6),
        _row("Nice", se=1.8, pe=0.4),
    ]
    gap = auroc_report(rows, source="fixture")["criteria"]["se_minus_pe_auroc"]
    assert gap["value"] >= 0.05
    assert gap["ci"]["low"] <= 0
    assert gap["met"] is False


# --- criterion 3 -----------------------------------------------------------


def _uplift_rows(n_fixed: int, n_broken: int, n_unchanged: int) -> list[dict]:
    fixed = [
        _row("Lyon", se=2.0, pe=1.0, final="Paris", escalated=True, source=JUDGE)
        for _ in range(n_fixed)
    ]
    broken = [
        _row("Paris", se=2.0, pe=1.0, final="Lyon", escalated=True, source=JUDGE)
        for _ in range(n_broken)
    ]
    same = [_row("Paris", se=0.1, pe=0.1) for _ in range(n_unchanged)]
    return fixed + broken + same


def test_uplift_report_measures_a_significant_gain():
    report = uplift_report(_uplift_rows(n_fixed=12, n_broken=1, n_unchanged=87), "fixture")

    criterion = report["criteria"]["escalation_uplift_pp"]
    assert report["greedy_accuracy_pct"] == 88.0
    assert report["final_accuracy_pct"] == 99.0
    assert criterion["value"] == 11.0
    assert criterion["mcnemar_p_value"] < 0.05
    assert criterion["ci"]["low"] > 0
    assert criterion["met"] is True
    assert report["n_fixed_by_escalation"] == 12
    assert report["n_broken_by_escalation"] == 1


def test_uplift_above_the_bar_but_not_significant_is_not_met():
    report = uplift_report(_uplift_rows(n_fixed=2, n_broken=0, n_unchanged=8), "fixture")

    criterion = report["criteria"]["escalation_uplift_pp"]
    assert criterion["value"] == 20.0
    assert criterion["mcnemar_p_value"] >= 0.05
    assert criterion["met"] is False


def test_uplift_report_flags_much_longer_final_answers():
    verbose = _row(
        "Lyon",
        se=2.0,
        pe=1.0,
        final="It could be Paris or maybe Lyon",
        escalated=True,
        source=JUDGE,
    )
    report = uplift_report([verbose] * 4, source="fixture")
    assert report["mean_answer_words"]["length_warning"] is True


def test_uplift_report_fails_fast_without_escalations():
    with pytest.raises(RuntimeError, match="was escalated"):
        uplift_report([_row("Paris", se=0.1, pe=0.1)], source="fixture")


def test_uplift_report_fails_fast_when_judge_never_answered():
    rows = [_row("Lyon", se=2.0, pe=1.0, escalated=True, source=SLM)]
    with pytest.raises(RuntimeError, match="without --no-judge"):
        uplift_report(rows, source="fixture")


def test_escalation_accuracy_averages_ties_at_the_cutoff():
    # Rows 0 and 1 tie for the single escalation slot: each is escalated with
    # probability 1/2. Row 0 gains from the judge, row 1 is right either way.
    scores = np.array([1.0, 1.0, 0.0])
    greedy = np.array([False, True, True])
    judge = np.array([True, True, True])
    assert escalation_accuracy(scores, greedy, judge, k=1) == pytest.approx(2.5 / 3)
    assert escalation_accuracy(scores, greedy, judge, k=0) == pytest.approx(2 / 3)
    assert escalation_accuracy(scores, greedy, judge, k=3) == pytest.approx(1.0)


def test_random_escalation_accuracy_is_the_expected_value():
    greedy = np.array([False, True])
    judge = np.array([True, True])
    assert random_escalation_accuracy(greedy, judge, k=1) == pytest.approx(0.75)


def _trigger_rows(informative: bool) -> list[dict]:
    """10 rows the SLM gets wrong and 90 it gets right. The judge fixes the
    10 and breaks the 90. An informative trigger escalates exactly the 10."""
    wrong = [_row("Lyon", se=2.0 if informative else 0.1, pe=0.5, judge="Paris") for _ in range(10)]
    right = [_row("Paris", se=0.1, pe=0.5, judge="Lyon") for _ in range(90)]
    rows = wrong + right
    escalate = range(10) if informative else range(10, 20)
    for i in escalate:
        rows[i] |= {"escalated": True, "final_answer_source": JUDGE}
        rows[i]["final_answer"] = rows[i]["judge_answer"]
    return rows


def test_informative_trigger_beats_random_escalation_at_the_same_rate():
    report = uplift_report(_trigger_rows(informative=True), "fixture")

    vs_random = report["criteria"]["escalation_beats_random_pp"]
    # Random at 10%: 0.9 + 0.1 * mean(judge - greedy) = 0.9 + 0.1 * (-0.8) = 0.82.
    assert report["at_run_rate"]["escalation_rate"] == 0.1
    assert report["at_run_rate"]["random_accuracy_pct"] == 82.0
    assert vs_random["value"] == 18.0
    assert vs_random["ci"]["low"] > 0
    assert vs_random["met"] is True
    # The same rate, chosen by each detector's score: SE picks the 10, PE is flat.
    assert report["at_run_rate"]["detector_accuracy_pct"] == {SE: 100.0, PE: 82.0}


def test_uninformative_trigger_does_not_beat_random():
    report = uplift_report(_trigger_rows(informative=False), "fixture")
    assert report["criteria"]["escalation_beats_random_pp"]["met"] is False


def test_uplift_report_gives_an_accuracy_vs_escalation_rate_curve():
    curve = uplift_report(_trigger_rows(informative=True), "fixture")["accuracy_vs_rate"]
    assert curve[0] == {"rate": 0.0, "random": 90.0, SE: 90.0, PE: 90.0}
    assert curve[-1] == {"rate": 1.0, "random": 10.0, SE: 10.0, PE: 10.0}
    assert curve[1]["rate"] == 0.1 and curve[1][SE] == 100.0


def test_uplift_report_fails_fast_without_judge_answers_on_every_row():
    rows = _trigger_rows(informative=True)
    rows[50]["judge_answer"] = None
    with pytest.raises(RuntimeError, match="judge_answer"):
        uplift_report(rows, "fixture")


# --- criterion 4 -----------------------------------------------------------


def test_separation_report_detects_higher_adversarial_scores():
    adversarial = [_row("x", se=se, pe=0.0, aliases=[]) for se in (1.6, 1.8, 2.0, 2.1, 2.3)]
    reference = [_row("Paris", se=se, pe=0.0) for se in (0.0, 0.1, 0.2, 0.3, 0.4)]
    reference.append(_row("Lyon", se=3.0, pe=0.0))  # wrong answer: excluded

    report = separation_report(adversarial, reference, detector=SE)

    criterion = report["criteria"]["adversarial_separation"]
    assert report["n_adversarial"] == 5
    assert report["n_correct_reference"] == 5
    assert criterion["effect_size_auc"] == 1.0
    assert criterion["met"] is True


def test_significant_but_small_separation_is_not_met():
    """Many rows make a weak shift significant; the effect size catches it."""
    rng = np.random.default_rng(0)
    adversarial = [_row("x", se=float(v), pe=0.0, aliases=[]) for v in rng.normal(0.3, 1, 2000)]
    reference = [_row("Paris", se=float(v), pe=0.0) for v in rng.normal(0.0, 1, 2000)]

    criterion = separation_report(adversarial, reference, detector=SE)["criteria"][
        "adversarial_separation"
    ]
    assert criterion["p_value"] < 0.01
    assert criterion["effect_size_auc"] < 0.75
    assert criterion["met"] is False


def test_separation_report_fails_fast_on_missing_detector():
    rows = [_row("Paris", se=0.1, pe=0.1)]
    with pytest.raises(RuntimeError, match="no 'semantic_entropy_probe' score"):
        separation_report(rows, rows, detector=str(DetectorName.SEMANTIC_ENTROPY_PROBE))


def test_separation_report_refuses_runs_from_different_models():
    adversarial = [_row("x", se=2.0, pe=0.0, aliases=[], run={**RUN, "model_id": "other"})]
    reference = [_row("Paris", se=0.1, pe=0.0)]
    with pytest.raises(RuntimeError, match="not comparable"):
        separation_report(adversarial, reference, detector=SE)


# --- run comparability -----------------------------------------------------


def test_run_config_rejects_mixed_rows():
    rows = [
        _row("Paris", se=0.1, pe=0.1),
        _row("Paris", se=0.1, pe=0.1, run={**RUN, "sample_n": 5}),
    ]
    with pytest.raises(RuntimeError, match="One results file must hold one run"):
        run_config(rows, "fixture")


def test_run_config_rejects_rows_from_before_it_was_recorded():
    row = _row("Paris", se=0.1, pe=0.1)
    del row["model_id"]
    with pytest.raises(RuntimeError, match="Re-run eval.run_benchmark"):
        run_config([row], "fixture")


# --- TriviaQA slices and SEP probe provenance -------------------------------


def test_slice_bounds_are_checked():
    assert slice_bounds(500, 1000, 17944) == range(500, 1500)
    with pytest.raises(ValueError, match="exceeds"):
        slice_bounds(17000, 1000, 17944)
    with pytest.raises(ValueError, match="offset >= 0 and n > 0"):
        slice_bounds(-1, 10, 100)


def test_provenance_rejects_overlap_and_accepts_disjoint_ids(tmp_path):
    probe = tmp_path / "probe"
    probe.mkdir()
    write_training_ids(probe, ["q1", "q2"], "m", tmp_path / "calib.jsonl")

    assert_disjoint(probe, ["q3", "q4"])
    with pytest.raises(RuntimeError, match="1 benchmark questions were used"):
        assert_disjoint(probe, ["q2", "q3"])


def test_provenance_ignores_a_stale_sidecar_beside_the_probe_dir(tmp_path):
    # The pre-classifier `<probe>.train_ids.json` sat beside the probe. A
    # leftover one must not stand in for the directory probe's own ids.
    probe = tmp_path / "probe"
    probe.mkdir()
    write_training_ids(probe, ["q1"], "m", tmp_path / "calib.jsonl")
    (tmp_path / "probe.train_ids.json").write_text(json.dumps({"question_ids": ["other"]}))

    assert training_ids_path(probe) == probe / SEP_TRAIN_IDS_FILE
    with pytest.raises(RuntimeError, match="1 benchmark questions were used"):
        assert_disjoint(probe, ["q1"])
