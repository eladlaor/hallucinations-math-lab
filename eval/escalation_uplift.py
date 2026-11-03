"""Measure what judge escalation buys, and whether the detector deserves credit (criterion 3).

Over the labeled rows of a run_benchmark.py output, compares:
  - greedy accuracy: the small model's own answer, never escalated;
  - final accuracy: the answer actually returned (the judge's on escalated rows).
The uplift is their difference in percentage points, reported with a paired
bootstrap 95% CI and McNemar's exact test (the two accuracies are measured on
the same questions, so they must be compared as pairs, not as two samples).

An uplift alone does not show the detector is useful: escalating *any* 30%
of questions to a stronger judge raises accuracy. So the report also asks
what escalating the same number of questions would have given if they were
picked
  - at random (an exact expectation, no simulation), or
  - by each recorded detector's score (top-k, ties at the cutoff averaged).
This needs the judge's answer to every labeled question, not only the
escalated ones; run_benchmark.py records it as `judge_answer` whenever the
judge is enabled. Criterion 3b is met when the run's escalation beats random
escalation at the same rate, with a paired-bootstrap CI that excludes zero.
`accuracy_vs_rate` gives the whole trade-off curve for every detector.

Answer-length caveat: longer answers mention more candidate names, so they
match an alias more often even under whole-token matching. The report gives
mean answer length for both sides and warns when the final answers are
markedly longer, since part of the uplift may then come from verbosity
rather than knowledge.

Fails fast when the run cannot answer the question at all: if no row was
escalated, if escalated rows were never answered by the judge (a
`--no-judge` run), or if any labeled row lacks a `judge_answer`.

Usage:
    uv run python -m eval.escalation_uplift --in eval/results/triviaqa_1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from eval.benchmark_results import (
    ALIASES_FIELD,
    DETECTOR_SCORES_FIELD,
    ESCALATED_FIELD,
    FINAL_ANSWER_FIELD,
    FINAL_ANSWER_SOURCE_FIELD,
    GREEDY_ANSWER_FIELD,
    JUDGE_ANSWER_FIELD,
    BenchmarkRow,
    is_correct,
    is_labeled,
    load_rows,
    run_config,
)
from eval.inference_stats import CI_LEVEL, mcnemar_exact_p, paired_bootstrap
from eval.success_criteria import ESCALATION_SIGNIFICANCE_LEVEL, ESCALATION_UPLIFT_PP_TARGET
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, AnswerSource

logger = logging.getLogger(__name__)

PERCENT: Final[float] = 100.0
REPORT_DECIMALS: Final[int] = 2
P_VALUE_DECIMALS: Final[int] = 6
# Final answers this many times longer than greedy ones (in words) trigger the
# verbosity warning.
LENGTH_RATIO_WARNING: Final[float] = 1.5
# Escalation rates at which the accuracy-vs-rate curve is reported.
CURVE_RATES: Final[tuple[float, ...]] = tuple(i / 10 for i in range(11))
RANDOM_BASELINE_KEY: Final[str] = "random"


def _mean_words(answers: list[str | None]) -> float:
    return float(np.mean([len((a or "").split()) for a in answers]))


def escalation_accuracy(
    scores: np.ndarray, greedy_correct: np.ndarray, judge_correct: np.ndarray, k: int
) -> float:
    """Accuracy when the `k` highest-scoring rows go to the judge.

    Rows tied with the k-th score share the remaining escalation slots
    equally (each is escalated with the same probability), so a detector
    that outputs many identical scores is not credited with an arbitrary
    tie-break.
    """
    n = len(scores)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} escalations is outside [0, {n}].")
    escalate_prob = np.zeros(n, dtype=np.float64)
    if k > 0:
        cutoff = np.sort(scores)[::-1][k - 1]
        above = scores > cutoff
        tied = scores == cutoff
        escalate_prob[above] = 1.0
        escalate_prob[tied] = (k - int(above.sum())) / int(tied.sum())
    per_row = escalate_prob * judge_correct + (1.0 - escalate_prob) * greedy_correct
    return float(per_row.mean())


def random_escalation_accuracy(
    greedy_correct: np.ndarray, judge_correct: np.ndarray, k: int
) -> float:
    """Expected accuracy when `k` of the rows, chosen uniformly at random, go to the judge."""
    n = len(greedy_correct)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} escalations is outside [0, {n}].")
    gain = judge_correct.astype(np.float64) - greedy_correct.astype(np.float64)
    return float(greedy_correct.mean() + (k / n) * gain.mean())


def _pct(value: float) -> float:
    return round(value * PERCENT, REPORT_DECIMALS)


def _require_judge_answers(labeled: list[BenchmarkRow], source: str) -> None:
    missing = [i for i, row in enumerate(labeled) if row.get(JUDGE_ANSWER_FIELD) is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} labeled rows in {source} have no {JUDGE_ANSWER_FIELD!r} (first at "
            f"labeled row {missing[0]}). Comparing against random escalation needs the judge's "
            "answer to every question; re-run eval.run_benchmark with the judge enabled."
        )


def uplift_report(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    """Greedy vs final accuracy over labeled rows, the uplift, and the same-rate baselines."""
    config = run_config(rows, source)
    labeled = [row for row in rows if is_labeled(row)]
    if not labeled:
        raise RuntimeError(f"No labeled rows in {source} (no ground-truth aliases found).")

    escalated = [row for row in labeled if row[ESCALATED_FIELD]]
    if not escalated:
        raise RuntimeError(
            f"No labeled row in {source} was escalated, so the judge never ran and the "
            "uplift is 0 by construction. Lower the escalation threshold or use a harder set."
        )
    judged = [
        row for row in escalated if row[FINAL_ANSWER_SOURCE_FIELD] == str(AnswerSource.JUDGE_MODEL)
    ]
    if not judged:
        raise RuntimeError(
            f"{len(escalated)} rows in {source} were escalated but none was answered by the "
            "judge. Re-run the benchmark without --no-judge."
        )
    _require_judge_answers(labeled, source)

    def correct(field: str) -> np.ndarray:
        return np.array([is_correct(r[field], r[ALIASES_FIELD]) for r in labeled], dtype=bool)

    greedy_correct = correct(GREEDY_ANSWER_FIELD)
    final_correct = correct(FINAL_ANSWER_FIELD)
    judge_correct = correct(JUDGE_ANSWER_FIELD)
    n, k = len(labeled), len(escalated)

    per_row_gain = final_correct.astype(float) - greedy_correct.astype(float)
    uplift_pp = float(per_row_gain.mean()) * PERCENT
    ci = paired_bootstrap(n, lambda idx: float(per_row_gain[idx].mean()) * PERCENT)
    p_value = mcnemar_exact_p(greedy_correct, final_correct)

    # Criterion 3b: the run's escalation vs escalating the same share at random.
    rate = k / n
    random_acc = random_escalation_accuracy(greedy_correct, judge_correct, k)
    vs_random_pp = (float(final_correct.mean()) - random_acc) * PERCENT

    def vs_random_on(idx: np.ndarray) -> float:
        k_resampled = round(rate * len(idx))
        baseline = random_escalation_accuracy(greedy_correct[idx], judge_correct[idx], k_resampled)
        return (float(final_correct[idx].mean()) - baseline) * PERCENT

    vs_random_ci = paired_bootstrap(n, vs_random_on)

    detector_scores = {
        name: np.array([float(r[DETECTOR_SCORES_FIELD][name]) for r in labeled])
        for name in labeled[0][DETECTOR_SCORES_FIELD]
    }
    curve: list[dict[str, float]] = []
    for curve_rate in CURVE_RATES:
        k_at = round(curve_rate * n)
        point = {
            "rate": curve_rate,
            RANDOM_BASELINE_KEY: _pct(
                random_escalation_accuracy(greedy_correct, judge_correct, k_at)
            ),
        }
        for name, scores in detector_scores.items():
            point[name] = _pct(escalation_accuracy(scores, greedy_correct, judge_correct, k_at))
        curve.append(point)

    greedy_words = _mean_words([r[GREEDY_ANSWER_FIELD] for r in labeled])
    final_words = _mean_words([r[FINAL_ANSWER_FIELD] for r in labeled])
    judge_words = _mean_words([r[JUDGE_ANSWER_FIELD] for r in labeled])
    length_ratio = final_words / greedy_words if greedy_words > 0 else float("inf")
    length_warning = length_ratio > LENGTH_RATIO_WARNING
    if length_warning:
        logger.warning(
            "Final answers are much longer than greedy ones; alias matching may "
            "credit verbosity as accuracy.",
            extra={
                LOG_EXTRA_KEY: {
                    "source": source,
                    "mean_words_greedy": round(greedy_words, REPORT_DECIMALS),
                    "mean_words_final": round(final_words, REPORT_DECIMALS),
                }
            },
        )

    return {
        "run_config": config,
        "n_rows": n,
        "n_escalated": k,
        "n_judge_answered": len(judged),
        "greedy_accuracy_pct": _pct(float(greedy_correct.mean())),
        "final_accuracy_pct": _pct(float(final_correct.mean())),
        "judge_accuracy_pct": _pct(float(judge_correct.mean())),
        "n_fixed_by_escalation": int(np.sum(~greedy_correct & final_correct)),
        "n_broken_by_escalation": int(np.sum(greedy_correct & ~final_correct)),
        "at_run_rate": {
            "escalation_rate": round(rate, 4),
            "random_accuracy_pct": _pct(random_acc),
            "detector_accuracy_pct": {
                name: _pct(escalation_accuracy(scores, greedy_correct, judge_correct, k))
                for name, scores in detector_scores.items()
            },
        },
        "accuracy_vs_rate": curve,
        "mean_answer_words": {
            "greedy": round(greedy_words, REPORT_DECIMALS),
            "final": round(final_words, REPORT_DECIMALS),
            "judge": round(judge_words, REPORT_DECIMALS),
            "length_warning": length_warning,
        },
        "criteria": {
            "escalation_uplift_pp": {
                "value": round(uplift_pp, REPORT_DECIMALS),
                "ci": {
                    "level": CI_LEVEL,
                    "low": round(ci.low, REPORT_DECIMALS),
                    "high": round(ci.high, REPORT_DECIMALS),
                },
                "mcnemar_p_value": round(p_value, P_VALUE_DECIMALS),
                "target": ESCALATION_UPLIFT_PP_TARGET,
                "significance_level": ESCALATION_SIGNIFICANCE_LEVEL,
                "met": uplift_pp >= ESCALATION_UPLIFT_PP_TARGET
                and p_value < ESCALATION_SIGNIFICANCE_LEVEL,
            },
            "escalation_beats_random_pp": {
                "value": round(vs_random_pp, REPORT_DECIMALS),
                "ci": {
                    "level": CI_LEVEL,
                    "low": round(vs_random_ci.low, REPORT_DECIMALS),
                    "high": round(vs_random_ci.high, REPORT_DECIMALS),
                },
                "met": vs_random_ci.low > 0.0,
            },
        },
    }


def main(path: Path) -> None:
    print(json.dumps(uplift_report(load_rows(path), source=str(path)), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Accuracy uplift from judge escalation.")
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
