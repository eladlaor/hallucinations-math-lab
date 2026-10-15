"""Measure how much judge escalation improves end-to-end accuracy (criterion 3).

Over the labeled rows of a run_benchmark.py output, compares:
  - greedy accuracy: the small model's own answer, never escalated;
  - final accuracy: the answer actually returned (the judge's on escalated rows).
The uplift is their difference in percentage points, reported with a paired
bootstrap 95% CI and McNemar's exact test (the two accuracies are measured on
the same questions, so they must be compared as pairs, not as two samples).

Answer-length caveat: correctness is a substring match against the accepted
aliases, which favours longer answers ("Canberra, not Sydney" contains more
candidates than "Canberra"). The report gives mean answer length for both
sides and warns when the final answers are markedly longer, since part of the
uplift may then come from verbosity rather than knowledge.

Fails fast when the run cannot answer the question at all: if no row was
escalated, or escalated rows were never answered by the judge (a `--no-judge`
run), the uplift is zero by construction rather than measured.

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
    ESCALATED_FIELD,
    FINAL_ANSWER_FIELD,
    FINAL_ANSWER_SOURCE_FIELD,
    GREEDY_ANSWER_FIELD,
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
# substring-bias warning.
LENGTH_RATIO_WARNING: Final[float] = 1.5


def _mean_words(answers: list[str | None]) -> float:
    return float(np.mean([len((a or "").split()) for a in answers]))


def uplift_report(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    """Greedy vs final accuracy over labeled rows, the uplift and its uncertainty."""
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

    greedy_correct = np.array(
        [is_correct(r[GREEDY_ANSWER_FIELD], r[ALIASES_FIELD]) for r in labeled], dtype=bool
    )
    final_correct = np.array(
        [is_correct(r[FINAL_ANSWER_FIELD], r[ALIASES_FIELD]) for r in labeled], dtype=bool
    )
    per_row_gain = final_correct.astype(float) - greedy_correct.astype(float)
    uplift_pp = float(per_row_gain.mean()) * PERCENT
    ci = paired_bootstrap(len(labeled), lambda idx: float(per_row_gain[idx].mean()) * PERCENT)
    p_value = mcnemar_exact_p(greedy_correct, final_correct)

    greedy_words = _mean_words([r[GREEDY_ANSWER_FIELD] for r in labeled])
    final_words = _mean_words([r[FINAL_ANSWER_FIELD] for r in labeled])
    length_ratio = final_words / greedy_words if greedy_words > 0 else float("inf")
    length_warning = length_ratio > LENGTH_RATIO_WARNING
    if length_warning:
        logger.warning(
            "Final answers are much longer than greedy ones; substring matching may "
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
        "n_rows": len(labeled),
        "n_escalated": len(escalated),
        "n_judge_answered": len(judged),
        "greedy_accuracy_pct": round(float(greedy_correct.mean()) * PERCENT, REPORT_DECIMALS),
        "final_accuracy_pct": round(float(final_correct.mean()) * PERCENT, REPORT_DECIMALS),
        "n_fixed_by_escalation": int(np.sum(~greedy_correct & final_correct)),
        "n_broken_by_escalation": int(np.sum(greedy_correct & ~final_correct)),
        "mean_answer_words": {
            "greedy": round(greedy_words, REPORT_DECIMALS),
            "final": round(final_words, REPORT_DECIMALS),
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
            }
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
