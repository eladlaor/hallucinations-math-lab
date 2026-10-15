"""Measure how much judge escalation improves end-to-end accuracy (criterion 3).

Over the labeled rows of a run_benchmark.py output, compares:
  - greedy accuracy: the small model's own answer, never escalated;
  - final accuracy: the answer actually returned (the judge's on escalated rows).
The uplift is their difference in percentage points.

Fails fast when the run cannot answer the question at all: if no row was
escalated, or escalated rows were never answered by the judge (a `--no-judge`
run), the uplift is zero by construction rather than measured.

Usage:
    uv run python -m eval.escalation_uplift --in eval/results/triviaqa_1000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
)
from eval.success_criteria import ESCALATION_UPLIFT_PP_TARGET
from hallucinations_math_lab.constants import AnswerSource

PERCENT = 100.0
PP_DECIMALS = 2


def uplift_report(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    """Greedy vs final accuracy over labeled rows, and the uplift in pp."""
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

    n = len(labeled)
    greedy_acc = sum(is_correct(r[GREEDY_ANSWER_FIELD], r[ALIASES_FIELD]) for r in labeled) / n
    final_acc = sum(is_correct(r[FINAL_ANSWER_FIELD], r[ALIASES_FIELD]) for r in labeled) / n
    uplift_pp = round((final_acc - greedy_acc) * PERCENT, PP_DECIMALS)
    return {
        "n_rows": n,
        "n_escalated": len(escalated),
        "n_judge_answered": len(judged),
        "greedy_accuracy_pct": round(greedy_acc * PERCENT, PP_DECIMALS),
        "final_accuracy_pct": round(final_acc * PERCENT, PP_DECIMALS),
        "criteria": {
            "escalation_uplift_pp": {
                "value": uplift_pp,
                "target": ESCALATION_UPLIFT_PP_TARGET,
                "met": uplift_pp >= ESCALATION_UPLIFT_PP_TARGET,
            }
        },
    }


def main(path: Path) -> None:
    print(json.dumps(uplift_report(load_rows(path), source=str(path)), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Accuracy uplift from judge escalation.")
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
