"""Test whether a detector scores adversarial prompts higher (criterion 4).

Compares one detector's scores on two run_benchmark.py outputs:
  - every row of the adversarial run (false premises, unanswerable questions);
  - the correctly answered rows of a labeled run (e.g. TriviaQA).
A one-sided Mann-Whitney U test asks whether adversarial scores tend to be
higher. It compares ranks, so it needs no assumption about the score
distribution — entropies are bounded, skewed and often tied at zero.

Usage:
    uv run python -m eval.adversarial_separation \\
        --adversarial eval/results/adversarial.jsonl \\
        --reference eval/results/triviaqa_1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import mannwhitneyu

from eval.benchmark_results import (
    ALIASES_FIELD,
    GREEDY_ANSWER_FIELD,
    BenchmarkRow,
    detector_score,
    is_correct,
    is_labeled,
    load_rows,
)
from eval.success_criteria import ADVERSARIAL_P_VALUE_TARGET
from hallucinations_math_lab.constants import DetectorName

MANN_WHITNEY_ALTERNATIVE = "greater"
STAT_DECIMALS = 4


def separation_report(
    adversarial_rows: list[BenchmarkRow],
    reference_rows: list[BenchmarkRow],
    detector: str,
) -> dict[str, Any]:
    """One-sided Mann-Whitney U: adversarial scores > correctly answered scores."""
    adversarial = [
        detector_score(row, detector, where=f"adversarial row {i}")
        for i, row in enumerate(adversarial_rows, start=1)
    ]
    correct = [
        detector_score(row, detector, where=f"reference row {i}")
        for i, row in enumerate(reference_rows, start=1)
        if is_labeled(row) and is_correct(row[GREEDY_ANSWER_FIELD], row[ALIASES_FIELD])
    ]
    if not adversarial:
        raise RuntimeError("The adversarial run has no rows.")
    if not correct:
        raise RuntimeError(
            "The reference run has no correctly answered labeled rows to compare against."
        )

    result = mannwhitneyu(adversarial, correct, alternative=MANN_WHITNEY_ALTERNATIVE)
    p_value = float(result.pvalue)
    return {
        "detector": detector,
        "n_adversarial": len(adversarial),
        "n_correct_reference": len(correct),
        "median_adversarial": round(statistics.median(adversarial), STAT_DECIMALS),
        "median_correct_reference": round(statistics.median(correct), STAT_DECIMALS),
        "mann_whitney_u": float(result.statistic),
        "criteria": {
            "adversarial_p_value": {
                "value": p_value,
                "target": ADVERSARIAL_P_VALUE_TARGET,
                "met": p_value < ADVERSARIAL_P_VALUE_TARGET,
            }
        },
    }


def main(adversarial_path: Path, reference_path: Path, detector: str) -> None:
    report = separation_report(load_rows(adversarial_path), load_rows(reference_path), detector)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Adversarial vs correct-answer score separation.")
    p.add_argument("--adversarial", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(
        "--detector",
        choices=[str(d) for d in DetectorName],
        default=str(DetectorName.SEMANTIC_ENTROPY),
    )
    args = p.parse_args()
    main(args.adversarial, args.reference, args.detector)
