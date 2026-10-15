"""Compute one AUROC per detector from run_benchmark.py output.

A 'hallucination' is defined as: the greedy answer does NOT contain any
ground-truth alias (see eval/benchmark_results.py). Rows without aliases
(e.g. the adversarial set) are skipped — they carry no label.

AUROC answers: if you pick one hallucinated row and one correct row at random,
how often does the detector give the hallucinated one the higher score?
0.5 = coin flip, 1.0 = perfect ranking.

When both Semantic Entropy and predictive entropy were scored, the report also
checks success criteria 1 and 2 (SE AUROC, and SE's margin over PE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

from eval.benchmark_results import (
    ALIASES_FIELD,
    DETECTOR_SCORES_FIELD,
    GREEDY_ANSWER_FIELD,
    BenchmarkRow,
    is_correct,
    is_labeled,
    load_rows,
)
from eval.success_criteria import SE_AUROC_TARGET, SE_MINUS_PE_AUROC_TARGET
from hallucinations_math_lab.constants import DetectorName

AUROC_DECIMALS = 4


def auroc_report(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    """Per-detector AUROC over the labeled rows, plus criteria 1–2 when scorable."""
    scores: dict[str, list[float]] = {}
    is_hallucination: list[int] = []

    for row_no, row in enumerate(rows, start=1):
        if not is_labeled(row):
            continue
        row_scores: dict[str, float] = row[DETECTOR_SCORES_FIELD]
        if scores and set(row_scores) != set(scores):
            raise RuntimeError(
                f"{source} row {row_no} scored detectors {sorted(row_scores)}, but earlier rows "
                f"scored {sorted(scores)}. Re-run the benchmark with one detector config."
            )
        for name, value in row_scores.items():
            scores.setdefault(name, []).append(value)
        correct = is_correct(row[GREEDY_ANSWER_FIELD], row[ALIASES_FIELD])
        is_hallucination.append(0 if correct else 1)

    if not is_hallucination:
        raise RuntimeError(f"No scorable rows in {source} (no ground-truth aliases found).")
    if len(set(is_hallucination)) < 2:
        raise RuntimeError(
            f"AUROC needs both correct and hallucinated rows; all {len(is_hallucination)} "
            "scorable rows have the same label. Use a larger or harder question set."
        )

    auroc = {
        name: round(float(roc_auc_score(is_hallucination, values)), AUROC_DECIMALS)
        for name, values in scores.items()
    }
    report: dict[str, Any] = {
        "n_rows": len(is_hallucination),
        "n_hallucinations": sum(is_hallucination),
        "auroc": auroc,
    }

    se = auroc.get(str(DetectorName.SEMANTIC_ENTROPY))
    pe = auroc.get(str(DetectorName.PREDICTIVE_ENTROPY))
    criteria: dict[str, Any] = {}
    if se is not None:
        criteria["se_auroc"] = {
            "value": se,
            "target": SE_AUROC_TARGET,
            "met": se >= SE_AUROC_TARGET,
        }
    if se is not None and pe is not None:
        gap = round(se - pe, AUROC_DECIMALS)
        criteria["se_minus_pe_auroc"] = {
            "value": gap,
            "target": SE_MINUS_PE_AUROC_TARGET,
            "met": gap >= SE_MINUS_PE_AUROC_TARGET,
        }
    if criteria:
        report["criteria"] = criteria
    return report


def main(path: Path) -> None:
    print(json.dumps(auroc_report(load_rows(path), source=str(path)), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-detector AUROC from benchmark output.")
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
