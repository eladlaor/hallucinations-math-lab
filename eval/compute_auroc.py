"""Compute one AUROC per detector from run_benchmark.py output, with 95% CIs.

A 'hallucination' is defined as: the greedy answer does NOT contain any
ground-truth alias (see eval/benchmark_results.py). Rows without aliases
(e.g. the adversarial set) are skipped — they carry no label.

AUROC answers: if you pick one hallucinated row and one correct row at random,
how often does the detector give the hallucinated one the higher score?
0.5 = coin flip, 1.0 = perfect ranking. Each AUROC comes with a percentile
bootstrap 95% CI over rows.

When both Semantic Entropy and predictive entropy were scored, the report also
checks success criteria 1 and 2. Criterion 2 (SE's margin over PE) uses a
*paired* bootstrap: both AUROCs are recomputed on the same resampled rows, and
the criterion is met only if the point gap clears the target and the CI of
the gap excludes zero.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np

from eval.benchmark_results import (
    ALIASES_FIELD,
    DETECTOR_SCORES_FIELD,
    GREEDY_ANSWER_FIELD,
    BenchmarkRow,
    answer_word_counts,
    is_correct,
    is_labeled,
    load_rows,
    run_config,
)
from eval.inference_stats import CI_LEVEL, BootstrapCI, auroc, paired_bootstrap
from eval.success_criteria import SE_AUROC_TARGET, SE_MINUS_PE_AUROC_TARGET
from hallucinations_math_lab.constants import DetectorName

REPORT_DECIMALS: Final[int] = 4


def _ci_dict(ci: BootstrapCI) -> dict[str, Any]:
    return {
        "level": CI_LEVEL,
        "low": round(ci.low, REPORT_DECIMALS),
        "high": round(ci.high, REPORT_DECIMALS),
        "n_resamples": ci.n_resamples,
        "n_degenerate_resamples": ci.n_degenerate,
    }


def _auroc_on(labels: np.ndarray, values: np.ndarray) -> Callable[[np.ndarray], float | None]:
    """AUROC of `values` restricted to resampled row indices."""
    return lambda idx: auroc(labels[idx], values[idx])


def _labeled_matrix(
    rows: list[BenchmarkRow], source: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Hallucination labels and per-detector score arrays over labeled rows."""
    scores: dict[str, list[float]] = {}
    labels: list[int] = []
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
        labels.append(0 if is_correct(row[GREEDY_ANSWER_FIELD], row[ALIASES_FIELD]) else 1)

    if not labels:
        raise RuntimeError(f"No scorable rows in {source} (no ground-truth aliases found).")
    if len(set(labels)) < 2:
        raise RuntimeError(
            f"AUROC needs both correct and hallucinated rows; all {len(labels)} "
            "scorable rows have the same label. Use a larger or harder question set."
        )
    return np.asarray(labels), {name: np.asarray(v) for name, v in scores.items()}


def auroc_report(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    """Per-detector AUROC with CIs over the labeled rows, plus criteria 1–2."""
    config = run_config(rows, source)
    labels, scores = _labeled_matrix(rows, source)

    point: dict[str, float] = {}
    detectors: dict[str, Any] = {}
    for name, values in scores.items():
        value = auroc(labels, values)
        if value is None:  # unreachable: both classes checked above
            raise RuntimeError(f"AUROC undefined for {name} in {source}.")
        point[name] = value
        ci = paired_bootstrap(len(labels), _auroc_on(labels, values))
        detectors[name] = {"auroc": round(value, REPORT_DECIMALS), "ci": _ci_dict(ci)}

    report: dict[str, Any] = {
        "run_config": config,
        "n_rows": len(labels),
        "n_hallucinations": int(labels.sum()),
        "greedy_answer_words": answer_word_counts(
            [row[GREEDY_ANSWER_FIELD] for row in rows if is_labeled(row)]
        ),
        "detectors": detectors,
    }

    se_name, pe_name = str(DetectorName.SEMANTIC_ENTROPY), str(DetectorName.PREDICTIVE_ENTROPY)
    criteria: dict[str, Any] = {}
    if se_name in point:
        criteria["se_auroc"] = {
            "value": round(point[se_name], REPORT_DECIMALS),
            "ci": detectors[se_name]["ci"],
            "target": SE_AUROC_TARGET,
            "met": point[se_name] >= SE_AUROC_TARGET,
        }
    if se_name in point and pe_name in point:
        se_values, pe_values = scores[se_name], scores[pe_name]

        def gap_on(idx: np.ndarray) -> float | None:
            se_auc = auroc(labels[idx], se_values[idx])
            pe_auc = auroc(labels[idx], pe_values[idx])
            return None if se_auc is None or pe_auc is None else se_auc - pe_auc

        gap = point[se_name] - point[pe_name]
        gap_ci = paired_bootstrap(len(labels), gap_on)
        criteria["se_minus_pe_auroc"] = {
            "value": round(gap, REPORT_DECIMALS),
            "ci": _ci_dict(gap_ci),
            "target": SE_MINUS_PE_AUROC_TARGET,
            "met": gap >= SE_MINUS_PE_AUROC_TARGET and gap_ci.low > 0.0,
        }
    if criteria:
        report["criteria"] = criteria
    return report


def main(path: Path) -> None:
    print(json.dumps(auroc_report(load_rows(path), source=str(path)), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-detector AUROC with bootstrap CIs.")
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
