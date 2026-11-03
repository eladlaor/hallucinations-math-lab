"""Choose the escalation threshold τ (SE_THRESHOLD) on a dev slice.

The default `SE_THRESHOLD=1.0` is a placeholder, not a measured value. This
script picks τ from a run_benchmark.py output over a dev slice that shares
no question with the test slice (see load_triviaqa.py `--offset`):

    uv run python -m eval.calibrate_threshold --dev eval/results/dev.jsonl \\
        --detector semantic_entropy --test eval/results/test.jsonl

τ maximizes Youden's J = TPR − FPR for flagging hallucinations, where a row
is flagged when `score > τ` (the evaluator's rule). Of equally good τ, the
highest is kept, since it escalates the fewest questions. J weighs a missed
hallucination and a needless escalation equally; when the judge's cost
matters more, read `escalation_rate` and the accuracy-vs-rate curve from
escalation_uplift.py instead.

Set the printed τ as SE_THRESHOLD before benchmarking the test slice, and
report the test result once, at that τ.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from eval.benchmark_results import (
    ALIASES_FIELD,
    GREEDY_ANSWER_FIELD,
    ID_FIELD,
    BenchmarkRow,
    detector_score,
    is_correct,
    is_labeled,
    load_rows,
    run_config,
)

REPORT_DECIMALS: Final[int] = 4


def youden_threshold(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, float, float]:
    """(τ, TPR, FPR) maximizing TPR − FPR, flagging `score > τ`; ties go to the higher τ.

    `labels` are 1 for a hallucination, 0 for a correct answer.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Calibration needs both hallucinated and correct rows; got {n_pos} and {n_neg}."
        )
    # Each distinct score is a candidate τ (flag everything above it), plus
    # one just below the minimum (flag everything).
    candidates = np.concatenate([[np.nextafter(s.min(), -np.inf)], np.unique(s)])
    best: tuple[float, float, float, float] | None = None  # (J, τ, tpr, fpr)
    for tau in candidates:
        flagged = s > tau
        tpr = float((flagged & (y == 1)).sum()) / n_pos
        fpr = float((flagged & (y == 0)).sum()) / n_neg
        j = tpr - fpr
        if best is None or j > best[0] or (j == best[0] and tau > best[1]):
            best = (j, float(tau), tpr, fpr)
    assert best is not None  # candidates is never empty
    return best[1], best[2], best[3]


def calibrate_report(
    rows: list[BenchmarkRow],
    detector: str,
    source: str,
    test_ids: set[str] | None = None,
) -> dict[str, Any]:
    """τ for `detector` over the labeled dev rows, after checking dev/test disjointness."""
    config = run_config(rows, source)
    labeled = [row for row in rows if is_labeled(row)]
    if not labeled:
        raise RuntimeError(f"No labeled rows in {source} (no ground-truth aliases found).")
    if test_ids is not None:
        overlap = sorted({str(row[ID_FIELD]) for row in labeled} & test_ids)
        if overlap:
            raise RuntimeError(
                f"{len(overlap)} questions in {source} are also in the test slice "
                f"(e.g. {overlap[:5]}). Calibrate on a disjoint slice."
            )

    labels = [0 if is_correct(r[GREEDY_ANSWER_FIELD], r[ALIASES_FIELD]) else 1 for r in labeled]
    scores = [
        detector_score(row, detector, f"{source} row {i}") for i, row in enumerate(labeled, 1)
    ]
    tau, tpr, fpr = youden_threshold(labels, scores)
    rate = float(np.mean(np.asarray(scores) > tau))
    return {
        "run_config": config,
        "detector": detector,
        "n_rows": len(labeled),
        "n_hallucinations": int(sum(labels)),
        "tau": round(tau, REPORT_DECIMALS),
        "tpr": round(tpr, REPORT_DECIMALS),
        "fpr": round(fpr, REPORT_DECIMALS),
        "escalation_rate": round(rate, REPORT_DECIMALS),
    }


def main(dev: Path, detector: str, test: Path | None) -> None:
    test_ids = (
        {str(row[ID_FIELD]) for row in load_rows(test) if row.get(ID_FIELD) is not None}
        if test is not None
        else None
    )
    print(json.dumps(calibrate_report(load_rows(dev), detector, str(dev), test_ids), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Choose SE_THRESHOLD on a dev slice.")
    p.add_argument("--dev", type=Path, required=True, help="run_benchmark output, dev slice.")
    p.add_argument("--detector", default="semantic_entropy")
    p.add_argument(
        "--test", type=Path, default=None, help="Test-slice output; refused if ids overlap."
    )
    args = p.parse_args()
    main(args.dev, args.detector, args.test)
