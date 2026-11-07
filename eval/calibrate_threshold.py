"""Choose a detector's escalation threshold τ on a dev slice.

The defaults (`SE_THRESHOLD=1.0` for the nats-scored detectors,
`SEP_THRESHOLD=0.5` for the SEP probe) are placeholders, not measured values.
This script picks τ from a run_benchmark.py output over a dev slice that shares
no question with the test slice (see load_triviaqa.py `--offset`):

    uv run python -m eval.calibrate_threshold --dev eval/results/dev.jsonl \\
        --detector semantic_entropy --test eval/results/test.jsonl

τ maximizes Youden's J = TPR − FPR for flagging hallucinations, where a row
is flagged when `score > τ` (the evaluator's rule). Of equally good τ, the
highest is kept, since it escalates the fewest questions. J weighs a missed
hallucination and a needless escalation equally; when the judge's cost
matters more, read `escalation_rate` and the accuracy-vs-rate curve from
escalation_uplift.py instead.

Set the printed τ as the printed `env_var` before benchmarking the test
slice, and report the test result once, at that τ. A detector with no
escalation threshold (an LLM judge) is refused. τ is printed unrounded,
since it is the value to set and the reported rates hold at exactly it.
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
from hallucinations_math_lab.config import escalation_threshold_env_var
from hallucinations_math_lab.constants import ESCALATION_THRESHOLDED_DETECTORS, DetectorName

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
    """τ for `detector` over the labeled dev rows, after checking dev/test disjointness.

    Raises ValueError for a detector that has no escalation threshold to set.
    """
    env_var = escalation_threshold_env_var(DetectorName(detector))
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
        "env_var": env_var,
        "n_rows": len(labeled),
        "n_hallucinations": int(sum(labels)),
        "tau": tau,  # unrounded: it is the value to set, and the rates are at it
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


def build_parser() -> argparse.ArgumentParser:
    """The CLI; `--detector` offers only the detectors that have a threshold to set."""
    p = argparse.ArgumentParser(
        description="Choose a detector's escalation threshold on a dev slice."
    )
    p.add_argument("--dev", type=Path, required=True, help="run_benchmark output, dev slice.")
    p.add_argument(
        "--detector",
        default=str(DetectorName.SEMANTIC_ENTROPY),
        choices=sorted(str(d) for d in ESCALATION_THRESHOLDED_DETECTORS),
    )
    p.add_argument(
        "--test", type=Path, default=None, help="Test-slice output; refused if ids overlap."
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args.dev, args.detector, args.test)
