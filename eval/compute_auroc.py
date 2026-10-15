"""Compute one AUROC per detector from run_benchmark.py output.

A 'hallucination' is defined as: the greedy answer does NOT contain any
ground-truth alias (lowercased substring match, the TriviaQA convention).
Rows without aliases (e.g. the adversarial set) are skipped — they are stress
tests read by hand, not labeled data.

AUROC answers: if you pick one hallucinated row and one correct row at random,
how often does the detector give the hallucinated one the higher score?
0.5 = coin flip, 1.0 = perfect ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


def _is_correct(answer: str, aliases: list[str]) -> bool:
    a = (answer or "").lower().strip()
    if not a:
        return False
    return any(alias.lower().strip() in a for alias in aliases if alias)


def main(path: Path) -> None:
    scores: dict[str, list[float]] = {}
    is_hallucination: list[int] = []

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            aliases = row.get("ground_truth_aliases", [])
            if not aliases:
                continue
            row_scores: dict[str, float] = row["detector_scores"]
            if scores and set(row_scores) != set(scores):
                raise RuntimeError(
                    f"{path}:{line_no} scored detectors {sorted(row_scores)}, but earlier rows "
                    f"scored {sorted(scores)}. Re-run the benchmark with one detector config."
                )
            for name, value in row_scores.items():
                scores.setdefault(name, []).append(value)
            is_hallucination.append(0 if _is_correct(row["greedy_answer"], aliases) else 1)

    if not is_hallucination:
        raise RuntimeError(f"No scorable rows in {path} (no ground-truth aliases found).")
    if len(set(is_hallucination)) < 2:
        raise RuntimeError(
            f"AUROC needs both correct and hallucinated rows; all {len(is_hallucination)} "
            "scorable rows have the same label. Use a larger or harder question set."
        )

    print(
        json.dumps(
            {
                "n_rows": len(is_hallucination),
                "n_hallucinations": sum(is_hallucination),
                "auroc": {
                    name: round(float(roc_auc_score(is_hallucination, values)), 4)
                    for name, values in scores.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-detector AUROC from benchmark output.")
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
