"""Compute AUROC for SE-as-hallucination-detector and the PE baseline.

A 'hallucination' is defined as: the greedy answer does NOT match any ground-truth
alias (lowercased substring match, the TriviaQA convention).
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
    se_scores: list[float] = []
    pe_scores: list[float] = []
    is_hallucination: list[int] = []

    with path.open() as f:
        for line in f:
            row = json.loads(line)
            aliases = row.get("ground_truth_aliases", [])
            if not aliases:
                # Adversarial rows have no aliases; skip from AUROC (they are stress
                # tests scored separately).
                continue
            correct = _is_correct(row["greedy_answer"], aliases)
            is_hallucination.append(0 if correct else 1)
            se_scores.append(row["semantic_entropy"])
            pe_scores.append(row["predictive_entropy"])

    if not is_hallucination:
        raise RuntimeError("No scorable rows in input (no ground-truth aliases found).")

    se_auroc = roc_auc_score(is_hallucination, se_scores)
    pe_auroc = roc_auc_score(is_hallucination, pe_scores)

    print(json.dumps(
        {
            "n_rows": len(is_hallucination),
            "n_hallucinations": sum(is_hallucination),
            "semantic_entropy_auroc": round(float(se_auroc), 4),
            "predictive_entropy_auroc": round(float(pe_auroc), 4),
            "se_minus_pe": round(float(se_auroc - pe_auroc), 4),
        },
        indent=2,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
