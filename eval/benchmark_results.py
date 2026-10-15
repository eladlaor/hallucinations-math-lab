"""Reading and labeling run_benchmark.py output, shared by the scoring scripts.

One labeling rule for every script: an answer is correct when it contains any
ground-truth alias (case-insensitive substring match, the TriviaQA
convention). Rows without aliases (e.g. the adversarial set) carry no label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

# Field names written by run_benchmark.py.
ALIASES_FIELD: Final[str] = "ground_truth_aliases"
GREEDY_ANSWER_FIELD: Final[str] = "greedy_answer"
FINAL_ANSWER_FIELD: Final[str] = "final_answer"
FINAL_ANSWER_SOURCE_FIELD: Final[str] = "final_answer_source"
DETECTOR_SCORES_FIELD: Final[str] = "detector_scores"
ESCALATED_FIELD: Final[str] = "escalated"

BenchmarkRow = dict[str, Any]


def load_rows(path: Path) -> list[BenchmarkRow]:
    """Every row of a run_benchmark.py JSONL file; fails fast on a malformed line."""
    rows: list[BenchmarkRow] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{line_no} is not valid JSON: {e}") from e
    if not rows:
        raise RuntimeError(f"{path} contains no benchmark rows.")
    return rows


def is_labeled(row: BenchmarkRow) -> bool:
    return bool(row.get(ALIASES_FIELD))


def is_correct(answer: str | None, aliases: list[str]) -> bool:
    """True when `answer` contains any alias (case-insensitive substring)."""
    a = (answer or "").lower().strip()
    if not a:
        return False
    return any(alias.lower().strip() in a for alias in aliases if alias)


def detector_score(row: BenchmarkRow, detector: str, where: str) -> float:
    """One detector's score from a row; fails fast if the run did not record it."""
    scores: dict[str, float] = row[DETECTOR_SCORES_FIELD]
    if detector not in scores:
        raise RuntimeError(
            f"{where}: no {detector!r} score (recorded: {sorted(scores)}). "
            "Re-run the benchmark with that detector enabled."
        )
    return float(scores[detector])
