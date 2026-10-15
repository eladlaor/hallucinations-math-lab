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
ID_FIELD: Final[str] = "id"
# What produced the scores. Two runs are comparable only if these match.
MODEL_ID_FIELD: Final[str] = "model_id"
SAMPLE_N_FIELD: Final[str] = "sample_n"
SAMPLE_TEMPERATURE_FIELD: Final[str] = "sample_temperature"
RUN_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    MODEL_ID_FIELD,
    SAMPLE_N_FIELD,
    SAMPLE_TEMPERATURE_FIELD,
)

BenchmarkRow = dict[str, Any]
RunConfig = dict[str, Any]


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


def run_config(rows: list[BenchmarkRow], source: str) -> RunConfig:
    """The model and sampling settings behind a run; fails fast if rows disagree.

    Mixing rows from different models or sampling settings would make every
    statistic computed over them meaningless, so a file must be one run.
    """
    config: RunConfig | None = None
    for row_no, row in enumerate(rows, start=1):
        missing = [field for field in RUN_CONFIG_FIELDS if field not in row]
        if missing:
            raise RuntimeError(
                f"{source} row {row_no} lacks {missing}; it was written before runs recorded "
                "their model and sampling settings. Re-run eval.run_benchmark."
            )
        row_config = {field: row[field] for field in RUN_CONFIG_FIELDS}
        if config is None:
            config = row_config
        elif row_config != config:
            raise RuntimeError(
                f"{source} row {row_no} was produced with {row_config}, but earlier rows with "
                f"{config}. One results file must hold one run."
            )
    if config is None:
        raise RuntimeError(f"{source} contains no benchmark rows.")
    return config


def assert_same_run_config(first: RunConfig, second: RunConfig, what: str) -> None:
    """Fail fast when two runs being compared used different models or sampling."""
    if first != second:
        raise RuntimeError(
            f"{what}: the runs are not comparable — {first} vs {second}. Benchmark both "
            "question sets with the same model and sampling settings."
        )


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
