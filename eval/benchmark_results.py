"""Reading and labeling run_benchmark.py output, shared by the scoring scripts.

One labeling rule for every script: an answer is correct when some
ground-truth alias appears in it as a run of whole tokens, after SQuAD-style
normalization of both (see `is_correct`). Rows without aliases (e.g. the
adversarial set) carry no label.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Final

import numpy as np

# Field names written by run_benchmark.py.
ALIASES_FIELD: Final[str] = "ground_truth_aliases"
GREEDY_ANSWER_FIELD: Final[str] = "greedy_answer"
FINAL_ANSWER_FIELD: Final[str] = "final_answer"
FINAL_ANSWER_SOURCE_FIELD: Final[str] = "final_answer_source"
# The judge's answer to the question, recorded on every row when the judge is
# enabled (not only escalated rows), so escalation can be compared to random.
JUDGE_ANSWER_FIELD: Final[str] = "judge_answer"
DETECTOR_SCORES_FIELD: Final[str] = "detector_scores"
ESCALATED_FIELD: Final[str] = "escalated"
ID_FIELD: Final[str] = "id"
# What produced the scores. Two runs are comparable only if these match.
MODEL_ID_FIELD: Final[str] = "model_id"
SAMPLE_N_FIELD: Final[str] = "sample_n"
SAMPLE_TEMPERATURE_FIELD: Final[str] = "sample_temperature"
MAX_TOKENS_FIELD: Final[str] = "max_tokens"
ANSWER_INSTRUCTION_FIELD: Final[str] = "answer_instruction"
RUN_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    MODEL_ID_FIELD,
    SAMPLE_N_FIELD,
    SAMPLE_TEMPERATURE_FIELD,
    MAX_TOKENS_FIELD,
    ANSWER_INSTRUCTION_FIELD,
)

_PUNCTUATION: Final[frozenset[str]] = frozenset(string.punctuation)
_ARTICLES_RE: Final[re.Pattern[str]] = re.compile(r"\b(a|an|the)\b")

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


def normalize_answer_tokens(text: str) -> list[str]:
    """SQuAD-style normalization, split into tokens.

    Lowercase, drop punctuation, drop the articles a/an/the, collapse
    whitespace: the normalization of the SQuAD v1.1 evaluation script, which
    TriviaQA's own evaluation reuses.
    """
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return no_articles.split()


def _contains_token_run(haystack: list[str], needle: list[str]) -> bool:
    width = len(needle)
    return any(haystack[i : i + width] == needle for i in range(len(haystack) - width + 1))


def is_correct(answer: str | None, aliases: list[str]) -> bool:
    """True when some alias appears in `answer` as a run of whole normalized tokens.

    Both sides are SQuAD-normalized, so "The Beatles." matches alias
    "Beatles". Whole tokens only, so alias "2" does not match "It was 1921."
    and "Paris" does not match "Parisian". Aliases that normalize to nothing
    (e.g. "The") are ignored.
    """
    answer_tokens = normalize_answer_tokens(answer or "")
    if not answer_tokens:
        return False
    for alias in aliases:
        alias_tokens = normalize_answer_tokens(alias)
        if alias_tokens and _contains_token_run(answer_tokens, alias_tokens):
            return True
    return False


def answer_word_counts(answers: list[str | None]) -> dict[str, float | int]:
    """Median, 90th percentile and max answer length in words.

    Correctness matching and NLI clustering both degrade on long answers, and
    a long tail here means the brief-answer instruction is not being followed
    or `max_tokens` is truncating answers, so every report carries it.
    """
    if not answers:
        raise ValueError("answer_word_counts needs at least one answer.")
    counts = np.array([len((a or "").split()) for a in answers], dtype=np.float64)
    return {
        "median": float(np.median(counts)),
        "p90": round(float(np.percentile(counts, 90)), 4),
        "max": int(counts.max()),
    }


def detector_score(row: BenchmarkRow, detector: str, where: str) -> float:
    """One detector's score from a row; fails fast if the run did not record it."""
    scores: dict[str, float] = row[DETECTOR_SCORES_FIELD]
    if detector not in scores:
        raise RuntimeError(
            f"{where}: no {detector!r} score (recorded: {sorted(scores)}). "
            "Re-run the benchmark with that detector enabled."
        )
    return float(scores[detector])
