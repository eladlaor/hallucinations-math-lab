"""Reading and writing a SEP dataset directory (SEP_CLASSIFIER.md, "The dataset directory").

    rows.parquet               one row per question: labels and answers
    hidden_states.safetensors  `tbg` and `slt`, each float32 [rows, L+1, d]
    collection_config.json     backbone, prompt and labels (SEPCollectionConfig)
    training_log.jsonl         one line per question, appended as collection runs

Tensors go in safetensors rather than parquet list columns: tens of thousands
of floats per row bloat parquet. The tensor header holds the ordered question
ids, and loading checks them against the parquet row order, so a hidden state
can never be paired with another question's label.

This module only reads and writes; it computes nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from hallucinations_math_lab.constants import (
    SEP_COLLECTION_CONFIG_FILE,
    SEP_DATASET_ROWS_FILE,
    SEP_DATASET_TENSORS_FILE,
    SEP_TRAINING_LOG_FILE,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import SEPCollectionConfig

# safetensors header metadata key holding the JSON list of question ids, in row order.
TENSOR_IDS_METADATA_KEY: Final[str] = "question_ids"
_TENSOR_FRAMEWORK: Final[str] = "numpy"
_HIDDEN_STATE_DTYPE: Final = np.float32


@dataclass(frozen=True)
class SEPDatasetRow:
    """One calibration question's row in rows.parquet."""

    id: str
    question: str
    ground_truth_aliases: list[str]
    sampled_answers: list[str]
    cluster_ids: list[int]  # cluster index per sampled answer, in sample order
    se_discrete: float
    se_likelihood_weighted: float
    n_clusters: int
    greedy_answer: str
    greedy_correct: bool


# Column order and types of rows.parquet; names must equal SEPDatasetRow's fields.
ROWS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("id", pa.string()),
        ("question", pa.string()),
        ("ground_truth_aliases", pa.list_(pa.string())),
        ("sampled_answers", pa.list_(pa.string())),
        ("cluster_ids", pa.list_(pa.int64())),
        ("se_discrete", pa.float64()),
        ("se_likelihood_weighted", pa.float64()),
        ("n_clusters", pa.int64()),
        ("greedy_answer", pa.string()),
        ("greedy_correct", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class TrainingLogEntry:
    """One line of training_log.jsonl."""

    id: str
    se_discrete: float
    se_likelihood_weighted: float
    n_clusters: int
    sampled_answer_word_counts: list[int]
    greedy_answer_word_count: int
    greedy_correct: bool
    elapsed_s: float


@dataclass(frozen=True)
class SEPDataset:
    """A whole dataset directory in memory: rows in order, and per position
    the hidden states of those rows, float32 [rows, L+1, hidden_size]."""

    config: SEPCollectionConfig
    rows: list[SEPDatasetRow]
    hidden_states: dict[TokenPosition, np.ndarray]


def check_dataset(dataset: SEPDataset) -> None:
    """Fail fast unless there are no more rows than questions requested, ids are
    unique, every row matches the config's sample_n, and every position's
    tensor is float32 [rows, L+1, hidden_size] for the config's backbone."""
    ids = [row.id for row in dataset.rows]
    if len(ids) > dataset.config.n_questions_requested:
        raise ValueError(
            f"SEP dataset has {len(ids)} rows but its config requested only "
            f"{dataset.config.n_questions_requested} questions."
        )
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"SEP dataset has duplicate question ids: {duplicates}.")
    sample_n = dataset.config.labels.sample_n
    for row in dataset.rows:
        check_row(row, sample_n)
    if set(dataset.hidden_states) != set(TokenPosition):
        raise ValueError(
            f"SEP dataset needs hidden states for {sorted(str(p) for p in TokenPosition)}; "
            f"got {sorted(str(p) for p in dataset.hidden_states)}."
        )
    backbone = dataset.config.backbone
    expected = (len(ids), backbone.num_hidden_layers + 1, backbone.hidden_size)
    for position, states in dataset.hidden_states.items():
        if states.shape != expected or states.dtype != _HIDDEN_STATE_DTYPE:
            raise ValueError(
                f"Hidden states for {position} are {states.dtype} {states.shape}; expected "
                f"{np.dtype(_HIDDEN_STATE_DTYPE)} {expected} (rows, L+1, hidden_size) for "
                f"{backbone.model_id}."
            )


def check_row(row: SEPDatasetRow, sample_n: int) -> None:
    """Raise unless the row has `sample_n` answers, one cluster id per answer,
    and cluster ids that are exactly 0..n_clusters-1."""
    if len(row.sampled_answers) != sample_n or len(row.cluster_ids) != sample_n:
        raise ValueError(
            f"SEP dataset row {row.id!r} has {len(row.sampled_answers)} sampled answers and "
            f"{len(row.cluster_ids)} cluster ids; the config's sample_n is {sample_n}."
        )
    if set(row.cluster_ids) != set(range(row.n_clusters)):
        raise ValueError(
            f"SEP dataset row {row.id!r} has n_clusters={row.n_clusters} but cluster ids "
            f"{row.cluster_ids}; expected ids 0..n_clusters-1."
        )


def prepare_dataset_dir(directory: Path) -> None:
    """Create `directory`, refusing one that already holds files: a collection
    costs hours of GPU time and is never overwritten or mixed with another."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"SEP dataset directory {directory} is not empty. Move it away or pass another "
            "--output-dir; a collection never overwrites an earlier one."
        )
    directory.mkdir(parents=True, exist_ok=True)


def append_training_log(directory: Path, entry: TrainingLogEntry) -> None:
    """Append one question's line to training_log.jsonl, so a record of progress
    survives a crash (the log cannot resume a collection)."""
    path = directory / SEP_TRAINING_LOG_FILE
    try:
        with path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
    except OSError as e:
        raise RuntimeError(f"Failed to append question {entry.id!r} to {path}: {e!s}") from e


def write_sep_dataset(directory: Path, dataset: SEPDataset) -> None:
    """Write rows.parquet, hidden_states.safetensors and collection_config.json.

    Everything is checked before the first file is written."""
    check_dataset(dataset)
    ids = [row.id for row in dataset.rows]
    try:
        table = pa.Table.from_pylist([asdict(row) for row in dataset.rows], schema=ROWS_SCHEMA)
        pq.write_table(table, directory / SEP_DATASET_ROWS_FILE)
        save_file(
            {str(position): states for position, states in dataset.hidden_states.items()},
            directory / SEP_DATASET_TENSORS_FILE,
            metadata={TENSOR_IDS_METADATA_KEY: json.dumps(ids)},
        )
        (directory / SEP_COLLECTION_CONFIG_FILE).write_text(
            dataset.config.model_dump_json(indent=2) + "\n"
        )
    except (OSError, pa.ArrowException, SafetensorError) as e:
        raise RuntimeError(f"Failed to write the SEP dataset to {directory}: {e!s}") from e


def load_sep_dataset(directory: Path, *, allow_incomplete: bool = False) -> SEPDataset:
    """Load a dataset directory, checking its config, shapes and id order.

    A collection that stopped early holds fewer rows than it requested; it is
    refused unless `allow_incomplete`, so nothing fits on it by accident."""
    config_path = directory / SEP_COLLECTION_CONFIG_FILE
    try:
        config = SEPCollectionConfig.model_validate_json(config_path.read_text())
    except (OSError, ValidationError) as e:
        raise ValueError(f"Cannot read SEP dataset config {config_path}: {e!s}") from e

    rows_path = directory / SEP_DATASET_ROWS_FILE
    try:
        table = pq.read_table(rows_path)
    except (OSError, pa.ArrowException) as e:
        raise ValueError(f"Cannot read SEP dataset rows {rows_path}: {e!s}") from e
    if table.schema.names != ROWS_SCHEMA.names or table.schema.types != ROWS_SCHEMA.types:
        raise ValueError(f"{rows_path} has columns {table.schema}; expected {ROWS_SCHEMA}.")
    rows = [SEPDatasetRow(**record) for record in table.to_pylist()]

    tensors_path = directory / SEP_DATASET_TENSORS_FILE
    try:
        with safe_open(tensors_path, framework=_TENSOR_FRAMEWORK) as f:
            metadata = f.metadata() or {}
            hidden_states = {TokenPosition(key): f.get_tensor(key) for key in f.keys()}  # noqa: SIM118 (not a dict)
    except (OSError, SafetensorError, ValueError) as e:
        raise ValueError(f"Cannot read SEP dataset hidden states {tensors_path}: {e!s}") from e
    if TENSOR_IDS_METADATA_KEY not in metadata:
        raise ValueError(f"{tensors_path} header lacks {TENSOR_IDS_METADATA_KEY!r}.")
    try:
        tensor_ids = json.loads(metadata[TENSOR_IDS_METADATA_KEY])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Cannot read SEP dataset hidden states {tensors_path}: header "
            f"{TENSOR_IDS_METADATA_KEY!r} is not JSON: {e!s}"
        ) from e
    check_id_order(tensor_ids, [row.id for row in rows])

    dataset = SEPDataset(config=config, rows=rows, hidden_states=hidden_states)
    check_dataset(dataset)
    if not allow_incomplete and len(rows) < config.n_questions_requested:
        raise ValueError(
            f"SEP dataset {directory} is incomplete: {len(rows)} of "
            f"{config.n_questions_requested} requested questions were collected before the run "
            "stopped. Pass allow_incomplete=True to load it anyway."
        )
    return dataset


def check_id_order(tensor_ids: list[str], row_ids: list[str]) -> None:
    """Raise unless the tensor header's ids equal the parquet rows' ids, in order."""
    if tensor_ids == row_ids:
        return
    if len(tensor_ids) != len(row_ids):
        raise ValueError(
            f"{SEP_DATASET_TENSORS_FILE} holds {len(tensor_ids)} question ids but "
            f"{SEP_DATASET_ROWS_FILE} has {len(row_ids)} rows."
        )
    row, tensor_id, row_id = next(
        (i, t, r) for i, (t, r) in enumerate(zip(tensor_ids, row_ids, strict=True)) if t != r
    )
    raise ValueError(
        f"{SEP_DATASET_TENSORS_FILE} and {SEP_DATASET_ROWS_FILE} disagree at row {row}: "
        f"tensor id {tensor_id!r}, parquet id {row_id!r}. Hidden states would be paired "
        "with the wrong labels."
    )
