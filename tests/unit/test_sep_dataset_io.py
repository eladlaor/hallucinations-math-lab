"""SEP dataset directory: writing and loading.

Contract pinned here (SEP_CLASSIFIER.md, "The dataset directory"):
  - A written dataset loads back identical: rows, every position's hidden
    states and the collection config.
  - The tensor file's header holds the ordered question ids; loading raises
    when they differ from rows.parquet's order, so no hidden state can be
    paired with another question's label.
  - Shapes are checked against the config's backbone on write and on load.
  - Every row has sample_n answers and cluster ids, and cluster ids
    0..n_clusters-1, on write and on load.
  - The collection config forbids unknown fields and unknown schema versions.
"""

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from safetensors.numpy import load_file, save_file

from eval.sep_dataset_io import (
    ROWS_SCHEMA,
    SEPDataset,
    SEPDatasetRow,
    load_sep_dataset,
    write_sep_dataset,
)
from hallucinations_math_lab.constants import (
    SEP_COLLECTION_CONFIG_FILE,
    SEP_DATASET_ROWS_FILE,
    SEP_DATASET_TENSORS_FILE,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPBackbone,
    SEPCollectionConfig,
    SEPCollectionLabels,
    SEPPromptSpec,
)

_NUM_LAYERS = 2
_HIDDEN = 3
_IDS = ["q0", "q1", "q2"]


def _config(n_questions_requested: int = len(_IDS)) -> SEPCollectionConfig:
    return SEPCollectionConfig(
        calibration_set="questions.jsonl",
        calibration_set_sha256="d" * 64,
        n_questions_requested=n_questions_requested,
        provider="hf_transformers",
        transformers_version="5.0.0",
        torch_version="2.14.1+cu126",
        torch_cuda_version="12.6",
        backbone=SEPBackbone(
            model_id="org/model",
            revision="a" * 40,
            num_hidden_layers=_NUM_LAYERS,
            hidden_size=_HIDDEN,
            dtype="bfloat16",
        ),
        prompt=SEPPromptSpec(chat_template_sha256="b" * 64, answer_instruction="Be brief."),
        labels=SEPCollectionLabels(
            sample_n=2,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=64,
            nli_model_id="org/nli",
            nli_revision="c" * 40,
        ),
    )


def _row(question_id: str) -> SEPDatasetRow:
    return SEPDatasetRow(
        id=question_id,
        question=f"Question {question_id}?",
        ground_truth_aliases=["A", "a."],
        sampled_answers=["A", "B"],
        cluster_ids=[0, 1],
        se_discrete=0.69,
        se_likelihood_weighted=0.5,
        n_clusters=2,
        greedy_answer="A",
        greedy_correct=True,
    )


def _dataset() -> SEPDataset:
    shape = (len(_IDS), _NUM_LAYERS + 1, _HIDDEN)
    return SEPDataset(
        config=_config(),
        rows=[_row(i) for i in _IDS],
        hidden_states={
            TokenPosition.TBG: np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
            TokenPosition.SLT: -np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        },
    )


def test_rows_schema_matches_row_fields() -> None:
    assert ROWS_SCHEMA.names == [f.name for f in fields(SEPDatasetRow)]


def test_write_then_load_roundtrips(tmp_path: Path) -> None:
    written = _dataset()
    write_sep_dataset(tmp_path, written)
    loaded = load_sep_dataset(tmp_path)

    assert loaded.config == written.config
    assert loaded.rows == written.rows
    for position in TokenPosition:
        np.testing.assert_array_equal(
            loaded.hidden_states[position], written.hidden_states[position]
        )


def test_tensor_ids_match_parquet_order(tmp_path: Path) -> None:
    write_sep_dataset(tmp_path, _dataset())
    tensors_path = tmp_path / SEP_DATASET_TENSORS_FILE
    tensors = load_file(tensors_path)
    save_file(tensors, tensors_path, metadata={"question_ids": json.dumps(["q1", "q0", "q2"])})

    with pytest.raises(ValueError, match=r"row 0.*'q1'.*'q0'"):
        load_sep_dataset(tmp_path)


def test_write_rejects_a_shape_that_disagrees_with_the_backbone(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset.hidden_states[TokenPosition.SLT] = np.zeros(
        (len(_IDS), _NUM_LAYERS, _HIDDEN), dtype=np.float32
    )
    with pytest.raises(ValueError, match=r"slt.*\(3, 3, 3\)"):
        write_sep_dataset(tmp_path, dataset)
    assert not any(tmp_path.iterdir())


def test_write_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset.rows[1] = _row("q0")
    with pytest.raises(ValueError, match="duplicate question ids"):
        write_sep_dataset(tmp_path, dataset)


def test_write_requires_every_position(tmp_path: Path) -> None:
    dataset = _dataset()
    del dataset.hidden_states[TokenPosition.SLT]
    with pytest.raises(ValueError, match="slt"):
        write_sep_dataset(tmp_path, dataset)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"sampled_answers": ["A"]}, "1 sampled answers and 2 cluster ids.*sample_n is 2"),
        ({"cluster_ids": [0, 1, 1]}, "2 sampled answers and 3 cluster ids"),
        ({"n_clusters": 3}, r"n_clusters=3 but cluster ids \[0, 1\]"),
        ({"cluster_ids": [0, 2]}, r"n_clusters=2 but cluster ids \[0, 2\]"),
    ],
)
def test_write_rejects_a_row_inconsistent_with_sample_n(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    dataset = _dataset()
    dataset.rows[1] = replace(_row("q1"), **changes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=rf"'q1'.*{match}"):
        write_sep_dataset(tmp_path, dataset)
    assert not any(tmp_path.iterdir())


def test_load_rejects_a_row_inconsistent_with_sample_n(tmp_path: Path) -> None:
    write_sep_dataset(tmp_path, _dataset())
    rows_path = tmp_path / SEP_DATASET_ROWS_FILE
    table = pq.read_table(rows_path)
    truncated = table.set_column(
        table.schema.get_field_index("sampled_answers"),
        "sampled_answers",
        pa.array([["A"]] * len(_IDS), type=pa.list_(pa.string())),
    )
    pq.write_table(truncated, rows_path)
    with pytest.raises(ValueError, match="'q0' has 1 sampled answers"):
        load_sep_dataset(tmp_path)


def test_load_names_the_file_when_the_id_header_is_not_json(tmp_path: Path) -> None:
    write_sep_dataset(tmp_path, _dataset())
    tensors_path = tmp_path / SEP_DATASET_TENSORS_FILE
    save_file(load_file(tensors_path), tensors_path, metadata={"question_ids": "[q0"})
    with pytest.raises(ValueError, match=rf"{SEP_DATASET_TENSORS_FILE}.*not JSON"):
        load_sep_dataset(tmp_path)


def test_config_rejects_unknown_fields_and_schema_versions(tmp_path: Path) -> None:
    payload = _config().model_dump(mode="json")
    with pytest.raises(ValidationError):
        SEPCollectionConfig.model_validate({**payload, "extra": 1})
    with pytest.raises(ValidationError):
        SEPCollectionConfig.model_validate({**payload, "schema_version": 2})

    write_sep_dataset(tmp_path, _dataset())
    (tmp_path / SEP_COLLECTION_CONFIG_FILE).write_text(json.dumps({**payload, "extra": 1}))
    with pytest.raises(ValueError, match=SEP_COLLECTION_CONFIG_FILE):
        load_sep_dataset(tmp_path)


def test_load_refuses_an_incomplete_dataset_unless_allowed(tmp_path: Path) -> None:
    """A run that stopped early must not pass for the full calibration set."""
    write_sep_dataset(tmp_path, replace(_dataset(), config=_config(n_questions_requested=5)))
    with pytest.raises(ValueError, match=r"incomplete: 3 of 5 requested"):
        load_sep_dataset(tmp_path)
    assert [r.id for r in load_sep_dataset(tmp_path, allow_incomplete=True).rows] == _IDS


def test_write_rejects_more_rows_than_questions_requested(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"3 rows but its config requested only 2"):
        write_sep_dataset(tmp_path, replace(_dataset(), config=_config(n_questions_requested=2)))
    assert not any(tmp_path.iterdir())
