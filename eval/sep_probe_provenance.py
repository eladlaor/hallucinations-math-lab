"""Which questions a SEP probe was trained on, and a guard against evaluating on them.

A probe scored on its own training prompts reports an optimistic AUROC: the
linear map has seen those exact hidden states. train_sep_probe.py writes a
sidecar next to the probe listing the training question ids; run_benchmark.py
refuses to score any of those ids with that probe.

Sidecar: `<probe stem>.train_ids.json` next to `<probe stem>.pt`, holding
`model_id`, `calibration_set` and `question_ids`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

SIDECAR_SUFFIX: Final[str] = ".train_ids.json"
MODEL_ID_KEY: Final[str] = "model_id"
CALIBRATION_SET_KEY: Final[str] = "calibration_set"
QUESTION_IDS_KEY: Final[str] = "question_ids"
# How many overlapping ids to name in the error message.
MAX_IDS_IN_ERROR: Final[int] = 5


def training_ids_path(probe_path: Path) -> Path:
    return probe_path.with_name(probe_path.stem + SIDECAR_SUFFIX)


def write_training_ids(
    probe_path: Path, question_ids: list[str], model_id: str, calibration_set: Path
) -> Path:
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Calibration set contains duplicate question ids.")
    sidecar = training_ids_path(probe_path)
    sidecar.write_text(
        json.dumps(
            {
                MODEL_ID_KEY: model_id,
                CALIBRATION_SET_KEY: str(calibration_set),
                QUESTION_IDS_KEY: question_ids,
            },
            indent=2,
        )
    )
    return sidecar


def read_training_ids(probe_path: Path) -> set[str]:
    sidecar = training_ids_path(probe_path)
    if not sidecar.is_file():
        raise RuntimeError(
            f"SEP probe {probe_path} has no training-id sidecar ({sidecar.name}), so it cannot "
            "be shown not to overlap the benchmark. Retrain it with eval.train_sep_probe."
        )
    try:
        return set(json.loads(sidecar.read_text())[QUESTION_IDS_KEY])
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Malformed SEP probe sidecar {sidecar}: {e!s}") from e


def assert_disjoint(probe_path: Path, benchmark_ids: list[str]) -> None:
    """Fail fast if any benchmark question was used to train the probe."""
    overlap = sorted(set(benchmark_ids) & read_training_ids(probe_path))
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} benchmark questions were used to train the SEP probe "
            f"{probe_path} (e.g. {overlap[:MAX_IDS_IN_ERROR]}); its scores on them would be "
            "optimistic. Build the benchmark from a disjoint slice (load_triviaqa --offset)."
        )
