"""Builders for small, valid SEP probe directories, shared by the detector,
evaluator and benchmark tests. A probe written here loads exactly like one
eval.train_sep_probe writes: `probe.safetensors` plus `probe_config.json`, and
`write_train_ids` adds its `train_ids.json`.
`ProbeBackboneProvider` gives a fake sampling provider the backbone these
probes were trained on, so the evaluator's compatibility check passes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hallucinations_math_lab.config import Settings
from hallucinations_math_lab.constants import (
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_TRAIN_IDS_FILE,
    SEPLayerSelection,
    SEPPaperDeviation,
    SEPStandardization,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPBackbone,
    SEPClassifierSpec,
    SEPFeatures,
    SEPOutputSpec,
    SEPProbeConfig,
    SEPProbeLabels,
    SEPPromptSpec,
    SEPTrainIds,
    SEPTrainingRecord,
)
from hallucinations_math_lab.services.sampling_provider import check_layers
from hallucinations_math_lab.services.sep_probe_weights import ProbeWeights, write_probe_weights

PROBE_MODEL_ID = "org/model"
PROBE_REVISION = "a" * 40
PROBE_NUM_HIDDEN_LAYERS = 2
PROBE_HIDDEN_SIZE = 4
# The configured default, which the evaluator checks a probe against.
PROBE_ANSWER_INSTRUCTION = Settings.model_fields["answer_instruction"].default
PROBE_TEMPLATE_SHA256 = "b" * 64
PROBE_RECOMMENDED_THRESHOLD = 0.4
_N_HIGH = 5
_N_LOW = 5


class ProbeBackboneProvider:
    """Base for fake sampling providers: reports the backbone and prompt the
    probes built here were trained on, and serves any valid layers."""

    model_id: str = PROBE_MODEL_ID
    model_revision: str = PROBE_REVISION
    num_hidden_layers: int = PROBE_NUM_HIDDEN_LAYERS
    hidden_size: int = PROBE_HIDDEN_SIZE
    chat_template_sha256: str = PROBE_TEMPLATE_SHA256

    def check_hidden_state_request(self, layers: tuple[int, ...]) -> None:
        check_layers(layers, self.num_hidden_layers)


def probe_config(
    *,
    model_id: str = PROBE_MODEL_ID,
    revision: str = PROBE_REVISION,
    num_hidden_layers: int = PROBE_NUM_HIDDEN_LAYERS,
    hidden_size: int = PROBE_HIDDEN_SIZE,
    layers: tuple[int, ...] = (PROBE_NUM_HIDDEN_LAYERS,),
    token_position: TokenPosition = TokenPosition.TBG,
) -> SEPProbeConfig:
    """A probe config for `model_id` reading `layers` at `token_position`."""
    return SEPProbeConfig(
        backbone=SEPBackbone(
            model_id=model_id,
            revision=revision,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            dtype="bfloat16",
        ),
        features=SEPFeatures(
            token_position=token_position,
            layers=layers,
            final_norm_applied=tuple(layer == num_hidden_layers for layer in layers),
        ),
        prompt=SEPPromptSpec(
            chat_template_sha256=PROBE_TEMPLATE_SHA256,
            answer_instruction=PROBE_ANSWER_INSTRUCTION,
        ),
        labels=SEPProbeLabels(
            sample_n=10,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=64,
            nli_model_id="org/nli",
            nli_revision="c" * 40,
            gamma_nats=0.5,
            n_high=_N_HIGH,
            n_low=_N_LOW,
        ),
        classifier=SEPClassifierSpec(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=100,
            standardization=SEPStandardization.NONE,
            layer_selection=SEPLayerSelection.FIXED,
            cv_folds=5,
            seed=0,
            sklearn_version="1.8.0",
        ),
        output=SEPOutputSpec(recommended_threshold=PROBE_RECOMMENDED_THRESHOLD),
        training=SEPTrainingRecord(
            provider="hf_transformers",
            transformers_version="5.0.0",
            n_rows=_N_HIGH + _N_LOW,
            calibration_set_sha256="d" * 64,
            repo_commit="e" * 40,
            repo_dirty=False,
            created_at=datetime(2026, 11, 7, tzinfo=UTC),
        ),
        deviations_from_paper=(SEPPaperDeviation.CHAT_TEMPLATE_PROMPT,),
    )


def write_probe_dir(
    directory: Path,
    config: SEPProbeConfig | None = None,
    weight: np.ndarray | None = None,
    bias: float = 0.0,
) -> Path:
    """Write a loadable probe directory; `weight` defaults to all ones."""
    config = config or probe_config()
    n_features = len(config.features.layers) * config.backbone.hidden_size
    directory.mkdir(parents=True, exist_ok=True)
    write_probe_weights(
        directory / SEP_PROBE_WEIGHTS_FILE,
        ProbeWeights(
            weight=(np.ones(n_features) if weight is None else weight).astype(np.float32),
            bias=np.array([bias], dtype=np.float32),
        ),
    )
    (directory / SEP_PROBE_CONFIG_FILE).write_text(config.model_dump_json(indent=2) + "\n")
    return directory


def write_train_ids(
    directory: Path, question_ids: list[str], model_id: str = PROBE_MODEL_ID
) -> Path:
    """Write the probe directory's `train_ids.json`, as eval.train_sep_probe does."""
    path = directory / SEP_TRAIN_IDS_FILE
    train_ids = SEPTrainIds(
        model_id=model_id, calibration_set="calib.jsonl", question_ids=tuple(question_ids)
    )
    path.write_text(train_ids.model_dump_json(indent=2) + "\n")
    return path
