"""Config blocks that tie SEP hidden states to the exact backbone, prompt and
labels that produced them (SEP_CLASSIFIER.md, "Artifact format").

A collected dataset's `collection_config.json` is `SEPCollectionConfig`. The
blocks are shared with the trained probe's config (`SEPProbeConfig`), so a
probe fitted on a dataset carries the same backbone and prompt it was
collected with. The probe directory's `train_ids.json` is `SEPTrainIds` and
its `metrics.json` is `SEPProbeMetrics`. Every block forbids unknown fields: a
misspelled or foreign field fails at load instead of being ignored.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from hallucinations_math_lab.constants import (
    GIT_COMMIT_SHA_PATTERN,
    HUB_COMMIT_SHA_PATTERN,
    SEP_DATASET_SCHEMA_VERSION,
    SEP_PROBE_SCHEMA_VERSION,
    SemanticEntropyEstimator,
    SEPBinarization,
    SEPClassifierType,
    SEPFeatureConcat,
    SEPLayerConvention,
    SEPLayerSelection,
    SEPOutputSemantics,
    SEPPaperDeviation,
    SEPStandardization,
    SEPThresholdSource,
    TokenPosition,
)

_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


class _StrictBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SEPBackbone(_StrictBlock):
    """The model whose hidden states were read, pinned to one Hub commit."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=HUB_COMMIT_SHA_PATTERN)
    num_hidden_layers: int = Field(ge=1)  # L; tensors hold layers 0..L
    hidden_size: int = Field(ge=1)
    dtype: str = Field(min_length=1)


class SEPPromptSpec(_StrictBlock):
    """How each question was formatted: a changed template or instruction
    moves every hidden state."""

    chat_template_sha256: str = Field(pattern=_SHA256_HEX_PATTERN)
    add_generation_prompt: Literal[True] = True
    answer_instruction: str = Field(min_length=1)


class SEPCollectionLabels(_StrictBlock):
    """How the SE labels were produced.

    `top_p` and `top_k` are the values sampling actually used. The sampler
    sends only the temperature, so `generate` takes both from the
    checkpoint's generation_config (deviation D8)."""

    sample_n: int = Field(ge=2)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    max_tokens: int = Field(ge=1)
    nli_model_id: str = Field(min_length=1)
    nli_revision: str = Field(pattern=HUB_COMMIT_SHA_PATTERN)


class SEPCollectionConfig(_StrictBlock):
    """`collection_config.json`: the calibration set, and the backbone, prompt
    and labels blocks of a collected SEP dataset. It holds no γ: the label
    split is a fit-time choice.

    `calibration_set` is the question file's name and `calibration_set_sha256`
    the SHA-256 of its bytes, so the fit, which reads only the dataset
    directory, can name the exact questions it trained on.
    `n_questions_requested` is the number of questions in that set. A run that
    stopped early writes fewer rows, and the count tells that partial dataset
    apart from a complete one."""

    schema_version: int = SEP_DATASET_SCHEMA_VERSION
    calibration_set: str = Field(min_length=1)
    calibration_set_sha256: str = Field(pattern=_SHA256_HEX_PATTERN)
    n_questions_requested: int = Field(ge=1)
    # The SLM_PROVIDER kind and transformers release that produced the hidden
    # states; the fit runs neither, so the probe copies both from here.
    provider: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    backbone: SEPBackbone
    prompt: SEPPromptSpec
    labels: SEPCollectionLabels

    @field_validator("schema_version")
    @classmethod
    def _known_schema_version(cls, value: int) -> int:
        if value != SEP_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unknown SEP dataset schema_version {value}; this code reads "
                f"version {SEP_DATASET_SCHEMA_VERSION}."
            )
        return value


class SEPFeatures(_StrictBlock):
    """Which hidden states a probe reads: `layers` at `token_position`, concatenated
    in increasing layer order. `final_norm_applied[i]` says whether `layers[i]` is
    layer L, the only one HF returns with the final norm applied."""

    token_position: TokenPosition
    layers: tuple[int, ...] = Field(min_length=1)
    layer_convention: SEPLayerConvention = SEPLayerConvention.AFTER_K_DECODER_LAYERS
    final_norm_applied: tuple[bool, ...]
    concat: SEPFeatureConcat = SEPFeatureConcat.INCREASING_LAYER

    @model_validator(mode="after")
    def _increasing_layers_one_flag_each(self) -> SEPFeatures:
        if any(b <= a for a, b in pairwise(self.layers)) or self.layers[0] < 1:
            raise ValueError(
                f"Probe layers must be strictly increasing and at least 1 (layer 0, the "
                f"embeddings, is never probed); got {list(self.layers)}."
            )
        if len(self.final_norm_applied) != len(self.layers):
            raise ValueError(
                f"final_norm_applied has {len(self.final_norm_applied)} flags for "
                f"{len(self.layers)} layers; it needs one per layer."
            )
        return self


class SEPProbeLabels(SEPCollectionLabels):
    """The collection's labels block plus the fit-time choices: which SE the
    target came from and where it was split (γ*, in nats)."""

    se_estimator: Literal[SemanticEntropyEstimator.DISCRETE] = SemanticEntropyEstimator.DISCRETE
    binarization: SEPBinarization = SEPBinarization.BEST_SPLIT_MIDPOINTS
    gamma_nats: float = Field(ge=0.0, allow_inf_nan=False)
    n_high: int = Field(ge=1)
    n_low: int = Field(ge=1)


class SEPClassifierSpec(_StrictBlock):
    """The logistic regression's settings, and how its layers were chosen."""

    type: SEPClassifierType = SEPClassifierType.LOGISTIC_REGRESSION
    C: float = Field(gt=0.0)
    penalty: str = Field(min_length=1)
    solver: str = Field(min_length=1)
    max_iter: int = Field(ge=1)
    standardization: SEPStandardization
    layer_selection: SEPLayerSelection
    cv_folds: int = Field(ge=2)
    seed: int
    sklearn_version: str = Field(min_length=1)


class SEPOutputSpec(_StrictBlock):
    """What the score means, and the threshold training recommends (never applied
    implicitly: SEP_THRESHOLD is the threshold)."""

    semantics: SEPOutputSemantics = SEPOutputSemantics.P_HIGH_SE
    recommended_threshold: float = Field(allow_inf_nan=False)


class SEPTrainingRecord(_StrictBlock):
    """Where the probe came from: the collection run, its rows, and this repo's state."""

    provider: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    n_rows: int = Field(ge=2)
    calibration_set_sha256: str = Field(pattern=_SHA256_HEX_PATTERN)
    repo_commit: str = Field(pattern=GIT_COMMIT_SHA_PATTERN)
    repo_dirty: bool
    created_at: AwareDatetime


class SEPProbeConfig(_StrictBlock):
    """`probe_config.json`: everything needed to check that a probe fits the
    model, prompt and labels it is used with, and to tell how it was made."""

    schema_version: int = SEP_PROBE_SCHEMA_VERSION
    backbone: SEPBackbone
    features: SEPFeatures
    prompt: SEPPromptSpec
    labels: SEPProbeLabels
    classifier: SEPClassifierSpec
    output: SEPOutputSpec
    training: SEPTrainingRecord
    deviations_from_paper: tuple[SEPPaperDeviation, ...] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def _known_schema_version(cls, value: int) -> int:
        if value != SEP_PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"Unknown SEP probe schema_version {value}; this code reads "
                f"version {SEP_PROBE_SCHEMA_VERSION}."
            )
        return value

    @model_validator(mode="after")
    def _consistent_with_backbone_and_rows(self) -> SEPProbeConfig:
        num_layers = self.backbone.num_hidden_layers
        if self.features.layers[-1] > num_layers:
            raise ValueError(
                f"Probe layers {list(self.features.layers)} exceed the backbone's "
                f"{num_layers} decoder layers."
            )
        expected_norm = tuple(layer == num_layers for layer in self.features.layers)
        if self.features.final_norm_applied != expected_norm:
            raise ValueError(
                f"final_norm_applied {list(self.features.final_norm_applied)} disagrees with "
                f"layers {list(self.features.layers)}: only layer L={num_layers} is normed."
            )
        if self.labels.n_high + self.labels.n_low != self.training.n_rows:
            raise ValueError(
                f"n_high={self.labels.n_high} and n_low={self.labels.n_low} do not add up to "
                f"n_rows={self.training.n_rows}."
            )
        return self


class SEPTrainIds(_StrictBlock):
    """`train_ids.json`: the questions a probe was fitted on, so a benchmark can
    refuse to score the probe on any of them."""

    model_id: str = Field(min_length=1)
    calibration_set: str = Field(min_length=1)
    question_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("question_ids")
    @classmethod
    def _unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Training question ids must be unique.")
        return value


class SEPAurocSummary(_StrictBlock):
    """AUROC across CV test folds: the mean and the population std."""

    mean: float = Field(ge=0.0, le=1.0)
    std: float = Field(ge=0.0)


class SEPFoldAuroc(SEPAurocSummary):
    """A CV AUROC with every fold's value."""

    per_fold: tuple[float, ...] = Field(min_length=2)


class SEPNestedCVAuroc(SEPFoldAuroc):
    """Nested-CV AUROC against binarized SE, with the window each outer fold chose."""

    window_per_fold: tuple[tuple[int, ...], ...] = Field(min_length=2)


class SEPLayerSweepMetrics(_StrictBlock):
    """One single-layer probe's CV AUROC; `auroc_vs_error` is None without error labels."""

    auroc_vs_binarized_se: SEPAurocSummary
    auroc_vs_error: SEPAurocSummary | None


class SEPSelectionMetrics(_StrictBlock):
    """One position's selected probe. `window` and `inner_cv_auroc` come from the
    selection run on all rows (`inner_cv_auroc` is None for fixed layers); the
    nested-CV AUROCs estimate the whole procedure, selection included."""

    window: tuple[int, ...] = Field(min_length=1)
    inner_cv_auroc: float | None = Field(ge=0.0, le=1.0)
    nested_cv_auroc_vs_binarized_se: SEPNestedCVAuroc
    nested_cv_auroc_vs_error: SEPFoldAuroc | None


class SEPThresholdMetrics(_StrictBlock):
    """The recommended threshold on p(high SE) and how it flags model errors."""

    recommended: float = Field(allow_inf_nan=False)
    tpr: float = Field(ge=0.0, le=1.0)
    fpr: float = Field(ge=0.0, le=1.0)
    escalation_rate: float = Field(ge=0.0, le=1.0)
    source: SEPThresholdSource


class SEPProbeMetrics(_StrictBlock):
    """`metrics.json` as the trainer writes it. Every per-position block holds
    both collected positions; `final_layer_auroc` is the sweep's layer-L entry,
    the old last-layer probe's baseline. `converged` and `n_iter` describe the
    served probe's final fit."""

    layer_sweep: dict[TokenPosition, dict[int, SEPLayerSweepMetrics]]
    final_layer_auroc: dict[TokenPosition, SEPLayerSweepMetrics]
    selection: dict[TokenPosition, SEPSelectionMetrics]
    threshold: SEPThresholdMetrics
    converged: bool
    n_iter: int = Field(ge=0)
