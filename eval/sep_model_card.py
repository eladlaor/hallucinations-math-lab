"""The Hub cards of a published SEP probe: the model card, the held-out report it
shows, the dataset card, and the license terms both state.

`render_model_card` and `render_dataset_card` are pure: they fill
eval/templates/SEP_MODEL_CARD.md and SEP_DATASET_CARD.md from the probe's config
(the model card also from its metrics and compute_auroc's held-out report) and
where the release goes. eval.publish_sep_probe checks those inputs agree before
calling them. A backbone with no entry in BACKBONE_LICENSES renders no card.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.datasets.load_triviaqa import DATASET_CONFIG, DATASET_NAME, DATASET_SPLIT
from eval.sep_dataset_io import TENSOR_IDS_METADATA_KEY
from eval.sep_recipe import MAX_ITER_FLAG, STANDARDIZE_FLAG
from hallucinations_math_lab.constants import (
    SEP_COLLECTION_CONFIG_FILE,
    SEP_DATASET_ROWS_FILE,
    SEP_DATASET_TENSORS_FILE,
    SEP_LAYERS_FLAG,
    SEP_LOGREG_MAX_ITER,
    SEP_METRICS_FILE,
    SEP_SERVED_TOKEN_POSITION,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    DetectorName,
    SEPLayerSelection,
    SEPLayersMode,
    SEPStandardization,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPProbeConfig,
    SEPProbeMetrics,
    SEPPublishedMetrics,
    SEPTrainingRecord,
)
from hallucinations_math_lab.services.semantic_entropy_probe import HubProbeRef

# Where the cards' templates live.
_TEMPLATES_DIR: Final[Path] = Path(__file__).parent / "templates"
MODEL_CARD_TEMPLATE: Final[Path] = _TEMPLATES_DIR / "SEP_MODEL_CARD.md"
DATASET_CARD_TEMPLATE: Final[Path] = _TEMPLATES_DIR / "SEP_DATASET_CARD.md"
# The code the cards link to: the commit that fitted the probe, in this repository.
REPO_URL: Final[str] = "https://github.com/eladlaor/hallucinations-math-lab"
# This repository's own license, and its file, which the cards link at the fitting commit.
CODE_LICENSE_NAME: Final[str] = "MIT"
CODE_LICENSE_FILE: Final[str] = "LICENSE"
HUB_URL: Final[str] = "https://huggingface.co"
# The license the Hub lists for the TriviaQA repo the calibration questions come from.
QUESTIONS_SOURCE_HUB_LICENSE: Final[str] = "unknown"
# Kossen et al. label the same hidden-state tuple 1-based, so our layer k is
# probably their layer k + 1 (likely, not verified); the card prints both.
PAPER_LAYER_OFFSET: Final[int] = 1
CARD_DECIMALS: Final[int] = 3
# The trainer invocation the card's refit section prints, before the probe's own options.
REFIT_COMMAND: Final[str] = (
    "uv run python -m eval.train_sep_probe --dataset sep-dataset --output-dir sep-probe"
)
# eval.train_sep_probe's `--layers <a>-<b>` range form.
_LAYER_RANGE_SEPARATOR: Final[str] = "-"
_PERCENT: Final[int] = 100


@dataclass(frozen=True)
class ReleaseLicense:
    """A license as the Hub names it (`license:` front matter) and as a reader reads it."""

    hub_id: str
    name: str
    url: str


@dataclass(frozen=True)
class BackboneLicense:
    """Who released a backbone, under which license, and the backbone's own license page."""

    licensor: str
    license: ReleaseLicense
    terms_url: str


APACHE_2_0: Final[ReleaseLicense] = ReleaseLicense(
    hub_id="apache-2.0",
    name="Apache License 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
# The license_link every Gemma 4 checkpoint's Hub repo lists; it redirects to Google's
# Apache 2.0 page.
GEMMA_4_LICENSE_URL: Final[str] = "https://ai.google.dev/gemma/docs/gemma_4_license"
_GEMMA_4_LICENSE: Final[BackboneLicense] = BackboneLicense(
    licensor="Google", license=APACHE_2_0, terms_url=GEMMA_4_LICENSE_URL
)
# Per backbone, the terms its Hub repo lists. A probe is released under its backbone's
# license; a backbone missing here is refused rather than published under unstated terms.
BACKBONE_LICENSES: Final[dict[str, BackboneLicense]] = {
    "google/gemma-4-E2B-it": _GEMMA_4_LICENSE,
    "google/gemma-4-E4B-it": _GEMMA_4_LICENSE,
    "google/gemma-4-26B-A4B-it": _GEMMA_4_LICENSE,
    "google/gemma-4-31B-it": _GEMMA_4_LICENSE,
}
# The license of the calibration data this repo derives: sampled answers, cluster ids,
# SE labels, hidden states and logs. The questions themselves are TriviaQA's.
DERIVED_DATA_LICENSE: Final[ReleaseLicense] = APACHE_2_0


def backbone_license(model_id: str) -> BackboneLicense:
    """The license terms recorded for `model_id`; raises if none is recorded."""
    terms = BACKBONE_LICENSES.get(model_id)
    if terms is None:
        raise ValueError(
            f"No license is recorded for backbone {model_id} (known: "
            f"{sorted(BACKBONE_LICENSES)}). Add the license its Hub repo lists to "
            "eval.sep_model_card.BACKBONE_LICENSES before publishing a probe for it."
        )
    return terms


@dataclass(frozen=True)
class ReleaseCards:
    """The two rendered cards of a release: the model repo's and the dataset repo's."""

    model: str
    dataset: str


@dataclass(frozen=True)
class PublishTarget:
    """Where a release goes: the probe's model repo, the dataset repo, and the tag on both."""

    model_repo: str
    dataset_repo: str
    tag: str

    @property
    def probe_uri(self) -> str:
        """The SEP_PROBE_PATH that loads the released probe."""
        return HubProbeRef(repo_id=self.model_repo, revision=self.tag).uri


class _HeldoutView(BaseModel):
    """The parts of compute_auroc's report the publisher checks and the card shows.
    Other fields are ignored here; metrics.json keeps the report verbatim."""

    model_config = ConfigDict(frozen=True)


class HeldoutCI(_HeldoutView):
    level: float = Field(gt=0.0, lt=1.0)
    low: float
    high: float


class HeldoutDetector(_HeldoutView):
    auroc: float = Field(ge=0.0, le=1.0)
    ci: HeldoutCI


class HeldoutProbe(_HeldoutView):
    """The SEP probe the held-out run scored (eval.sep_probe_provenance.SEPProbeRunRecord)."""

    ref: str = Field(min_length=1)
    weights_sha256: str
    config_sha256: str


class HeldoutRunConfig(_HeldoutView):
    model_id: str = Field(min_length=1)
    sep_probe: HeldoutProbe | None


class HeldoutReport(_HeldoutView):
    run_config: HeldoutRunConfig
    n_rows: int = Field(ge=1)
    n_hallucinations: int = Field(ge=0)
    detectors: dict[DetectorName, HeldoutDetector]
    # compute_auroc --json-out's ids of the scored rows, one per row in n_rows.
    question_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _one_id_per_scored_row(self) -> HeldoutReport:
        if len(self.question_ids) != self.n_rows:
            raise ValueError(
                f"The report scored {self.n_rows} rows but lists {len(self.question_ids)} "
                "question ids; regenerate it with eval.compute_auroc --json-out."
            )
        return self

    @model_validator(mode="after")
    def _one_ci_level(self) -> HeldoutReport:
        levels = {scores.ci.level for scores in self.detectors.values()}
        if len(levels) > 1:
            raise ValueError(
                f"The detectors' confidence intervals have different levels {sorted(levels)}; "
                "the card states one."
            )
        return self

    @property
    def ci_level(self) -> float:
        """The confidence level every detector's interval shares."""
        if not self.detectors:
            raise ValueError("The held-out report scored no detector.")
        return next(iter(self.detectors.values())).ci.level


# The cross-validation section's estimate and table column, by how the layers were chosen.
_CV_ESTIMATE: Final[dict[SEPLayerSelection, str]] = {
    SEPLayerSelection.NESTED_CV: (
        "Nested cross-validation ({folds} folds in both the outer and the inner loop) "
        "estimates the whole fitting procedure, including the layer selection, against "
        "binarized SE."
    ),
    SEPLayerSelection.FIXED: (
        "Cross-validation ({folds} folds), with the layers fixed in advance, estimates the "
        "fitting procedure against binarized SE."
    ),
}
_CV_COLUMN: Final[dict[SEPLayerSelection, str]] = {
    SEPLayerSelection.NESTED_CV: "Nested-CV AUROC vs binarized SE",
    SEPLayerSelection.FIXED: "CV AUROC vs binarized SE",
}


def _number(value: float) -> str:
    return f"{value:.{CARD_DECIMALS}f}"


def _torch_build(training: SEPTrainingRecord) -> str:
    """The torch build that produced the hidden states, as the cards print it:
    "2.14.1+cu126 (CUDA 12.6)", or "2.14.1 (CPU-only)" for a build without CUDA."""
    cuda = training.torch_cuda_version
    runtime = "CPU-only" if cuda is None else f"CUDA {cuda}"
    return f"{training.torch_version} ({runtime})"


def _layers(layers: Sequence[int], offset: int = 0) -> str:
    return ", ".join(str(layer + offset) for layer in layers)


def _heldout_table(heldout: HeldoutReport) -> str:
    """One row per detector the held-out run scored, in DetectorName order."""
    return "\n".join(
        f"| `{name}` | {_number(scores.auroc)} | {_number(scores.ci.low)}–"
        f"{_number(scores.ci.high)} |"
        for name in DetectorName
        if (scores := heldout.detectors.get(name)) is not None
    )


def _selection_table(metrics: SEPProbeMetrics) -> str:
    """One row per collected token position; the served one is marked."""
    rows = []
    for position in TokenPosition:
        selection = metrics.selection[position]
        nested = selection.nested_cv_auroc_vs_binarized_se
        final = metrics.final_layer_auroc[position].auroc_vs_binarized_se
        served = " (served)" if position == SEP_SERVED_TOKEN_POSITION else ""
        rows.append(
            f"| {str(position).upper()}{served} | {_layers(selection.window)} | "
            f"{_layers(selection.window, PAPER_LAYER_OFFSET)} | {_number(nested.mean)} ± "
            f"{_number(nested.std)} | {_number(final.mean)} ± {_number(final.std)} |"
        )
    return "\n".join(rows)


def refit_command(config: SEPProbeConfig) -> str:
    """The eval.train_sep_probe command that refits this probe: the default command plus
    every option the probe was fitted with (fixed layers, max_iter, standardization)."""
    classifier, layers = config.classifier, config.features.layers
    options: list[str] = []
    if classifier.layer_selection == SEPLayerSelection.FIXED:
        if layers == (config.backbone.num_hidden_layers,):
            options.append(f"{SEP_LAYERS_FLAG} {SEPLayersMode.FINAL}")
        elif layers == tuple(range(layers[0], layers[-1] + 1)):
            options.append(f"{SEP_LAYERS_FLAG} {layers[0]}{_LAYER_RANGE_SEPARATOR}{layers[-1]}")
        else:
            raise ValueError(
                f"The probe's fixed layers {list(layers)} are not a contiguous range, so no "
                f"{SEP_LAYERS_FLAG} value refits them."
            )
    if classifier.max_iter != SEP_LOGREG_MAX_ITER:
        options.append(f"{MAX_ITER_FLAG} {classifier.max_iter}")
    if classifier.standardization != SEPStandardization.NONE:
        options.append(STANDARDIZE_FLAG)
    return " ".join((REFIT_COMMAND, *options))


def _render(template: Path, fields: dict[str, object]) -> str:
    """`template` with every field filled; a field the template names but `fields` lacks fails."""
    try:
        return Template(template.read_text()).substitute(fields)
    except (OSError, KeyError, ValueError) as e:
        raise RuntimeError(f"Cannot render the card from {template}: {e!r}") from e


def _shared_fields(config: SEPProbeConfig, target: PublishTarget) -> dict[str, object]:
    """The fields both cards show: where the release goes, the backbone and its license,
    the code that fitted the probe, and the command that refits the probe."""
    backbone, training = config.backbone, config.training
    terms = backbone_license(backbone.model_id)
    return {
        "model_repo": target.model_repo,
        "dataset_repo": target.dataset_repo,
        "tag": target.tag,
        "hub_url": HUB_URL,
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "num_hidden_layers": backbone.num_hidden_layers,
        "hidden_size": backbone.hidden_size,
        "backbone_licensor": terms.licensor,
        "backbone_license_name": terms.license.name,
        "backbone_license_url": terms.terms_url,
        "repo_url": REPO_URL,
        "repo_commit": training.repo_commit,
        "code_license_name": CODE_LICENSE_NAME,
        "code_license_url": f"{REPO_URL}/blob/{training.repo_commit}/{CODE_LICENSE_FILE}",
        "refit_command": refit_command(config),
    }


def render_model_card(
    config: SEPProbeConfig,
    metrics: SEPPublishedMetrics,
    heldout: HeldoutReport,
    target: PublishTarget,
) -> str:
    """The model card: pure, from the probe's config and metrics, the held-out report
    and where the release goes. Every template field must be filled."""
    backbone, features, labels = config.backbone, config.features, config.labels
    classifier, training, threshold = config.classifier, config.training, metrics.threshold
    # The probe is released under its backbone's license.
    release_license = backbone_license(backbone.model_id).license
    fields = {
        **_shared_fields(config, target),
        "license": release_license.hub_id,
        "license_name": release_license.name,
        "license_url": release_license.url,
        "probe_uri": target.probe_uri,
        "train_ids_file": SEP_TRAIN_IDS_FILE,
        "training_log_file": SEP_TRAINING_LOG_FILE,
        "metrics_file": SEP_METRICS_FILE,
        "dtype": backbone.dtype,
        "token_position": str(features.token_position).upper(),
        "layers": _layers(features.layers),
        "paper_layers": _layers(features.layers, PAPER_LAYER_OFFSET),
        "layer_selection": classifier.layer_selection,
        "answer_instruction": config.prompt.answer_instruction,
        "chat_template_sha256": config.prompt.chat_template_sha256,
        "heldout_n_rows": heldout.n_rows,
        "heldout_n_hallucinations": heldout.n_hallucinations,
        "ci_percent": f"{heldout.ci_level * _PERCENT:g}",
        "heldout_table": _heldout_table(heldout),
        "selection_table": _selection_table(metrics),
        # Unrounded: it is the value to set, and the rates below hold at exactly it.
        "recommended_threshold": repr(threshold.recommended),
        "threshold_tpr": _number(threshold.tpr),
        "threshold_fpr": _number(threshold.fpr),
        "threshold_escalation_rate": _number(threshold.escalation_rate),
        "sample_n": labels.sample_n,
        "temperature": labels.temperature,
        "top_p": labels.top_p,
        "top_k": labels.top_k,
        "max_tokens": labels.max_tokens,
        "nli_model_id": labels.nli_model_id,
        "nli_revision": labels.nli_revision,
        "gamma_nats": _number(labels.gamma_nats),
        "n_high": labels.n_high,
        "n_low": labels.n_low,
        "n_rows": training.n_rows,
        "C": classifier.C,
        "penalty": classifier.penalty,
        "solver": classifier.solver,
        "max_iter": classifier.max_iter,
        "n_iter": metrics.n_iter,
        "standardization": classifier.standardization,
        "cv_estimate": _CV_ESTIMATE[classifier.layer_selection].format(folds=classifier.cv_folds),
        "cv_column": _CV_COLUMN[classifier.layer_selection],
        "sklearn_version": classifier.sklearn_version,
        "provider": training.provider,
        "transformers_version": training.transformers_version,
        "torch_build": _torch_build(training),
        "created_at": training.created_at.isoformat(),
        "deviations": ", ".join(str(d) for d in config.deviations_from_paper),
    }
    return _render(MODEL_CARD_TEMPLATE, fields)


def render_dataset_card(config: SEPProbeConfig, target: PublishTarget) -> str:
    """The dataset repo's card: pure, from the config of the probe the dataset trained
    (eval.publish_sep_probe checks they agree) and where the release goes."""
    backbone, labels, training = config.backbone, config.labels, config.training
    fields = {
        **_shared_fields(config, target),
        "license": DERIVED_DATA_LICENSE.hub_id,
        "license_name": DERIVED_DATA_LICENSE.name,
        "license_url": DERIVED_DATA_LICENSE.url,
        "source_dataset": DATASET_NAME,
        "source_dataset_config": DATASET_CONFIG,
        "source_dataset_split": DATASET_SPLIT,
        "source_dataset_hub_license": QUESTIONS_SOURCE_HUB_LICENSE,
        "n_rows": training.n_rows,
        "num_states": backbone.num_hidden_layers + 1,
        "tbg": TokenPosition.TBG,
        "slt": TokenPosition.SLT,
        "ids_metadata_key": TENSOR_IDS_METADATA_KEY,
        "rows_file": SEP_DATASET_ROWS_FILE,
        "tensors_file": SEP_DATASET_TENSORS_FILE,
        "collection_config_file": SEP_COLLECTION_CONFIG_FILE,
        "training_log_file": SEP_TRAINING_LOG_FILE,
        "sample_n": labels.sample_n,
        "temperature": labels.temperature,
        "top_p": labels.top_p,
        "top_k": labels.top_k,
        "max_tokens": labels.max_tokens,
        "nli_model_id": labels.nli_model_id,
        "nli_revision": labels.nli_revision,
        # The trainer copies these from the dataset's collection config.
        "provider": training.provider,
        "transformers_version": training.transformers_version,
        "torch_build": _torch_build(training),
    }
    return _render(DATASET_CARD_TEMPLATE, fields)
