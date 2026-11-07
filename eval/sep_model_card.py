"""The model card of a published SEP probe, and the held-out report it shows.

`render_model_card` is pure: it fills eval/templates/SEP_MODEL_CARD.md from the
probe's config and metrics, compute_auroc's held-out report, and where the
release goes. eval.publish_sep_probe checks those inputs agree before calling it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eval.sep_recipe import MAX_ITER_FLAG, STANDARDIZE_FLAG
from hallucinations_math_lab.constants import (
    SEP_LAYERS_FLAG,
    SEP_LOGREG_MAX_ITER,
    SEP_SERVED_TOKEN_POSITION,
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
)
from hallucinations_math_lab.services.semantic_entropy_probe import HubProbeRef

# Where the card's template lives.
MODEL_CARD_TEMPLATE: Final[Path] = Path(__file__).parent / "templates" / "SEP_MODEL_CARD.md"
# The code the card links to: the commit that fitted the probe, in this repository.
REPO_URL: Final[str] = "https://github.com/eladlaor/hallucinations-math-lab"
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
    fields = {
        "model_repo": target.model_repo,
        "dataset_repo": target.dataset_repo,
        "tag": target.tag,
        "probe_uri": target.probe_uri,
        "model_id": backbone.model_id,
        "revision": backbone.revision,
        "num_hidden_layers": backbone.num_hidden_layers,
        "hidden_size": backbone.hidden_size,
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
        "created_at": training.created_at.isoformat(),
        "repo_url": REPO_URL,
        "repo_commit": training.repo_commit,
        "refit_command": refit_command(config),
        "deviations": ", ".join(str(d) for d in config.deviations_from_paper),
    }
    try:
        return Template(MODEL_CARD_TEMPLATE.read_text()).substitute(fields)
    except (OSError, KeyError, ValueError) as e:
        raise RuntimeError(f"Cannot render the model card from {MODEL_CARD_TEMPLATE}: {e!r}") from e
