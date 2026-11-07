"""Fit a Semantic Entropy Probe from a collected SEP dataset, on CPU.

The input is a dataset directory written by eval.collect_sep_dataset; nothing
else is read, so a refit from a published dataset reproduces the probe. For
each collected token position (TBG and SLT):
  1. a single-layer sweep: the CV AUROC of a probe at every layer 1..L,
     against binarized SE and against model error (descriptive only);
  2. nested CV of the whole procedure: each outer fold picks its layer window
     by an inner CV on its own training rows, so the estimate pays for the
     choice (deviation D4);
  3. the final window, chosen by the same inner procedure on all rows.

The served probe reads TBG (v1 serves no SLT probe): γ* and the logistic
regression are fitted on all rows at the final window. Its recommended
threshold is Youden's J on the nested CV's out-of-fold scores against model
error (deviation D9).

The output is a probe directory (SEP_CLASSIFIER.md, "Artifact format"), the
same layout on disk and on the Hub: probe.safetensors, probe_config.json,
train_ids.json, training_log.jsonl (the dataset's, copied) and metrics.json.

Usage:
    uv run python -m eval.train_sep_probe --dataset DIR [--output-dir DIR]
        [--layers auto|final|<a>-<b>] [--max-iter 100|1000] [--standardize]

`--layers final` reads layer L only, the one layer the vLLM embed runner can
serve; `<a>-<b>` fixes the window to layers a..b. Both skip the inner selection.
`--max-iter 1000`, then `--standardize`, are the pre-registered fallbacks when
the paper's fit does not converge (deviations D10 and D12). Without
`--output-dir`, the probe goes to data/probes/<model slug>/, which must be
empty or absent.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import sklearn

from eval.repo_state import RepoState, read_repo_state
from eval.sep_dataset_io import SEPDataset, load_sep_dataset
from eval.sep_layer_selection import (
    FIRST_PROBED_LAYER,
    CVAuroc,
    LayerSweepEntry,
    NestedCVResult,
    ProbeFitOptions,
    ThresholdRecommendation,
    Window,
    WindowChoice,
    WindowProbe,
    choose_window,
    fit_window_probe,
    layer_sweep,
    nested_cv,
    recommend_threshold,
)
from eval.sep_recipe import HIGH_SE_LABEL, MAX_ITER_FLAG, STANDARDIZE_FLAG, binarize
from hallucinations_math_lab.constants import (
    LOG_EXTRA_KEY,
    RANDOM_SEED,
    SEP_CV_FOLDS,
    SEP_LAYERS_FLAG,
    SEP_LOGREG_C,
    SEP_LOGREG_FALLBACK_MAX_ITER,
    SEP_LOGREG_MAX_ITER,
    SEP_LOGREG_PENALTY,
    SEP_LOGREG_SOLVER,
    SEP_METRICS_FILE,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_DIR,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_SERVED_TOKEN_POSITION,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    SEPLayerSelection,
    SEPLayersMode,
    SEPPaperDeviation,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPAurocSummary,
    SEPClassifierSpec,
    SEPFeatures,
    SEPFoldAuroc,
    SEPLayerSweepMetrics,
    SEPNestedCVAuroc,
    SEPOutputSpec,
    SEPProbeConfig,
    SEPProbeLabels,
    SEPProbeMetrics,
    SEPSelectionMetrics,
    SEPThresholdMetrics,
    SEPTrainIds,
    SEPTrainingRecord,
)
from hallucinations_math_lab.services.semantic_entropy_probe import model_slug
from hallucinations_math_lab.services.sep_probe_weights import (
    PROBE_DTYPE,
    ProbeWeights,
    write_probe_weights,
)

logger = logging.getLogger("train_sep_probe")

# `--layers <a>-<b>`: an inclusive range of decoder layers.
_LAYER_RANGE_PATTERN: Final[str] = r"(\d+)-(\d+)"
# Deviations every probe carries; the rest depend on how it was fitted.
_RECIPE_DEVIATIONS: Final[frozenset[SEPPaperDeviation]] = frozenset(SEPPaperDeviation) - {
    SEPPaperDeviation.WINDOW_BY_INNER_CV,
    SEPPaperDeviation.WINDOW_RANKED_BY_BINARIZED_SE,
    SEPPaperDeviation.FALLBACK_MAX_ITER,
    SEPPaperDeviation.FOLDED_STANDARDIZATION,
}

# The only lbfgs budgets a probe may use: the recipe's, and its pre-registered fallback (D10).
_PRE_REGISTERED_MAX_ITERS: Final[tuple[int, ...]] = (
    SEP_LOGREG_MAX_ITER,
    SEP_LOGREG_FALLBACK_MAX_ITER,
)

LayersArg = SEPLayersMode | Window


@dataclass(frozen=True, eq=False)
class PositionFit:
    """One token position's measurements. `choice` is None for fixed layers."""

    sweep: dict[int, LayerSweepEntry]
    nested: NestedCVResult
    choice: WindowChoice | None
    window: Window


def parse_layers(value: str) -> LayersArg:
    """`--layers`: `auto`, `final`, or an inclusive range `<a>-<b>` with 1 <= a <= b."""
    if value in set(SEPLayersMode):
        return SEPLayersMode(value)
    match = re.fullmatch(_LAYER_RANGE_PATTERN, value)
    if match is None:
        raise ValueError(
            f"{SEP_LAYERS_FLAG} must be {SEPLayersMode.AUTO}, {SEPLayersMode.FINAL} or a range "
            f"<a>-<b>; got {value!r}."
        )
    first, last = int(match.group(1)), int(match.group(2))
    if not FIRST_PROBED_LAYER <= first <= last:
        raise ValueError(
            f"{SEP_LAYERS_FLAG} {value}: the range needs {FIRST_PROBED_LAYER} <= a <= b "
            "(layer 0, the embeddings, is never probed)."
        )
    return tuple(range(first, last + 1))


def _parse_layers_arg(value: str) -> LayersArg:
    """parse_layers for argparse, which prints only an ArgumentTypeError's message."""
    try:
        return parse_layers(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def fixed_window(layers: LayersArg, num_hidden_layers: int) -> Window | None:
    """The layers every fit reads, or None when the window is selected (`auto`).
    A range past layer L raises before any fit runs."""
    if isinstance(layers, tuple):
        if layers[-1] > num_hidden_layers:
            raise ValueError(
                f"{SEP_LAYERS_FLAG} {layers[0]}-{layers[-1]}: the backbone has "
                f"{num_hidden_layers} decoder layers, so b must be <= {num_hidden_layers}."
            )
        return layers
    return None if layers == SEPLayersMode.AUTO else (num_hidden_layers,)


def fit_position(
    hidden_states: npt.NDArray[np.float32],
    se: npt.NDArray[np.float64],
    error: npt.NDArray[np.int64],
    fixed: Window | None,
    options: ProbeFitOptions,
) -> PositionFit:
    """Sweep, nested CV and the final window choice for one position's states."""
    sweep = layer_sweep(hidden_states, se, error, options=options)
    nested = nested_cv(hidden_states, se, error, fixed_layers=fixed, options=options)
    if fixed is not None:
        return PositionFit(sweep=sweep, nested=nested, choice=None, window=fixed)
    choice = choose_window(hidden_states, se, options=options)
    return PositionFit(sweep=sweep, nested=nested, choice=choice, window=choice.window)


def active_deviations(
    layer_selection: SEPLayerSelection, options: ProbeFitOptions
) -> tuple[SEPPaperDeviation, ...]:
    """The paper deviations this probe carries, in table order."""
    active = set(_RECIPE_DEVIATIONS)
    if layer_selection == SEPLayerSelection.NESTED_CV:
        active |= {
            SEPPaperDeviation.WINDOW_BY_INNER_CV,
            SEPPaperDeviation.WINDOW_RANKED_BY_BINARIZED_SE,
        }
    if options.max_iter != SEP_LOGREG_MAX_ITER:
        active.add(SEPPaperDeviation.FALLBACK_MAX_ITER)
    if options.standardize:
        active.add(SEPPaperDeviation.FOLDED_STANDARDIZATION)
    return tuple(d for d in SEPPaperDeviation if d in active)


def _summary(auroc: CVAuroc) -> SEPAurocSummary:
    return SEPAurocSummary(mean=auroc.mean, std=auroc.std)


def _fold_auroc(auroc: CVAuroc) -> SEPFoldAuroc:
    return SEPFoldAuroc(mean=auroc.mean, std=auroc.std, per_fold=auroc.per_fold)


def _sweep_metrics(entry: LayerSweepEntry) -> SEPLayerSweepMetrics:
    return SEPLayerSweepMetrics(
        auroc_vs_binarized_se=_summary(entry.auroc_vs_binarized_se),
        auroc_vs_error=None if entry.auroc_vs_error is None else _summary(entry.auroc_vs_error),
    )


def _selection_metrics(fit: PositionFit) -> SEPSelectionMetrics:
    se_auroc = fit.nested.auroc_vs_binarized_se
    error_auroc = fit.nested.auroc_vs_error
    return SEPSelectionMetrics(
        window=fit.window,
        inner_cv_auroc=None if fit.choice is None else fit.choice.inner_cv_auroc,
        nested_cv_auroc_vs_binarized_se=SEPNestedCVAuroc(
            mean=se_auroc.mean,
            std=se_auroc.std,
            per_fold=se_auroc.per_fold,
            window_per_fold=fit.nested.window_per_fold,
        ),
        nested_cv_auroc_vs_error=None if error_auroc is None else _fold_auroc(error_auroc),
    )


def build_metrics(
    fits: dict[TokenPosition, PositionFit],
    probe: WindowProbe,
    threshold: ThresholdRecommendation,
    num_hidden_layers: int,
) -> SEPProbeMetrics:
    """metrics.json: the measured outputs, in SEP_CLASSIFIER.md's schema."""
    return SEPProbeMetrics(
        layer_sweep={
            position: {k: _sweep_metrics(entry) for k, entry in fit.sweep.items()}
            for position, fit in fits.items()
        },
        final_layer_auroc={
            position: _sweep_metrics(fit.sweep[num_hidden_layers]) for position, fit in fits.items()
        },
        selection={position: _selection_metrics(fit) for position, fit in fits.items()},
        threshold=SEPThresholdMetrics(
            recommended=threshold.recommended,
            tpr=threshold.tpr,
            fpr=threshold.fpr,
            escalation_rate=threshold.escalation_rate,
            source=threshold.source,
        ),
        # fit_logistic raises on non-convergence, so a written probe converged.
        converged=True,
        n_iter=probe.fit.n_iter,
    )


def build_probe_config(
    dataset: SEPDataset,
    se: npt.NDArray[np.float64],
    probe: WindowProbe,
    layer_selection: SEPLayerSelection,
    options: ProbeFitOptions,
    threshold: ThresholdRecommendation,
    repo: RepoState,
    created_at: datetime,
) -> SEPProbeConfig:
    """probe_config.json for the served probe fitted on `dataset`."""
    collected = dataset.config
    num_layers = collected.backbone.num_hidden_layers
    n_high = int((binarize(se, probe.gamma) == HIGH_SE_LABEL).sum())
    return SEPProbeConfig(
        backbone=collected.backbone,
        features=SEPFeatures(
            token_position=SEP_SERVED_TOKEN_POSITION,
            layers=probe.window,
            final_norm_applied=tuple(layer == num_layers for layer in probe.window),
        ),
        prompt=collected.prompt,
        labels=SEPProbeLabels(
            **collected.labels.model_dump(),
            gamma_nats=probe.gamma,
            n_high=n_high,
            n_low=se.size - n_high,
        ),
        classifier=SEPClassifierSpec(
            C=SEP_LOGREG_C,
            penalty=SEP_LOGREG_PENALTY,
            solver=SEP_LOGREG_SOLVER,
            max_iter=probe.fit.max_iter,
            standardization=probe.fit.standardization,
            layer_selection=layer_selection,
            cv_folds=SEP_CV_FOLDS,
            seed=RANDOM_SEED,
            sklearn_version=sklearn.__version__,
        ),
        output=SEPOutputSpec(recommended_threshold=threshold.recommended),
        training=SEPTrainingRecord(
            provider=collected.provider,
            transformers_version=collected.transformers_version,
            n_rows=se.size,
            calibration_set_sha256=collected.calibration_set_sha256,
            repo_commit=repo.commit,
            repo_dirty=repo.dirty,
            created_at=created_at,
        ),
        deviations_from_paper=active_deviations(layer_selection, options),
    )


def _prepare_probe_dir(directory: Path) -> None:
    """Create `directory`, refusing one that already holds files."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"SEP probe directory {directory} is not empty. Move it away or pass another "
            "--output-dir; a fit never overwrites an earlier probe."
        )
    directory.mkdir(parents=True, exist_ok=True)


def write_probe_dir(
    directory: Path,
    probe: WindowProbe,
    config: SEPProbeConfig,
    metrics: SEPProbeMetrics,
    train_ids: SEPTrainIds,
    dataset_dir: Path,
) -> None:
    """Write the five files of a probe directory; the training log is the dataset's."""
    weights = ProbeWeights(
        weight=probe.fit.weight.astype(PROBE_DTYPE),
        bias=np.array([probe.fit.bias], dtype=PROBE_DTYPE),
    )
    write_probe_weights(directory / SEP_PROBE_WEIGHTS_FILE, weights)
    try:
        (directory / SEP_PROBE_CONFIG_FILE).write_text(config.model_dump_json(indent=2) + "\n")
        (directory / SEP_METRICS_FILE).write_text(metrics.model_dump_json(indent=2) + "\n")
        (directory / SEP_TRAIN_IDS_FILE).write_text(train_ids.model_dump_json(indent=2) + "\n")
        shutil.copyfile(dataset_dir / SEP_TRAINING_LOG_FILE, directory / SEP_TRAINING_LOG_FILE)
    except OSError as e:
        raise RuntimeError(f"Failed to write the SEP probe directory {directory}: {e!s}") from e


def train_sep_probe(
    dataset_dir: Path,
    output_dir: Path | None,
    *,
    layers: LayersArg = SEPLayersMode.AUTO,
    max_iter: int = SEP_LOGREG_MAX_ITER,
    standardize: bool = False,
    repo: RepoState,
    created_at: datetime,
) -> SEPProbeConfig:
    """Fit the probe on `dataset_dir` and write its directory to `output_dir`
    (default: data/probes/<model slug>/). Returns the written config."""
    if max_iter not in _PRE_REGISTERED_MAX_ITERS:
        raise ValueError(
            f"max_iter={max_iter} is not pre-registered; use one of {_PRE_REGISTERED_MAX_ITERS}."
        )
    dataset = load_sep_dataset(dataset_dir)
    backbone = dataset.config.backbone
    fixed = fixed_window(layers, backbone.num_hidden_layers)
    directory = output_dir or Path(SEP_PROBE_DIR) / model_slug(backbone.model_id)
    _prepare_probe_dir(directory)

    se = np.array([row.se_discrete for row in dataset.rows], dtype=np.float64)
    error = np.array([0 if row.greedy_correct else 1 for row in dataset.rows], dtype=np.int64)
    options = ProbeFitOptions(max_iter=max_iter, standardize=standardize)
    ctx = {"dataset_dir": str(dataset_dir), "n_rows": se.size, "model_id": backbone.model_id}
    logger.info("Fitting SEP probe.", extra={LOG_EXTRA_KEY: {**ctx, "layers": str(layers)}})

    fits: dict[TokenPosition, PositionFit] = {}
    for position in TokenPosition:
        fit = fit_position(dataset.hidden_states[position], se, error, fixed, options)
        fits[position] = fit
        logger.info(
            "Measured one token position.",
            extra={
                LOG_EXTRA_KEY: {
                    **ctx,
                    "position": str(position),
                    "window": list(fit.window),
                    "nested_cv_auroc_vs_binarized_se": fit.nested.auroc_vs_binarized_se.mean,
                }
            },
        )

    served = fits[SEP_SERVED_TOKEN_POSITION]
    probe = fit_window_probe(
        dataset.hidden_states[SEP_SERVED_TOKEN_POSITION], se, served.window, options=options
    )
    threshold = recommend_threshold(served.nested.oof_p_high_se, error)
    config = build_probe_config(
        dataset, se, probe, served.nested.layer_selection, options, threshold, repo, created_at
    )
    write_probe_dir(
        directory,
        probe,
        config,
        build_metrics(fits, probe, threshold, backbone.num_hidden_layers),
        SEPTrainIds(
            model_id=backbone.model_id,
            calibration_set=dataset.config.calibration_set,
            question_ids=tuple(row.id for row in dataset.rows),
        ),
        dataset_dir,
    )
    logger.info(
        "Saved SEP probe.",
        extra={
            LOG_EXTRA_KEY: {
                **ctx,
                "output_dir": str(directory),
                "layers": list(probe.window),
                "gamma_nats": probe.gamma,
                "recommended_threshold": threshold.recommended,
            }
        },
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a SEP probe from a collected dataset.")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory (collect_sep_dataset)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Empty or absent directory; default: {SEP_PROBE_DIR}/<model slug>/.",
    )
    parser.add_argument(
        SEP_LAYERS_FLAG,
        type=_parse_layers_arg,
        default=SEPLayersMode.AUTO,
        help=f"{SEPLayersMode.AUTO} (nested-CV window), {SEPLayersMode.FINAL} (layer L), "
        "or <a>-<b>.",
    )
    parser.add_argument(
        MAX_ITER_FLAG,
        type=int,
        choices=_PRE_REGISTERED_MAX_ITERS,
        default=SEP_LOGREG_MAX_ITER,
        help="lbfgs iterations; raise only as the pre-registered fallback (deviation D10).",
    )
    parser.add_argument(
        STANDARDIZE_FLAG,
        action="store_true",
        help="Fold a standard scaler into the weights; fallback only (deviation D12).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train_sep_probe(
        args.dataset,
        args.output_dir,
        layers=args.layers,
        max_iter=args.max_iter,
        standardize=args.standardize,
        repo=read_repo_state(),
        created_at=datetime.now(UTC),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
