"""Which layers a SEP reads: per-layer sweep, nested-CV window selection, OOF threshold.

Pure functions over one token position's hidden states, shaped [rows, L+1, d]:
index k is the vector after k decoder layers, so index 0 is the embeddings and is
never probed (deviation D13). A window is a tuple of strictly increasing layers in
1..L; its features are those layers' vectors concatenated in that order.

Every fit goes through `sep_recipe`: γ* is refitted on that fit's own training rows
and the probe is the paper's logistic regression (deviation D3). Fold assignment is
stratified on labels from a γ fitted on the rows being split; it reads SE only,
never features. Windows are ranked by inner-CV AUROC against binarized SE
(deviation D5) on outer-training rows only, so the reported nested-CV AUROC carries
the cost of choosing (deviation D4). The recommended escalation threshold is
Youden's J on the out-of-fold p(high SE) against model error (deviation D9).

No I/O, no model, no logging. See knowledge/plans/SEP_CLASSIFIER.md, "How is the
layer window chosen without peeking at the test rows?".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from eval.calibrate_threshold import youden_threshold
from eval.sep_recipe import (
    HIGH_SE_LABEL,
    LOW_SE_LABEL,
    FloatArray,
    LabelArray,
    LogisticProbeFit,
    best_split,
    binarize,
    fit_logistic,
    predict_p_high_se,
)
from hallucinations_math_lab.constants import (
    RANDOM_SEED,
    SEP_CV_FOLDS,
    SEP_LAYER_WINDOW,
    SEP_LOGREG_MAX_ITER,
    SEPLayerSelection,
    SEPThresholdSource,
)

# Layer 0 is the token embeddings; at TBG it is the same vector for every prompt (D13).
FIRST_PROBED_LAYER: Final[int] = 1
HIDDEN_STATES_NDIM: Final[int] = 3
MIN_CV_FOLDS: Final[int] = 2
SPREAD_HINT: Final[str] = (
    "Folds need both classes: need a spread of easy and hard questions "
    "(more rows, or a calibration set mixing easy and hard questions)."
)
SE_TARGET: Final[str] = "binarized SE"
ERROR_TARGET: Final[str] = "model error"
# Model-error labels: 1 = the greedy answer was wrong, 0 = it was right.
ERROR_LABEL_VALUES: Final[frozenset[int]] = frozenset({0, 1})

IndexArray = npt.NDArray[np.intp]
Window = tuple[int, ...]
Fold = tuple[IndexArray, IndexArray]


@dataclass(frozen=True)
class ProbeFitOptions:
    """Settings applied to every probe fit: the paper's, or a pre-registered fallback."""

    max_iter: int = SEP_LOGREG_MAX_ITER
    standardize: bool = False


# The paper's settings: max_iter=100, no scaler.
PAPER_FIT_OPTIONS: Final[ProbeFitOptions] = ProbeFitOptions()


@dataclass(frozen=True)
class CVAuroc:
    """AUROC across CV test folds; `std` is the population std of `per_fold`."""

    mean: float
    std: float
    per_fold: tuple[float, ...]

    @classmethod
    def of(cls, per_fold: list[float]) -> CVAuroc:
        values = np.asarray(per_fold, dtype=np.float64)
        return cls(mean=float(values.mean()), std=float(values.std()), per_fold=tuple(per_fold))


@dataclass(frozen=True)
class LayerSweepEntry:
    """One single-layer probe's CV AUROC; `auroc_vs_error` is None without error labels."""

    auroc_vs_binarized_se: CVAuroc
    auroc_vs_error: CVAuroc | None


@dataclass(frozen=True)
class WindowChoice:
    """The window with the best mean inner-CV AUROC against binarized SE."""

    window: Window
    inner_cv_auroc: float


@dataclass(frozen=True, eq=False)
class NestedCVResult:
    """Outer-fold scores of the whole procedure (selection included).

    `fold_test_rows[i]` are the outer test rows of fold i, whose window was
    `window_per_fold[i]`. `oof_p_high_se` holds each row's p(high SE) from the
    one outer fit that did not train on it.
    """

    layer_selection: SEPLayerSelection
    auroc_vs_binarized_se: CVAuroc
    auroc_vs_error: CVAuroc | None
    window_per_fold: tuple[Window, ...]
    fold_test_rows: tuple[IndexArray, ...]
    oof_p_high_se: FloatArray


@dataclass(frozen=True, eq=False)
class WindowProbe:
    """A probe fitted on all given rows, with the γ* its labels came from."""

    window: Window
    gamma: float
    fit: LogisticProbeFit


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Youden's J threshold on p(high SE) for flagging model errors (`score > recommended`)."""

    recommended: float
    tpr: float
    fpr: float
    escalation_rate: float
    source: SEPThresholdSource = SEPThresholdSource.OOF_YOUDEN


@dataclass(frozen=True, eq=False)
class _FoldScore:
    p_high_se: FloatArray
    auroc_vs_binarized_se: float
    auroc_vs_error: float | None


def _validated_hidden_states(hidden_states: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """`hidden_states` as a finite float [rows, L+1, d] array with L >= 1."""
    states = np.asarray(hidden_states)
    if states.ndim != HIDDEN_STATES_NDIM:
        raise ValueError(
            f"Hidden states must be a 3-D [rows, L+1, d] array; got shape {states.shape}."
        )
    if states.shape[1] <= FIRST_PROBED_LAYER:
        raise ValueError(
            f"Hidden states need at least one decoder layer after the embeddings; got "
            f"{states.shape[1]} layer entries."
        )
    if not np.issubdtype(states.dtype, np.floating):
        raise ValueError(f"Hidden states must be floats; got dtype {states.dtype}.")
    if not np.all(np.isfinite(states)):
        raise ValueError(
            f"Hidden states must be finite; got {int((~np.isfinite(states)).sum())} "
            "non-finite values."
        )
    return states


def _validated_inputs(
    hidden_states: npt.ArrayLike, se: npt.ArrayLike
) -> tuple[npt.NDArray[np.floating], FloatArray]:
    """Hidden states and 1-D SE, aligned row for row. SE's own checks run in `best_split`."""
    se_values = np.asarray(se, dtype=np.float64)
    if se_values.ndim != 1:
        raise ValueError(f"SE must be a 1-D array; got shape {se_values.shape}.")
    states = _validated_hidden_states(hidden_states)
    if states.shape[0] != se_values.shape[0]:
        raise ValueError(
            f"Hidden states have {states.shape[0]} rows but SE has {se_values.shape[0]} rows; "
            "they must align row for row."
        )
    return states, se_values


def _num_layers(hidden_states: npt.NDArray[np.floating]) -> int:
    """L: the number of decoder layers (the array holds L+1 entries)."""
    return int(hidden_states.shape[1]) - 1


def _validated_window(window: Window, num_layers: int) -> Window:
    """`window` if it is non-empty, integral, strictly increasing and inside 1..L."""
    if not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k in window):
        raise ValueError(f"Window layers must be integers; got {list(window)}.")
    layers = tuple(int(k) for k in window)
    if not layers:
        raise ValueError("A layer window needs at least one layer.")
    if any(b <= a for a, b in itertools.pairwise(layers)):
        raise ValueError(f"Window layers must be strictly increasing; got {list(layers)}.")
    if layers[0] < FIRST_PROBED_LAYER or layers[-1] > num_layers:
        raise ValueError(
            f"Window layers must lie in {FIRST_PROBED_LAYER}..{num_layers} (layer 0, the "
            f"embeddings, is never probed); got {list(layers)}."
        )
    return layers


def _validated_cv_folds(cv_folds: int) -> int:
    if cv_folds < MIN_CV_FOLDS:
        raise ValueError(f"cv_folds must be at least {MIN_CV_FOLDS}; got {cv_folds}.")
    return cv_folds


def _validated_error(error: npt.ArrayLike, n_rows: int) -> LabelArray:
    """Model-error labels (1 = wrong answer) as a 1-D 0/1 array, one per row."""
    values = np.asarray(error)
    if values.shape != (n_rows,):
        raise ValueError(
            f"Error labels must be 1-D with one entry per row ({n_rows} rows); "
            f"got shape {values.shape}."
        )
    present = set(np.unique(values).tolist())
    if not present <= ERROR_LABEL_VALUES:
        raise ValueError(f"Error labels must be 0/1 (1 = wrong answer); got {sorted(present)}.")
    return values.astype(np.int64)


def _gather(
    hidden_states: npt.NDArray[np.floating], rows: IndexArray, window: Window
) -> FloatArray:
    """[len(rows), len(window) * d] features: the window's vectors concatenated in order."""
    picked = hidden_states[rows[:, None], np.asarray(window)[None, :], :]
    features: FloatArray = picked.reshape(rows.size, -1).astype(np.float64)
    return features


def window_features(hidden_states: npt.ArrayLike, window: Window) -> FloatArray:
    """Every row's features for `window`, layers concatenated in increasing k."""
    states = _validated_hidden_states(hidden_states)
    layers = _validated_window(window, _num_layers(states))
    return _gather(states, np.arange(states.shape[0]), layers)


def candidate_windows(num_layers: int, window_size: int = SEP_LAYER_WINDOW) -> tuple[Window, ...]:
    """Every window [k, k + size − 1] inside 1..L, shallowest first."""
    last_start = num_layers - window_size + 1
    if window_size < 1 or last_start < FIRST_PROBED_LAYER:
        raise ValueError(
            f"A {window_size}-layer window does not fit in layers "
            f"{FIRST_PROBED_LAYER}..{num_layers}."
        )
    return tuple(
        tuple(range(start, start + window_size))
        for start in range(FIRST_PROBED_LAYER, last_start + 1)
    )


def _stratified_folds(se: FloatArray, rows: IndexArray, cv_folds: int) -> list[Fold]:
    """(train, test) row indices for a CV over `rows`, stratified on SE-only labels.

    The labels come from a γ fitted on exactly these rows; features are never read.
    """
    labels = binarize(se[rows], best_split(se[rows]))
    n_high = int((labels == HIGH_SE_LABEL).sum())
    n_low = int((labels == LOW_SE_LABEL).sum())
    if min(n_high, n_low) < cv_folds:
        raise ValueError(
            f"Cannot split {rows.size} rows into {cv_folds} stratified folds: {n_high} high-SE "
            f"and {n_low} low-SE rows, and each class needs at least {cv_folds}. {SPREAD_HINT}"
        )
    splitter = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_SEED)
    return [
        (rows[train], rows[test])
        for train, test in splitter.split(np.zeros((rows.size, 1)), labels)
    ]


def _auroc(labels: LabelArray, scores: FloatArray, target: str) -> float:
    """AUROC of `scores` against 0/1 `labels`; raises with the hint on a single-class fold."""
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"A CV test fold of {labels.size} rows has {n_pos} positive and {n_neg} negative "
            f"{target} labels, so AUROC is undefined. {SPREAD_HINT}"
        )
    return float(roc_auc_score(labels, scores))


def _fit_rows(
    hidden_states: npt.NDArray[np.floating],
    se: FloatArray,
    rows: IndexArray,
    window: Window,
    options: ProbeFitOptions,
) -> WindowProbe:
    """A probe on `rows` only, labeled by a γ* fitted on those same rows."""
    gamma = best_split(se[rows])
    fit = fit_logistic(
        _gather(hidden_states, rows, window),
        binarize(se[rows], gamma),
        max_iter=options.max_iter,
        standardize=options.standardize,
    )
    return WindowProbe(window=window, gamma=gamma, fit=fit)


def _score_fold(
    hidden_states: npt.NDArray[np.floating],
    se: FloatArray,
    error: LabelArray | None,
    fold: Fold,
    window: Window,
    options: ProbeFitOptions,
) -> _FoldScore:
    """Fit on the fold's train rows; score its test rows against the train γ's labels."""
    train, test = fold
    probe = _fit_rows(hidden_states, se, train, window, options)
    p_high_se = predict_p_high_se(probe.fit, _gather(hidden_states, test, window))
    return _FoldScore(
        p_high_se=p_high_se,
        auroc_vs_binarized_se=_auroc(binarize(se[test], probe.gamma), p_high_se, SE_TARGET),
        auroc_vs_error=None if error is None else _auroc(error[test], p_high_se, ERROR_TARGET),
    )


def _choose_window(
    hidden_states: npt.NDArray[np.floating],
    se: FloatArray,
    rows: IndexArray,
    candidates: tuple[Window, ...],
    cv_folds: int,
    options: ProbeFitOptions,
) -> WindowChoice:
    """The candidate with the best inner-CV AUROC on `rows`; ties go to the deeper window."""
    folds = _stratified_folds(se, rows, cv_folds)
    best: WindowChoice | None = None
    for window in candidates:
        per_fold = [_score_fold(hidden_states, se, None, fold, window, options) for fold in folds]
        score = float(np.mean([fold.auroc_vs_binarized_se for fold in per_fold]))
        # Candidates run shallowest first, so >= hands a tie to the deeper window.
        if best is None or score >= best.inner_cv_auroc:
            best = WindowChoice(window=window, inner_cv_auroc=score)
    if best is None:
        raise ValueError("Window selection needs at least one candidate window.")
    return best


def layer_sweep(
    hidden_states: npt.ArrayLike,
    se: npt.ArrayLike,
    error: npt.ArrayLike | None = None,
    *,
    cv_folds: int = SEP_CV_FOLDS,
    options: ProbeFitOptions = PAPER_FIT_OPTIONS,
) -> dict[int, LayerSweepEntry]:
    """CV AUROC of a single-layer probe at every layer k in 1..L. Descriptive only.

    All layers share one stratified split of all rows. `error` (1 = wrong answer)
    adds AUROC against model error for the same out-of-fold probabilities.
    """
    states, se_values = _validated_inputs(hidden_states, se)
    errors = None if error is None else _validated_error(error, se_values.shape[0])
    folds = _stratified_folds(
        se_values, np.arange(se_values.shape[0]), _validated_cv_folds(cv_folds)
    )
    sweep: dict[int, LayerSweepEntry] = {}
    for layer in range(FIRST_PROBED_LAYER, _num_layers(states) + 1):
        scores = [_score_fold(states, se_values, errors, fold, (layer,), options) for fold in folds]
        sweep[layer] = LayerSweepEntry(
            auroc_vs_binarized_se=CVAuroc.of([s.auroc_vs_binarized_se for s in scores]),
            auroc_vs_error=_error_summary(scores),
        )
    return sweep


def _error_summary(scores: list[_FoldScore]) -> CVAuroc | None:
    """CV AUROC against model error, or None when no fold was scored against it."""
    per_fold = [s.auroc_vs_error for s in scores]
    if any(value is None for value in per_fold):
        return None
    return CVAuroc.of([value for value in per_fold if value is not None])


def nested_cv(
    hidden_states: npt.ArrayLike,
    se: npt.ArrayLike,
    error: npt.ArrayLike | None = None,
    *,
    fixed_layers: Window | None = None,
    window_size: int = SEP_LAYER_WINDOW,
    cv_folds: int = SEP_CV_FOLDS,
    options: ProbeFitOptions = PAPER_FIT_OPTIONS,
) -> NestedCVResult:
    """Outer-CV estimate of the whole procedure, window selection included.

    Without `fixed_layers`, each outer fold picks its window by an inner CV over its
    own training rows (`_choose_window`); with them, every fold uses those layers.
    """
    states, se_values = _validated_inputs(hidden_states, se)
    n_rows = se_values.shape[0]
    errors = None if error is None else _validated_error(error, n_rows)
    folds_n = _validated_cv_folds(cv_folds)
    num_layers = _num_layers(states)
    fixed = None if fixed_layers is None else _validated_window(fixed_layers, num_layers)
    candidates = candidate_windows(num_layers, window_size) if fixed is None else ()

    oof = np.full(n_rows, np.nan)
    windows: list[Window] = []
    scores: list[_FoldScore] = []
    outer = _stratified_folds(se_values, np.arange(n_rows), folds_n)
    for fold in outer:
        window = (
            fixed
            if fixed is not None
            else _choose_window(states, se_values, fold[0], candidates, folds_n, options).window
        )
        score = _score_fold(states, se_values, errors, fold, window, options)
        oof[fold[1]] = score.p_high_se
        windows.append(window)
        scores.append(score)
    return NestedCVResult(
        layer_selection=SEPLayerSelection.NESTED_CV if fixed is None else SEPLayerSelection.FIXED,
        auroc_vs_binarized_se=CVAuroc.of([s.auroc_vs_binarized_se for s in scores]),
        auroc_vs_error=_error_summary(scores),
        window_per_fold=tuple(windows),
        fold_test_rows=tuple(test for _, test in outer),
        oof_p_high_se=oof,
    )


def choose_window(
    hidden_states: npt.ArrayLike,
    se: npt.ArrayLike,
    *,
    window_size: int = SEP_LAYER_WINDOW,
    cv_folds: int = SEP_CV_FOLDS,
    options: ProbeFitOptions = PAPER_FIT_OPTIONS,
) -> WindowChoice:
    """The final probe's window: the inner selection procedure run on all rows."""
    states, se_values = _validated_inputs(hidden_states, se)
    return _choose_window(
        states,
        se_values,
        np.arange(se_values.shape[0]),
        candidate_windows(_num_layers(states), window_size),
        _validated_cv_folds(cv_folds),
        options,
    )


def fit_window_probe(
    hidden_states: npt.ArrayLike,
    se: npt.ArrayLike,
    window: Window,
    *,
    options: ProbeFitOptions = PAPER_FIT_OPTIONS,
) -> WindowProbe:
    """The final probe: γ* and the logistic fit on all rows, reading `window`."""
    states, se_values = _validated_inputs(hidden_states, se)
    layers = _validated_window(window, _num_layers(states))
    return _fit_rows(states, se_values, np.arange(se_values.shape[0]), layers, options)


def recommend_threshold(
    oof_p_high_se: npt.ArrayLike, error: npt.ArrayLike
) -> ThresholdRecommendation:
    """Youden's J τ on out-of-fold p(high SE) for flagging model errors (1 = wrong)."""
    scores = np.asarray(oof_p_high_se, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(
            f"Out-of-fold scores must be a finite 1-D array; got shape {scores.shape}."
        )
    errors = _validated_error(error, scores.shape[0])
    tau, tpr, fpr = youden_threshold(errors.tolist(), scores.tolist())
    return ThresholdRecommendation(
        recommended=tau,
        tpr=tpr,
        fpr=fpr,
        escalation_rate=float(np.mean(scores > tau)),
    )
