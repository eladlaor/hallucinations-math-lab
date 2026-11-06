"""The SEP training recipe as pure functions: no I/O, no model, no logging.

This module turns per-question semantic entropy (SE) into the binary target a
Semantic Entropy Probe (SEP) is trained on (Kossen et al. 2024, arXiv:2406.15927,
Sec. 4, "Binarization"), and fits the probe itself.

Binarization follows the paper's best-split rule: γ* minimizes the total
within-group squared error of SE, and a row is labeled high-SE (1) when
SE > γ*. The search differs from the paper's code on purpose (deviation D11 in
knowledge/plans/SEP_CLASSIFIER.md): instead of a 100-point grid that ends at
max(SE), every midpoint between consecutive distinct SE values is tried. No SE
value can then equal γ*, so every row gets a 0/1 label, and the search is exact.

SE is rounded to `SE_SPLIT_DECIMALS` before any comparison. The collector sums
cluster entropies in cluster-discovery order, so one partition (or two partitions
with the same true entropy) can yield SE values a few ulp apart. Without
rounding they would count as distinct levels that γ could fall between, and
values exactly 1 ulp apart would have no midpoint at all.

The probe is the paper notebook's bare `LogisticRegression()` on raw hidden
states (see `make_logistic_regression`). Non-convergence is an error, never a
silently shipped probe. The pre-registered fallbacks are a larger max_iter
(deviation D10), then a standard scaler folded into the linear weights (D12).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hallucinations_math_lab.constants import (
    RANDOM_SEED,
    SEP_LOGREG_C,
    SEP_LOGREG_FALLBACK_MAX_ITER,
    SEP_LOGREG_L1_RATIO,
    SEP_LOGREG_MAX_ITER,
    SEP_LOGREG_SOLVER,
    SEPStandardization,
)

MIN_DISTINCT_SE_VALUES: Final[int] = 2
# Far coarser than summation-order noise (~1e-16), far finer than the gap between
# distinct discrete SE levels: >= 2.9e-3 for N = 10 samples, >= 1e-4 pooled over N <= 10.
SE_SPLIT_DECIMALS: Final[int] = 10
HIGH_SE_LABEL: Final[int] = 1
LOW_SE_LABEL: Final[int] = 0
N_PROBE_CLASSES: Final[int] = 2
FEATURE_MATRIX_NDIM: Final[int] = 2
# The fit CLI's flags for the pre-registered convergence fallbacks (D10, then D12).
MAX_ITER_FLAG: Final[str] = "--max-iter"
STANDARDIZE_FLAG: Final[str] = "--standardize"

FloatArray = npt.NDArray[np.float64]
LabelArray = npt.NDArray[np.int64]


def _canonical_se(se: npt.ArrayLike) -> FloatArray:
    """`se` validated as 1-D, finite and non-negative, rounded to `SE_SPLIT_DECIMALS`."""
    values = np.asarray(se, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"SE must be a 1-D array; got shape {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"SE values must be finite and non-negative; got "
            f"{int((~np.isfinite(values)).sum())} non-finite and {int((values < 0).sum())} "
            "negative values."
        )
    return np.round(values, SE_SPLIT_DECIMALS)


def _validated_gamma(gamma: float) -> float:
    """`gamma` if it is finite and non-negative, else a ValueError."""
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError(f"γ must be finite and non-negative; got {gamma}.")
    return float(gamma)


def split_sse(se: npt.ArrayLike, gamma: float) -> float:
    """Total within-group squared error of SE split into SE <= γ and SE > γ.

    SE is rounded as in `best_split` before splitting.
    """
    values = _canonical_se(se)
    gamma = _validated_gamma(gamma)
    low, high = values[values <= gamma], values[values > gamma]
    if low.size == 0 or high.size == 0:
        raise ValueError(
            f"γ={gamma} leaves a group empty: {low.size} rows at or below it, {high.size} above it."
        )
    return float(((low - low.mean()) ** 2).sum() + ((high - high.mean()) ** 2).sum())


def best_split(se: npt.ArrayLike) -> float:
    """γ* minimizing `split_sse` over midpoints of consecutive distinct SE values.

    Of equally good midpoints, the lowest wins. Raises when SE has fewer than
    two distinct values, since no split then exists.
    """
    values = _canonical_se(se)
    distinct = np.unique(values)
    if distinct.size < MIN_DISTINCT_SE_VALUES:
        raise ValueError(
            f"Best-split binarization needs at least two distinct SE values; got "
            f"{distinct.size} distinct among {values.size} rows ({distinct.tolist()})."
        )
    midpoints = (distinct[:-1] + distinct[1:]) / 2
    # Guard: two adjacent floats have no float strictly between them, so the
    # midpoint would equal a data value. Rounding rules this out for realistic
    # SE magnitudes; it can only fire for values far beyond any entropy of N samples.
    collides = (midpoints == distinct[:-1]) | (midpoints == distinct[1:])
    if np.any(collides):
        raise ValueError(
            f"SE values {distinct[:-1][collides].tolist()} and {distinct[1:][collides].tolist()} "
            "are adjacent floats with no midpoint between them even after rounding to "
            f"{SE_SPLIT_DECIMALS} decimals."
        )
    errors = np.array([split_sse(values, float(m)) for m in midpoints])
    # argmin returns the first minimum, i.e. the lowest γ among ties.
    return float(midpoints[int(np.argmin(errors))])


def binarize(se: npt.ArrayLike, gamma: float) -> LabelArray:
    """1 where SE > γ (high SE), else 0, comparing SE rounded as in `best_split`."""
    values = _canonical_se(se)
    gamma = _validated_gamma(gamma)
    return np.where(values > gamma, HIGH_SE_LABEL, LOW_SE_LABEL).astype(np.int64)


class ProbeConvergenceError(RuntimeError):
    """lbfgs reported non-convergence (any nonzero status); the probe must not be used."""


@dataclass(frozen=True, eq=False)
class LogisticProbeFit:
    """A fitted SEP probe as one linear layer: p(high SE) = sigmoid(x · weight + bias).

    `weight` is in raw-feature space even when a scaler was used (it is folded in),
    and read-only. Equality is identity: compare fields explicitly.
    """

    weight: FloatArray
    bias: float
    n_iter: int
    max_iter: int
    standardization: SEPStandardization


def make_logistic_regression(max_iter: int = SEP_LOGREG_MAX_ITER) -> LogisticRegression:
    """The paper's classifier: L2, C=1, lbfgs, seeded, no scaler.

    `max_iter` defaults to the paper's SEP_LOGREG_MAX_ITER (100); the D10 fallback passes 1000.
    """
    return LogisticRegression(
        C=SEP_LOGREG_C,
        l1_ratio=SEP_LOGREG_L1_RATIO,
        solver=SEP_LOGREG_SOLVER,
        max_iter=max_iter,
        random_state=RANDOM_SEED,
    )


def _validated_features(features: npt.ArrayLike) -> FloatArray:
    """`features` as a finite float64 [rows, F] matrix, else a ValueError."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != FEATURE_MATRIX_NDIM:
        raise ValueError(f"Probe features must be a 2-D [rows, F] array; got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"Probe features must be finite; got {int((~np.isfinite(matrix)).sum())} "
            "non-finite values."
        )
    return matrix


def _validated_labels(labels: npt.ArrayLike, n_rows: int) -> LabelArray:
    """`labels` as a 1-D 0/1 array with one entry per row and both classes present."""
    values = np.asarray(labels)
    if values.shape != (n_rows,):
        raise ValueError(
            f"Labels must be 1-D with one entry per feature row ({n_rows} rows); "
            f"got shape {values.shape}."
        )
    present = set(np.unique(values).tolist())
    if not present <= {LOW_SE_LABEL, HIGH_SE_LABEL}:
        raise ValueError(f"Labels must be 0/1; got values {sorted(present)}.")
    if len(present) < N_PROBE_CLASSES:
        raise ValueError(f"Labels must contain both classes; got only {sorted(present)}.")
    return values.astype(np.int64)


def _next_fallback_advice(max_iter: int, standardize: bool) -> str:
    """The next step of the pre-registered ladder (D10, then D12) not yet taken."""
    raise_max_iter = f"{MAX_ITER_FLAG} {SEP_LOGREG_FALLBACK_MAX_ITER} (deviation D10)"
    add_scaler = f"{STANDARDIZE_FLAG} (deviation D12)"
    if max_iter < SEP_LOGREG_FALLBACK_MAX_ITER and not standardize:
        return f"Refit with {raise_max_iter}; if that still fails, add {add_scaler}."
    if max_iter < SEP_LOGREG_FALLBACK_MAX_ITER:
        return f"Refit with {raise_max_iter}."
    if not standardize:
        return f"Refit adding {add_scaler}."
    return (
        "Both pre-registered fallbacks (deviations D10 and D12) are already in effect; "
        "this probe cannot be fitted under the pre-registered recipe."
    )


def _fit_converged(
    model: LogisticRegression, features: FloatArray, labels: LabelArray, *, standardize: bool
) -> LogisticRegression:
    """`model` fitted, with a ConvergenceWarning raised as ProbeConvergenceError."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            return model.fit(features, labels)
    except ConvergenceWarning as exc:
        raise ProbeConvergenceError(
            f"Logistic probe: lbfgs did not converge (max_iter={model.max_iter}, "
            f"{features.shape[0]} rows, {features.shape[1]} features, "
            f"standardize={standardize}). Solver report: {str(exc).strip()}"
            f"\n\nNext pre-registered fallback: {_next_fallback_advice(model.max_iter, standardize)}"
        ) from exc


def fit_logistic(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    max_iter: int = SEP_LOGREG_MAX_ITER,
    standardize: bool = False,
) -> LogisticProbeFit:
    """Fit the SEP probe on [rows, F] features and 0/1 high-SE labels.

    With `standardize`, a StandardScaler is fitted on these rows and folded into
    the weights (w' = w/σ, b' = b − Σ w·μ/σ). A column constant on these rows
    scales to (near-)zeros, so the L2 fit itself leaves its weight at (near-)0;
    no mask is applied.
    Raises ProbeConvergenceError when lbfgs reports non-convergence.
    """
    matrix = _validated_features(features)
    targets = _validated_labels(labels, matrix.shape[0])
    if max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter}.")
    model = make_logistic_regression(max_iter)
    if not standardize:
        model = _fit_converged(model, matrix, targets, standardize=False)
        weight, bias = model.coef_[0].astype(np.float64), float(model.intercept_[0])
        standardization = SEPStandardization.NONE
    else:
        scaler = StandardScaler().fit(matrix)
        model = _fit_converged(model, scaler.transform(matrix), targets, standardize=True)
        scaled_weight = model.coef_[0]
        weight = scaled_weight / scaler.scale_
        bias = float(model.intercept_[0] - np.sum(scaled_weight * scaler.mean_ / scaler.scale_))
        standardization = SEPStandardization.FOLDED_INTO_WEIGHTS
    weight.flags.writeable = False
    return LogisticProbeFit(
        weight=weight,
        bias=bias,
        n_iter=int(model.n_iter_[0]),
        max_iter=max_iter,
        standardization=standardization,
    )


def predict_p_high_se(fit: LogisticProbeFit, features: npt.ArrayLike) -> FloatArray:
    """p(high SE) per row: sigmoid(features · weight + bias)."""
    matrix = _validated_features(features)
    if matrix.shape[1] != fit.weight.shape[0]:
        raise ValueError(
            f"Probe expects {fit.weight.shape[0]} features per row; got {matrix.shape[1]}."
        )
    scores: FloatArray = expit(matrix @ fit.weight + fit.bias)
    return scores
