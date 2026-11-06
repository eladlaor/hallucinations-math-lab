"""The SEP training recipe as pure functions: no I/O, no model, no logging.

This module turns per-question semantic entropy (SE) into the binary target a
Semantic Entropy Probe (SEP) is trained on (Kossen et al. 2024, arXiv:2406.15927,
Sec. 4, "Binarization").

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
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

MIN_DISTINCT_SE_VALUES: Final[int] = 2
# Far coarser than summation-order noise (~1e-16), far finer than the gap between
# distinct discrete SE levels: >= 2.9e-3 for N = 10 samples, >= 1e-4 pooled over N <= 10.
SE_SPLIT_DECIMALS: Final[int] = 10
HIGH_SE_LABEL: Final[int] = 1
LOW_SE_LABEL: Final[int] = 0

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
