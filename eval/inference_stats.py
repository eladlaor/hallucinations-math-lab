"""Uncertainty for the success-criteria numbers: AUROC, bootstrap CIs, McNemar.

A point estimate compared to a fixed bar can flip on sampling noise, so every
criterion is reported with a 95% interval, and the comparative criteria (SE
vs PE, judge vs greedy) also require the improvement to be significant.

Bootstrap CIs are percentile intervals over row resamples. "Paired" means
both quantities being compared are recomputed on the *same* resample, which
keeps their correlation (the same rows are easy or hard for both detectors)
and gives a much tighter, honest interval for the difference than two
independent intervals would.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.stats import binomtest, rankdata

from hallucinations_math_lab.constants import RANDOM_SEED

N_BOOTSTRAP: Final[int] = 2000
CI_LEVEL: Final[float] = 0.95
# McNemar's test is two-sided: it asks whether greedy and final answers differ
# in accuracy at all, not presupposing the direction.
MCNEMAR_ALTERNATIVE: Final[str] = "two-sided"
MCNEMAR_NULL_PROPORTION: Final[float] = 0.5


@dataclass(frozen=True)
class BootstrapCI:
    low: float
    high: float
    n_resamples: int
    # Resamples where the statistic is undefined (e.g. only one class drawn).
    n_degenerate: int


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC via the Mann-Whitney rank formula, ties counted as half.

    Returns None when `labels` holds a single class (AUROC is undefined).
    Equals sklearn's roc_auc_score; written out so the bootstrap can call it
    thousands of times cheaply.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def paired_bootstrap(
    n_rows: int,
    statistic: Callable[[np.ndarray], float | None],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> BootstrapCI:
    """Percentile CI of `statistic(row_indices)` over resampled row indices.

    `statistic` receives an index array drawn with replacement and returns
    None when undefined on that resample; those draws are counted, not used.
    """
    if n_rows < 2:
        raise ValueError(f"Bootstrap needs at least 2 rows, got {n_rows}.")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n_degenerate = 0
    for _ in range(n_bootstrap):
        value = statistic(rng.integers(0, n_rows, size=n_rows))
        if value is None:
            n_degenerate += 1
        else:
            values.append(value)
    if not values:
        raise ValueError("Every bootstrap resample was degenerate; the CI is undefined.")
    tail = (1.0 - CI_LEVEL) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return BootstrapCI(float(low), float(high), len(values), n_degenerate)


def mcnemar_exact_p(before_correct: np.ndarray, after_correct: np.ndarray) -> float:
    """Exact McNemar test on paired correctness (binomial test on discordant pairs).

    Only rows where the two answers disagree in correctness carry information:
    b = right before, wrong after; c = wrong before, right after. Under no
    difference, c ~ Binomial(b + c, 0.5). With no discordant pairs the two
    are indistinguishable and p = 1.
    """
    b = int(np.sum(before_correct & ~after_correct))
    c = int(np.sum(~before_correct & after_correct))
    if b + c == 0:
        return 1.0
    return float(
        binomtest(c, b + c, MCNEMAR_NULL_PROPORTION, alternative=MCNEMAR_ALTERNATIVE).pvalue
    )
