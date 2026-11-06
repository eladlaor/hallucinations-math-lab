"""SEP training recipe: binarizing semantic entropy into high/low labels.

Contract pinned here (SEP_CLASSIFIER.md, "Where exactly is the high/low split?", D11):
  - γ* minimizes the total within-group squared error of SE over every midpoint
    between consecutive distinct SE values; equal errors go to the lower γ.
  - No SE value ever equals γ*, so every row gets a 0/1 label.
  - The label is 1 exactly when SE > γ*.
  - Fewer than two distinct SE values cannot be split and raise.
  - SE is rounded to SE_SPLIT_DECIMALS before searching and labeling, so values
    that differ only by the collector's summation order count as one level.
  - Distinct rounded SE values that are adjacent floats (no midpoint between them)
    raise.
"""

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest
import torch

from eval.sep_recipe import SE_SPLIT_DECIMALS, best_split, binarize, split_sse
from hallucinations_math_lab.constants import SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropy

LN_10 = math.log(10)


# Copied from the paper's official notebook, OATML/semantic-entropy-probes,
# semantic_entropy_probes/train-latent-probe.ipynb, cell 16. Changes: the function
# is renamed from `best_split` to `paper_best_split`; one mypy `type: ignore` is
# added; the trailing `plt.plot(...)` line is dropped (it draws a figure and does
# not affect the return value); whitespace follows ruff format.
# Best split for SE binarization.
def paper_best_split(entropy: torch.Tensor, label="Dx"):
    """
    Identify best split for minimizing reconstruction error via low and high SE mean estimates,
    as discussed in Section 4. Binarization of paper (ArXiv: 2406.15927)
    """
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split

        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])

        mse = np.sum((ents[low_idxs] - low_mean) ** 2) + np.sum((ents[high_idxs] - high_mean) ** 2)
        mse = np.sum(mse)

        split_mses.append(mse)

    split_mses = np.array(split_mses)  # type: ignore[assignment]

    return splits[np.argmin(split_mses)]


def _brute_force_sse(se: npt.NDArray[np.float64], gamma: float) -> float:
    low, high = se[se <= gamma], se[se > gamma]
    return float(((low - low.mean()) ** 2).sum() + ((high - high.mean()) ** 2).sum())


class _ExactMatchNLI:
    """Bidirectional entailment iff the two strings are equal."""

    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a == b


def _collector_se(cluster_sizes: tuple[int, ...]) -> float:
    """Discrete SE from the collector's own code path (`compute_semantic_entropy`),
    with clusters discovered in the given order."""
    answers = [f"cluster-{i}" for i, size in enumerate(cluster_sizes) for _ in range(size)]
    sample_set = SampleSet(prompt="q", samples=[Sample(text=a) for a in answers])
    result = compute_semantic_entropy(
        sample_set, scorer=_ExactMatchNLI(), estimator=SemanticEntropyEstimator.DISCRETE
    )
    return result.semantic_entropy


# SE values a discrete estimator with N=10 can produce. Each partition is summed in
# every cluster order, as real data arrives in discovery order: [7, 2, 1] gives two
# floats one ulp apart, and [4, 2, 1, 1, 1, 1] and [2] * 5 (both ln 5) differ by a
# few ulp. The last partition is "all 10 answers distinct", i.e. ln 10.
_PARTITIONS = [
    (10,),
    (9, 1),
    (8, 2),
    (7, 2, 1),
    (7, 3),
    (5, 5),
    (4, 3, 3),
    (4, 2, 1, 1, 1, 1),
    (2,) * 5,
    (2,) + (1,) * 8,
    (1,) * 10,
]
_RAW_SE_LEVELS = sorted(
    {_collector_se(order) for p in _PARTITIONS for order in set(itertools.permutations(p))}
)
DISCRETE_SE_LEVELS = np.array(_RAW_SE_LEVELS)


def test_best_split_minimizes_sse_over_all_midpoints():
    rng = np.random.default_rng(0)
    for _ in range(20):
        se = rng.choice(np.round(rng.uniform(0.0, 2.5, size=12), 2), size=40)
        distinct = np.unique(se)
        assert distinct.size >= 2
        midpoints = (distinct[:-1] + distinct[1:]) / 2
        brute = min(_brute_force_sse(se, float(m)) for m in midpoints)
        gamma = best_split(se)
        assert gamma in midpoints
        assert split_sse(se, gamma) == pytest.approx(brute, rel=1e-12, abs=1e-12)


def test_best_split_ties_go_to_the_lower_gamma():
    # Symmetric data: splitting after 0 or before 2 give the same error.
    se = np.array([0.0, 1.0, 2.0])
    assert best_split(se) == pytest.approx(0.5)


def test_best_split_never_ties_with_data():
    rng = np.random.default_rng(1)
    all_distinct = _collector_se((1,) * 10)
    assert all_distinct == pytest.approx(LN_10)
    se = np.concatenate([rng.choice(DISCRETE_SE_LEVELS, size=200), np.full(150, all_distinct)])
    gamma = best_split(se)
    assert not np.any(se == gamma)
    assert set(np.unique(binarize(se, gamma)).tolist()) == {0, 1}


def test_best_split_sse_not_worse_than_paper_grid():
    rng = np.random.default_rng(2)
    # Pre-rounded so both searches see the same values (best_split rounds internally).
    se = np.round(np.concatenate([[0.0], rng.uniform(0.0, 2.2, size=300)]), SE_SPLIT_DECIMALS)
    paper_gamma = float(paper_best_split(torch.from_numpy(se)))
    assert not np.any(se == paper_gamma), "fixture must avoid the paper's tie case"
    paper_low, paper_high = se[se < paper_gamma], se[se >= paper_gamma]
    paper_sse = float(
        ((paper_low - paper_low.mean()) ** 2).sum() + ((paper_high - paper_high.mean()) ** 2).sum()
    )
    assert split_sse(se, best_split(se)) <= paper_sse


def test_binarize_strictly_greater():
    se = np.array([0.0, 0.5, 0.5000001, 1.0])
    labels = binarize(se, 0.5)
    assert labels.tolist() == [0, 0, 1, 1]
    assert labels.dtype == np.int64


@pytest.mark.parametrize("se", [[], [0.7], [1.2, 1.2, 1.2]])
def test_best_split_raises_on_fewer_than_two_distinct_values(se: list[float]):
    with pytest.raises(ValueError, match="at least two distinct"):
        best_split(np.array(se, dtype=np.float64))


@pytest.mark.parametrize("bad", [[0.1, math.nan], [0.1, math.inf], [0.1, -0.2]])
def test_best_split_rejects_non_finite_or_negative_se(bad: list[float]):
    with pytest.raises(ValueError, match="finite and non-negative"):
        best_split(np.array(bad))


def test_best_split_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1-D"):
        best_split(np.zeros((2, 2)))


def test_fixture_holds_summation_order_noise():
    # Guards the fixture itself: without ulp-apart variants, the test below proves nothing.
    assert len(_RAW_SE_LEVELS) > len(np.unique(np.round(DISCRETE_SE_LEVELS, SE_SPLIT_DECIMALS)))


def test_best_split_merges_summation_order_variants_of_one_level():
    seven_two_one = sorted({_collector_se(o) for o in itertools.permutations((7, 2, 1))})
    assert len(seven_two_one) > 1
    se = np.array([0.0, 0.0, *seven_two_one, LN_10])
    gamma = best_split(se)
    labels = binarize(se, gamma)
    # Every variant of [7, 2, 1] lands on the same side of γ.
    assert len(set(labels[2:-1].tolist())) == 1


def test_best_split_raises_on_adjacent_floats_with_no_midpoint():
    # Rounding cannot separate adjacent floats this large (ulp > 10^-SE_SPLIT_DECIMALS).
    big = 1e8
    se = np.array([big, np.nextafter(big, 2 * big)])
    with pytest.raises(ValueError, match="adjacent floats"):
        best_split(se)


@pytest.mark.parametrize("gamma", [math.nan, math.inf, -0.1])
def test_binarize_and_split_sse_reject_invalid_gamma(gamma: float):
    se = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="γ must be finite"):
        binarize(se, gamma)
    with pytest.raises(ValueError, match="γ must be finite"):
        split_sse(se, gamma)


def test_split_sse_rejects_a_gamma_that_leaves_a_group_empty():
    with pytest.raises(ValueError, match="empty"):
        split_sse(np.array([0.0, 1.0]), 1.0)
