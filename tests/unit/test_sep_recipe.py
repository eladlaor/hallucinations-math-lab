"""SEP training recipe: binarizing semantic entropy, then fitting the logistic probe.

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

Logistic fit (SEP_CLASSIFIER.md, "Which classifier, and why no feature scaling?"):
  - The paper's bare LogisticRegression: L2, C=1, lbfgs, max_iter=100, no scaler.
  - Non-convergence raises and never returns a probe. The message names the next
    pre-registered fallback not yet taken (--max-iter 1000, then --standardize),
    or says both are exhausted.
  - The optional scaler is folded into the weights, so the probe stays one
    linear layer and reproduces StandardScaler + LogisticRegression.
  - A zero-variance column ends up with weight (near-)0 under the folded scaler:
    StandardScaler's mean carries rounding error, so the scaled column is not exact
    zeros but rounding residue proportional to the constant's size (~1e-17 to ~1e-13
    here), and nothing masks it.
  - The returned weight is read-only.
"""

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval import sep_recipe
from eval.sep_recipe import (
    MAX_ITER_FLAG,
    SE_SPLIT_DECIMALS,
    STANDARDIZE_FLAG,
    ProbeConvergenceError,
    best_split,
    binarize,
    fit_logistic,
    make_logistic_regression,
    predict_p_high_se,
    split_sse,
)
from hallucinations_math_lab.constants import (
    RANDOM_SEED,
    SEP_LOGREG_C,
    SEP_LOGREG_MAX_ITER,
    SemanticEntropyEstimator,
    SEPStandardization,
)
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


def _probe_data(
    seed: int, n_rows: int = 120, n_features: int = 6, *, wide_scales: bool = True
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Noisy labels driven by features 0 and 3.

    With `wide_scales`, columns span four orders of magnitude around large offsets
    (lbfgs then does not converge unscaled at the paper's max_iter=100); without it, every column is standard normal.
    """
    rng = np.random.default_rng(seed)
    scales = np.logspace(-2, 2, n_features) if wide_scales else np.ones(n_features)
    offsets = rng.uniform(-50.0, 50.0, size=n_features) if wide_scales else np.zeros(n_features)
    features = rng.normal(size=(n_rows, n_features)) * scales + offsets
    logits = (features[:, 0] - offsets[0]) / scales[0] - (features[:, 3] - offsets[3]) / scales[3]
    labels = (logits + rng.normal(scale=0.5, size=n_rows) > 0).astype(np.int64)
    return features, labels


def test_fit_logistic_uses_paper_defaults():
    params = make_logistic_regression().get_params()
    assert params["C"] == SEP_LOGREG_C == 1.0
    # Pure L2: l1_ratio=0 is sklearn >= 1.8's spelling of the paper's penalty="l2".
    assert params["l1_ratio"] == 0.0
    assert params["solver"] == "lbfgs"
    assert params["max_iter"] == SEP_LOGREG_MAX_ITER == 100
    assert params["random_state"] == RANDOM_SEED
    assert params["fit_intercept"] is True
    assert params["class_weight"] is None

    features, labels = _probe_data(seed=3, wide_scales=False)
    fit = fit_logistic(features, labels)
    reference = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=100).fit(
        features, labels
    )
    # No scaler: the weights are the bare estimator's, untouched.
    assert fit.standardization == SEPStandardization.NONE
    assert fit.max_iter == SEP_LOGREG_MAX_ITER
    np.testing.assert_array_equal(fit.weight, reference.coef_[0])
    assert fit.bias == reference.intercept_[0]
    assert fit.n_iter == reference.n_iter_[0]


def test_convergence_warning_raises_naming_max_iter_and_solver_status():
    features, labels = _probe_data(seed=4, wide_scales=False)
    # The solver's own status line is kept: lbfgs can also stop early (e.g. line search).
    with pytest.raises(
        ProbeConvergenceError, match=rf"(?s)max_iter=1\b.*status=\d+.*{MAX_ITER_FLAG}"
    ):
        fit_logistic(features, labels, max_iter=1)


@pytest.mark.parametrize(
    ("fallback_max_iter", "standardize", "named_flags"),
    [
        (1000, False, (MAX_ITER_FLAG, STANDARDIZE_FLAG)),
        (1000, True, (MAX_ITER_FLAG,)),
        (1, False, (STANDARDIZE_FLAG,)),
        (1, True, ()),
    ],
)
def test_convergence_error_advises_only_the_next_untaken_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fallback_max_iter: int,
    standardize: bool,
    named_flags: tuple[str, ...],
):
    # Lowering the fallback max_iter to 1 makes a max_iter=1 fit count as "D10 already taken".
    monkeypatch.setattr(sep_recipe, "SEP_LOGREG_FALLBACK_MAX_ITER", fallback_max_iter)
    features, labels = _probe_data(seed=4, wide_scales=False)
    with pytest.raises(ProbeConvergenceError) as excinfo:
        fit_logistic(features, labels, max_iter=1, standardize=standardize)
    message = str(excinfo.value)
    for flag in (MAX_ITER_FLAG, STANDARDIZE_FLAG):
        assert (flag in message) == (flag in named_flags), message
    assert ("already in effect" in message) == (not named_flags), message


def test_unscaled_wide_features_fail_at_default_max_iter_but_fit_standardized():
    # The pre-registered fallback in miniature: raw features on wildly different
    # scales stall lbfgs at the paper's max_iter, and the folded scaler fixes it.
    features, labels = _probe_data(seed=4)
    with pytest.raises(ProbeConvergenceError, match=MAX_ITER_FLAG):
        fit_logistic(features, labels)
    fit = fit_logistic(features, labels, standardize=True)
    assert fit.standardization == SEPStandardization.FOLDED_INTO_WEIGHTS
    assert fit.max_iter == SEP_LOGREG_MAX_ITER


def test_folded_standardize_matches_pipeline_probs():
    features, labels = _probe_data(seed=5)
    fit = fit_logistic(features, labels, standardize=True)
    pipeline = make_pipeline(StandardScaler(), make_logistic_regression()).fit(features, labels)
    assert fit.standardization == SEPStandardization.FOLDED_INTO_WEIGHTS
    unseen, _ = _probe_data(seed=6)
    for rows in (features, unseen):
        np.testing.assert_allclose(
            predict_p_high_se(fit, rows), pipeline.predict_proba(rows)[:, 1], rtol=0, atol=1e-6
        )


# A constant column's weight is rounding residue, not an exact zero (no mask is applied).
ZERO_VARIANCE_WEIGHT_ATOL = 1e-9


# Non-dyadic constants: their column mean is inexact, so the scaled column is not all zeros.
@pytest.mark.parametrize("constant", [0.1, 0.7, 1234.567])
def test_zero_variance_column_gets_near_zero_weight(constant: float):
    features, labels = _probe_data(seed=7)
    features[:, 2] = constant
    fit = fit_logistic(features, labels, standardize=True)
    assert abs(fit.weight[2]) < ZERO_VARIANCE_WEIGHT_ATOL
    assert np.all(fit.weight[[0, 1, 3, 4, 5]] != 0.0)


@pytest.mark.parametrize("standardize", [False, True])
def test_fit_weight_is_read_only(standardize: bool):
    features, labels = _probe_data(seed=11)
    fit = fit_logistic(features, labels, max_iter=1000, standardize=standardize)
    with pytest.raises(ValueError, match="read-only"):
        fit.weight[0] = 99.0


def test_fit_compares_by_identity_without_raising():
    features, labels = _probe_data(seed=3, wide_scales=False)
    first, second = fit_logistic(features, labels), fit_logistic(features, labels)
    # Identity equality: comparing or hashing a fit must not hit ndarray truthiness.
    assert first != second
    assert len({first, second, first}) == 2


def test_predict_p_high_se_is_sigmoid_of_the_linear_score():
    features, labels = _probe_data(seed=8, wide_scales=False)
    fit = fit_logistic(features, labels)
    reference = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=100).fit(
        features, labels
    )
    np.testing.assert_allclose(
        predict_p_high_se(fit, features), reference.predict_proba(features)[:, 1], atol=1e-12
    )


@pytest.mark.parametrize(
    ("features", "labels", "match"),
    [
        (np.zeros(4), np.array([0, 1, 0, 1]), "2-D"),
        (np.zeros((4, 2)), np.array([0, 1, 0]), "rows"),
        (np.array([[0.0], [np.nan], [1.0], [2.0]]), np.array([0, 1, 0, 1]), "finite"),
        (np.zeros((4, 1)), np.array([0, 2, 0, 1]), "0/1"),
        (np.zeros((4, 1)), np.array([1, 1, 1, 1]), "both classes"),
    ],
)
def test_fit_logistic_rejects_malformed_inputs(
    features: npt.NDArray[np.float64], labels: npt.NDArray[np.int64], match: str
):
    with pytest.raises(ValueError, match=match):
        fit_logistic(features, labels)


def test_fit_logistic_rejects_non_positive_max_iter():
    features, labels = _probe_data(seed=9)
    with pytest.raises(ValueError, match="max_iter"):
        fit_logistic(features, labels, max_iter=0)


def test_predict_p_high_se_rejects_wrong_feature_width():
    features, labels = _probe_data(seed=10, wide_scales=False)
    fit = fit_logistic(features, labels)
    with pytest.raises(ValueError, match="features"):
        predict_p_high_se(fit, features[:, :-1])
