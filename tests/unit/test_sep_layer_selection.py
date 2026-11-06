"""SEP layer sweep, nested-CV window selection and the out-of-fold threshold.

Contract pinned here (SEP_CLASSIFIER.md, "How is the layer window chosen without
peeking at the test rows?", D3, D4, D5, D9, D13):
  - Layer k is the vector after k decoder layers; layer 0 (the embeddings) is never
    probed, so the sweep and every candidate window live in 1..L.
  - A window's features are its layers' vectors concatenated in increasing k.
  - The per-layer sweep reports 5-fold CV AUROC of single-layer probes against
    binarized SE and, when given, against model error.
  - Nested CV: an inner CV on the outer-training rows only picks the window by mean
    AUROC against binarized SE; ties go to the deeper window. No fit ever sees the
    outer-test rows. With fixed layers there is no inner selection.
  - γ is refitted on each fit's own training rows; fold assignment is stratified on
    SE-only labels and never reads features.
  - A fold without both classes raises with a "spread of easy and hard questions" hint.
  - The recommended threshold is Youden's J on the out-of-fold p(high SE) against
    model error, through calibrate_threshold.youden_threshold.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from eval import sep_layer_selection
from eval.calibrate_threshold import youden_threshold
from eval.sep_layer_selection import (
    candidate_windows,
    choose_window,
    fit_window_probe,
    layer_sweep,
    nested_cv,
    recommend_threshold,
    window_features,
)
from eval.sep_recipe import (
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
    SEPLayerSelection,
    SEPThresholdSource,
)

N_LAYERS = 10  # L: hidden_states has L + 1 entries, index 0 = embeddings.
SIGNAL_LAYER = 7
WIDTH = 4
# Hidden-state dimension that carries each row's index in recording tests.
ROW_ID_DIM = WIDTH - 1
SPREAD_HINT_PATTERN = "spread of easy and hard questions"

FloatArray = npt.NDArray[np.float64]


def _signal_data(
    seed: int = 0, n_rows: int = 200, n_layers: int = N_LAYERS
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """(hidden_states [rows, L+1, d], SE, error): SE and error driven by layer 7 only."""
    rng = np.random.default_rng(seed)
    hidden_states = rng.standard_normal((n_rows, n_layers + 1, WIDTH))
    latent = hidden_states[:, SIGNAL_LAYER, 0]
    se = np.log(10.0) * expit(2.0 * latent + rng.normal(scale=0.3, size=n_rows))
    error = (latent + rng.normal(scale=0.5, size=n_rows) > 0).astype(np.int64)
    return hidden_states, se, error


def _with_row_ids(hidden_states: FloatArray) -> FloatArray:
    """A copy whose ROW_ID_DIM holds the row index at every layer."""
    tagged = hidden_states.copy()
    tagged[:, :, ROW_ID_DIM] = np.arange(hidden_states.shape[0])[:, None]
    return tagged


@dataclass(frozen=True)
class _RecordedFit:
    rows: frozenset[int]
    labels: npt.NDArray[np.int64]
    row_order: tuple[int, ...]


@pytest.fixture
def recorded_fits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_RecordedFit]]:
    """Records every probe fit's training rows (read from ROW_ID_DIM) and labels.

    The row-id columns are zeroed before the real fit, so they carry no signal.
    """
    records: list[_RecordedFit] = []

    def recording_fit(
        features: npt.ArrayLike, labels: npt.ArrayLike, *, max_iter: int, standardize: bool
    ) -> LogisticProbeFit:
        matrix = np.array(features, dtype=np.float64)
        row_ids = matrix[:, ROW_ID_DIM].astype(int)
        records.append(
            _RecordedFit(
                rows=frozenset(row_ids.tolist()),
                labels=np.asarray(labels, dtype=np.int64),
                row_order=tuple(row_ids.tolist()),
            )
        )
        matrix[:, ROW_ID_DIM::WIDTH] = 0.0
        return fit_logistic(matrix, labels, max_iter=max_iter, standardize=standardize)

    monkeypatch.setattr(sep_layer_selection, "fit_logistic", recording_fit)
    yield records


def test_single_layer_sweep_peaks_at_signal_layer():
    hidden_states, se, error = _signal_data()
    sweep = layer_sweep(hidden_states, se, error)

    assert sorted(sweep) == list(range(1, N_LAYERS + 1))
    by_se = {k: entry.auroc_vs_binarized_se.mean for k, entry in sweep.items()}
    assert max(by_se, key=by_se.__getitem__) == SIGNAL_LAYER
    assert by_se[SIGNAL_LAYER] > 0.9
    errors = {k: entry.auroc_vs_error for k, entry in sweep.items()}
    assert all(summary is not None for summary in errors.values())
    signal_error = errors[SIGNAL_LAYER]
    assert signal_error is not None and signal_error.mean > 0.75
    for entry in sweep.values():
        assert len(entry.auroc_vs_binarized_se.per_fold) == SEP_CV_FOLDS
        assert entry.auroc_vs_binarized_se.std >= 0.0


def test_sweep_without_error_labels_reports_no_error_auroc():
    hidden_states, se, _ = _signal_data()
    sweep = layer_sweep(hidden_states, se)
    assert all(entry.auroc_vs_error is None for entry in sweep.values())


def test_window_selection_contains_signal_layer():
    hidden_states, se, error = _signal_data()
    result = nested_cv(hidden_states, se, error)

    assert result.layer_selection == SEPLayerSelection.NESTED_CV
    assert len(result.window_per_fold) == SEP_CV_FOLDS
    assert all(SIGNAL_LAYER in window for window in result.window_per_fold)
    assert result.auroc_vs_binarized_se.mean > 0.85
    assert result.auroc_vs_error is not None and result.auroc_vs_error.mean > 0.7

    final = choose_window(hidden_states, se)
    assert SIGNAL_LAYER in final.window
    assert len(final.window) == SEP_LAYER_WINDOW
    assert final.inner_cv_auroc > 0.85


def test_nested_cv_never_fits_on_outer_test_rows(recorded_fits: list[_RecordedFit]):
    hidden_states, se, error = _signal_data()
    result = nested_cv(_with_row_ids(hidden_states), se, error)

    n_windows = len(candidate_windows(N_LAYERS, SEP_LAYER_WINDOW))
    fits_per_outer_fold = n_windows * SEP_CV_FOLDS + 1  # inner selection, then the outer fit
    assert len(recorded_fits) == SEP_CV_FOLDS * fits_per_outer_fold
    all_rows = frozenset(range(se.size))
    for fold, test_rows in enumerate(result.fold_test_rows):
        chunk = recorded_fits[fold * fits_per_outer_fold : (fold + 1) * fits_per_outer_fold]
        outer_test = frozenset(test_rows.tolist())
        assert all(fit.rows.isdisjoint(outer_test) for fit in chunk)
        # The outer fit uses exactly the outer-training rows.
        assert chunk[-1].rows == all_rows - outer_test
    covered = sorted(row for rows in result.fold_test_rows for row in rows.tolist())
    assert covered == list(range(se.size))


def test_gamma_fitted_on_train_fold_only(recorded_fits: list[_RecordedFit]):
    hidden_states, se, error = _signal_data()
    nested_cv(_with_row_ids(hidden_states), se, error)

    global_labels = binarize(se, best_split(se))
    differs_from_global = False
    for fit in recorded_fits:
        rows = np.array(fit.row_order)
        expected = binarize(se[rows], best_split(se[rows]))
        np.testing.assert_array_equal(fit.labels, expected)
        differs_from_global |= not np.array_equal(fit.labels, global_labels[rows])
    # Guard: the data must make a leaky all-rows γ distinguishable from per-fold γ.
    assert differs_from_global


def test_stratification_uses_no_features():
    hidden_states, se, _ = _signal_data(seed=0)
    other_states, _, _ = _signal_data(seed=1)
    fixed = (N_LAYERS,)
    first = nested_cv(hidden_states, se, fixed_layers=fixed)
    second = nested_cv(other_states, se, fixed_layers=fixed)

    reference = StratifiedKFold(SEP_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(
        np.zeros((se.size, 1)), binarize(se, best_split(se))
    )
    for a, b, (_, expected) in zip(
        first.fold_test_rows, second.fold_test_rows, reference, strict=True
    ):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, expected)


def test_single_class_test_fold_raises_with_hint():
    hidden_states, _, _ = _signal_data(n_rows=40)
    # Only 3 high-SE rows: fewer than the 5 folds, so some test fold has no high row.
    se = np.zeros(40)
    se[:3] = 1.0
    with pytest.raises(ValueError, match=rf"(?s)3 high.*37 low.*{SPREAD_HINT_PATTERN}"):
        nested_cv(hidden_states, se)
    with pytest.raises(ValueError, match=SPREAD_HINT_PATTERN):
        layer_sweep(hidden_states, se)


def test_test_fold_left_single_class_by_train_gamma_raises_with_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    hidden_states, se, _ = _signal_data()
    real_best_split = sep_layer_selection.best_split

    def gamma_above_most_rows(values: npt.ArrayLike) -> float:
        array = np.asarray(values, dtype=np.float64)
        if array.size == se.size:  # outer stratification: the real split
            return real_best_split(array)
        # A training-fold γ that leaves only that fold's top row high.
        top_two = np.sort(array)[-2:]
        return float(top_two.mean())

    monkeypatch.setattr(sep_layer_selection, "best_split", gamma_above_most_rows)
    with pytest.raises(ValueError, match=SPREAD_HINT_PATTERN):
        nested_cv(hidden_states, se, fixed_layers=(N_LAYERS,))


def test_error_fold_without_both_classes_raises_with_hint():
    hidden_states, se, _ = _signal_data()
    error = np.zeros(se.size, dtype=np.int64)
    error[0] = 1
    with pytest.raises(ValueError, match=rf"(?s)model error.*{SPREAD_HINT_PATTERN}"):
        nested_cv(hidden_states, se, error, fixed_layers=(N_LAYERS,))


def test_concat_order_increasing_layer():
    hidden_states = np.broadcast_to(
        np.arange(N_LAYERS + 1, dtype=np.float64)[None, :, None], (3, N_LAYERS + 1, WIDTH)
    )
    features = window_features(hidden_states, (3, 4, 5))
    assert features.shape == (3, 3 * WIDTH)
    np.testing.assert_array_equal(features[0], np.repeat([3.0, 4.0, 5.0], WIDTH))
    with pytest.raises(ValueError, match="increasing"):
        window_features(hidden_states, (5, 4, 3))
    with pytest.raises(ValueError, match="increasing"):
        window_features(hidden_states, (4, 4))


def test_layer_zero_excluded():
    hidden_states, se, _ = _signal_data()
    assert min(layer_sweep(hidden_states, se)) == 1
    windows = candidate_windows(N_LAYERS, SEP_LAYER_WINDOW)
    assert windows[0] == (1, 2, 3, 4, 5)
    assert windows[-1] == (6, 7, 8, 9, 10)
    assert all(min(window) >= 1 for window in windows)
    with pytest.raises(ValueError, match="1..10"):
        window_features(hidden_states, (0, 1))
    with pytest.raises(ValueError, match="1..10"):
        nested_cv(hidden_states, se, fixed_layers=(0,))
    with pytest.raises(ValueError, match="1..10"):
        window_features(hidden_states, (N_LAYERS + 1,))


@pytest.mark.parametrize("window", [(3.5,), (3, 4.2), (True, 2)])
def test_non_integer_layers_rejected(window: tuple[float | bool, ...]):
    hidden_states, se, _ = _signal_data()
    with pytest.raises(ValueError, match="integers"):
        window_features(hidden_states, window)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integers"):
        nested_cv(hidden_states, se, fixed_layers=window)  # type: ignore[arg-type]


def test_candidate_windows_need_enough_layers():
    with pytest.raises(ValueError, match="window"):
        candidate_windows(SEP_LAYER_WINDOW - 1, SEP_LAYER_WINDOW)


def test_ties_prefer_deeper_window():
    hidden_states, se, _ = _signal_data()
    # Every layer identical: every window has identical features and scores.
    tied = np.repeat(hidden_states[:, SIGNAL_LAYER : SIGNAL_LAYER + 1, :], N_LAYERS + 1, axis=1)
    deepest = candidate_windows(N_LAYERS, SEP_LAYER_WINDOW)[-1]

    assert choose_window(tied, se).window == deepest
    assert set(nested_cv(tied, se).window_per_fold) == {deepest}


def test_fixed_layers_skip_inner_selection(recorded_fits: list[_RecordedFit]):
    hidden_states, se, error = _signal_data()
    fixed = (SIGNAL_LAYER, SIGNAL_LAYER + 1)
    result = nested_cv(_with_row_ids(hidden_states), se, error, fixed_layers=fixed)

    assert result.layer_selection == SEPLayerSelection.FIXED
    assert len(recorded_fits) == SEP_CV_FOLDS  # one outer fit per fold, no inner fits
    assert result.window_per_fold == (fixed,) * SEP_CV_FOLDS


def test_pure_noise_nested_auroc_near_half(monkeypatch: pytest.MonkeyPatch):
    # 12 layers of 16 noise features: 8 candidate windows, enough room for selection bias.
    rng = np.random.default_rng(0)
    n_rows, n_layers, width = 200, 12, 16
    hidden_states = rng.standard_normal((n_rows, n_layers + 1, width))
    se = rng.uniform(0.0, np.log(10.0), n_rows)
    clean = nested_cv(hidden_states, se).auroc_vs_binarized_se.mean

    # The leak this test exists to catch: windows ranked on all rows, outer-test rows too.
    choose_on_train_rows = sep_layer_selection._choose_window
    all_rows = np.arange(n_rows)

    def choose_on_all_rows(
        states: FloatArray,
        se_values: FloatArray,
        rows: npt.NDArray[np.intp],
        candidates: tuple[tuple[int, ...], ...],
        cv_folds: int,
        options: sep_layer_selection.ProbeFitOptions,
    ) -> sep_layer_selection.WindowChoice:
        return choose_on_train_rows(states, se_values, all_rows, candidates, cv_folds, options)

    monkeypatch.setattr(sep_layer_selection, "_choose_window", choose_on_all_rows)
    leaky = nested_cv(hidden_states, se).auroc_vs_binarized_se.mean

    assert abs(clean - 0.5) < 0.05
    assert leaky - clean > 0.05


def test_oof_probabilities_cover_every_row_once():
    hidden_states, se, error = _signal_data()
    window = (SIGNAL_LAYER,)
    result = nested_cv(hidden_states, se, error, fixed_layers=window)
    assert result.oof_p_high_se.shape == se.shape
    covered = np.sort(np.concatenate(result.fold_test_rows))
    np.testing.assert_array_equal(covered, np.arange(se.size))
    for test_rows in result.fold_test_rows:
        train_rows = np.setdiff1d(np.arange(se.size), test_rows)
        outer_fit = fit_window_probe(hidden_states[train_rows], se[train_rows], window).fit
        expected = predict_p_high_se(outer_fit, window_features(hidden_states[test_rows], window))
        np.testing.assert_allclose(result.oof_p_high_se[test_rows], expected)


def test_recommended_threshold_is_youden_on_oof_probs():
    hidden_states, se, error = _signal_data()
    result = nested_cv(hidden_states, se, error, fixed_layers=(SIGNAL_LAYER,))
    recommendation = recommend_threshold(result.oof_p_high_se, error)

    tau, tpr, fpr = youden_threshold(error.tolist(), result.oof_p_high_se.tolist())
    assert (recommendation.recommended, recommendation.tpr, recommendation.fpr) == (tau, tpr, fpr)
    assert recommendation.escalation_rate == float(np.mean(result.oof_p_high_se > tau))
    assert recommendation.source == SEPThresholdSource.OOF_YOUDEN == "oof_youden"


def test_recommend_threshold_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="one entry per"):
        recommend_threshold(np.array([0.1, 0.9]), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="0/1"):
        recommend_threshold(np.array([0.1, 0.9]), np.array([0, 2]))


def test_fit_window_probe_uses_gamma_from_all_rows():
    hidden_states, se, _ = _signal_data()
    window = (SIGNAL_LAYER,)
    probe = fit_window_probe(hidden_states, se, window)

    assert probe.window == window
    assert probe.gamma == best_split(se)
    reference = fit_logistic(window_features(hidden_states, window), binarize(se, probe.gamma))
    np.testing.assert_array_equal(probe.fit.weight, reference.weight)


@pytest.mark.parametrize(
    ("shape", "match"),
    [((10, N_LAYERS + 1), "3-D"), ((10, 1, WIDTH), "layer")],
)
def test_rejects_malformed_hidden_states(shape: tuple[int, ...], match: str):
    se = np.linspace(0.0, 2.0, shape[0])
    with pytest.raises(ValueError, match=match):
        layer_sweep(np.zeros(shape), se)


def test_rejects_row_count_mismatch_and_bad_folds():
    hidden_states, se, _ = _signal_data()
    with pytest.raises(ValueError, match="rows"):
        layer_sweep(hidden_states, se[:-1])
    with pytest.raises(ValueError, match="cv_folds"):
        layer_sweep(hidden_states, se, cv_folds=1)
