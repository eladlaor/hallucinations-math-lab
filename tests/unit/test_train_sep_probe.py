"""The SEP trainer: from a dataset directory to a probe directory, CPU only.

Contract pinned here:
  - The probe directory holds exactly the five files of SEP_CLASSIFIER.md's
    "Artifact format": probe.safetensors, probe_config.json, train_ids.json,
    training_log.jsonl (the dataset's, copied) and metrics.json.
  - probe_config.json ties the probe to the dataset's backbone and prompt, and
    records its layers, position, γ*, classifier and active paper deviations.
  - The weights round-trip through safetensors alone, with identical scores.
  - A refit from the same dataset reproduces the weights.
  - The served probe, its window and its threshold come from the TBG states.
  - `--layers final` reads layer L only, with no inner selection.
  - The repo commit and dirty flag are recorded.

The dataset is synthetic: SE signal sits in a different layer at each position
(TBG deep, SLT shallow), far enough apart that no 5-layer window holds both.
"""

import json
import pickle
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from scipy.special import expit

from demo.model_registry import ProviderKind
from eval.repo_state import RepoState, read_repo_state
from eval.sep_dataset_io import (
    SEPDataset,
    SEPDatasetRow,
    TrainingLogEntry,
    append_training_log,
    load_sep_dataset,
    write_sep_dataset,
)
from eval.sep_layer_selection import (
    fit_window_probe,
    nested_cv,
    recommend_threshold,
    window_features,
)
from eval.sep_recipe import best_split
from eval.train_sep_probe import build_parser, parse_layers, train_sep_probe
from hallucinations_math_lab.constants import (
    RANDOM_SEED,
    SEP_CV_FOLDS,
    SEP_LAYERS_FLAG,
    SEP_LOGREG_C,
    SEP_LOGREG_MAX_ITER,
    SEP_LOGREG_PENALTY,
    SEP_LOGREG_SOLVER,
    SEP_METRICS_FILE,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    SemanticEntropyEstimator,
    SEPLayerSelection,
    SEPLayersMode,
    SEPPaperDeviation,
    SEPStandardization,
    SEPThresholdSource,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPBackbone,
    SEPCollectionConfig,
    SEPCollectionLabels,
    SEPProbeConfig,
    SEPProbeMetrics,
    SEPPromptSpec,
    SEPTrainIds,
)
from hallucinations_math_lab.services.sep_probe_weights import (
    ProbeWeights,
    read_probe_weights,
    write_probe_weights,
)

_NUM_LAYERS = 10
_HIDDEN = 2
_ROWS = 60
_SIGNAL_LAYER = {TokenPosition.TBG: 9, TokenPosition.SLT: 2}
_REPO = RepoState(commit="e" * 40, dirty=False)
_CREATED_AT = datetime(2026, 11, 7, tzinfo=UTC)
_PROBE_FILES = {
    SEP_PROBE_WEIGHTS_FILE,
    SEP_PROBE_CONFIG_FILE,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    SEP_METRICS_FILE,
}


def _collection_config(torch_cuda_version: str | None = "12.6") -> SEPCollectionConfig:
    return SEPCollectionConfig(
        calibration_set="calibration.jsonl",
        calibration_set_sha256="d" * 64,
        n_questions_requested=_ROWS,
        provider="hf_transformers",
        transformers_version="5.0.0",
        torch_version="2.14.1+cu126",
        torch_cuda_version=torch_cuda_version,
        backbone=SEPBackbone(
            model_id="org/model",
            revision="a" * 40,
            num_hidden_layers=_NUM_LAYERS,
            hidden_size=_HIDDEN,
            dtype="bfloat16",
        ),
        prompt=SEPPromptSpec(chat_template_sha256="b" * 64, answer_instruction="Be brief."),
        labels=SEPCollectionLabels(
            sample_n=2,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=64,
            nli_model_id="org/nli",
            nli_revision="c" * 40,
        ),
    )


def _write_dataset(directory: Path, torch_cuda_version: str | None = "12.6") -> SEPDataset:
    """A dataset whose SE is linearly readable from one layer per position and
    whose greedy answers are wrong mostly when SE is high."""
    rng = np.random.default_rng(0)
    se = rng.uniform(0.0, 2.0, size=_ROWS)
    correct = (se + rng.normal(scale=0.4, size=_ROWS)) < 1.0
    states = {}
    for position in TokenPosition:
        noise = rng.normal(size=(_ROWS, _NUM_LAYERS + 1, _HIDDEN))
        noise[:, _SIGNAL_LAYER[position], 0] += 3.0 * se
        states[position] = noise.astype(np.float32)
    rows = [
        SEPDatasetRow(
            id=f"q{i}",
            question=f"Question {i}?",
            ground_truth_aliases=["A"],
            sampled_answers=["A", "B"],
            cluster_ids=[0, 1],
            se_discrete=float(se[i]),
            se_likelihood_weighted=float(se[i]),
            n_clusters=2,
            greedy_answer="A",
            greedy_correct=bool(correct[i]),
        )
        for i in range(_ROWS)
    ]
    dataset = SEPDataset(
        config=_collection_config(torch_cuda_version), rows=rows, hidden_states=states
    )
    directory.mkdir(parents=True)
    for row in rows:
        append_training_log(
            directory,
            TrainingLogEntry(
                id=row.id,
                se_discrete=row.se_discrete,
                se_likelihood_weighted=row.se_likelihood_weighted,
                n_clusters=row.n_clusters,
                sampled_answer_word_counts=[1, 1],
                greedy_answer_word_count=1,
                greedy_correct=row.greedy_correct,
                elapsed_s=1.0,
            ),
        )
    write_sep_dataset(directory, dataset)
    return dataset


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("sep") / "dataset"
    _write_dataset(directory)
    return directory


@pytest.fixture(scope="module")
def probe_dir(dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("probe") / "probe"
    train_sep_probe(dataset_dir, output, repo=_REPO, created_at=_CREATED_AT)
    return output


def _config(probe_dir: Path) -> SEPProbeConfig:
    return SEPProbeConfig.model_validate_json((probe_dir / SEP_PROBE_CONFIG_FILE).read_text())


def test_train_writes_all_five_files(probe_dir: Path, dataset_dir: Path) -> None:
    assert {p.name for p in probe_dir.iterdir()} == _PROBE_FILES
    assert (probe_dir / SEP_TRAINING_LOG_FILE).read_text() == (
        dataset_dir / SEP_TRAINING_LOG_FILE
    ).read_text()
    train_ids = SEPTrainIds.model_validate_json((probe_dir / SEP_TRAIN_IDS_FILE).read_text())
    assert train_ids.model_id == "org/model"
    assert train_ids.calibration_set == "calibration.jsonl"
    assert train_ids.question_ids == tuple(f"q{i}" for i in range(_ROWS))


def test_config_holds_backbone_revision_layers_position_gamma_classifier_prompt(
    probe_dir: Path,
) -> None:
    config = _config(probe_dir)
    collected = _collection_config()
    assert config.backbone == collected.backbone
    assert config.prompt == collected.prompt
    assert config.features.token_position == TokenPosition.TBG
    assert _SIGNAL_LAYER[TokenPosition.TBG] in config.features.layers
    assert _SIGNAL_LAYER[TokenPosition.SLT] not in config.features.layers
    assert config.features.final_norm_applied == tuple(
        layer == _NUM_LAYERS for layer in config.features.layers
    )
    se = np.random.default_rng(0).uniform(0.0, 2.0, size=_ROWS)
    assert config.labels.gamma_nats == best_split(se)
    assert config.labels.n_high + config.labels.n_low == _ROWS
    assert config.labels.se_estimator == SemanticEntropyEstimator.DISCRETE
    assert config.labels.nli_revision == collected.labels.nli_revision
    assert config.classifier.C == SEP_LOGREG_C
    assert config.classifier.penalty == SEP_LOGREG_PENALTY
    assert config.classifier.solver == SEP_LOGREG_SOLVER
    assert config.classifier.max_iter == SEP_LOGREG_MAX_ITER
    assert config.classifier.standardization == SEPStandardization.NONE
    assert config.classifier.layer_selection == SEPLayerSelection.NESTED_CV
    assert config.classifier.cv_folds == SEP_CV_FOLDS
    assert config.classifier.seed == RANDOM_SEED
    assert config.training.provider == ProviderKind.HF_TRANSFORMERS
    assert config.training.transformers_version == "5.0.0"
    assert config.training.torch_version == "2.14.1+cu126"
    assert config.training.torch_cuda_version == "12.6"
    assert config.training.n_rows == _ROWS
    assert config.training.calibration_set_sha256 == "d" * 64
    assert config.training.created_at == _CREATED_AT
    metrics = SEPProbeMetrics.model_validate_json((probe_dir / SEP_METRICS_FILE).read_text())
    assert config.output.recommended_threshold == metrics.threshold.recommended


def test_served_threshold_comes_from_tbg_out_of_fold_scores(
    probe_dir: Path, dataset_dir: Path
) -> None:
    dataset = load_sep_dataset(dataset_dir)
    se = np.array([row.se_discrete for row in dataset.rows])
    error = np.array([0 if row.greedy_correct else 1 for row in dataset.rows])
    nested = nested_cv(dataset.hidden_states[TokenPosition.TBG], se, error)
    expected = recommend_threshold(nested.oof_p_high_se, error).recommended
    assert _config(probe_dir).output.recommended_threshold == expected


def test_probe_config_lists_deviations(probe_dir: Path) -> None:
    deviations = _config(probe_dir).deviations_from_paper
    # Paper settings, nested-CV selection: every deviation but the two fallbacks.
    assert set(deviations) == set(SEPPaperDeviation) - {
        SEPPaperDeviation.FALLBACK_MAX_ITER,
        SEPPaperDeviation.FOLDED_STANDARDIZATION,
    }
    assert list(deviations) == sorted(deviations, key=list(SEPPaperDeviation).index)


def test_fallbacks_are_recorded_as_deviations(dataset_dir: Path, tmp_path: Path) -> None:
    train_sep_probe(
        dataset_dir,
        tmp_path / "probe",
        layers=SEPLayersMode.FINAL,
        max_iter=1000,
        standardize=True,
        repo=_REPO,
        created_at=_CREATED_AT,
    )
    config = _config(tmp_path / "probe")
    assert config.classifier.max_iter == 1000
    assert config.classifier.standardization == SEPStandardization.FOLDED_INTO_WEIGHTS
    assert SEPPaperDeviation.FALLBACK_MAX_ITER in config.deviations_from_paper
    assert SEPPaperDeviation.FOLDED_STANDARDIZATION in config.deviations_from_paper
    # No window was chosen, so no window-selection deviation applies.
    assert SEPPaperDeviation.WINDOW_BY_INNER_CV not in config.deviations_from_paper
    assert SEPPaperDeviation.WINDOW_RANKED_BY_BINARIZED_SE not in config.deviations_from_paper


@pytest.mark.parametrize("max_iter", [50, 0, -5])
def test_max_iter_other_than_the_pre_registered_ones_is_refused(
    max_iter: int, dataset_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "x", "--max-iter", str(max_iter)])
    with pytest.raises(ValueError, match="pre-registered"):
        train_sep_probe(
            dataset_dir,
            tmp_path / "probe",
            max_iter=max_iter,
            repo=_REPO,
            created_at=_CREATED_AT,
        )
    assert not (tmp_path / "probe").exists()


def test_safetensors_roundtrip_identical_scores(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    written = ProbeWeights(
        weight=rng.normal(size=8).astype(np.float32), bias=np.array([0.25], dtype=np.float32)
    )
    path = tmp_path / SEP_PROBE_WEIGHTS_FILE
    write_probe_weights(path, written)
    loaded = read_probe_weights(path)

    features = rng.normal(size=(5, 8)).astype(np.float32)
    np.testing.assert_array_equal(
        expit(features @ loaded.weight + loaded.bias),
        expit(features @ written.weight + written.bias),
    )


def test_trained_probe_scores_match_its_fit(probe_dir: Path, dataset_dir: Path) -> None:
    """The written weights are the final TBG fit at the config's window, cast to float32."""
    config = _config(probe_dir)
    weights = read_probe_weights(probe_dir / SEP_PROBE_WEIGHTS_FILE)
    assert weights.weight.shape == (len(config.features.layers) * _HIDDEN,)

    dataset = load_sep_dataset(dataset_dir)
    states = dataset.hidden_states[TokenPosition.TBG]
    se = np.array([row.se_discrete for row in dataset.rows])
    fit = fit_window_probe(states, se, config.features.layers).fit
    np.testing.assert_allclose(weights.weight, fit.weight.astype(np.float32))
    np.testing.assert_allclose(weights.bias, np.array([fit.bias], dtype=np.float32))

    features = window_features(states, config.features.layers)
    scores = expit(features @ weights.weight + weights.bias)
    np.testing.assert_allclose(scores, expit(features @ fit.weight + fit.bias), rtol=1e-5)
    high = se > config.labels.gamma_nats
    assert scores[high].mean() > scores[~high].mean()


def test_probe_loader_uses_safetensors_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the probe loader must not unpickle")

    path = tmp_path / SEP_PROBE_WEIGHTS_FILE
    write_probe_weights(
        path, ProbeWeights(weight=np.ones(4, dtype=np.float32), bias=np.zeros(1, np.float32))
    )
    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(torch, "load", refuse)
    read_probe_weights(path)

    pickled = tmp_path / "pickled.safetensors"
    pickled.write_bytes(pickle.dumps({"weight": [1.0], "bias": [0.0]}))
    with pytest.raises(ValueError, match="Cannot read SEP probe weights"):
        read_probe_weights(pickled)


def test_probe_loader_rejects_a_foreign_header(tmp_path: Path) -> None:

    path = tmp_path / SEP_PROBE_WEIGHTS_FILE
    save_file(
        {"weight": np.ones(4, dtype=np.float32), "bias": np.zeros(1, dtype=np.float32)},
        path,
        metadata={"format": "sep-logistic", "schema_version": "2"},
    )
    with pytest.raises(ValueError, match="header"):
        read_probe_weights(path)


def test_refit_from_export_reproduces_weights(
    probe_dir: Path, dataset_dir: Path, tmp_path: Path
) -> None:
    train_sep_probe(dataset_dir, tmp_path / "refit", repo=_REPO, created_at=_CREATED_AT)
    first = read_probe_weights(probe_dir / SEP_PROBE_WEIGHTS_FILE)
    refit = read_probe_weights(tmp_path / "refit" / SEP_PROBE_WEIGHTS_FILE)
    np.testing.assert_allclose(refit.weight, first.weight)
    np.testing.assert_allclose(refit.bias, first.bias)
    assert _config(tmp_path / "refit") == _config(probe_dir)


def test_layers_final_gives_L_and_fixed_selection(dataset_dir: Path, tmp_path: Path) -> None:
    train_sep_probe(
        dataset_dir,
        tmp_path / "probe",
        layers=SEPLayersMode.FINAL,
        repo=_REPO,
        created_at=_CREATED_AT,
    )
    config = _config(tmp_path / "probe")
    assert config.features.layers == (_NUM_LAYERS,)
    assert config.features.final_norm_applied == (True,)
    assert config.classifier.layer_selection == SEPLayerSelection.FIXED
    metrics = SEPProbeMetrics.model_validate_json(
        (tmp_path / "probe" / SEP_METRICS_FILE).read_text()
    )
    for position in TokenPosition:
        selection = metrics.selection[position]
        assert selection.window == (_NUM_LAYERS,)
        assert selection.inner_cv_auroc is None
        assert selection.nested_cv_auroc_vs_binarized_se.window_per_fold == ((_NUM_LAYERS,),) * (
            SEP_CV_FOLDS
        )


def test_parse_layers_accepts_auto_final_and_a_range() -> None:
    assert parse_layers("auto") == SEPLayersMode.AUTO
    assert parse_layers("final") == SEPLayersMode.FINAL
    assert parse_layers("3-5") == (3, 4, 5)
    assert build_parser().parse_args(["--dataset", "d", SEP_LAYERS_FLAG, "2-2"]).layers == (2,)
    for bad in ("5-3", "x", "0-2", "1-"):
        with pytest.raises(ValueError, match=SEP_LAYERS_FLAG):
            parse_layers(bad)


def test_cli_reports_why_a_layer_range_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "d", SEP_LAYERS_FLAG, "0-2"])
    assert "layer 0, the embeddings, is never probed" in capsys.readouterr().err


def test_layer_range_outside_the_backbone_raises_before_any_fit(
    dataset_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "probe"
    with pytest.raises(ValueError, match=rf"{SEP_LAYERS_FLAG} 9-11.*<= {_NUM_LAYERS}"):
        train_sep_probe(dataset_dir, output, layers=(9, 10, 11), repo=_REPO, created_at=_CREATED_AT)
    assert not output.exists()


def test_refuses_a_non_empty_output_dir(dataset_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "probe"
    output.mkdir()
    (output / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        train_sep_probe(dataset_dir, output, repo=_REPO, created_at=_CREATED_AT)


def test_records_repo_commit_and_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    assert read_repo_state(repo) == RepoState(commit=head, dirty=False)
    (repo / "a.txt").write_text("changed")
    assert read_repo_state(repo) == RepoState(commit=head, dirty=True)


def test_read_repo_state_outside_a_repo_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="git"):
        read_repo_state(tmp_path)


def test_config_records_the_given_repo_state(dataset_dir: Path, tmp_path: Path) -> None:
    dirty = RepoState(commit="f" * 40, dirty=True)
    train_sep_probe(
        dataset_dir,
        tmp_path / "probe",
        layers=SEPLayersMode.FINAL,
        repo=dirty,
        created_at=_CREATED_AT,
    )
    training = _config(tmp_path / "probe").training
    assert training.repo_commit == "f" * 40
    assert training.repo_dirty is True


def test_config_copies_a_cpu_only_torch_build(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "dataset", torch_cuda_version=None)
    train_sep_probe(
        tmp_path / "dataset",
        tmp_path / "probe",
        layers=SEPLayersMode.FINAL,
        repo=_REPO,
        created_at=_CREATED_AT,
    )
    training = _config(tmp_path / "probe").training
    assert training.torch_version == "2.14.1+cu126"
    assert training.torch_cuda_version is None


def test_metrics_json_matches_schema(probe_dir: Path) -> None:
    raw = json.loads((probe_dir / SEP_METRICS_FILE).read_text())
    assert set(raw) == {
        "layer_sweep",
        "final_layer_auroc",
        "selection",
        "threshold",
        "converged",
        "n_iter",
    }
    metrics = SEPProbeMetrics.model_validate(raw)
    positions = set(TokenPosition)
    assert set(metrics.layer_sweep) == positions
    assert set(metrics.final_layer_auroc) == positions
    assert set(metrics.selection) == positions
    for position in TokenPosition:
        sweep = metrics.layer_sweep[position]
        assert sorted(sweep) == list(range(1, _NUM_LAYERS + 1))
        assert metrics.final_layer_auroc[position] == sweep[_NUM_LAYERS]
        assert set(raw["layer_sweep"][position]["1"]["auroc_vs_binarized_se"]) == {"mean", "std"}
        signal_layer = _SIGNAL_LAYER[position]
        assert sweep[signal_layer].auroc_vs_binarized_se.mean > 0.9
        assert sweep[signal_layer].auroc_vs_error is not None
        selection = metrics.selection[position]
        assert signal_layer in selection.window
        assert selection.inner_cv_auroc is not None
        nested = selection.nested_cv_auroc_vs_binarized_se
        assert len(nested.per_fold) == len(nested.window_per_fold) == SEP_CV_FOLDS
        assert selection.nested_cv_auroc_vs_error is not None
    assert metrics.threshold.source == SEPThresholdSource.OOF_YOUDEN
    assert 0.0 <= metrics.threshold.escalation_rate <= 1.0
    assert metrics.converged is True
    assert metrics.n_iter >= 1
