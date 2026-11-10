"""Publishing a SEP probe and its dataset to the Hugging Face Hub.

Contract pinned here:
  - The model card is rendered from the probe's config and metrics plus the
    held-out benchmark, purely; a golden file pins its text.
  - Nothing is published without a held-out benchmark that scored this exact
    probe: both its weights_sha256 and config_sha256 must match the files
    being published, its model must be the probe's backbone, and none of its
    scored question ids may be a training id.
  - The dataset uploaded beside the probe must be the one that trained it.
  - A probe fitted from uncommitted code, or recorded under an absolute local
    path, is refused; so is a malformed repo id or tag, before anything is written.
  - The card describes nested CV only for a probe whose layers were selected.
  - The token comes from `hf auth login`, never from the command line.
  - A real run creates both repos, uploads the probe folder with its card and the
    dataset folder with its own card (README.md), and tags both commits;
    a dry run uploads nothing, a tag either repo already has uploads nothing, and
    a failed model tag deletes the dataset tag it follows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from huggingface_hub import CommitInfo, CommitOperationAdd, GitRefInfo, GitRefs

from eval import publish_sep_probe
from eval.publish_sep_probe import (
    CARD_FILE,
    build_parser,
    check_target,
    load_heldout,
    main,
    stage_release,
)
from eval.sep_dataset_io import SEPDataset, SEPDatasetRow, write_sep_dataset
from eval.sep_model_card import (
    GEMMA_4_LICENSE_URL,
    REFIT_COMMAND,
    REPO_URL,
    PublishTarget,
    ReleaseCards,
    refit_command,
    render_dataset_card,
    render_model_card,
)
from hallucinations_math_lab.constants import (
    SEP_COLLECTION_CONFIG_FILE,
    SEP_DATASET_DIR_FILES,
    SEP_DATASET_ROWS_FILE,
    SEP_DATASET_TENSORS_FILE,
    SEP_METRICS_FILE,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_DIR_FILES,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    DetectorName,
    SEPLayerSelection,
    SEPStandardization,
    SEPThresholdSource,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPAurocSummary,
    SEPCollectionConfig,
    SEPCollectionLabels,
    SEPLayerSweepMetrics,
    SEPNestedCVAuroc,
    SEPProbeConfig,
    SEPProbeMetrics,
    SEPPublishedMetrics,
    SEPSelectionMetrics,
    SEPThresholdMetrics,
)
from tests.unit.sep_probe_fixtures import (
    PROBE_NUM_HIDDEN_LAYERS,
    probe_config,
    write_probe_dir,
    write_train_ids,
)

_GOLDEN_CARD = Path(__file__).parent / "fixtures" / "sep_model_card.golden.md"
_TARGET = PublishTarget(
    model_repo="owner/sep-probe-model", dataset_repo="owner/sep-calibration-model", tag="v1"
)
# The card states the backbone's license, so the release is for a backbone with one recorded.
_MODEL_ID = "google/gemma-4-E2B-it"
_WINDOW = (1, 2)
_N_ROWS = 10
_IDS = [f"q{i}" for i in range(_N_ROWS)]
_SAMPLE_N = 10
_HELDOUT_ROWS = 200


def _config(**training: Any) -> SEPProbeConfig:
    config = probe_config(layers=_WINDOW, model_id=_MODEL_ID)
    if not training:
        return config
    return config.model_copy(update={"training": config.training.model_copy(update=training)})


def _sweep(mean: float) -> SEPLayerSweepMetrics:
    return SEPLayerSweepMetrics(
        auroc_vs_binarized_se=SEPAurocSummary(mean=mean, std=0.05),
        auroc_vs_error=SEPAurocSummary(mean=mean - 0.1, std=0.06),
    )


def _metrics(window: tuple[int, ...] = _WINDOW) -> SEPProbeMetrics:
    sweep = {
        TokenPosition.TBG: {1: _sweep(0.71), 2: _sweep(0.75)},
        TokenPosition.SLT: {1: _sweep(0.68), 2: _sweep(0.66)},
    }
    return SEPProbeMetrics(
        layer_sweep=sweep,
        final_layer_auroc={position: layers[2] for position, layers in sweep.items()},
        selection={
            position: SEPSelectionMetrics(
                window=window,
                inner_cv_auroc=None,
                nested_cv_auroc_vs_binarized_se=SEPNestedCVAuroc(
                    mean=mean,
                    std=0.04,
                    per_fold=(mean - 0.02, mean + 0.02),
                    window_per_fold=(window, window),
                ),
                nested_cv_auroc_vs_error=None,
            )
            for position, mean in ((TokenPosition.TBG, 0.77), (TokenPosition.SLT, 0.69))
        },
        threshold=SEPThresholdMetrics(
            recommended=0.41234567891,  # unrounded on the card
            tpr=0.7,
            fpr=0.3,
            escalation_rate=0.45,
            source=SEPThresholdSource.OOF_YOUDEN,
        ),
        converged=True,
        n_iter=12,
    )


def _write_probe(directory: Path, config: SEPProbeConfig, metrics: SEPProbeMetrics) -> Path:
    write_probe_dir(directory, config)
    write_train_ids(directory, _IDS, model_id=config.backbone.model_id)
    (directory / SEP_TRAINING_LOG_FILE).write_text('{"id": "q0"}\n')
    (directory / SEP_METRICS_FILE).write_text(metrics.model_dump_json(indent=2) + "\n")
    return directory


def _write_dataset(directory: Path, config: SEPProbeConfig, ids: list[str] = _IDS) -> Path:
    labels = config.labels
    collection = SEPCollectionConfig(
        calibration_set="calib.jsonl",
        calibration_set_sha256=config.training.calibration_set_sha256,
        n_questions_requested=len(ids),
        provider=config.training.provider,
        transformers_version=config.training.transformers_version,
        torch_version=config.training.torch_version,
        torch_cuda_version=config.training.torch_cuda_version,
        backbone=config.backbone,
        prompt=config.prompt,
        labels=SEPCollectionLabels(
            sample_n=_SAMPLE_N,
            temperature=labels.temperature,
            top_p=labels.top_p,
            top_k=labels.top_k,
            max_tokens=labels.max_tokens,
            nli_model_id=labels.nli_model_id,
            nli_revision=labels.nli_revision,
        ),
    )
    rows = [
        SEPDatasetRow(
            id=qid,
            question=f"Question {qid}?",
            ground_truth_aliases=["A"],
            sampled_answers=["A"] * _SAMPLE_N,
            cluster_ids=[0] * _SAMPLE_N,
            se_discrete=0.0,
            se_likelihood_weighted=0.0,
            n_clusters=1,
            greedy_answer="A",
            greedy_correct=True,
        )
        for qid in ids
    ]
    shape = (len(ids), config.backbone.num_hidden_layers + 1, config.backbone.hidden_size)
    states = {position: np.zeros(shape, dtype=np.float32) for position in TokenPosition}
    directory.mkdir(parents=True)
    (directory / SEP_TRAINING_LOG_FILE).write_text('{"id": "q0"}\n')
    write_sep_dataset(directory, SEPDataset(config=collection, rows=rows, hidden_states=states))
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heldout(probe_dir: Path, **sep_probe: Any) -> dict[str, Any]:
    """compute_auroc --json-out's report for a run that scored the probe in `probe_dir`."""
    model_id = sep_probe.pop("model_id", _MODEL_ID)

    def detector(auroc: float) -> dict[str, Any]:
        return {
            "auroc": auroc,
            "ci": {
                "level": 0.95,
                "low": auroc - 0.05,
                "high": auroc + 0.05,
                "n_resamples": 2000,
                "n_degenerate_resamples": 0,
            },
        }

    return {
        "run_config": {
            "model_id": model_id,
            "sample_n": 10,
            "sample_temperature": 1.0,
            "max_tokens": 64,
            "answer_instruction": "Answer briefly.",
            "sep_probe": {
                "ref": f"hf://{_TARGET.model_repo}@{_TARGET.tag}",
                "revision": "f" * 40,
                "weights_sha256": _sha256(probe_dir / SEP_PROBE_WEIGHTS_FILE),
                "config_sha256": _sha256(probe_dir / SEP_PROBE_CONFIG_FILE),
                **sep_probe,
            },
        },
        "n_rows": _HELDOUT_ROWS,
        "n_hallucinations": 80,
        "greedy_answer_words": {"mean": 2.1},
        "detectors": {
            str(DetectorName.SEMANTIC_ENTROPY): detector(0.8),
            str(DetectorName.SEMANTIC_ENTROPY_PROBE): detector(0.74),
            str(DetectorName.PREDICTIVE_ENTROPY): detector(0.7),
            str(DetectorName.DISCRETE_SEMANTIC_ENTROPY): detector(0.78),
        },
        "criteria": {"se_auroc": {"value": 0.8, "met": True}},
        "question_ids": [f"b{i}" for i in range(_HELDOUT_ROWS)],
    }


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


class _Release:
    """The three inputs of a release, written under one temporary directory."""

    def __init__(
        self,
        root: Path,
        config: SEPProbeConfig | None = None,
        dataset_config: SEPProbeConfig | None = None,
    ) -> None:
        config = config or _config()
        self.probe_dir = _write_probe(root / "probe", config, _metrics())
        self.dataset_dir = _write_dataset(root / "dataset", dataset_config or config)
        self.heldout_path = _write_json(root / "heldout.json", _heldout(self.probe_dir))
        self.staging_dir = root / "staging"

    def stage(self) -> ReleaseCards:
        return stage_release(
            self.probe_dir, self.dataset_dir, self.heldout_path, _TARGET, self.staging_dir
        )

    def argv(self, *extra: str) -> list[str]:
        return [
            "--probe-dir",
            str(self.probe_dir),
            "--dataset",
            str(self.dataset_dir),
            "--heldout",
            str(self.heldout_path),
            "--repo",
            _TARGET.model_repo,
            "--dataset-repo",
            _TARGET.dataset_repo,
            "--tag",
            _TARGET.tag,
            *extra,
        ]


@pytest.fixture
def release(tmp_path: Path) -> _Release:
    return _Release(tmp_path)


def test_model_card_snapshot(release: _Release) -> None:
    raw, heldout = load_heldout(release.heldout_path)
    metrics = SEPPublishedMetrics(**_metrics().model_dump(), heldout_benchmark=raw)
    card = render_model_card(_config(), metrics, heldout, _TARGET)
    assert card == _GOLDEN_CARD.read_text()


@pytest.mark.parametrize(
    ("torch_version", "cuda_version", "shown"),
    [
        ("2.14.1+cu126", "12.6", "torch 2.14.1+cu126 (CUDA 12.6)"),
        ("2.14.1+cpu", None, "torch 2.14.1+cpu (CPU-only)"),
    ],
)
def test_model_card_names_the_torch_build(
    release: _Release, torch_version: str, cuda_version: str | None, shown: str
) -> None:
    raw, heldout = load_heldout(release.heldout_path)
    metrics = SEPPublishedMetrics(**_metrics().model_dump(), heldout_benchmark=raw)
    config = _config(torch_version=torch_version, torch_cuda_version=cuda_version)
    card = render_model_card(config, metrics, heldout, _TARGET)
    assert " ".join(card.split()).count(shown) == 1


def test_model_card_carries_the_backbone_license_and_links_the_code_license(
    release: _Release,
) -> None:
    raw, heldout = load_heldout(release.heldout_path)
    metrics = SEPPublishedMetrics(**_metrics().model_dump(), heldout_benchmark=raw)
    config = _config()
    card = render_model_card(config, metrics, heldout, _TARGET)
    _, front, _ = card.split("---\n", 2)
    assert "license: apache-2.0" in front.splitlines()
    assert f"license_link: {GEMMA_4_LICENSE_URL}" in front.splitlines()
    assert f"{REPO_URL}/blob/{config.training.repo_commit}/LICENSE" in card
    # Every file that carries TriviaQA ids is named as keeping TriviaQA's terms.
    _, license_section = card.split("## What license is it under?", 1)
    for name in (SEP_TRAIN_IDS_FILE, SEP_TRAINING_LOG_FILE, SEP_METRICS_FILE):
        assert f"`{name}`" in license_section.split("\n\n")[1]


def test_model_card_refuses_a_backbone_without_a_recorded_license(release: _Release) -> None:
    raw, heldout = load_heldout(release.heldout_path)
    metrics = SEPPublishedMetrics(**_metrics().model_dump(), heldout_benchmark=raw)
    with pytest.raises(ValueError, match="org/model"):
        render_model_card(probe_config(layers=_WINDOW), metrics, heldout, _TARGET)


def _classifier(config: SEPProbeConfig, **classifier: Any) -> SEPProbeConfig:
    return config.model_copy(update={"classifier": config.classifier.model_copy(update=classifier)})


@pytest.mark.parametrize(
    ("layers", "classifier", "options"),
    [
        (_WINDOW, {}, "--layers 1-2"),
        ((PROBE_NUM_HIDDEN_LAYERS,), {}, "--layers final"),
        (_WINDOW, {"layer_selection": SEPLayerSelection.NESTED_CV}, ""),
        (
            (PROBE_NUM_HIDDEN_LAYERS,),
            {"max_iter": 1000, "standardization": SEPStandardization.FOLDED_INTO_WEIGHTS},
            "--layers final --max-iter 1000 --standardize",
        ),
    ],
    ids=["fixed_range", "fixed_final", "nested_cv", "fallbacks"],
)
def test_card_refit_command_carries_the_probe_options(
    layers: tuple[int, ...], classifier: dict[str, Any], options: str
) -> None:
    config = _classifier(probe_config(layers=layers), **classifier)
    assert refit_command(config) == f"{REFIT_COMMAND} {options}".rstrip()


def test_card_refit_command_refuses_a_non_contiguous_window() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        refit_command(probe_config(layers=(1, 3), num_hidden_layers=3))


def test_stage_writes_the_probe_files_and_merged_metrics(release: _Release) -> None:
    cards = release.stage()
    staged = release.staging_dir
    # The cards are uploaded from memory, so the staged folder holds only the probe's files.
    assert {p.name for p in staged.iterdir()} == set(SEP_PROBE_DIR_FILES)
    assert cards.model == _GOLDEN_CARD.read_text()
    assert cards.dataset == render_dataset_card(_config(), _TARGET)
    for name in (SEP_PROBE_WEIGHTS_FILE, SEP_PROBE_CONFIG_FILE):
        assert (staged / name).read_bytes() == (release.probe_dir / name).read_bytes()
    published = SEPPublishedMetrics.model_validate_json((staged / SEP_METRICS_FILE).read_text())
    assert published.heldout_benchmark == json.loads(release.heldout_path.read_text())
    assert SEPProbeMetrics(**published.model_dump(exclude={"heldout_benchmark"})) == _metrics()


@pytest.mark.parametrize("name", [SEP_TRAIN_IDS_FILE, SEP_TRAINING_LOG_FILE])
def test_stage_refuses_a_file_changed_after_the_checks(
    release: _Release, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    copyfile = publish_sep_probe.shutil.copyfile

    def copy_then_change(src: Path, dst: Path) -> None:
        copyfile(src, dst)
        if Path(dst).name == name:
            Path(dst).write_text(Path(dst).read_text().replace("q0", "z0"))

    monkeypatch.setattr(publish_sep_probe.shutil, "copyfile", copy_then_change)
    with pytest.raises(RuntimeError, match=name):
        release.stage()


@pytest.mark.parametrize(
    "drop",
    [str(DetectorName.SEMANTIC_ENTROPY_PROBE), "sep_probe"],
    ids=["no_sep_auroc", "no_probe"],
)
def test_publish_refuses_without_heldout(release: _Release, drop: str) -> None:
    report = _heldout(release.probe_dir)
    if drop == "sep_probe":
        report["run_config"]["sep_probe"] = None
    else:
        del report["detectors"][drop]
    _write_json(release.heldout_path, report)
    with pytest.raises(ValueError, match="held-out"):
        release.stage()
    assert not release.staging_dir.exists()


@pytest.mark.parametrize("field", ["weights_sha256", "config_sha256"])
def test_heldout_refuses_probe_sha_mismatch(release: _Release, field: str) -> None:
    _write_json(release.heldout_path, _heldout(release.probe_dir, **{field: "0" * 64}))
    with pytest.raises(ValueError, match=field) as excinfo:
        release.stage()
    assert "0" * 64 in str(excinfo.value)
    assert not release.staging_dir.exists()


def test_heldout_refuses_absolute_probe_ref(release: _Release) -> None:
    _write_json(
        release.heldout_path, _heldout(release.probe_dir, ref=str(release.probe_dir.resolve()))
    )
    with pytest.raises(ValueError, match="absolute"):
        release.stage()


def test_publish_refuses_model_id_mismatch_in_heldout(release: _Release) -> None:
    _write_json(release.heldout_path, _heldout(release.probe_dir, model_id="org/other"))
    with pytest.raises(ValueError, match="org/other"):
        release.stage()


def test_publish_refuses_model_id_mismatch(tmp_path: Path) -> None:
    other = probe_config(layers=_WINDOW, model_id="org/other")
    release = _Release(tmp_path, dataset_config=other)
    with pytest.raises(ValueError, match="org/other"):
        release.stage()


def test_publish_refuses_dataset_that_did_not_train_the_probe(tmp_path: Path) -> None:
    release = _Release(tmp_path)
    write_train_ids(release.probe_dir, [f"x{i}" for i in range(_N_ROWS)])
    _write_json(release.heldout_path, _heldout(release.probe_dir))
    with pytest.raises(ValueError, match="question ids"):
        release.stage()


def test_publish_refuses_dataset_without_its_training_log(tmp_path: Path) -> None:
    release = _Release(tmp_path)
    (release.dataset_dir / SEP_TRAINING_LOG_FILE).unlink()
    with pytest.raises(ValueError, match=SEP_TRAINING_LOG_FILE):
        release.stage()


def test_publish_refuses_probe_log_that_is_not_the_dataset_log(tmp_path: Path) -> None:
    release = _Release(tmp_path)
    (release.probe_dir / SEP_TRAINING_LOG_FILE).write_text('{"id": "other"}\n')
    with pytest.raises(ValueError, match=SEP_TRAINING_LOG_FILE):
        release.stage()


def test_publish_refuses_metrics_for_another_window(tmp_path: Path) -> None:
    release = _Release(tmp_path)
    (release.probe_dir / SEP_METRICS_FILE).write_text(_metrics(window=(2,)).model_dump_json())
    with pytest.raises(ValueError, match="window"):
        release.stage()


def test_publish_refuses_repo_dirty(tmp_path: Path) -> None:
    release = _Release(tmp_path, config=_config(repo_dirty=True))
    with pytest.raises(ValueError, match="uncommitted"):
        release.stage()


def test_publish_has_no_token_arg() -> None:
    options = {option for action in build_parser()._actions for option in action.option_strings}
    assert not any("token" in option for option in options)


def _mock_hf_api(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock, dict[str, dict[str, bytes]]]:
    """Replace HfApi; returns the class, its instance, and per repo type the files uploaded
    (path in the repo -> content)."""
    uploaded: dict[str, dict[str, bytes]] = {}

    def create_commit(**kwargs: Any) -> CommitInfo:
        operations: list[CommitOperationAdd] = kwargs["operations"]
        assert all(isinstance(op, CommitOperationAdd) for op in operations)
        uploaded[kwargs["repo_type"]] = {
            op.path_in_repo: (
                op.path_or_fileobj
                if isinstance(op.path_or_fileobj, bytes)
                else Path(str(op.path_or_fileobj)).read_bytes()
            )
            for op in operations
        }
        return CommitInfo(
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/abc",
            commit_message="m",
            commit_description="",
            oid=f"oid-{kwargs['repo_type']}",
        )

    api = MagicMock()
    api.create_commit.side_effect = create_commit
    api.repo_exists.return_value = False
    api_class = MagicMock(return_value=api)
    monkeypatch.setattr(publish_sep_probe, "HfApi", api_class)
    return api_class, api, uploaded


def test_upload_calls_mocked_hfapi_with_both_repos_and_tag(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_class, api, uploaded = _mock_hf_api(monkeypatch)

    assert main(release.argv()) == 0

    api_class.assert_called_once_with()  # the token comes from `hf auth login`
    repos = {(_TARGET.model_repo, "model"), (_TARGET.dataset_repo, "dataset")}
    assert {
        (c.kwargs["repo_id"], c.kwargs["repo_type"]) for c in api.create_repo.call_args_list
    } == repos
    assert {
        (c.kwargs["repo_id"], c.kwargs["repo_type"]) for c in api.create_commit.call_args_list
    } == repos
    assert {
        (c.args[0], c.kwargs["repo_type"], c.kwargs["tag"], c.kwargs["revision"])
        for c in api.create_tag.call_args_list
    } == {
        (_TARGET.model_repo, "model", _TARGET.tag, "oid-model"),
        (_TARGET.dataset_repo, "dataset", _TARGET.tag, "oid-dataset"),
    }
    assert set(uploaded["model"]) == {*SEP_PROBE_DIR_FILES, CARD_FILE}
    assert set(uploaded["dataset"]) == {
        SEP_DATASET_ROWS_FILE,
        SEP_DATASET_TENSORS_FILE,
        SEP_COLLECTION_CONFIG_FILE,
        SEP_TRAINING_LOG_FILE,
        CARD_FILE,
    }
    assert uploaded["model"][CARD_FILE].decode() == _GOLDEN_CARD.read_text()
    assert uploaded["dataset"][CARD_FILE].decode() == render_dataset_card(_config(), _TARGET)
    for name in SEP_DATASET_DIR_FILES:
        assert uploaded["dataset"][name] == (release.dataset_dir / name).read_bytes()


def test_dry_run_never_uploads(
    release: _Release, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api_class, _, _ = _mock_hf_api(monkeypatch)
    assert main(release.argv("--dry-run")) == 0
    api_class.assert_not_called()
    out = capsys.readouterr().out
    assert f"hf://{_TARGET.model_repo}@{_TARGET.tag}" in out
    assert render_dataset_card(_config(), _TARGET) in out


def test_upload_failure_names_the_repo_and_tags_nothing(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)
    api.create_repo.side_effect = OSError("403 Forbidden")
    with pytest.raises(RuntimeError, match=_TARGET.dataset_repo):
        main(release.argv())
    api.create_tag.assert_not_called()


def test_dataset_upload_failure_leaves_the_model_repo_untouched(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)

    def create_repo(**kwargs: Any) -> None:
        if kwargs["repo_type"] == "dataset":
            raise OSError("403 Forbidden")

    api.create_repo.side_effect = create_repo
    with pytest.raises(RuntimeError, match=_TARGET.dataset_repo):
        main(release.argv())
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


@pytest.mark.parametrize("tagged_repo_type", ["model", "dataset"])
def test_existing_tag_is_refused_before_any_upload(
    release: _Release, monkeypatch: pytest.MonkeyPatch, tagged_repo_type: str
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)
    api.repo_exists.return_value = True

    def list_repo_refs(repo_id: str, *, repo_type: str) -> GitRefs:
        tags = (
            [GitRefInfo(name=_TARGET.tag, ref=f"refs/tags/{_TARGET.tag}", target_commit="c")]
            if repo_type == tagged_repo_type
            else []
        )
        return GitRefs(branches=[], converts=[], tags=tags)

    api.list_repo_refs.side_effect = list_repo_refs
    with pytest.raises(ValueError, match="already has tag"):
        main(release.argv())
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


def test_upload_commit_messages_name_the_tag_not_the_staging_dir(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)
    assert main(release.argv()) == 0
    messages = [c.kwargs["commit_message"] for c in api.create_commit.call_args_list]
    assert len(messages) == 2
    assert all(_TARGET.tag in m and "tmp" not in m for m in messages)


def test_heldout_with_mixed_ci_levels_is_refused(tmp_path: Path) -> None:
    release = _Release(tmp_path)
    raw = json.loads(release.heldout_path.read_text())
    raw["detectors"][str(DetectorName.SEMANTIC_ENTROPY)]["ci"]["level"] = 0.9
    _write_json(release.heldout_path, raw)
    with pytest.raises(ValueError, match="different levels"):
        load_heldout(release.heldout_path)


@pytest.mark.parametrize(
    ("selection", "estimate", "column"),
    [
        (
            SEPLayerSelection.NESTED_CV,
            "Nested cross-validation (5 folds in both the outer and the inner loop)",
            "| Nested-CV AUROC vs binarized SE |",
        ),
        (
            SEPLayerSelection.FIXED,
            "Cross-validation (5 folds), with the layers fixed in advance",
            "| CV AUROC vs binarized SE |",
        ),
    ],
    ids=["nested_cv", "fixed"],
)
def test_card_describes_the_cross_validation_that_ran(
    release: _Release, selection: SEPLayerSelection, estimate: str, column: str
) -> None:
    raw, heldout = load_heldout(release.heldout_path)
    metrics = SEPPublishedMetrics(**_metrics().model_dump(), heldout_benchmark=raw)
    card = render_model_card(
        _classifier(_config(), layer_selection=selection), metrics, heldout, _TARGET
    )
    assert estimate in card
    assert column in card
    assert ("inner loop" in card) == (selection == SEPLayerSelection.NESTED_CV)


def test_heldout_refuses_a_training_question(release: _Release) -> None:
    report = _heldout(release.probe_dir)
    report["question_ids"][-1] = _IDS[3]
    _write_json(release.heldout_path, report)
    with pytest.raises(ValueError, match=r"1 held-out benchmark questions trained.*'q3'"):
        release.stage()
    assert not release.staging_dir.exists()


@pytest.mark.parametrize("ids", [None, ["b0"]], ids=["missing", "fewer_than_rows"])
def test_heldout_needs_one_question_id_per_scored_row(
    release: _Release, ids: list[str] | None
) -> None:
    report = _heldout(release.probe_dir)
    if ids is None:
        del report["question_ids"]
    else:
        report["question_ids"] = ids
    _write_json(release.heldout_path, report)
    with pytest.raises(ValueError, match="question_ids|question ids"):
        load_heldout(release.heldout_path)


@pytest.mark.parametrize(
    ("flag", "target"),
    [
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "")),
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "v 1")),
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "v1..2")),
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "v1~")),
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "v1@x")),
        ("--tag", PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, "v1.lock")),
        ("--repo", PublishTarget("owner", _TARGET.dataset_repo, _TARGET.tag)),
        ("--repo", PublishTarget("a/b/c", _TARGET.dataset_repo, _TARGET.tag)),
        ("--dataset-repo", PublishTarget(_TARGET.model_repo, "own er/x", _TARGET.tag)),
    ],
)
def test_bad_target_is_refused_before_anything_is_written(
    release: _Release, flag: str, target: PublishTarget
) -> None:
    with pytest.raises(ValueError, match=flag):
        stage_release(
            release.probe_dir,
            release.dataset_dir,
            release.heldout_path,
            target,
            release.staging_dir,
        )
    assert not release.staging_dir.exists()


@pytest.mark.parametrize("tag", ["v1", "v1.0", "release_2026-11", "1"])
def test_valid_tags_are_accepted(tag: str) -> None:
    check_target(PublishTarget(_TARGET.model_repo, _TARGET.dataset_repo, tag))


def _fail_model_tag(kwargs: dict[str, Any]) -> None:
    if kwargs["repo_type"] == "model":
        raise OSError("500 Server Error")


def test_model_tag_failure_deletes_the_dataset_tag(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)
    api.create_tag.side_effect = lambda *args, **kwargs: _fail_model_tag(kwargs)
    with pytest.raises(RuntimeError, match=_TARGET.model_repo):
        main(release.argv())
    api.delete_tag.assert_called_once_with(
        _TARGET.dataset_repo, tag=_TARGET.tag, repo_type="dataset"
    )


def test_failed_tag_rollback_names_the_tag_left_behind(
    release: _Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, _ = _mock_hf_api(monkeypatch)
    api.create_tag.side_effect = lambda *args, **kwargs: _fail_model_tag(kwargs)
    api.delete_tag.side_effect = OSError("403 Forbidden")
    with pytest.raises(RuntimeError, match="Rolling back also failed") as excinfo:
        main(release.argv())
    assert f"Delete tag {_TARGET.tag} from dataset repo {_TARGET.dataset_repo}" in str(
        excinfo.value
    )
