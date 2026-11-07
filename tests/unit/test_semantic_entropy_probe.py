"""Semantic Entropy Probe detector tests.

Pin the probe's contract end to end:
  - A probe directory (or an `hf://` reference to one) loads into the
    detector; a legacy Ridge `.pt` probe is refused with a retrain hint.
  - The probe's config must match the backbone and prompt it is used with;
    every mismatch names both values.
  - The score is p(high SE) = sigmoid(w·x + b) over the concatenated layers.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy.special import expit

from hallucinations_math_lab.constants import (
    LOG_EXTRA_KEY,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_DIR,
    SEP_PROBE_DIR_FILES,
    SEP_PROBE_WEIGHTS_FILE,
    DetectorName,
    TokenPosition,
)
from hallucinations_math_lab.services import semantic_entropy_probe
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    DetectorPrerequisiteError,
)
from hallucinations_math_lab.services.semantic_entropy_probe import (
    HubProbeRef,
    ObsoleteSEPProbeError,
    ProbeSource,
    SemanticEntropyProbeDetector,
    SEPProbeMismatchError,
    default_probe_dir,
    load_probe_dir,
    parse_hf_probe_uri,
    resolve_probe_source,
)
from tests.unit.sep_probe_fixtures import (
    PROBE_ANSWER_INSTRUCTION,
    PROBE_HIDDEN_SIZE,
    PROBE_MODEL_ID,
    PROBE_NUM_HIDDEN_LAYERS,
    PROBE_REVISION,
    PROBE_TEMPLATE_SHA256,
    probe_config,
    write_probe_dir,
)

_MATCHING_BACKBONE: dict[str, Any] = {
    "model_id": PROBE_MODEL_ID,
    "revision": PROBE_REVISION,
    "num_hidden_layers": PROBE_NUM_HIDDEN_LAYERS,
    "hidden_size": PROBE_HIDDEN_SIZE,
    "answer_instruction": PROBE_ANSWER_INSTRUCTION,
    "chat_template_sha256": PROBE_TEMPLATE_SHA256,
}


@pytest.fixture
def probe_dir(tmp_path: Path) -> Path:
    return write_probe_dir(tmp_path / "probe")


def _ctx(hidden: torch.Tensor | None) -> DetectionContext:
    return DetectionContext(prompt="q", samples=None, pooled_hidden_state=hidden)


# --- Where the probe comes from ---------------------------------------------


def test_hf_uri_parses_repo_and_revision() -> None:
    assert parse_hf_probe_uri("hf://a/b") == HubProbeRef(repo_id="a/b", revision=None)
    assert parse_hf_probe_uri("hf://a/b@v1") == HubProbeRef(repo_id="a/b", revision="v1")
    for malformed in (
        "hf://a",
        "hf://a/b/c",
        "hf://a/b@",
        "hf:///b",
        "hf://a/",
        "a/b",
        "hf://a b/c",
    ):
        with pytest.raises(ValueError, match="hf://"):
            parse_hf_probe_uri(malformed)


def test_hub_download_mocked_loads_matching_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(write_probe_dir(tmp_path / "hub-cache"))

    monkeypatch.setattr(semantic_entropy_probe, "snapshot_download", fake_snapshot_download)

    source = resolve_probe_source("hf://owner/sep-probe@v1", PROBE_MODEL_ID)
    detector = SemanticEntropyProbeDetector(source)
    detector.check_compatible(**_MATCHING_BACKBONE)

    (call,) = calls
    assert call["repo_id"] == "owner/sep-probe"
    assert call["revision"] == "v1"
    assert sorted(call["allow_patterns"]) == sorted(SEP_PROBE_DIR_FILES)
    assert detector.probe_dir == tmp_path / "hub-cache"
    assert detector.source == source


def test_hub_download_logs_the_resolved_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    commit = "c" * 40

    def fake_snapshot_download(repo_id: str, **_kwargs: Any) -> str:
        return str(write_probe_dir(tmp_path / "snapshots" / commit))

    monkeypatch.setattr(semantic_entropy_probe, "snapshot_download", fake_snapshot_download)
    with caplog.at_level(logging.INFO, logger=semantic_entropy_probe.logger.name):
        source = resolve_probe_source("hf://owner/sep-probe", PROBE_MODEL_ID)
    assert source.ref == "hf://owner/sep-probe"
    assert source.revision == commit

    (record,) = [r for r in caplog.records if r.name == semantic_entropy_probe.logger.name]
    context = getattr(record, LOG_EXTRA_KEY)
    assert context["probe_ref"] == "hf://owner/sep-probe"
    assert context["resolved_commit"] == commit


def test_hub_download_failure_names_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(repo_id: str, **_kwargs: Any) -> str:
        raise OSError("no network")

    monkeypatch.setattr(semantic_entropy_probe, "snapshot_download", offline)
    with pytest.raises(RuntimeError, match=r"hf://owner/sep-probe@v1.*no network"):
        resolve_probe_source("hf://owner/sep-probe@v1", PROBE_MODEL_ID)


def test_explicit_directory_is_used_as_is(probe_dir: Path) -> None:
    assert resolve_probe_source(str(probe_dir), "any/model") == ProbeSource(
        directory=probe_dir, ref=str(probe_dir), revision=None
    )


def test_a_relative_directory_is_recorded_as_given(
    probe_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never resolved: a published run config must not carry this machine's absolute path."""
    monkeypatch.chdir(probe_dir.parent)
    source = resolve_probe_source(probe_dir.name, "any/model")
    assert source.directory == Path(probe_dir.name)
    assert source.ref == probe_dir.name


def test_missing_explicit_directory_raises_with_training_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"eval\.collect_sep_dataset"):
        resolve_probe_source(str(tmp_path / "absent"), PROBE_MODEL_ID)


def test_default_directory_follows_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = Path(SEP_PROBE_DIR) / "google-gemma-4-e2b-it"
    assert default_probe_dir("google/gemma-4-E2B-it") == expected
    with pytest.raises(FileNotFoundError, match="google-gemma-4-e2b-it"):
        resolve_probe_source("", "google/gemma-4-E2B-it")
    write_probe_dir(expected)
    source = resolve_probe_source("", "google/gemma-4-E2B-it")
    assert source.directory == expected
    assert source.ref == f"{SEP_PROBE_DIR}/google-gemma-4-e2b-it"


def test_pt_file_raises_obsolete_with_retrain_hint(tmp_path: Path) -> None:
    legacy = tmp_path / "probe.pt"
    legacy.write_bytes(b"pickled ridge")
    for reference in (legacy, tmp_path / "never-written.pt"):
        with pytest.raises(ObsoleteSEPProbeError, match=r"Ridge.*eval\.train_sep_probe"):
            resolve_probe_source(str(reference), PROBE_MODEL_ID)
    with pytest.raises(ObsoleteSEPProbeError, match="p\\(high SE\\)"):
        SemanticEntropyProbeDetector(legacy)


def test_regular_file_is_refused_as_obsolete(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "probe.bin"
    not_a_dir.write_bytes(b"x")
    with pytest.raises(ObsoleteSEPProbeError, match="must be a probe directory") as excinfo:
        resolve_probe_source(str(not_a_dir), PROBE_MODEL_ID)
    assert "Ridge" not in str(excinfo.value)


def test_file_inside_a_probe_dir_names_the_directory(probe_dir: Path) -> None:
    weights_file = probe_dir / SEP_PROBE_WEIGHTS_FILE
    with pytest.raises(ObsoleteSEPProbeError, match="point at that") as excinfo:
        resolve_probe_source(str(weights_file), PROBE_MODEL_ID)
    assert str(probe_dir) in str(excinfo.value)
    assert "Ridge" not in str(excinfo.value)


def test_missing_default_dir_with_legacy_pt_raises_obsolete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy = Path(SEP_PROBE_DIR) / "org-model.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"pickled ridge")
    with pytest.raises(ObsoleteSEPProbeError, match="org-model.pt"):
        resolve_probe_source("", PROBE_MODEL_ID)


# --- What the probe is -------------------------------------------------------


def test_loads_config_and_weights(probe_dir: Path) -> None:
    probe = load_probe_dir(probe_dir)
    assert probe.config == probe_config()
    assert probe.weights.weight.shape == (PROBE_HIDDEN_SIZE,)


def test_detector_hashes_the_bytes_it_loaded(probe_dir: Path) -> None:
    """Hashes come from load time, so rewriting the directory later does not change them."""
    weights_sha256 = hashlib.sha256((probe_dir / SEP_PROBE_WEIGHTS_FILE).read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256((probe_dir / SEP_PROBE_CONFIG_FILE).read_bytes()).hexdigest()
    detector = SemanticEntropyProbeDetector(probe_dir)
    (probe_dir / SEP_PROBE_CONFIG_FILE).write_text("retrained")
    (probe_dir / SEP_PROBE_WEIGHTS_FILE).write_bytes(b"retrained")
    assert detector.weights_sha256 == weights_sha256
    assert detector.config_sha256 == config_sha256


def test_missing_probe_file_raises_naming_it(probe_dir: Path) -> None:
    (probe_dir / SEP_PROBE_CONFIG_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=SEP_PROBE_CONFIG_FILE):
        SemanticEntropyProbeDetector(probe_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("unexpected_field", True)],
    ids=["unknown_schema", "extra_field"],
)
def test_unknown_schema_or_extra_field_raises(probe_dir: Path, field: str, value: object) -> None:
    config_path = probe_dir / SEP_PROBE_CONFIG_FILE
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        SemanticEntropyProbeDetector(probe_dir)


def test_weights_must_match_layers_times_hidden_size(tmp_path: Path) -> None:
    directory = write_probe_dir(tmp_path / "probe", weight=np.ones(PROBE_HIDDEN_SIZE + 1))
    with pytest.raises(ValueError, match="features"):
        SemanticEntropyProbeDetector(directory)


def test_slt_probe_refused_in_v1(tmp_path: Path) -> None:
    directory = write_probe_dir(tmp_path / "probe", probe_config(token_position=TokenPosition.SLT))
    with pytest.raises(ValueError, match="SLT serving is a follow-up"):
        SemanticEntropyProbeDetector(directory)


# --- Does it fit the model under test? ---------------------------------------


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("model_id", "org/other-model"),
        ("revision", "f" * 40),
        ("num_hidden_layers", PROBE_NUM_HIDDEN_LAYERS + 1),
        ("hidden_size", PROBE_HIDDEN_SIZE * 2),
        ("answer_instruction", "Answer at length."),
        ("chat_template_sha256", "0" * 64),
    ],
    ids=["model_id", "revision", "layers", "hidden_size", "answer_instruction", "template"],
)
def test_mismatch_raises_naming_both(probe_dir: Path, field: str, other: object) -> None:
    detector = SemanticEntropyProbeDetector(probe_dir)
    with pytest.raises(SEPProbeMismatchError) as excinfo:
        detector.check_compatible(**{**_MATCHING_BACKBONE, field: other})
    # The exact fragment: bare values could match digits in the probe's tmp path.
    expected = f"{field}: probe {_MATCHING_BACKBONE[field]!r}, model under test {other!r}"
    assert expected in str(excinfo.value)


def test_matching_backbone_passes(probe_dir: Path) -> None:
    SemanticEntropyProbeDetector(probe_dir).check_compatible(**_MATCHING_BACKBONE)


# --- How it scores -----------------------------------------------------------


def test_detector_declares_correct_prerequisites(probe_dir: Path) -> None:
    detector = SemanticEntropyProbeDetector(probe_dir)
    assert detector.name == DetectorName.SEMANTIC_ENTROPY_PROBE
    assert detector.requires_samples is False
    assert detector.requires_hidden_state is True
    assert detector.hidden_state_layers == (PROBE_NUM_HIDDEN_LAYERS,)


def test_score_is_sigmoid_in_unit_interval(tmp_path: Path) -> None:
    layers = (1, 2)
    weight = np.arange(len(layers) * PROBE_HIDDEN_SIZE, dtype=np.float32) / 10.0
    directory = write_probe_dir(
        tmp_path / "probe", probe_config(layers=layers), weight=weight, bias=-1.0
    )
    detector = SemanticEntropyProbeDetector(directory)
    for scale in (-100.0, -1.0, 0.0, 1.0, 100.0):
        hidden = scale * torch.ones(len(layers), PROBE_HIDDEN_SIZE)
        score = detector.score(_ctx(hidden)).score
        expected = expit(scale * float(weight.sum()) - 1.0)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(expected, abs=1e-6)


def test_score_concatenates_layers_in_increasing_order(tmp_path: Path) -> None:
    layers = (1, 2)
    # Only the second layer's first feature counts.
    weight = np.zeros(len(layers) * PROBE_HIDDEN_SIZE, dtype=np.float32)
    weight[PROBE_HIDDEN_SIZE] = 1.0
    detector = SemanticEntropyProbeDetector(
        write_probe_dir(tmp_path / "probe", probe_config(layers=layers), weight=weight)
    )
    hidden = torch.zeros(len(layers), PROBE_HIDDEN_SIZE)
    hidden[1, 0] = 2.0
    assert detector.score(_ctx(hidden)).score == pytest.approx(expit(2.0))


def test_score_raises_without_hidden_state(probe_dir: Path) -> None:
    detector = SemanticEntropyProbeDetector(probe_dir)
    with pytest.raises(DetectorPrerequisiteError, match="pooled hidden state"):
        detector.score(_ctx(None))


@pytest.mark.parametrize(
    "shape",
    [(PROBE_HIDDEN_SIZE,), (2, PROBE_HIDDEN_SIZE), (1, PROBE_HIDDEN_SIZE * 2)],
    ids=["1d", "too_many_layers", "wrong_hidden_size"],
)
def test_score_raises_on_wrong_hidden_state_shape(probe_dir: Path, shape: tuple[int, ...]) -> None:
    detector = SemanticEntropyProbeDetector(probe_dir)
    with pytest.raises(RuntimeError, match=r"expects \[1, 4\]"):
        detector.score(_ctx(torch.zeros(shape)))
