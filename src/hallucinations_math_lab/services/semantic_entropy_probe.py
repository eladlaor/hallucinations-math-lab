"""Semantic Entropy Probe (SEP) detector.

Kossen et al. (2024): a logistic probe over the SLM's hidden states at the
last prompt token predicts whether Semantic Entropy would be high, from one
forward pass instead of N generations plus O(N) NLI calls.

  - The score is p(high SE) = sigmoid(w·x + b), where x concatenates the
    probe's layers in increasing order. It is not a hallucination
    probability and not in nats; SEP_THRESHOLD is its escalation threshold.

  - A probe is a directory (`eval.train_sep_probe` writes it, the Hub hosts
    the same layout): `probe.safetensors` holds the weights, and
    `probe_config.json` pins the backbone, layers, token position and prompt
    it was trained on. Nothing in it is unpickled, so a downloaded probe
    cannot run code.

  - SEP_PROBE_PATH picks the directory: a local path, `hf://<owner>/<repo>[@<revision>]`,
    or unset for `data/probes/<model slug>/`. A pre-classifier Ridge `.pt`
    probe is refused, never converted: it predicted SE in nats.

  - The probe is loaded and validated at construction, and
    `check_compatible` refuses a model or prompt it was not trained on, so a
    misconfiguration surfaces before the first request.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import ValidationError

from hallucinations_math_lab.constants import (
    HF_PROBE_REVISION_SEPARATOR,
    HF_PROBE_URI_PREFIX,
    HF_REPO_ID_SEPARATOR,
    LEGACY_SEP_PROBE_SUFFIX,
    LOG_EXTRA_KEY,
    MODEL_SLUG_PATTERN,
    MODEL_SLUG_SEPARATOR,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_DIR,
    SEP_PROBE_DIR_FILES,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_SERVED_TOKEN_POSITION,
    DetectorCategory,
    DetectorName,
)
from hallucinations_math_lab.custom_types.sep_probe_types import SEPProbeConfig
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorPrerequisiteError,
    DetectorScore,
)
from hallucinations_math_lab.services.sep_probe_weights import ProbeWeights, read_probe_weights

logger = logging.getLogger(__name__)

_HF_URI_FORM: Final[str] = (
    f"{HF_PROBE_URI_PREFIX}<owner>/<repo>[{HF_PROBE_REVISION_SEPARATOR}<revision>]"
)
_TRAIN_HINT: Final[str] = (
    "Collect a dataset with `uv run python -m eval.collect_sep_dataset`, then train with "
    "`uv run python -m eval.train_sep_probe --dataset DIR`, or point SEP_PROBE_PATH at a "
    f"probe directory or a Hub probe ({_HF_URI_FORM})."
)
_HUB_REPO_ID_PARTS: Final[int] = 2


class ObsoleteSEPProbeError(ValueError):
    """A file where a probe directory belongs, never loaded: typically a pre-classifier
    Ridge `.pt` probe, which predicts SE in nats."""


class SEPProbeMismatchError(ValueError):
    """The probe was trained on another backbone or prompt than the one under test."""


def model_slug(model_id: str) -> str:
    """Filesystem-safe name for a model id: `google/gemma-4-E2B-it` -> `google-gemma-4-e2b-it`."""
    slug = re.sub(MODEL_SLUG_PATTERN, MODEL_SLUG_SEPARATOR, model_id.lower()).strip(
        MODEL_SLUG_SEPARATOR
    )
    if not slug:
        raise ValueError(f"model_slug: model id {model_id!r} has no letters or digits.")
    return slug


def default_probe_dir(model_id: str) -> Path:
    """Where the SEP probe for `model_id` lives unless SEP_PROBE_PATH overrides it."""
    return Path(SEP_PROBE_DIR) / model_slug(model_id)


def _legacy_probe_path(model_id: str) -> Path:
    return Path(SEP_PROBE_DIR) / f"{model_slug(model_id)}{LEGACY_SEP_PROBE_SUFFIX}"


def _refuse_probe_file(path: Path) -> None:
    """Raise ObsoleteSEPProbeError if `path` is a `.pt` reference or any regular file."""
    if path.suffix == LEGACY_SEP_PROBE_SUFFIX:
        raise ObsoleteSEPProbeError(
            f"SEP probe {path} is a pre-classifier Ridge probe (predicts SE in nats), not a "
            f"probe directory. SEP now outputs p(high SE). {_TRAIN_HINT}"
        )
    if path.is_file():
        inside = (
            f" Its directory {path.parent} holds a {SEP_PROBE_CONFIG_FILE}; point at that."
            if (path.parent / SEP_PROBE_CONFIG_FILE).is_file()
            else ""
        )
        raise ObsoleteSEPProbeError(
            f"SEP probe {path} is a file; SEP_PROBE_PATH must be a probe directory or "
            f"{_HF_URI_FORM}.{inside}"
        )


@dataclass(frozen=True)
class HubProbeRef:
    """A probe directory on the Hub: a model repo, at `revision` or its default branch."""

    repo_id: str
    revision: str | None

    @property
    def uri(self) -> str:
        suffix = "" if self.revision is None else HF_PROBE_REVISION_SEPARATOR + self.revision
        return f"{HF_PROBE_URI_PREFIX}{self.repo_id}{suffix}"


def parse_hf_probe_uri(uri: str) -> HubProbeRef:
    """Parse `hf://<owner>/<repo>[@<revision>]`; raises ValueError on any other form."""
    repo_id, separator, revision = uri.removeprefix(HF_PROBE_URI_PREFIX).partition(
        HF_PROBE_REVISION_SEPARATOR
    )
    parts = repo_id.split(HF_REPO_ID_SEPARATOR)
    if (
        not uri.startswith(HF_PROBE_URI_PREFIX)
        or len(parts) != _HUB_REPO_ID_PARTS
        or not all(parts)
        or (separator and not revision)
    ):
        raise ValueError(f"SEP probe reference {uri!r} is malformed; expected {_HF_URI_FORM}.")
    try:
        validate_repo_id(repo_id)
    except HFValidationError as e:
        raise ValueError(
            f"SEP probe reference {uri!r} names an invalid Hub repo id; expected "
            f"{_HF_URI_FORM}: {e!s}"
        ) from e
    return HubProbeRef(repo_id=repo_id, revision=revision or None)


def download_probe(ref: HubProbeRef) -> Path:
    """Fetch the probe directory's files from the Hub; returns the local snapshot.

    The snapshot folder is named after the commit it holds, which is logged. If the Hub
    cannot be reached, `snapshot_download` serves the cached snapshot of `ref.revision`
    (or of the default branch when unpinned) without raising, so an unpinned reference may
    then load an older probe; pin `@<revision>` for reproducible runs."""
    try:
        local = snapshot_download(
            ref.repo_id, revision=ref.revision, allow_patterns=list(SEP_PROBE_DIR_FILES)
        )
    except OSError as e:
        raise RuntimeError(f"Cannot download SEP probe {ref.uri}: {e!s}") from e
    if not isinstance(local, str):
        raise TypeError(f"snapshot_download returned {type(local).__name__} for {ref.uri}.")
    logger.info(
        "Downloaded SEP probe.",
        extra={
            LOG_EXTRA_KEY: {
                "probe_ref": ref.uri,
                "resolved_commit": Path(local).name,
                "local_dir": local,
            }
        },
    )
    return Path(local)


def resolve_probe_source(sep_probe_path: str, model_id: str) -> Path:
    """The local probe directory for SEP_PROBE_PATH (empty = the default for `model_id`).

    An `hf://` reference is downloaded (see download_probe for its offline behaviour).
    There is no fallback between sources: a missing directory raises, and a legacy `.pt`
    probe raises ObsoleteSEPProbeError."""
    if sep_probe_path.startswith(HF_PROBE_URI_PREFIX):
        return download_probe(parse_hf_probe_uri(sep_probe_path))
    if sep_probe_path:
        directory = Path(sep_probe_path)
        _refuse_probe_file(directory)
    else:
        directory = default_probe_dir(model_id)
        legacy = _legacy_probe_path(model_id)
        if not directory.is_dir() and legacy.is_file():
            _refuse_probe_file(legacy)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"No SEP probe directory for {model_id} at {directory}. {_TRAIN_HINT}"
        )
    return directory


@dataclass(frozen=True)
class LoadedSEPProbe:
    """A probe directory's config and weights, checked against each other."""

    config: SEPProbeConfig
    weights: ProbeWeights


def _read_probe_config(path: Path) -> SEPProbeConfig:
    try:
        return SEPProbeConfig.model_validate_json(path.read_text())
    except OSError as e:
        raise RuntimeError(f"Cannot read SEP probe config {path}: {e!s}") from e
    except ValidationError as e:
        raise ValueError(f"SEP probe config {path} is invalid: {e!s}") from e


def load_probe_dir(directory: Path) -> LoadedSEPProbe:
    """Load and validate a probe directory; refuses a legacy `.pt` probe and SLT probes."""
    _refuse_probe_file(directory)
    for name in (SEP_PROBE_CONFIG_FILE, SEP_PROBE_WEIGHTS_FILE):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"SEP probe directory {directory} has no {name}. {_TRAIN_HINT}")
    config = _read_probe_config(directory / SEP_PROBE_CONFIG_FILE)
    position = config.features.token_position
    if position != SEP_SERVED_TOKEN_POSITION:
        raise ValueError(
            f"SEP probe {directory} reads token position {position!s}; only "
            f"{SEP_SERVED_TOKEN_POSITION!s} probes are served (SLT serving is a follow-up)."
        )
    weights = read_probe_weights(directory / SEP_PROBE_WEIGHTS_FILE)
    n_features = len(config.features.layers) * config.backbone.hidden_size
    if weights.weight.size != n_features:
        raise ValueError(
            f"SEP probe {directory} has {weights.weight.size} weights, but its config reads "
            f"{n_features} features ({len(config.features.layers)} layers x hidden size "
            f"{config.backbone.hidden_size})."
        )
    return LoadedSEPProbe(config=config, weights=weights)


class SemanticEntropyProbeDetector(Detector):
    """SEP detector implementing the `Detector` protocol.

    `probe_dir` is a local probe directory (see `resolve_probe_source`). The
    caller must pass the hidden states of `hidden_state_layers`, one row per
    layer, and should call `check_compatible` with the model under test."""

    name = DetectorName.SEMANTIC_ENTROPY_PROBE
    requires_samples = False
    requires_hidden_state = True
    metadata = DetectorMetadata(
        signal="logistic probe over the SLM's hidden states at the last prompt token, giving p(high SE)",
        assumption="whether SE will be high is linearly decodable from a single forward pass over the prompt",
        failure_modes=(
            "probe is SLM-specific — weights trained on model A do not transfer to model B",
            "underperforms full SE on adversarial / out-of-distribution prompts where the linear approximation breaks",
            "requires hidden-state access; useless against any provider that does not expose them",
        ),
        category=DetectorCategory.UNCERTAINTY,
    )

    def __init__(self, probe_dir: str | Path) -> None:
        self._probe_dir = Path(probe_dir)
        probe = load_probe_dir(self._probe_dir)
        self._config = probe.config
        self._weight = torch.from_numpy(probe.weights.weight)
        self._bias = float(probe.weights.bias[0])
        logger.info(
            "Loaded SEP probe.",
            extra={
                LOG_EXTRA_KEY: {
                    "probe_dir": str(self._probe_dir),
                    "model_id": self._config.backbone.model_id,
                    "layers": list(self._config.features.layers),
                }
            },
        )

    @property
    def probe_dir(self) -> Path:
        """The local probe directory the probe was loaded from."""
        return self._probe_dir

    @property
    def config(self) -> SEPProbeConfig:
        """The probe's `probe_config.json`."""
        return self._config

    @property
    def hidden_state_layers(self) -> tuple[int, ...]:
        """The layers whose hidden states the probe reads, in increasing order."""
        return self._config.features.layers

    def check_compatible(
        self,
        *,
        model_id: str,
        revision: str,
        num_hidden_layers: int,
        hidden_size: int,
        answer_instruction: str,
        chat_template_sha256: str,
    ) -> None:
        """Raise SEPProbeMismatchError, naming both values, for every field in
        which the model under test differs from the one the probe was trained on."""
        backbone, prompt = self._config.backbone, self._config.prompt
        pairs: dict[str, tuple[object, object]] = {
            "model_id": (backbone.model_id, model_id),
            "revision": (backbone.revision, revision),
            "num_hidden_layers": (backbone.num_hidden_layers, num_hidden_layers),
            "hidden_size": (backbone.hidden_size, hidden_size),
            "answer_instruction": (prompt.answer_instruction, answer_instruction),
            "chat_template_sha256": (prompt.chat_template_sha256, chat_template_sha256),
        }
        mismatches = [
            f"{field}: probe {trained!r}, model under test {actual!r}"
            for field, (trained, actual) in pairs.items()
            if trained != actual
        ]
        if mismatches:
            raise SEPProbeMismatchError(
                f"SEP probe {self._probe_dir} does not fit the model under test "
                f"({'; '.join(mismatches)}). A probe works only with the backbone and prompt "
                f"it was trained on. {_TRAIN_HINT}"
            )

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.pooled_hidden_state is None:
            raise DetectorPrerequisiteError(
                "SemanticEntropyProbeDetector requires a pooled hidden state; "
                "DetectionContext.pooled_hidden_state is None."
            )
        layers = self.hidden_state_layers
        hidden_size = self._config.backbone.hidden_size
        h = ctx.pooled_hidden_state.to(dtype=torch.float32)
        if tuple(h.shape) != (len(layers), hidden_size):
            raise RuntimeError(
                f"Pooled hidden state has shape {list(h.shape)}, but this probe reads layers "
                f"{list(layers)} and expects [{len(layers)}, {hidden_size}]."
            )
        # cost_usd is 0 (local CPU); latency_ms is the one dot product.
        start = time.perf_counter()
        p_high_se = torch.sigmoid(h.reshape(-1) @ self._weight + self._bias).item()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectorScore(
            name=self.name,
            score=float(p_high_se),
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
