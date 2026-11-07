"""Detector protocol and dispatch primitives.

All hallucination detectors (PE, SEP, SE, and the planned LLM-as-judge) implement
the `Detector` protocol and operate on a shared `DetectionContext`.
`evaluate_responses` builds the context once per request, checks that each
detector's declared inputs are present, and dispatches it to every configured
detector.

Why a protocol and not a base class:
  - The detectors share no implementation, only an interface.
  - SE depends on N samples + an NLI scorer, PE on per-token logprobs of those
    same samples, SEP on hidden states at the last prompt token, the
    LLM-as-judge on an external LLM call. The context carries the common
    inputs; each detector reads only what it needs.

Each detector declares:
  - `requires_samples` / `requires_hidden_state`: prerequisite flags so the
    orchestrator can fail fast when the context is missing what the detector
    claimed to need.
  - `metadata: DetectorMetadata`: self-description of signal, assumption, and
    failure modes. The eval harness reads this to render the detector report,
    so every detector documents itself honestly in one place.

Each `DetectorScore` carries `cost_usd` and `latency_ms` alongside the raw
score, so the eval harness can place the detector on the AUROC / cost /
latency Pareto frontier without external bookkeeping.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from hallucinations_math_lab.constants import DetectorCategory, DetectorName
from hallucinations_math_lab.custom_types.domain_types import Cluster, SampleSet


@dataclass(frozen=True)
class DetectionContext:
    """Per-request input bundle passed to every detector.

    `samples` is None when no provider sampled the SLM for this request (e.g. a
    SEP-only fast path). `pooled_hidden_state` has shape
    `[len(layers), hidden_size]`: one row per layer the hidden-state detector
    reads, at the last prompt token, in increasing layer order (see
    `services/sampling_provider.py` for the layer convention). It is None
    when the provider does not expose hidden states (e.g. plain vLLM
    OpenAI-compat without an embed endpoint configured)."""

    prompt: str
    samples: SampleSet | None
    pooled_hidden_state: torch.Tensor | None


@dataclass(frozen=True)
class DetectorMetadata:
    """Self-description of a detector. Read by the eval harness to render the
    detector report and by humans to understand what a score actually means.

    Fields are deliberately free-form strings (not enums) because the space of
    signals and failure modes is open-ended; the report renders them verbatim.
    """

    signal: str
    """One sentence describing what the detector measures (e.g. 'mean
    per-sample negative log-likelihood')."""

    assumption: str
    """The correlational link the detector relies on to flag hallucinations
    (e.g. 'low token-level confidence correlates with incorrect answers')."""

    failure_modes: tuple[str, ...]
    """Known cases where the signal misleads (e.g. 'misses confident
    hallucinations where the model is wrong with high token probability')."""

    category: DetectorCategory


@dataclass(frozen=True)
class DetectorScore:
    """Uniform detector output.

    `score` is the scalar fed to the AUROC computation; higher should mean
    'more likely a hallucination' by convention (detectors that produce a
    correctness-confidence signal must invert before returning).

    `cost_usd` and `latency_ms` are per-call measurements used to place the
    detector on the Pareto frontier. They are populated by the detector
    itself, not by the orchestrator, because only the detector knows its
    own resource model (NLI GPU time, Vertex endpoint $/hr, etc.).

    `clusters` is populated only by SE.
    """

    name: DetectorName
    score: float
    cost_usd: float
    latency_ms: float
    clusters: list[Cluster] | None = None
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    name: DetectorName
    metadata: DetectorMetadata
    requires_samples: bool
    requires_hidden_state: bool

    def score(self, ctx: DetectionContext) -> DetectorScore: ...


class DetectorPrerequisiteError(RuntimeError):
    """Raised when a detector's declared prerequisites are not present in the
    DetectionContext. Fail-fast at dispatch time, not deep inside the detector."""
