"""Detector protocol and dispatch primitives.

All hallucination detectors (PE, SEP, SE) implement the `Detector` protocol and
operate on a shared `DetectionContext`. `DetectorAgent` builds the context once
per request and dispatches it to every configured detector, regardless of
which detectors actually need samples or hidden states.

Why a protocol and not a base class:
  - The three detectors share no implementation, only an interface.
  - SE depends on N samples + an NLI scorer, PE on per-token logprobs of those
    same samples, SEP on a single pooled hidden state. The context carries all
    three; each detector reads only what it needs.

Detectors are responsible for declaring `requires_samples` and
`requires_hidden_state` so the orchestrator can fail fast when the context is
missing the prerequisites a detector claimed to need.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from taste.constants import DetectorName
from taste.custom_types.domain_types import Cluster, SampleSet


@dataclass(frozen=True)
class DetectionContext:
    """Per-request input bundle passed to every detector.

    `samples` is None when no provider sampled the SLM for this request (e.g. a
    SEP-only fast path). `pooled_hidden_state` is None when the provider does
    not expose hidden states (e.g. plain vLLM OpenAI-compat without an embed
    endpoint configured)."""

    prompt: str
    samples: SampleSet | None
    pooled_hidden_state: torch.Tensor | None


@dataclass(frozen=True)
class DetectorScore:
    """Uniform detector output. `clusters` is populated only by SE."""

    name: DetectorName
    score: float
    clusters: list[Cluster] | None = None


@runtime_checkable
class Detector(Protocol):
    name: DetectorName
    requires_samples: bool
    requires_hidden_state: bool

    def score(self, ctx: DetectionContext) -> DetectorScore: ...


class DetectorPrerequisiteError(RuntimeError):
    """Raised when a detector's declared prerequisites are not present in the
    DetectionContext. Fail-fast at dispatch time, not deep inside the detector."""
