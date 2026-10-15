"""SamplingProvider protocol — the library-side contract for "something that
can drive an LLM and hand back samples + optional pooled hidden state."

Why this exists separately from the demo's SLMProvider:
  - The library must not import from `demo/`. Consumers (e.g. langrag) build
    against the library; they have their own LLM and either pass already-
    sampled responses (response-shaped API) or wrap their own client to
    satisfy this protocol (provider-shaped API).
  - The demo's SLMProvider satisfies this protocol structurally (same method
    names + shapes). The two protocols are intentionally identical in
    surface; the duplication is the price of clean directionality
    (`hallucinations_math_lab` does not depend on `demo`).

Use this type, not the demo type, in any library-side function signature
that talks to "something that produces samples."
"""

from typing import Protocol, runtime_checkable

import torch

from hallucinations_math_lab.custom_types.domain_types import SampleSet


@runtime_checkable
class SamplingProvider(Protocol):
    """Anything that can sample, greedy-decode, and optionally expose a
    pooled hidden state for the prompt.

    Implementations must set `model_id` to whatever identifier they want
    surfaced in results (the library does not parse this string — it is
    propagated verbatim into logs and result metadata).

    `pooled_hidden_state` returns None when the implementation cannot
    expose hidden states. SEP-style detectors gracefully degrade in
    that case (the orchestrator skips them and reports `n/a`)."""

    model_id: str

    def sample(self, prompt: str) -> SampleSet: ...
    def greedy(self, prompt: str) -> str: ...
    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None: ...
