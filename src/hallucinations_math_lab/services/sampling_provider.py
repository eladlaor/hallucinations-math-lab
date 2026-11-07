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

Layer convention: layer k is the hidden state after k decoder layers. Layer 0
is the token embeddings and layer L (`num_hidden_layers`) is the last layer,
with the model's final norm applied. This is the index into the tuple HF
returns for the boolean `output_hidden_states=True`; the list form of that
argument omits the embeddings and shifts every index by one.
"""

from itertools import pairwise
from typing import Protocol, runtime_checkable

import torch

from hallucinations_math_lab.custom_types.domain_types import SampleSet


class UnsupportedHiddenStateRequest(ValueError):
    """Raised when a provider cannot serve the hidden-state layers asked of it
    (e.g. a vLLM embed runner, which pools only the last layer)."""


def final_norm_applied(layer: int, num_hidden_layers: int) -> bool:
    """Whether `layer` carries the model's final norm: true only for layer L,
    which HF overwrites with the normed `last_hidden_state`."""
    if not 0 <= layer <= num_hidden_layers:
        raise ValueError(
            f"Layer {layer} is outside 0..{num_hidden_layers} for a model with "
            f"{num_hidden_layers} decoder layers."
        )
    return layer == num_hidden_layers


def check_layers(layers: tuple[int, ...], num_hidden_layers: int) -> None:
    """Fail fast unless `layers` is a non-empty, strictly increasing selection
    of indices in 0..num_hidden_layers (the order features are concatenated in)."""
    in_range = all(0 <= layer <= num_hidden_layers for layer in layers)
    increasing = all(a < b for a, b in pairwise(layers))
    if not layers or not in_range or not increasing:
        raise ValueError(
            f"Hidden-state layers {list(layers)} must be non-empty, strictly increasing "
            f"and within 0..{num_hidden_layers}."
        )


@runtime_checkable
class SamplingProvider(Protocol):
    """Anything that can sample, greedy-decode, and optionally expose
    hidden states for the prompt.

    Implementations must set `model_id` to whatever identifier they want
    surfaced in results (the library does not parse this string — it is
    propagated verbatim into logs and result metadata).

    `num_hidden_layers` is the backbone's decoder depth L, the index of its
    last layer under the convention in this module's docstring.

    `pooled_hidden_state(prompt, layers)` takes `layers` non-empty and strictly
    increasing within 0..L (see `check_layers`) and returns one row per layer,
    in that order, read at the last token of the chat-formatted prompt: shape
    `[len(layers), hidden_size]`, float32. It returns None
    when the implementation cannot expose hidden states; SEP-style detectors
    gracefully degrade in that case (the orchestrator skips them and reports
    `n/a`). It raises `UnsupportedHiddenStateRequest` when it can expose
    hidden states but not the layers requested."""

    @property
    def model_id(self) -> str: ...

    @property
    def num_hidden_layers(self) -> int: ...

    def sample(self, prompt: str) -> SampleSet: ...
    def greedy(self, prompt: str) -> str: ...
    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]) -> torch.Tensor | None: ...
