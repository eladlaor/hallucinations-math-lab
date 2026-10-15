"""Judge protocol.

A Judge is "something that can answer the original prompt independently when
the detectors decide the small model is uncertain." The library defines the
shape; the consumer (or the demo) supplies the implementation. We ship
GeminiJudge as a reference, but consumers are free to plug in any model
behind this protocol — their own Gemini account, an Anthropic API client,
a stronger local model, anything.

Returns (answer, reasoning). Reasoning is best-effort — an empty string is
valid when the judge model declines to justify, and the orchestrator does
not interpret reasoning content."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    def answer(self, prompt: str) -> tuple[str, str]: ...
