"""Judges — escalation-time models that re-answer a prompt the detectors
flagged as uncertain. Library defines the protocol; consumers plug in any
model that implements it. `GeminiJudge` is the reference Vertex impl."""

from taste.judges.gemini import GeminiJudge
from taste.judges.protocol import Judge

__all__ = ["GeminiJudge", "Judge"]
