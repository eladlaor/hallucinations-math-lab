"""Escalation policies — pluggable decision rules over detector scores.

The single-detector path (compare one detector's score against one
threshold) is the historical default and remains the back-compat
behavior when no policy is supplied. These policies generalize that
decision so a consumer can combine detectors with OR / AND semantics
without the evaluator hardcoding a comparison.

No heavy imports here — policies are pure logic over a
`dict[DetectorName, float]` of already-computed scores.

A note on detector roles: a false-negative-prone detector (e.g.
predictive entropy) must never be used as a gate placed IN FRONT OF
semantic entropy (do NOT require it via `AllAbove`). It may only be an
additive OR trigger (`AnyAbove`) so it can raise escalation but never
suppress an SE-driven escalation.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hallucinations_math_lab.constants import DetectorName


@runtime_checkable
class EscalationPolicy(Protocol):
    """Decides whether a set of detector scores warrants escalation.

    `detectors_referenced` declares which detector names the policy reads,
    so the evaluator can fail-fast-validate they are present in the scores
    map before calling `should_escalate`."""

    detectors_referenced: tuple[DetectorName, ...]

    def should_escalate(self, scores: dict[DetectorName, float]) -> bool: ...

    def describe(self) -> str: ...


@dataclass(frozen=True)
class SingleDetector:
    """Back-compat policy: escalate when one detector's score strictly
    exceeds one threshold. Equivalent to the historical
    `scores[name] > threshold` decision."""

    name: DetectorName
    threshold: float

    def should_escalate(self, scores: dict[DetectorName, float]) -> bool:
        return scores[self.name] > self.threshold

    def describe(self) -> str:
        return f"{self.name} > {self.threshold}"

    @property
    def detectors_referenced(self) -> tuple[DetectorName, ...]:
        return (self.name,)


@dataclass(frozen=True)
class AnyAbove:
    """OR-safeguard: escalate if ANY referenced detector strictly exceeds
    its threshold. The safe way to add a false-negative-prone detector
    (e.g. PE) as an additive trigger — it can raise escalation but never
    suppress one."""

    thresholds: dict[DetectorName, float]

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError(
                "AnyAbove requires at least one detector threshold; got an empty mapping. "
                "An empty OR-policy would never escalate, which is almost certainly a misconfiguration."
            )

    def should_escalate(self, scores: dict[DetectorName, float]) -> bool:
        return any(scores[n] > t for n, t in self.thresholds.items())

    def describe(self) -> str:
        return "ANY(" + ", ".join(f"{n} > {t}" for n, t in self.thresholds.items()) + ")"

    @property
    def detectors_referenced(self) -> tuple[DetectorName, ...]:
        return tuple(self.thresholds.keys())


@dataclass(frozen=True)
class AllAbove:
    """AND-policy: escalate only if ALL referenced detectors strictly
    exceed their thresholds. Do NOT use this to gate a strong detector
    (SE) behind a false-negative-prone one (PE) — that would let PE
    suppress an SE-driven escalation."""

    thresholds: dict[DetectorName, float]

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError(
                "AllAbove requires at least one detector threshold; got an empty mapping. "
                "An empty AND-policy would always escalate (all() over no items is True), "
                "which is almost certainly a misconfiguration."
            )

    def should_escalate(self, scores: dict[DetectorName, float]) -> bool:
        return all(scores[n] > t for n, t in self.thresholds.items())

    def describe(self) -> str:
        return "ALL(" + ", ".join(f"{n} > {t}" for n, t in self.thresholds.items()) + ")"

    @property
    def detectors_referenced(self) -> tuple[DetectorName, ...]:
        return tuple(self.thresholds.keys())
