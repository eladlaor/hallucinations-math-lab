"""Taste — response evaluators for LLM-native projects.

Two public entry points:

  - `evaluate_responses(prompt, samples, ...)` — response-shaped. You
    already have N completions from your own LLM; pass them in and get
    detector scores back. The library never drives an LLM in this path.

  - `ResponseEvaluator(slm).evaluate(prompt)` — provider-shaped. You
    plug in a `SamplingProvider` and the evaluator drives sampling and
    scoring for you. Implemented in terms of `evaluate_responses`.

Detectors implement the `Detector` protocol and live under
`taste.services.*`. Three ship by default:

  - `PredictiveEntropyDetector` — token-logprob baseline (cheap).
  - `SemanticEntropyDetector` — Farquhar et al. NLI-clustered Shannon
    entropy over N samples (strong signal, N forward passes plus
    O(N * |clusters|) NLI calls).
  - `SemanticEntropyProbeDetector` — Kossen et al. linear probe over a
    pooled hidden state, single forward pass.

Escalation: pass a `Judge` implementation and the evaluator routes the
final answer through it when the chosen detector's score crosses a
threshold. `GeminiJudge` is the reference Vertex impl; consumers may
plug in any model that implements the `Judge` protocol.
"""

from taste.constants import AnswerSource, DetectorName, NLILabel
from taste.custom_types.domain_types import (
    Cluster,
    DetectionResult,
    Sample,
    SampleSet,
    TokenLogprob,
)
from taste.escalation import AllAbove, AnyAbove, EscalationPolicy, SingleDetector
from taste.evaluator import ResponseEvaluator, evaluate_responses
from taste.judges.protocol import Judge
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorPrerequisiteError,
    DetectorScore,
)
from taste.services.predictive_entropy import PredictiveEntropyDetector
from taste.services.sampling_provider import SamplingProvider
from taste.services.semantic_entropy import SemanticEntropyDetector
from taste.services.semantic_entropy_probe import SemanticEntropyProbeDetector

__all__ = [
    # Top-level entry points
    "evaluate_responses",
    "ResponseEvaluator",
    # Protocols
    "Detector",
    "Judge",
    "SamplingProvider",
    # Escalation policies
    "EscalationPolicy",
    "SingleDetector",
    "AnyAbove",
    "AllAbove",
    # Detector implementations
    "PredictiveEntropyDetector",
    "SemanticEntropyDetector",
    "SemanticEntropyProbeDetector",
    # Result + context types
    "DetectionContext",
    "DetectionResult",
    "DetectorScore",
    "DetectorPrerequisiteError",
    # Sample plumbing (consumers building their own SampleSet)
    "Sample",
    "SampleSet",
    "TokenLogprob",
    "Cluster",
    # Enums
    "AnswerSource",
    "DetectorName",
    "NLILabel",
]
