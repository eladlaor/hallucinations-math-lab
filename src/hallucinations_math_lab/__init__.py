"""hallucinations-math-lab — response evaluators for LLM-native projects.

Two public entry points:

  - `evaluate_responses(prompt, samples, ...)` — response-shaped. You
    already have N completions from your own LLM; pass them in and get
    detector scores back. The library never drives an LLM in this path.

  - `ResponseEvaluator(slm).evaluate(prompt)` — provider-shaped. You
    plug in a `SamplingProvider` and the evaluator drives sampling and
    scoring for you. Implemented in terms of `evaluate_responses`.

Detectors implement the `Detector` protocol and live under
`hallucinations_math_lab.services.*`. Four ship:

  - `PredictiveEntropyDetector` — token-logprob baseline (cheap).
  - `SemanticEntropyDetector` — Farquhar et al. NLI-clustered Shannon
    entropy over N samples, each meaning cluster weighted by its samples'
    sequence likelihoods (strong signal, N forward passes plus
    O(N * |clusters|) NLI calls).
  - `DiscreteSemanticEntropyDetector` — the same clusters weighted by count
    alone (|c| / N), for samples without token logprobs.
    The equivalence test is pluggable via the `EntailmentScorer` protocol;
    `NLIScorer` (DeBERTa-MNLI) is the default.
  - `SemanticEntropyProbeDetector` — Kossen et al. linear probe over a
    pooled hidden state, single forward pass.

Escalation: pass a `Judge` implementation and the evaluator routes the
final answer through it when the chosen detector's score crosses a
threshold. `GeminiJudge` is the reference Vertex impl; consumers may
plug in any model that implements the `Judge` protocol.
"""

from hallucinations_math_lab.constants import AnswerSource, DetectorName, NLILabel
from hallucinations_math_lab.custom_types.domain_types import (
    Cluster,
    DetectionResult,
    Sample,
    SampleSet,
    TokenLogprob,
)
from hallucinations_math_lab.escalation import AllAbove, AnyAbove, EscalationPolicy, SingleDetector
from hallucinations_math_lab.evaluator import ResponseEvaluator, evaluate_responses
from hallucinations_math_lab.judges.protocol import Judge
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorPrerequisiteError,
    DetectorScore,
)
from hallucinations_math_lab.services.nli import EntailmentScorer, NLIScorer
from hallucinations_math_lab.services.predictive_entropy import PredictiveEntropyDetector
from hallucinations_math_lab.services.sampling_provider import SamplingProvider
from hallucinations_math_lab.services.semantic_entropy import (
    DiscreteSemanticEntropyDetector,
    SemanticEntropyDetector,
)
from hallucinations_math_lab.services.semantic_entropy_probe import SemanticEntropyProbeDetector

__all__ = [
    # Top-level entry points
    "evaluate_responses",
    "ResponseEvaluator",
    # Protocols
    "Detector",
    "EntailmentScorer",
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
    "DiscreteSemanticEntropyDetector",
    "SemanticEntropyProbeDetector",
    # NLI backend for Semantic Entropy clustering
    "NLIScorer",
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
