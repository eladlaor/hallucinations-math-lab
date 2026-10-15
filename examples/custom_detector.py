"""Playground: add your own detector and run it next to PE and SE.

A detector is any object that satisfies the `Detector` protocol: a `name`,
a `metadata` self-description, two prerequisite flags, and a
`score(ctx) -> DetectorScore` method. Higher score = more likely a
hallucination, by convention.

This example implements the simplest possible "sample disagreement"
detector — the fraction of samples that differ from the most common
answer — and scores it alongside the built-ins. No GPU, no LLM.

Run:
    uv run python examples/custom_detector.py

To wire a new detector permanently, add its name to `DetectorName` in
src/hallucinations_math_lab/constants.py (the one place detector ids live),
put the class under src/hallucinations_math_lab/services/, register it in
`_build_default_detectors` in evaluator.py, and add a unit test next to
tests/unit/test_semantic_entropy.py.
"""

import time
from collections import Counter

from hallucinations_math_lab import (
    DetectionContext,
    DetectorName,
    DetectorPrerequisiteError,
    DetectorScore,
    PredictiveEntropyDetector,
    SampleSet,
    SemanticEntropyDetector,
    evaluate_responses,
)
from hallucinations_math_lab.constants import DetectorCategory
from hallucinations_math_lab.services.detector_protocol import DetectorMetadata

from score_precollected_samples import SCENARIOS, ExactMatchNLI, _normalize

# Reuses an existing id so the example needs no edit to the library. A real
# detector gets its own DetectorName member (see module docstring).
DISAGREEMENT_ID = DetectorName.LLAMA_JUDGE_QA_SAMPLES


class MajorityDisagreementDetector:
    """Fraction of samples that differ (after normalization) from the modal answer."""

    name = DISAGREEMENT_ID
    requires_samples = True
    requires_hidden_state = False
    metadata = DetectorMetadata(
        signal="fraction of samples whose normalized text differs from the most common sample",
        assumption="a model that knows the answer repeats it; one that guesses scatters",
        failure_modes=("treats every paraphrase as a disagreement, like PE",),
        category=DetectorCategory.UNCERTAINTY,
    )

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.samples is None:
            raise DetectorPrerequisiteError(
                "MajorityDisagreementDetector requires samples; DetectionContext.samples is None."
            )
        start = time.perf_counter()
        counts = Counter(_normalize(s.text) for s in ctx.samples.samples)
        modal_count = counts.most_common(1)[0][1]
        value = 1.0 - modal_count / len(ctx.samples.samples)
        return DetectorScore(
            name=self.name,
            score=value,
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


def main() -> None:
    detectors = [
        PredictiveEntropyDetector(),
        SemanticEntropyDetector(scorer=ExactMatchNLI()),
        MajorityDisagreementDetector(),
    ]
    print(f"{'scenario':<12} {'PE':>6} {'SE':>6} {'disagree':>9}")
    for label, (prompt, samples) in SCENARIOS.items():
        result = evaluate_responses(
            prompt=prompt,
            samples=SampleSet(prompt=prompt, samples=samples),
            detectors=detectors,
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=1.0,
            greedy_answer=samples[0].text,
        )
        s = result.detector_scores
        print(
            f"{label:<12} {s[DetectorName.PREDICTIVE_ENTROPY]:>6.3f} "
            f"{round(s[DetectorName.SEMANTIC_ENTROPY], 3) + 0.0:>6.3f} {s[DISAGREEMENT_ID]:>9.3f}"
        )


if __name__ == "__main__":
    main()
