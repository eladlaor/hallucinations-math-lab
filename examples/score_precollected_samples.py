"""Playground: score hand-written sample sets with PE and SE — no GPU, no answer-generating LLM.

Three toy prompts, each with N=10 "sampled" answers and made-up token
logprobs, chosen to expose where the detectors agree and where they split:

  - consistent:  every sample says the same thing, confidently.
  - paraphrased: every sample means the same thing, worded differently.
  - divergent:   the samples disagree in meaning (the hallucination shape).

Run:
    uv run python examples/score_precollected_samples.py

SE groups the samples by meaning with the NLI model the library uses by
default (microsoft/deberta-large-mnli, ~1.6 GB download on first run; CPU is
fine), so the paraphrases land in one cluster while PE still reads their
varied wording as uncertainty.
"""

import logging

from hallucinations_math_lab import (
    DetectorName,
    NLIScorer,
    PredictiveEntropyDetector,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
)

SE_THRESHOLD_NATS = 1.0


def _sample(text: str, logprob: float) -> Sample:
    """One sample whose every token carries the same logprob — enough for PE."""
    return Sample(
        text=text,
        token_logprobs=[TokenLogprob(token=tok, logprob=logprob) for tok in text.split()],
    )


SCENARIOS: dict[str, tuple[str, list[Sample]]] = {
    "consistent": (
        "What is the capital of France?",
        [_sample("Paris", -0.05) for _ in range(10)],
    ),
    "paraphrased": (
        "What is the capital of France?",
        [
            _sample(text, -0.6)
            for text in [
                "Paris",
                "The capital is Paris.",
                "It's Paris.",
                "Paris, France",
                "Paris is the capital of France.",
                "Paris",
                "The answer is Paris.",
                "France's capital is Paris.",
                "Paris.",
                "That would be Paris.",
            ]
        ],
    ),
    "divergent": (
        "Who was the first woman to walk on Mars?",
        [
            _sample(text, -0.4)
            for text in [
                "Sally Ride",
                "Valentina Tereshkova",
                "Peggy Whitson",
                "Sally Ride",
                "Christina Koch",
                "Mae Jemison",
                "No woman has walked on Mars.",
                "Valentina Tereshkova",
                "Eileen Collins",
                "Sally Ride",
            ]
        ],
    ),
}


def main() -> None:
    detectors = [PredictiveEntropyDetector(), SemanticEntropyDetector(scorer=NLIScorer())]

    print(f"escalation: SE > {SE_THRESHOLD_NATS} nats\n")
    print(f"{'scenario':<12} {'PE':>6} {'SE':>6} {'clusters':>9}  escalate")
    for label, (prompt, samples) in SCENARIOS.items():
        result = evaluate_responses(
            prompt=prompt,
            samples=SampleSet(prompt=prompt, samples=samples),
            detectors=detectors,
            escalation_detector=DetectorName.SEMANTIC_ENTROPY,
            escalation_threshold=SE_THRESHOLD_NATS,
            greedy_answer=samples[0].text,
        )
        scores = result.detector_scores
        n_clusters = len(result.clusters) if result.clusters else 0
        print(
            f"{label:<12} {scores[DetectorName.PREDICTIVE_ENTROPY]:>6.3f} "
            f"{round(scores[DetectorName.SEMANTIC_ENTROPY], 3) + 0.0:>6.3f} {n_clusters:>9}  {result.escalated}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
