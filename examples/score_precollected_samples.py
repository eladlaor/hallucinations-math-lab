"""Playground: score hand-written sample sets with PE and SE — no GPU, no LLM.

Three toy prompts, each with N=10 "sampled" answers and made-up token
logprobs, chosen to expose where the detectors agree and where they split:

  - consistent:  every sample says the same thing, confidently.
  - paraphrased: every sample means the same thing, worded differently.
  - divergent:   the samples disagree in meaning (the hallucination shape).

Run:
    uv run python examples/score_precollected_samples.py
    uv run python examples/score_precollected_samples.py --nli deberta

`--nli exact` (default) clusters by normalized string equality — instant, no
download, but it cannot see that two paraphrases mean the same thing, so the
"paraphrased" row gets a high SE. `--nli deberta` uses the real bidirectional
NLI model (microsoft/deberta-large-mnli, ~1.6 GB download on first run, CPU
is fine) and collapses those paraphrases into one meaning cluster. Comparing
the two runs is the whole point of Semantic Entropy in one table.
"""

import argparse
import logging

from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
)
from hallucinations_math_lab.services.nli import NLIScorer

NLI_EXACT = "exact"
NLI_DEBERTA = "deberta"
SE_THRESHOLD_NATS = 1.0


class ExactMatchNLI:
    """Stand-in NLI scorer: two answers 'entail each other' iff they are the
    same string after case/whitespace/punctuation normalization."""

    def bidirectional_entails(self, a: str, b: str) -> bool:
        return _normalize(a) == _normalize(b)


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nli", choices=[NLI_EXACT, NLI_DEBERTA], default=NLI_EXACT)
    args = parser.parse_args()

    scorer = ExactMatchNLI() if args.nli == NLI_EXACT else NLIScorer()
    detectors = [PredictiveEntropyDetector(), SemanticEntropyDetector(scorer=scorer)]

    print(f"NLI = {args.nli}   escalation: SE > {SE_THRESHOLD_NATS} nats\n")
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
