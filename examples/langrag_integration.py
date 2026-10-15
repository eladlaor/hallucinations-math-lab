"""Example: integrating hallucinations_math_lab into a langrag-style project.

This file is illustrative — it depends on `langrag.*` which is not a
declared dependency of hallucinations_math_lab, so it will not import unless langrag is
installed alongside. The point is to show the shape of the integration,
not to be a runnable demo.

The consumer pattern, in one paragraph:

  langrag (or any LLM-native consumer) already owns the LLM. It calls the
  model itself, gets N samples back, and just wants hallucinations_math_lab to score them.
  That's the `evaluate_responses` path — pure scoring, no provider plumbing.
  The consumer is in charge of *what* it sampled, *how* it sampled, and
  *whether* to escalate to its own judge. the library's job is to compute SE / PE
  / SEP over the samples it was handed, and to invoke the supplied judge
  when the chosen detector crosses the configured threshold.

Two integration points worth flagging:

1. Sample collection on the consumer's side. The library needs `text` + a list
   of `(token, logprob)` pairs for each sample. Every modern inference
   API exposes per-token logprobs (OpenAI: `logprobs`+`top_logprobs`; HF
   transformers: `output_scores=True` on .generate; vLLM: same as OpenAI
   over the openai-compat endpoint). The translation is mechanical.

2. SEP is opt-in. It needs a pooled hidden state at the final prompt
   token. Consumers without a way to extract that (e.g. anyone behind a
   chat-completions-only proxy) drop SEP from the detector list — the
   other two detectors still run.

Run this against the SAME slm the rest of langrag uses; don't introduce a
second model just to compute SE. The whole point is to characterize the
uncertainty of the answers langrag is actually shipping.
"""

from typing import Any

# These imports are real and stable for consumer use:
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
)

# --- The langrag side (sketched, not real imports) ------------------------


def langrag_sample_n(prompt: str, n: int) -> list[dict[str, Any]]:
    """Stand-in for whatever langrag uses to drive its LLM.

    Real langrag returns something with `text` and `logprobs` per choice.
    The exact shape varies by backend; the translation to the library's Sample
    is straightforward (see _to_sample_set below)."""
    raise NotImplementedError(
        "Replace this with the langrag inference call, e.g. "
        "langrag.llm.complete(prompt, n=n, temperature=1.0, logprobs=True)."
    )


def langrag_greedy(prompt: str) -> str:
    """Stand-in for langrag's deterministic single-sample decode."""
    raise NotImplementedError("Replace with langrag.llm.complete(prompt, n=1, temperature=0.0).")


# --- The translation layer ------------------------------------------------


def _to_sample_set(prompt: str, raw_choices: list[dict[str, Any]]) -> SampleSet:
    """Convert langrag's per-choice output into the library's SampleSet.

    Assumes each choice has shape:
        {"text": "...", "logprobs": [{"token": "Paris", "logprob": -0.13}, ...]}

    Adapt the field accesses to whatever your stack actually returns.
    The shape the library needs is intentionally narrow: a list of (token,
    logprob) pairs per sample, nothing else."""
    samples = [
        Sample(
            text=choice["text"],
            token_logprobs=[
                TokenLogprob(token=tlp["token"], logprob=tlp["logprob"])
                for tlp in choice["logprobs"]
            ],
        )
        for choice in raw_choices
    ]
    return SampleSet(prompt=prompt, samples=samples)


# --- The scoring entry point ---------------------------------------------


def score_langrag_response(
    prompt: str,
    n: int = 10,
    se_threshold: float = 1.0,
) -> dict[str, Any]:
    """Drive langrag's LLM, score the response with hallucinations_math_lab, return a flat dict
    suitable for logging or insertion into a langrag result envelope.

    Detectors: PE + SE. SEP is omitted because it requires hidden-state
    access that langrag's default backend does not expose. Add a
    SemanticEntropyProbeDetector + a pooled hidden state if your backend
    can supply one (e.g. vLLM with --task embed on a sibling port)."""
    raw_choices = langrag_sample_n(prompt, n=n)
    samples = _to_sample_set(prompt, raw_choices)
    greedy = langrag_greedy(prompt)

    result = evaluate_responses(
        prompt=prompt,
        samples=samples,
        detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector()],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=se_threshold,
        greedy_answer=greedy,
        judge=None,  # langrag may supply its own judge; pass it here to escalate
    )

    return {
        "answer": result.final_answer,
        "se_nats": result.detector_scores[DetectorName.SEMANTIC_ENTROPY],
        "pe_nats": result.detector_scores[DetectorName.PREDICTIVE_ENTROPY],
        "escalated": result.escalated,
        "n_meaning_clusters": len(result.clusters) if result.clusters else None,
    }


# --- Notes for langrag maintainers ---------------------------------------
#
# Threshold tuning. SE > 1.0 nats is a defensible "the model is uncertain"
# line on TriviaQA-style factoid prompts but is workload-dependent. For
# long-form generation, NLI clustering gets noisier and SE inflates
# uniformly — recalibrate against your own labeled set.
#
# Cost. PE is free (it's already paid for by the n samples you collected).
# SE adds one DeBERTa-MNLI call per (sample, cluster_rep) pair, capped at
# n × max_clusters. For n=10 on factoid prompts that's typically 10-30 NLI
# calls per question — runs in <1s on a CPU-only deploy with the default
# DeBERTa-MNLI checkpoint, faster on GPU.
#
# Failure modes worth flagging in your envelope:
#   - PredictiveEntropyDetector raises if no sample has token_logprobs.
#     If langrag's backend ever returns choices without logprobs, the
#     scoring call will fail fast — surface that as a degraded-mode flag
#     rather than swallowing it.
#   - SemanticEntropyDetector raises on an empty sample set. Same guidance.
