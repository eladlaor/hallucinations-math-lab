# PARETO_DETECTOR_FRAMEWORK

- [Summary](#summary)
- [Motivation](#motivation)
- [Scope](#scope)
- [Design](#design)
  - [DetectorMetadata contract](#detectormetadata-contract)
  - [Cost / latency reporting](#cost--latency-reporting)
  - [LlamaJudgeDetector](#llamajudgedetector)
  - [Pareto report](#pareto-report)
- [Success criteria (TDD)](#success-criteria-tdd)
- [Build order](#build-order)
- [Out of scope](#out-of-scope)
- [Open questions](#open-questions)

## Summary

Extend the detector layer from "two uncertainty estimators (PE, SE)" to a
**Pareto-framed detector comparison** over three axes: AUROC, $/query,
ms/query. Add a Llama 3.2 3B Instruct LLM-as-judge as a third detector family
(two variants: `qa_only`, `qa_plus_samples`). Each detector self-documents
*signal / assumption / failure modes* via a `DetectorMetadata` contract. The
eval harness emits a Pareto frontier table so the architectural decision
becomes **"where on the frontier do we want to operate?"** rather than
"which detector wins?".

## Motivation

- A single-number "best detector" framing is engineering-naive: real
  deployments operate under cost and latency budgets.
- An LLM-judge baseline gives a meaningful reference ceiling for the SE
  thesis. Without it, the "why not just ask a bigger model?" question is
  unanswered.
- Honest detector docs (signal / assumption / failure modes) force clarity
  about *what each detector is actually measuring* and where it breaks. This
  is the kind of artifact an architect should be able to hand a customer.

## Scope

In:

1. `DetectorMetadata` dataclass on `detector_protocol.py`. Fields:
   `signal: str`, `assumption: str`, `failure_modes: list[str]`,
   `category: DetectorCategory` (uncertainty / external-judgment).
2. `DetectorScore` extended with `cost_usd: float` and `latency_ms: float`.
3. Backfill metadata + cost/latency on `SemanticEntropyDetector`,
   `PredictiveEntropyDetector`, `SemanticEntropyProbeDetector`.
4. `LlamaJudgeDetector` with two configs (`qa_only`, `qa_plus_samples`).
   Yes/No-token logprob scoring. Stubbed judge client during dev.
5. Rename `eval/compute_auroc.py` -> `eval/compute_detector_report.py`.
   Output: per-detector AUROC + $/query + ms/query + `on_frontier: bool`.
6. `knowledge/PARETO_FRONTIER.md` (framing, how to read the report).
7. `knowledge/COST_COMPARISON.md` (Vertex L4/T4 vs RunPod, reasoning for
   staying on Vertex).
8. `serving/llama_3_3b/` vLLM container + Vertex deploy script targeting a
   **T4** endpoint. Provisioned only when ready to run real eval.

Out: see [Out of scope](#out-of-scope).

## Design

### DetectorMetadata contract

```python
class DetectorCategory(StrEnum):
    UNCERTAINTY = "uncertainty"        # PE, SE, SEP
    EXTERNAL_JUDGMENT = "external_judgment"  # LlamaJudge*


@dataclass(frozen=True)
class DetectorMetadata:
    signal: str           # what is measured, in one sentence
    assumption: str       # the correlational link to hallucination
    failure_modes: list[str]
    category: DetectorCategory
```

Detectors expose `metadata: DetectorMetadata` as a class attribute. The eval
harness reads it and renders it in the report.

### Cost / latency reporting

`DetectorScore` gains `cost_usd: float` and `latency_ms: float`. Each
detector measures its own per-call cost+latency at `score()` time:

- PE: ~0 cost (reuses sample logprobs), small CPU.
- SE: NLI inference cost (CPU/GPU), measured via wall-clock; cost = NLI GPU
  time * $/hr / 3600.
- SEP: hidden-state forward pass + probe inference.
- LlamaJudge*: Vertex T4 endpoint $/hr * latency-seconds / 3600, plus
  request overhead.

Cost rates live in `config.py` (per-resource $/hr), so the report stays
correct as prices change.

### LlamaJudgeDetector

Two configs, both implement `Detector`:

- `LlamaJudgeQADetector`: input = (question, greedy_answer). Prompt asks
  "Is this answer correct? Reply Yes or No." Score = P(No) from the
  Yes/No-token logprobs.
- `LlamaJudgeQASamplesDetector`: input = (question, N samples). Prompt
  shows all samples; same Yes/No scoring. Tests whether external judgment
  improves when given sample disagreement.

`requires_samples`: True for both (the qa-only variant uses
`samples.greedy` or `samples[0]`).
`requires_hidden_state`: False.

Judge client is stubbed during dev (records a fake yes/no logprob per
input). Vertex client wired in step 8.

### Pareto report

For each detector i, compute `(auroc_i, cost_i, latency_i)`. Detector i is
**on the Pareto frontier** iff no other detector j dominates it
(`auroc_j >= auroc_i AND cost_j <= cost_i AND latency_j <= latency_i`, with
at least one strict). Report:

```json
{
  "detectors": [
    {
      "name": "semantic_entropy",
      "auroc": 0.78,
      "cost_per_query_usd": 0.002,
      "latency_ms_p50": 800,
      "on_frontier": true,
      "metadata": { ... }
    },
    ...
  ],
  "frontier": ["predictive_entropy", "semantic_entropy", "llama_judge_qa_samples"]
}
```

## Success criteria (TDD)

Preserve existing TDD criteria from CLAUDE.md (SE AUROC >= 0.75, SE - PE >=
0.05, etc.). Add:

1. All detectors report `cost_usd` and `latency_ms` populated and > 0
   (except PE cost which may be ~0).
2. Pareto computation is correct on a synthetic 4-detector fixture
   (unit-tested).
3. LlamaJudge AUROC computable end-to-end against the stubbed client (no
   Vertex dependency for tests).
4. Both Llama variants appear in the Pareto report; their `on_frontier`
   flag matches the synthetic fixture's expected output.

## Build order

1. Plan doc (this file). DONE on commit of this file.
2. `DetectorMetadata` + `DetectorScore` extension in `detector_protocol.py`.
3. Backfill existing detectors.
4. `LlamaJudgeDetector` (stubbed client).
5. `compute_detector_report.py` (rename + Pareto).
6. `knowledge/PARETO_FRONTIER.md`.
7. `knowledge/COST_COMPARISON.md`.
8. `serving/llama_3_3b/` (deferred provisioning).

## Out of scope

- Multi-turn / streaming detector use.
- Fine-tuning the Llama judge.
- Replacing Gemini judge with Llama judge in the downstream escalation path
  (LlamaJudge here is a *detector*, not the orchestration Judge).
- RunPod or any non-Vertex serving.
- Local-only Llama inference (machine has no NVIDIA GPU; CPU inference is
  too slow for the eval scale).

## Open questions

- Cost rate freshness: do we want a `knowledge/COST_COMPARISON.md` update
  cadence, or is one-shot acceptable? Default: one-shot, dated.
- AUPRC alongside AUROC? Useful when hallucination rate is imbalanced.
  Deferred until we see the actual base rate on TriviaQA-1000.
