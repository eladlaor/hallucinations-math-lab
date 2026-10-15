# hallucinations-math-lab

A playground for **math-based response evaluators** that flag when an LLM is
likely hallucinating — Predictive Entropy, Semantic Entropy, and Semantic
Entropy Probes — plus an optional "escalate to a stronger judge" step.

Import name: `hallucinations_math_lab` · Python 3.12–3.13 · MIT license

## Table of contents

- [Summary](#summary)
- [Quickstart: no GPU, no LLM](#quickstart-no-gpu-no-llm)
- [What can I experiment with?](#what-can-i-experiment-with)
- [How do the three detectors differ?](#how-do-the-three-detectors-differ)
- [How do I score samples from my own model?](#how-do-i-score-samples-from-my-own-model)
- [How do I combine detectors when deciding to escalate?](#how-do-i-combine-detectors-when-deciding-to-escalate)
- [How do I add my own detector?](#how-do-i-add-my-own-detector)
- [How do I measure whether a detector actually works?](#how-do-i-measure-whether-a-detector-actually-works)
- [How do I run the live demo against a real model?](#how-do-i-run-the-live-demo-against-a-real-model)
- [Where does the theory live?](#where-does-the-theory-live)
- [Repository layout](#repository-layout)
- [Project status](#project-status)

## Summary

The idea: ask a model the same question several times. If it really knows
the answer, the answers agree in meaning. If it is making something up, they
scatter. This repo turns that idea into numbers you can compute, compare and
tune:

| Detector | What it reads | Cost |
|---|---|---|
| **PE** — Predictive Entropy | the token probabilities of the samples | free (a by-product of generation) |
| **SE** — Semantic Entropy ([Farquhar et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07421-0)) | how many *different meanings* appear across N samples | N generations + NLI checks |
| **SEP** — Semantic Entropy Probe ([Kossen et al., 2024](https://arxiv.org/abs/2406.15927)) | one hidden state of the model, through a trained linear probe | one extra matrix multiply |

When a chosen score crosses a threshold, the answer can be routed to a judge
model (Gemini on Vertex is the reference; any class with
`.answer(prompt) -> (answer, reasoning)` works).

Three ways in, from cheapest to most involved:

1. **Library, no model at all** — score samples you already have (or made up)
   with `evaluate_responses(...)`. Runs on a laptop CPU.
2. **Benchmark loop** — run the detectors over TriviaQA with a real small
   model and get one AUROC per detector.
3. **Live demo** — `python -m demo.cli chat "..."` against a Gemma 4 model,
   locally or on a Vertex endpoint, with Gemini escalation.

## Quickstart: no GPU, no LLM

```bash
git clone https://github.com/eladlaor/hallucinations-math-lab.git
cd hallucinations-math-lab
uv sync --all-extras
uv run python examples/score_precollected_samples.py
```

The script scores three hand-written sets of 10 "samples" each:

```
NLI = exact   escalation: SE > 1.0 nats

scenario         PE     SE  clusters  escalate
consistent    0.050  0.000         1  False
paraphrased   0.600  1.973         8  True
divergent     0.400  1.834         7  True
```

`--nli exact` (the default) decides "same meaning" by string equality, so
ten wordings of "Paris" look like eight different answers — a false alarm.
Now let a real NLI model decide (downloads `microsoft/deberta-large-mnli`,
~1.6 GB, once; CPU is fine):

```bash
uv run python examples/score_precollected_samples.py --nli deberta
```

```
scenario         PE     SE  clusters  escalate
consistent    0.050  0.000         1  False
paraphrased   0.600  0.802         3  False
divergent     0.400  1.834         7  True
```

The paraphrases collapse into a few meaning clusters and stop escalating,
while the genuinely contradictory answers still do. PE never notices the
difference — it only sees token probabilities. That gap is the whole case
for Semantic Entropy.

## What can I experiment with?

| Knob | Where | What to watch |
|---|---|---|
| The "same meaning?" judge | pass any object with `bidirectional_entails(a, b)` to `SemanticEntropyDetector(scorer=...)`; default is DeBERTa-MNLI (`NLI_MODEL_ID`) | how cluster counts change on paraphrases |
| Number of samples N, temperature | `SAMPLE_N`, `SAMPLE_TEMPERATURE` in `.env` | SE resolution: with N=10 the maximum is ln 10 ≈ 2.30 nats |
| Escalation threshold | `SE_THRESHOLD`, or `escalation_threshold=` | escalation rate vs. missed hallucinations |
| Escalation rule over several detectors | `AnyAbove`, `AllAbove`, `SingleDetector` | see [below](#how-do-i-combine-detectors-when-deciding-to-escalate) |
| Your own detector | the `Detector` protocol; [`examples/custom_detector.py`](examples/custom_detector.py) | its AUROC next to PE and SE |
| Which small model | `VLLM_MODEL_ID`, registry in [`demo/model_registry.py`](demo/model_registry.py) | how detectors behave per model size |
| The question set | [`eval/datasets/`](eval/datasets/) — TriviaQA loader + an adversarial set | AUROC on easy vs. trick questions |
| SEP probe | [`eval/train_sep_probe.py`](eval/train_sep_probe.py) | how much of SE's signal a single forward pass recovers |

All settings are read from `.env` (copy [`.env.example`](.env.example)); every
key is documented there.

## How do the three detectors differ?

Each one rests on a different assumption about *where* the hallucination
signal lives — in token probabilities, in the meaning of the samples, or
already in the model's hidden state. Those assumptions decide when each one
works and when it silently fails.

### Predictive Entropy (PE) — the token-probability baseline

- **What it measures.** The average, across samples, of each sample's
  length-normalized negative log-likelihood:
  `H_pred = (1/N) Σ_i ( -(1/T_i) Σ_t log p(tok_{i,t}) )`. With one sample it
  is just that sample's average per-token surprise.
- **Assumption.** When the model is wrong, it was already unsure about the
  tokens it produced. The signal is purely about *wording*, not meaning.
- **Fails when.** (a) The model is confidently wrong — every token is
  high-probability, PE stays low. (b) Answers are very short, so dividing by
  a tiny token count lets one unlikely token dominate. (c) There are many
  valid ways to word the same correct answer: PE reads that variety of
  wording as uncertainty even though every sample means the same thing.
- **Cost.** Nothing beyond generation — token probabilities come for free.

### Semantic Entropy (SE) — entropy over meanings

- **What it measures.** Sample N answers at temperature > 0, group them into
  meaning clusters (two answers share a cluster when an NLI model says each
  one entails the other), then take the entropy of the cluster sizes:
  `SE = -Σ_c (|c|/N) ln(|c|/N)` in nats. One cluster → 0; N distinct
  meanings → ln N.
- **Assumption.** When the model is making things up, independent samples
  disagree in *meaning*, even if each sounds fluent. It also assumes the NLI
  model is a good judge of "do these two answers mean the same thing?".
- **Fails when.** (a) Answers are long — NLI gets unreliable past a few
  sentences, so clusters split or merge wrongly. (b) The NLI model misses a
  paraphrase, inflating SE. (c) N is small: with N=10 rare divergent
  meanings can be missed entirely. (d) The task has no clusterable answer
  (essays, code), so "same meaning" is ill-defined.
- **Cost.** N generations plus up to N × (number of clusters) NLI calls.

### Semantic Entropy Probe (SEP) — SE from one forward pass

- **What it measures.** A single linear layer over the model's last-layer
  hidden state at the final prompt token, trained to predict SE. Its output
  is in nats, so SE's threshold carries over.
- **Assumption.** The model's internal state *before it answers* already
  encodes how scattered its answers will be, in a linearly readable way —
  a much stronger claim than SE's.
- **Fails when.** (a) Used on a different model than it was trained on —
  probes do not transfer. (b) Prompts are adversarial or unlike the training
  set. (c) The provider does not expose hidden states (most hosted APIs).
  (d) The SE labels it was trained on were themselves miscalibrated.
- **Cost.** One matrix-vector multiply — the cheapest by far.

**How they relate.** PE is the floor (free, wording-only). SE is the signal
ceiling at real cost. SEP tries to buy most of SE's signal at PE's price by
moving the work into offline probe training. The benchmark loop measures
whether that trade pays off for a given model and dataset.

## How do I score samples from my own model?

Collect N samples with per-token logprobs from any LLM, wrap them, and call
`evaluate_responses`. The library never calls an LLM on this path.

```python
from hallucinations_math_lab import (
    DetectorName, PredictiveEntropyDetector, Sample, SampleSet,
    SemanticEntropyDetector, TokenLogprob, evaluate_responses,
)

samples = SampleSet(
    prompt="Who wrote 1984?",
    samples=[
        Sample(text=text, token_logprobs=[TokenLogprob(token=t, logprob=lp) for t, lp in toks])
        for text, toks in my_llm_samples  # your N completions
    ],
)

result = evaluate_responses(
    prompt="Who wrote 1984?",
    samples=samples,
    detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector()],
    escalation_detector=DetectorName.SEMANTIC_ENTROPY,
    escalation_threshold=1.0,
    greedy_answer="George Orwell.",  # your model's temperature-0 answer
    judge=None,                      # or any object with .answer(prompt) -> (str, str)
)

result.detector_scores  # {predictive_entropy: ..., semantic_entropy: ...}
result.clusters         # the meaning clusters SE found
result.escalated        # did the rule fire?
result.final_answer     # greedy answer, or the judge's if it escalated
```

[`examples/langrag_integration.py`](examples/langrag_integration.py) shows the
translation layer for an existing RAG stack. If you would rather plug in a
model and let the library drive sampling, implement the `SamplingProvider`
protocol (`sample`, `greedy`, `pooled_hidden_state`) and use
`ResponseEvaluator(slm=...).evaluate(prompt)`.

## How do I combine detectors when deciding to escalate?

Pass an `escalation_policy`; it takes precedence over the single
`escalation_detector` / `escalation_threshold` pair:

```python
from hallucinations_math_lab import AnyAbove, DetectorName

result = evaluate_responses(
    ...,
    escalation_policy=AnyAbove({
        DetectorName.SEMANTIC_ENTROPY: 1.0,
        DetectorName.PREDICTIVE_ENTROPY: 2.0,
    }),
)
result.escalation_policy_desc  # "ANY(semantic_entropy > 1.0, predictive_entropy > 2.0)"
```

**The misconception to avoid:** using PE as a cheap pre-filter in front of
SE (`AllAbove({PE, SE})`). PE misses confident hallucinations, so requiring
it to fire first would *suppress* escalations SE would have caught. PE may
only add escalations (`AnyAbove`), never veto them.

## How do I add my own detector?

Write a class with a `name`, a `metadata` self-description (signal,
assumption, failure modes), two flags (`requires_samples`,
`requires_hidden_state`) and a `score(ctx) -> DetectorScore` method — higher
score means "more likely hallucinating". No base class needed.

```bash
uv run python examples/custom_detector.py
```

runs a toy "fraction of samples that disagree with the majority" detector
next to PE and SE. To make one permanent: add its id to `DetectorName` in
`src/hallucinations_math_lab/constants.py`, put it under
`src/hallucinations_math_lab/services/`, register it in
`_build_default_detectors` (`evaluator.py`), and add a unit test. The
benchmark and AUROC scripts pick it up with no further changes. See the
[Developer Guide](knowledge/usage_guides/DEVELOPER_GUIDE.md#how-do-i-add-a-detector).

## How do I measure whether a detector actually works?

The benchmark loop labels each question "hallucinated" when the model's
greedy answer contains none of the accepted answers, then reports one AUROC
per detector — the chance that a random hallucinated answer scores higher
than a random correct one (0.5 = coin flip, 1.0 = perfect).

```bash
cp .env.example .env    # defaults: local Gemma 4 E2B via transformers, PE + SE

# 1. Question set (fixed seed)
uv run python -m eval.datasets.load_triviaqa --n 200 --out eval/datasets/triviaqa_200.jsonl

# 2. Run the detectors (needs a model: GPU or Apple Silicon recommended; see the User Guide)
uv run python -m eval.run_benchmark --in eval/datasets/triviaqa_200.jsonl \
    --out eval/results/triviaqa_200.jsonl --no-judge

# 3. Score them
uv run python -m eval.compute_auroc --in eval/results/triviaqa_200.jsonl
```

```
{ "n_rows": <scorable questions>, "n_hallucinations": <wrong greedy answers>,
  "auroc": { "predictive_entropy": <0..1>, "semantic_entropy": <0..1> },
  "criteria": { "se_auroc": {...}, "se_minus_pe_auroc": {...} } }
```

The numbers are yours to find — that is the experiment. The project's own
targets (SE AUROC ≥ 0.75, SE beating PE by ≥ 0.05, a ≥ 5-point accuracy gain
from escalation, and SE separating adversarial prompts) and the script that
checks each one are in the
[QA Guide](knowledge/usage_guides/QA_GUIDE.md#what-does-success-look-like-for-the-detectors);
none has been measured on real hardware yet. The whole loop is covered by
unit tests with a fake model, so you can change it with confidence. Details
and the SEP training step are in the
[User Guide](knowledge/usage_guides/USER_GUIDE.md#how-do-i-run-a-benchmark).

## How do I run the live demo against a real model?

```bash
cp .env.example .env
uv run python -m demo.cli detect "Who wrote 1984?"                        # scores only, JSON
uv run python -m demo.cli chat "Who was the first woman to walk on Mars?" # + Gemini escalation
```

`detect` needs only the model (local by default). `chat` also calls Gemini on
Vertex, so it needs `GCP_PROJECT_ID` and Google credentials. To serve the
model with vLLM on a Vertex GPU endpoint instead, follow the
[DevOps Guide](knowledge/usage_guides/DEVOPS_GUIDE.md).

## Where does the theory live?

| Doc | Read it to learn |
|---|---|
| [Semantic Entropy walkthrough](knowledge/theory/SEMANTIC_ENTROPY_WALKTHROUGH.md) | the math step by step, each step next to the code that implements it |
| [Caveats: theory vs. production](knowledge/CAVEATS.md) | where the clean story breaks in practice |
| [Glossary](knowledge/GLOSSARY.md) | terms such as bidirectional entailment |
| [`knowledge/theory/artifacts/`](knowledge/theory/artifacts/) | two standalone HTML explainers (open in a browser) |
| [Pareto detector framework plan](knowledge/plans/PARETO_DETECTOR_FRAMEWORK.md) | the planned cost / latency / accuracy comparison |

## Repository layout

```
src/hallucinations_math_lab/   # the library — imports nothing from demo/
  evaluator.py                 # evaluate_responses + ResponseEvaluator
  escalation.py                # SingleDetector, AnyAbove, AllAbove
  constants.py, config.py      # DetectorName & co.; env-driven settings
  custom_types/                # Sample, SampleSet, Cluster, DetectionResult
  services/                    # detector protocol, PE, SE, SEP, NLI wrapper
  judges/                      # Judge protocol + GeminiJudge
demo/                          # CLI, SLM providers (vLLM, transformers), model registry
examples/                      # runnable playground scripts
eval/                          # datasets, benchmark, success-criteria scoring, SEP probe training
tests/unit/                    # math, dispatch, escalation, providers, benchmark loop
serving/, terraform/           # vLLM container and GCP infrastructure
knowledge/                     # guides, theory, caveats, plans
```

## Project status

Pre-1.0 and changing. The library API (`evaluate_responses`,
`ResponseEvaluator`, the three detectors, the `Judge` protocol and the
escalation policies) is what the tests pin down. An LLM-as-judge detector
and cost / latency reporting in the benchmark are planned — see the
[Pareto plan](knowledge/plans/PARETO_DETECTOR_FRAMEWORK.md). Changes are
tracked in [CHANGELOG.md](CHANGELOG.md). Running the tests and the release
checks is covered in the [QA Guide](knowledge/usage_guides/QA_GUIDE.md).
