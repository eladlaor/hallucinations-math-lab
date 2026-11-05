# hallucinations-math-lab

A playground for **math-based response evaluators** that flag when an LLM is
likely hallucinating — Predictive Entropy, Semantic Entropy, and Semantic
Entropy Probes — plus an optional "escalate to a stronger judge" step.

Import name: `hallucinations_math_lab` · Python 3.12–3.13 · MIT license

## Table of contents

- [Summary](#summary)
- [Which model am I testing, and how do I choose it?](#which-model-am-i-testing-and-how-do-i-choose-it)
- [Quickstart: no GPU, no LLM](#quickstart-no-gpu-no-llm)
- [What do I need before running a real model?](#what-do-i-need-before-running-a-real-model)
- [What does a single run actually do?](#what-does-a-single-run-actually-do)
- [How do the three detectors differ?](#how-do-the-three-detectors-differ)
- [How do I ask one question?](#how-do-i-ask-one-question)
- [What does a Semantic Entropy number mean?](#what-does-a-semantic-entropy-number-mean)
- [When does it escalate, and how do I tune it?](#when-does-it-escalate-and-how-do-i-tune-it)
- [How do I evaluate one model?](#how-do-i-evaluate-one-model)
- [How do I compare several models?](#how-do-i-compare-several-models)
- [How do I train and use a Semantic Entropy Probe?](#how-do-i-train-and-use-a-semantic-entropy-probe)
- [How do I score samples from my own model?](#how-do-i-score-samples-from-my-own-model)
- [How do I add my own detector?](#how-do-i-add-my-own-detector)
- [What can I experiment with?](#what-can-i-experiment-with)
- [What does it look like on real questions?](#what-does-it-look-like-on-real-questions)
- [What are the limitations?](#what-are-the-limitations)
- [Troubleshooting](#troubleshooting)
- [Where does the theory live?](#where-does-the-theory-live)
- [Repository layout](#repository-layout)
- [Project status](#project-status)

## Summary

The idea behind Semantic Entropy: ask a model the same question several
times. If it really knows the answer, the answers agree in meaning. If it is
making something up, they scatter. This repo turns that idea into numbers you
can compute, compare and tune — plus two neighbours: a cheaper baseline (PE)
and a probe that predicts the same signal without the repeated sampling (SEP):

| Detector | What it reads | Cost |
|---|---|---|
| **PE** — Predictive Entropy | the token probabilities of the samples | free (a by-product of generation) |
| **SE** — Semantic Entropy ([Farquhar et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07421-0)) | how many *different meanings* appear across N samples | N generations + NLI checks |
| **SEP** — Semantic Entropy Probes ([Kossen et al., 2024](https://arxiv.org/abs/2406.15927)) | one hidden state of the model, through a trained linear probe | one extra matrix multiply |

When a chosen score crosses a threshold, the answer can be routed to a judge
model (Gemini on Vertex is the reference; any class with
`.answer(prompt) -> (answer, reasoning)` works).

Three ways to use the repo, from cheapest to most involved:

| Way in | You bring | You get back | You need | Start with |
|---|---|---|---|---|
| **1. Library** — score answers you already have | nothing to start (the example ships three sample sets); later, N sampled answers with token logprobs from any LLM | PE and SE scores, the meaning clusters, and an escalate yes/no — the repo never calls a model | any laptop ([tier 0](#what-hardware-do-i-need-and-why)) | [`examples/score_precollected_samples.py`](#quickstart-no-gpu-no-llm), then [`evaluate_responses(...)`](#how-do-i-score-samples-from-my-own-model) |
| **2. Benchmark** — measure how well each detector spots wrong answers | nothing — 200 TriviaQA questions download on first run; you pick the Gemma 4 model(s) in `.env` | one AUROC per detector (per model), with confidence intervals | a machine that fits the model ([tiers 1–3](#what-hardware-do-i-need-and-why)), or a remote vLLM / Vertex server | [`eval.run_benchmark`](#how-do-i-evaluate-one-model), or [`--sweep` + `eval.compare_models`](#how-do-i-compare-several-models) |
| **3. Live demo** — ask one question end to end | a single question typed on the command line | the detector scores and an answer — the model's own, or Gemini's when the score crosses the threshold | as for the benchmark, plus Google credentials for the Gemini judge (`chat` only) | [`demo.cli detect` / `demo.cli chat`](#how-do-i-ask-one-question) |

Everything is configured in `.env` (copy [`.env.example`](.env.example);
every key is documented there).

## Which model am I testing, and how do I choose it?

The model under test is a **Gemma 4 instruction-tuned model**, in its native
bf16 precision (nothing is quantized). It must be one of the four entries in
[`demo/model_registry.py`](demo/model_registry.py), which pins how each one
is served:

| Model | Role | Hardware it is pinned to |
|---|---|---|
| `google/gemma-4-E2B-it` | **default** — runs in-process on a laptop | laptop / any single GPU |
| `google/gemma-4-E4B-it` | default for the Vertex deployment | 1× L4 ("one L4") |
| `google/gemma-4-26B-A4B-it` | larger comparison point | 1× A100 80GB ("one A100, eighty gig") |
| `google/gemma-4-31B-it` | largest comparison point | 2× A100 80GB ("two A100s, eighty gig each") |

**One model or several** — two `.env` keys, one per mode:

| You want to | Set | Run |
|---|---|---|
| test **one** model (demo, benchmark, SEP training) | `VLLM_MODEL_ID=google/gemma-4-E2B-it` | `demo.cli`, `eval.run_benchmark --out file.jsonl` |
| **compare** several models | `TARGET_MODEL_IDS=["google/gemma-4-E2B-it","google/gemma-4-E4B-it"]` (at least two) | `eval.run_benchmark --sweep --out dir/`, then `eval.compare_models` |

`VLLM_MODEL_ID` is used whichever provider runs the model — despite the
name, not only for vLLM.

**How the model is reached** — `SLM_PROVIDER`:

- `hf_transformers` (default) — downloads the model from Hugging Face and
  runs it in this process, on a CUDA GPU if present, otherwise on the CPU
  (slow, but fine for a handful of questions). Exposes hidden states, so SEP
  works.
- `vllm` — calls an OpenAI-compatible vLLM server at `VLLM_BASE_URL`: one you
  run (`VLLM_AUTH=api_key`) or a Vertex AI endpoint (`VLLM_AUTH=google`,
  which sends a refreshing Google access token). In a comparison, give each
  model its own server in `TARGET_MODEL_BASE_URLS`. SEP needs a second
  server running the embed runner, set in `VLLM_EMBED_BASE_URL`. Setup is in
  the [DevOps Guide](knowledge/usage_guides/DEVOPS_GUIDE.md).

Both providers feed SEP the same input (the hidden state at the last token
of the chat-formatted prompt), so a probe trained through one works through
the other — but only for the same model.

**The judge** is separate from the model under test: `JUDGE_MODEL_ID`
(default `gemini-3.5-flash-lite`), Gemini on Vertex AI through the
`google-genai` SDK. It returns a schema-validated reasoning and answer,
reasoning first so the answer is conditioned on it. Gemini 3.5 on Vertex
ignores temperature, so the judge sends no sampling parameters;
`JUDGE_THINKING_LEVEL` (default `LOW`) sets its reasoning depth.

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
paraphrased   0.600  0.000         1  False
divergent     0.400  1.834         7  True
```

The paraphrases collapse into one meaning cluster and stop escalating,
while the genuinely contradictory answers still do. PE never notices the
difference — it only sees token probabilities. That gap is the whole case
for Semantic Entropy.

## What do I need before running a real model?

```bash
uv sync --all-extras
cp .env.example .env
```

| You want to | You also need |
|---|---|
| the demo, the benchmark or SEP training with the local provider | disk for the model download; a GPU strongly recommended |
| the same against vLLM | a reachable vLLM server (`VLLM_BASE_URL`), or a Vertex endpoint plus `VLLM_AUTH=google` — see the [DevOps Guide](knowledge/usage_guides/DEVOPS_GUIDE.md#pointing-the-client-at-vertex) |
| `chat`, or a benchmark with the judge | `GCP_PROJECT_ID` in `.env` and `gcloud auth application-default login` |

Without Google credentials, escalation fails fast — there is no silent
fallback to another provider.

## What does a single run actually do?

It asks the model the same question N times (`SAMPLE_N`, default 10),
measures how much the answers agree **in meaning**, and either returns the
model's own answer or hands the question to the judge.

1. Sample N answers at temperature > 0 (`SAMPLE_TEMPERATURE`), keeping
   per-token logprobs.
2. Take one more answer at temperature 0 — the **greedy answer**, which is
   what you get back if nothing escalates.
3. Run every detector in `ENABLED_DETECTORS` (default PE and SE; SEP is
   opt-in).
4. Apply the escalation rule (default: `SE > SE_THRESHOLD`). If it fires and
   a judge is configured, the judge answers instead.

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
  meaning clusters, then take the entropy of the cluster sizes:
  `SE = -Σ_c (|c|/N) ln(|c|/N)` in nats. One cluster → 0; N distinct
  meanings → ln N. Two answers share a cluster when an NLI model says each
  one entails the other, with the question placed in front of both, so a
  bare "Paris" and "The capital of France is Paris." are judged in context.
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
moving the work into offline probe training. The benchmark measures whether
that trade pays off for a given model and dataset.

**Reading PE and SE together.** PE measures how sure the model was about its
*words*; SE measures whether it gave the same *answer*:

- **PE low, SE high** — fluent but inconsistent: the classic hallucination
  shape, and exactly what PE alone misses.
- **PE high, SE low** — hedged wording, same answer every time: usually
  fine, and exactly what PE alone false-alarms on.

## How do I ask one question?

```bash
uv run python -m demo.cli detect "Who wrote the novel 1984?"   # scores only, never escalates
uv run python -m demo.cli chat   "Who wrote the novel 1984?"   # scores + judge if needed
```

There are no flags; the model, sample count, threshold and detectors all
come from `.env`. `detect` needs only the model; `chat` also calls the judge.

`detect` prints the full `DetectionResult` as JSON: the prompt, the greedy
answer, the N sample texts, `detector_scores`, the SE `clusters` (each with
its member indices and a representative text), and the escalation decision.

`chat` prints two blocks, plus a third when the judge answered:

```
──────────── Detection ────────────
PE = 0.412  SE = 0.000
escalation_detector = semantic_entropy  threshold = 1.000  escalated = False  source = small_model_greedy
──────────── Final Answer ─────────
George Orwell.
```

- **PE / SE** — the detector scores, in nats. SEP appears between them when
  enabled.
- **escalated** — whether the escalation rule fired.
- **source** — `small_model_greedy` (the greedy answer) or `judge_model`.
- **Judge Reasoning** — the third block, printed only when the judge answered.

## What does a Semantic Entropy number mean?

SE is the entropy, in nats, of how the N answers split across meaning
clusters: 0 means every answer meant the same thing, and the maximum is
ln N (≈ 2.30 for N = 10), meaning every answer was different.

| Cluster split (N = 10) | SE (nats) | Reading |
|---|---|---|
| 10 | 0.00 | full agreement |
| 9 + 1 | 0.33 | one stray answer |
| 5 + 5 | 0.69 | two camps |
| 7 + 2 + 1 | 0.80 | real wobble |
| 4 + 3 + 2 + 1 | 1.28 | scattered |
| 1 × 10 | 2.30 | every answer different |

These are reference points, not verdicts — the threshold that separates
"fine" from "hallucinating" depends on the model and the question set, and
finding it is what the benchmark is for. The full derivation is in the
[walkthrough](knowledge/theory/SEMANTIC_ENTROPY_WALKTHROUGH.md).

## When does it escalate, and how do I tune it?

It escalates when the escalation rule fires — by default
`semantic_entropy > SE_THRESHOLD` (`ESCALATION_DETECTOR`, `SE_THRESHOLD`,
default `1.0`). The judge then receives **only the original question** (not
the samples), answers it independently, and its answer and reasoning replace
the greedy answer.

- **Lower threshold** → escalates more → fewer hallucinations get through,
  more judge calls (cost, latency).
- **Higher threshold** → escalates less → cheaper and faster, more
  hallucinations through.

Pick it from data: run the benchmark, then look at the rows in the results
file — each has every detector score and whether the greedy answer was
right — and choose the point on that trade-off you can afford. AUROC tells
you whether *any* threshold can separate the two groups well. A high
escalation rate is itself a finding: either the threshold is too tight, or
the questions are outside what the model knows.

**Combining detectors.** In the library, pass an `escalation_policy`; it
takes precedence over the single `escalation_detector` /
`escalation_threshold` pair:

```python
from hallucinations_math_lab import AnyAbove, DetectorName

result = evaluate_responses(
    ...,
    escalation_policy=AnyAbove(
        {
            DetectorName.SEMANTIC_ENTROPY: 1.0,
            DetectorName.PREDICTIVE_ENTROPY: 2.0,
        }
    ),
)
result.escalation_policy_desc  # "ANY(semantic_entropy > 1.0, predictive_entropy > 2.0)"
```

**The misconception to avoid:** using PE as a cheap pre-filter in front of
SE (`AllAbove({PE, SE})`). PE misses confident hallucinations, so requiring
it to fire first would *suppress* escalations SE would have caught. PE may
only add escalations (`AnyAbove`), never veto them.

## How do I evaluate one model?

The benchmark labels each question "hallucinated" when the model's greedy
answer contains none of the accepted answers, then reports one AUROC per
detector — the chance that a random hallucinated answer scores higher than a
random correct one (0.5 = coin flip, 1.0 = perfect). It runs the model in
`VLLM_MODEL_ID`.

1. **Questions come with the repo.** Without `--in`, the benchmark uses 200
   TriviaQA questions (positions 500–699 of a fixed-seed shuffle), downloaded
   once (~7 MB) to `data/questions/`. The first 500 are kept for SEP probe
   training, so the two sets never share a question. For a different slice,
   build one and pass it with `--in`:
   ```bash
   uv run python -m eval.datasets.load_triviaqa --offset 700 --n 1000 \
       --out data/questions/my_slice.jsonl
   ```
   [`eval/datasets/adversarial.jsonl`](eval/datasets/adversarial.jsonl)
   ships in the repo: a few false-premise and unknowable questions. They
   have no answer key, so AUROC skips them — read their rows by hand.
2. **Run the detectors.** `--limit 10` is a good first smoke test:
   ```bash
   uv run python -m eval.run_benchmark \
       --out eval/results/triviaqa_200.jsonl --no-judge
   ```
   Each output row carries the question, answer key, greedy and final
   answers, `detector_scores` (every enabled detector), `escalated`, the
   cluster count, the raw samples, and the run's `model_id`, `sample_n` and
   `sample_temperature`. Rows are written as they finish; the run stops on
   the first error, keeping the rows already written.
3. **Score it.**
   ```bash
   uv run python -m eval.compute_auroc --in eval/results/triviaqa_200.jsonl
   ```
   ```
   { "run_config": { "model_id": ..., "sample_n": ..., "sample_temperature": ... },
     "n_rows": <scorable questions>, "n_hallucinations": <wrong greedy answers>,
     "detectors": { "semantic_entropy": { "auroc": <0..1>, "ci": { "low": ..., "high": ... } }, ... },
     "criteria": { "se_auroc": {...}, "se_minus_pe_auroc": {...} } }
   ```
   A row counts as hallucinated when the greedy answer contains none of the
   accepted answers (case-insensitive substring — the usual TriviaQA rule).
   Each AUROC comes with a bootstrap 95% CI; with SE and PE both enabled it
   also reports SE's margin over PE with a paired-bootstrap CI, each against
   its target.
4. **Measure what escalation buys.** Needs a run *with* the judge (drop
   `--no-judge` in step 2):
   ```bash
   uv run python -m eval.escalation_uplift --in eval/results/triviaqa_200.jsonl
   ```
   Prints greedy accuracy, final accuracy, the uplift in percentage points
   with a 95% CI and McNemar's p-value, how many answers escalation fixed
   and broke, and the mean answer length on both sides (with a warning when
   the judge's answers are much longer, since substring matching rewards
   length). It refuses runs where nothing was escalated or the judge never
   answered, since the uplift there is zero by construction.
5. **Check the adversarial set.** Run step 2 on
   `eval/datasets/adversarial.jsonl` too, with the same model and settings,
   then:
   ```bash
   uv run python -m eval.adversarial_separation \
       --adversarial eval/results/adversarial.jsonl \
       --reference eval/results/triviaqa_200.jsonl
   ```
   A one-sided Mann-Whitney U test of whether SE (or `--detector <name>`) is
   higher on the adversarial prompts than on correctly answered TriviaQA
   questions, plus the effect size: the probability that a random
   adversarial prompt outscores a random correct one.

The scoring scripts refuse to mix rows from different runs, and
`adversarial_separation` refuses two runs made with different models or
sampling settings. The numbers are yours to find — that is the experiment.
The project's own targets (SE AUROC ≥ 0.75, SE beating PE by ≥ 0.05 with a CI
that excludes zero, a significant ≥ 5-point accuracy gain from escalation,
and SE separating adversarial prompts with a large effect size) and why each
needs more than a point estimate are in the
[QA Guide](knowledge/usage_guides/QA_GUIDE.md#what-does-success-look-like-for-the-detectors);
none has been measured on real hardware yet.

## How do I compare several models?

List the models in `TARGET_MODEL_IDS`, run the same question set through all
of them with `--sweep`, then compare:

```bash
# .env: TARGET_MODEL_IDS=["google/gemma-4-E2B-it","google/gemma-4-E4B-it"]

uv run python -m eval.run_benchmark --sweep \
    --out eval/results/sweep/ --no-judge

uv run python -m eval.compare_models --in eval/results/sweep/
```

```
200 questions, sampling {'sample_n': 10, 'sample_temperature': 1.0}

model                  scored  wrong  accuracy  predictive_entropy AUROC [95% CI]  semantic_entropy AUROC [95% CI]
google/gemma-4-E2B-it  ...     ...    ...       ...                                ...
google/gemma-4-E4B-it  ...     ...    ...       ...                                ...
```

- **What `--sweep` does.** It runs the models one after another, writing
  `<out>/<model slug>.jsonl` per model (e.g. `google-gemma-4-e2b-it.jsonl`).
  Only one model is loaded at a time. Before scoring anything it checks that
  every id is in the registry and, with SEP enabled, that every model has
  its own probe — so a sweep does not fail hours in.
- **What `compare_models` checks.** It refuses files that cover different
  questions, used different sampling settings, or repeat a model. `--json`
  prints the full report, including each model's CIs. Each file is also a
  normal single-model run, so `compute_auroc`, `escalation_uplift` and
  `adversarial_separation` work on it as well.
- **Hardware.** Every listed model must run on what you have. With the local
  provider, the two A100 models do not fit a laptop. With vLLM, each model
  needs its own server: map them in `TARGET_MODEL_BASE_URLS`, e.g.
  `{"google/gemma-4-E4B-it": "http://gpu-box:8080/v1"}`; models not listed
  there use `VLLM_BASE_URL`.
- **What a difference means.** Read the CIs before ranking: two models whose
  AUROC intervals overlap heavily are not shown to differ on this question
  set.

## How do I train and use a Semantic Entropy Probe?

A probe is trained per model: for each calibration question it computes SE
(the label) and takes the model's hidden state at the last prompt token (the
input), then fits a ridge regression — a linear fit with a penalty that
keeps the weights small.

1. **Calibration questions come with the repo.** Without `--calibration-set`,
   training uses the first 500 questions of the same fixed-seed shuffle,
   downloaded once to `data/questions/`. They never overlap the default
   benchmark questions. To use your own, pass any slice that does not
   overlap your benchmark.
2. **Train** for the model in `VLLM_MODEL_ID`. Needs a provider that exposes
   hidden states (the default `hf_transformers` does):
   ```bash
   uv run python -m eval.train_sep_probe
   ```
   It writes `data/probes/<model slug>.pt` (or `--output-path`), which is where the evaluator
   looks for that model unless `SEP_PROBE_PATH` says otherwise. Next to it,
   it writes `<model slug>.train_ids.json`, the ids of the questions it was
   trained on. Before saving, it logs the probe's 5-fold cross-validated
   Pearson r against SE (mean ± std, `--cv-folds`): how well the one-pass
   probe stands in for the ten-sample SE on questions it never saw.
3. **Enable it** in `.env` and benchmark on the disjoint slice:
   ```
   ENABLED_DETECTORS=["predictive_entropy","semantic_entropy_probe","semantic_entropy"]
   ```
   `run_benchmark` checks the benchmark questions against the probe's
   `.train_ids.json` and refuses to start on any overlap, or if the file is
   missing.

A probe trained on one model is meaningless on another. To include SEP in a
model comparison, train one probe per model (rerun step 2 with each
`VLLM_MODEL_ID`) and leave `SEP_PROBE_PATH` unset — `--sweep` refuses a
single explicit probe path.

## How do I score samples from my own model?

Collect N samples with per-token logprobs from any LLM, wrap them, and call
`evaluate_responses`. The library never calls an LLM on this path.

```python
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
    evaluate_responses,
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
    judge=None,  # or any object with .answer(prompt) -> (str, str)
)

result.detector_scores  # {predictive_entropy: ..., semantic_entropy: ...}
result.clusters  # the meaning clusters SE found
result.escalated  # did the rule fire?
result.final_answer  # greedy answer, or the judge's if it escalated
```

[`examples/langrag_integration.py`](examples/langrag_integration.py) shows the
translation layer for an existing RAG stack. If you would rather plug in a
model and let the library drive sampling, implement the `SamplingProvider`
protocol (`model_id`, `sample`, `greedy`, `pooled_hidden_state`) and use
`ResponseEvaluator(slm=...).evaluate(prompt)`.

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
benchmark, AUROC and comparison scripts pick it up with no further changes.
See the
[Developer Guide](knowledge/usage_guides/DEVELOPER_GUIDE.md#how-do-i-add-a-detector).

## What can I experiment with?

| Knob | Where | What to watch |
|---|---|---|
| Which model, or which set of models | `VLLM_MODEL_ID`; `TARGET_MODEL_IDS` with `--sweep` | how detectors behave per model size |
| The "same meaning?" judge | pass any object with `bidirectional_entails(a, b)` to `SemanticEntropyDetector(scorer=...)`; default is DeBERTa-MNLI (`NLI_MODEL_ID`) | how cluster counts change on paraphrases |
| Number of samples N, temperature | `SAMPLE_N` (2–50), `SAMPLE_TEMPERATURE` | SE resolution: with N=10 the maximum is ln 10 ≈ 2.30 nats; every question costs N generations |
| Escalation threshold | `SE_THRESHOLD`, or `escalation_threshold=` | escalation rate vs. missed hallucinations |
| Escalation rule over several detectors | `AnyAbove`, `AllAbove`, `SingleDetector` | see [above](#when-does-it-escalate-and-how-do-i-tune-it) |
| Your own detector | the `Detector` protocol; [`examples/custom_detector.py`](examples/custom_detector.py) | its AUROC next to PE and SE |
| The question set | [`eval/datasets/`](eval/datasets/) — TriviaQA loader + an adversarial set | AUROC on easy vs. trick questions |
| SEP probe | [`eval/train_sep_probe.py`](eval/train_sep_probe.py) | how much of SE's signal a single forward pass recovers |

## What does it look like on real questions?

The numbers below are illustrative, to show the shapes to look for.

**Confident and correct.** "Who wrote the novel 1984?" — all ten samples say
Orwell: one cluster, SE = 0, no escalation, the greedy answer is returned.

**False premise.** "Who was the first woman to walk on Mars?" — the samples
invent different names: many clusters, SE near ln 10, escalation fires, and
the judge can reject the premise instead of inventing an eleventh name.

**Ambiguous question.** "What is the population of Tel Aviv?" — some samples
answer for the city, some for the metro area: a few large clusters, a
middling SE. Here high SE signals an under-specified question, not
ignorance — a reminder that SE measures disagreement, not falsehood.

## What are the limitations?

- **Short answers only.** NLI clustering gets unreliable on long, multi-claim
  answers, so SE is hard to interpret there.
- **The judge can be wrong too.** Escalation lowers the hallucination rate;
  it does not remove it.
- **Single-turn.** Each call is independent; there is no conversation state.
- **The answer key is crude.** Substring matching against aliases mislabels
  some correct-but-differently-phrased answers; this caps measurable AUROC.
- **Only Gemma 4 is registered.** Another model needs a registry entry first
  (see the [Developer Guide](knowledge/usage_guides/DEVELOPER_GUIDE.md#how-do-i-add-a-model-to-the-registry)).
- More in [Caveats: theory vs. production](knowledge/CAVEATS.md).

## Troubleshooting

**Why does startup fail with `error parsing value for field "enabled_detectors"`?**
List and dict settings must be JSON in `.env`:
`ENABLED_DETECTORS=["predictive_entropy","semantic_entropy"]`, not a
comma-separated list. The same holds for `TARGET_MODEL_IDS` and
`TARGET_MODEL_BASE_URLS`.

**Why does `--sweep` fail with "not in the model registry"?**
Every id in `TARGET_MODEL_IDS` must match an entry in
`demo/model_registry.py` exactly, including case.

**Why does `SemanticEntropyProbeDetector` raise `FileNotFoundError`?**
SEP is enabled but no probe exists for the model being run. Train one
([above](#how-do-i-train-and-use-a-semantic-entropy-probe)) or set
`SEP_PROBE_PATH` (single-model runs only).

**Why do I get a connection error to `VLLM_BASE_URL`?**
`SLM_PROVIDER=vllm` but the server is not reachable. Check with
`curl $VLLM_BASE_URL/models`, or switch back to
`SLM_PROVIDER=hf_transformers`. On Vertex, add
`-H "Authorization: Bearer $(gcloud auth print-access-token)"`; a
name-resolution error right after deploying usually means the dedicated
endpoint's DNS name has not propagated yet.

**Why does escalation fail with a credentials error?**
`GCP_PROJECT_ID` is unset or `gcloud auth application-default login` has not
been run for an account with Vertex access. This is deliberate fail-fast.

**Why is SE always about 0?**
The samples are identical, which almost always means the temperature is 0
somewhere. Check `SAMPLE_TEMPERATURE` and, on vLLM, the server's settings.

**Why is SE always very high?**
Clustering is splitting paraphrases. Check `NLI_MODEL_ID`, and compare the
two modes of `examples/score_precollected_samples.py` to see what a healthy
NLI model does.

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
eval/                          # datasets, benchmark (one model or a sweep), model comparison,
                               # success-criteria scoring, SEP probe training
tests/unit/                    # math, dispatch, escalation, providers, benchmark loop
serving/, terraform/           # vLLM container and GCP infrastructure
knowledge/                     # developer / DevOps / QA guides, theory, caveats, plans
```

| Guide | For |
|---|---|
| this README | using the project: models, detectors, demo, benchmark, comparison |
| [Developer Guide](knowledge/usage_guides/DEVELOPER_GUIDE.md) | changing the code |
| [DevOps Guide](knowledge/usage_guides/DEVOPS_GUIDE.md) | serving models with vLLM on Vertex or your own GPU |
| [QA Guide](knowledge/usage_guides/QA_GUIDE.md) | tests, CI and the success criteria |

## Project status

Pre-1.0 and changing. The library API (`evaluate_responses`,
`ResponseEvaluator`, the three detectors, the `Judge` protocol and the
escalation policies) is what the tests pin down. An LLM-as-judge detector
and cost / latency reporting in the benchmark are planned — see the
[Pareto plan](knowledge/plans/PARETO_DETECTOR_FRAMEWORK.md). Changes are
tracked in [CHANGELOG.md](CHANGELOG.md).
