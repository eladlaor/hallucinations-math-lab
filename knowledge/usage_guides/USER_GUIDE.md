# User Guide

## Table of Contents

- [Summary](#summary)
- [What does a single run actually do?](#what-does-a-single-run-actually-do)
- [Which model am I running?](#which-model-am-i-running)
- [What do I need before running anything?](#what-do-i-need-before-running-anything)
- [How do I ask a question?](#how-do-i-ask-a-question)
- [How do I read the output?](#how-do-i-read-the-output)
- [What does a Semantic Entropy number mean?](#what-does-a-semantic-entropy-number-mean)
- [Why are there two numbers, SE and PE?](#why-are-there-two-numbers-se-and-pe)
- [When does it escalate, and to what?](#when-does-it-escalate-and-to-what)
- [How do I tune the threshold?](#how-do-i-tune-the-threshold)
- [How do I run a benchmark?](#how-do-i-run-a-benchmark)
- [How do I train and use a Semantic Entropy Probe?](#how-do-i-train-and-use-a-semantic-entropy-probe)
- [What does it look like on real questions?](#what-does-it-look-like-on-real-questions)
- [What are the limitations?](#what-are-the-limitations)
- [Troubleshooting](#troubleshooting)

## Summary

This guide is for running experiments with the detectors against a real
small model. If you have not yet run the no-model quickstart, start with the
[README](../../README.md#quickstart-no-gpu-no-llm).

- **Ask one question:** `uv run python -m demo.cli detect "..."` prints every
  detector score as JSON; `chat` adds Gemini escalation.
- **Measure detectors:** `eval/datasets/load_triviaqa.py` →
  `eval/run_benchmark.py` → `eval/compute_auroc.py` gives one AUROC per
  detector.
- **Try SEP:** train a probe with `eval/train_sep_probe.py`, then add
  `semantic_entropy_probe` to `ENABLED_DETECTORS`.
- Everything is configured through `.env` (copy `.env.example`).

## What does a single run actually do?

It asks the small model the same question N times (default 10), measures how
much the answers agree **in meaning**, and either returns the model's own
answer or hands the question to a stronger judge.

1. Sample N answers at temperature > 0, keeping per-token logprobs.
2. Take one more answer at temperature 0 — the **greedy answer**, which is
   what you get back if nothing escalates.
3. Run every enabled detector:
   - **PE** averages how surprised the model was by its own tokens.
   - **SE** groups the N answers into meaning clusters (two answers share a
     cluster when an NLI model says each entails the other) and takes the
     entropy over the clusters, each weighted by how likely the model found
     its answers. `discrete_semantic_entropy` weighs them by count alone.
   - **SEP** (opt-in) reads one hidden state through a trained linear probe.
4. Apply the escalation rule (default: `SE > SE_THRESHOLD`). If it fires and
   a judge is configured, the judge answers instead.

## Which model am I running?

The model is chosen by `VLLM_MODEL_ID` and must be an entry in
[`demo/model_registry.py`](../../demo/model_registry.py), which pins its
serving settings. All entries are Gemma 4 instruction-tuned models in their
native bf16 precision — nothing is quantized.

| Model | Role in the registry | Hardware it is pinned to |
|---|---|---|
| `google/gemma-4-E2B-it` | **local default** — loaded in-process with `transformers` | laptop / any single GPU |
| `google/gemma-4-E4B-it` | **deployed default** — served by vLLM on Vertex | 1× L4 ("one L4") |
| `google/gemma-4-26B-A4B-it` | sweep member | 1× A100 80GB |
| `google/gemma-4-31B-it` | sweep member | 2× A100 80GB |

`SLM_PROVIDER` decides *how* the model is reached:

- `hf_transformers` (default) — downloads the model from Hugging Face and runs
  it in this process. Uses a CUDA GPU if present, otherwise CPU (slow, but it
  works for a handful of questions). Exposes hidden states, so SEP works.
- `vllm` — calls an OpenAI-compatible vLLM server at `VLLM_BASE_URL`, either
  one you run (`VLLM_AUTH=api_key`) or a Vertex AI endpoint
  (`VLLM_AUTH=google`, which sends a refreshing Google access token). SEP
  needs a second vLLM server running the embed runner, set in
  `VLLM_EMBED_BASE_URL`; both providers feed SEP the same feature (the
  chat-templated prompt's last-token hidden state, unnormalized), so a probe
  trained on one works on the other.

To compare models, set `TARGET_MODEL_IDS` (at least two registry ids). The
judge (`JUDGE_MODEL_ID`, default `gemini-3.5-flash-lite`) is Gemini on
Vertex AI through the `google-genai` SDK; it returns a schema-validated
reasoning and answer (reasoning first, so the answer is conditioned on it).
Gemini 3.5 on Vertex ignores temperature, so the judge sends no sampling
parameters; `JUDGE_THINKING_LEVEL` (default `LOW`) sets its reasoning depth.

## What do I need before running anything?

```bash
uv sync --all-extras
cp .env.example .env
```

| You want to | You also need |
|---|---|
| `detect`, the benchmark, SEP training with the local model | disk for the model download; a GPU strongly recommended |
| the same against vLLM | a reachable vLLM server (`VLLM_BASE_URL`), or a Vertex endpoint plus `VLLM_AUTH=google` — see [DevOps Guide](DEVOPS_GUIDE.md#pointing-the-client-at-vertex) |
| `chat` or benchmark with a judge | `GCP_PROJECT_ID` in `.env` and `gcloud auth application-default login` |

Without Google credentials, escalation fails fast — there is no silent
fallback to another provider.

## How do I ask a question?

```bash
uv run python -m demo.cli detect "Who wrote the novel 1984?"   # scores only, never escalates
uv run python -m demo.cli chat   "Who wrote the novel 1984?"   # scores + judge if needed
```

There are no flags; sample count, threshold, detectors and models all come
from `.env`.

## How do I read the output?

`detect` prints the full `DetectionResult` as JSON: the prompt, the greedy
answer, the N sample texts, `detector_scores`, the SE `clusters` (each with
its member indices and a representative text), and the escalation decision.

`chat` prints three blocks:

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
- **Judge Reasoning** — a third block, printed only when the judge answered.

## What does a Semantic Entropy number mean?

SE is the entropy, in nats, of how the N answers split across meaning
clusters: 0 means every answer meant the same thing, and the maximum is
ln N (≈ 2.30 for N = 10), meaning every answer was different.

| Cluster split (N = 10) | SE (nats) | Reading |
|---|---|---|
| 10 | 0.00 | full agreement |
| 9 + 1 | 0.33 | one stray answer |
| 7 + 2 + 1 | 0.80 | real wobble |
| 5 + 5 | 0.69 | two camps |
| 4 + 3 + 2 + 1 | 1.28 | scattered |
| 1 × 10 | 2.30 | every answer different |

These are reference points, not verdicts — the threshold that separates
"fine" from "hallucinating" depends on the model and the question set, and
finding it is what the benchmark is for. The full derivation is in the
[walkthrough](../theory/SEMANTIC_ENTROPY_WALKTHROUGH.md).

## Why are there two numbers, SE and PE?

Because they measure different things and fail in different places. PE
measures how sure the model was about its *words*; SE measures whether it
gave the same *answer*.

- **PE low, SE high** — fluent but inconsistent: the classic hallucination
  shape, and exactly what PE alone misses.
- **PE high, SE low** — hedged wording, same answer every time: usually fine,
  and exactly what PE alone false-alarms on.

PE is shown as a cheap baseline. By default the escalation decision uses SE
only; you can add PE as an *extra* trigger with `AnyAbove`, but never as a
gate in front of SE (see the [README](../../README.md#how-do-i-combine-detectors-when-deciding-to-escalate)).

## When does it escalate, and to what?

It escalates when the escalation rule fires — by default
`semantic_entropy > SE_THRESHOLD`. The judge then receives **only the original
question** (not the samples), answers it independently, and its answer and
reasoning replace the greedy answer. The reference judge is Gemini on Vertex
(`JUDGE_MODEL_ID`); any object with `.answer(prompt) -> (answer, reasoning)`
can replace it when you use the library directly.

A high escalation rate is itself a finding: either the threshold is too
tight, or the questions are outside what the small model knows.

## How do I tune the threshold?

Set `SE_THRESHOLD` in `.env`. The default `1.0` is a placeholder, not a
measured value. To measure one, benchmark a dev slice that shares no
question with your test slice, then run
`uv run python -m eval.calibrate_threshold --dev <dev results> --test <test results>`.
It prints the τ that best separates hallucinated from correct answers.

- **Lower** → escalates more → fewer hallucinations get through, more judge
  calls (cost, latency).
- **Higher** → escalates less → cheaper and faster, more hallucinations
  through.

Pick it from data: run the benchmark, then look at the rows in the results
JSONL — each has every detector score and whether the greedy answer was
right — and choose the point on that trade-off you can afford. AUROC tells
you whether *any* threshold can separate the two groups well.

## How do I run a benchmark?

1. **Build a question set.** The loader shuffles TriviaQA's validation split
   once with a fixed seed and takes a slice of `--n` questions starting at
   `--offset`. Slices that do not overlap share no questions. Start the
   benchmark after the questions you will use to train a SEP probe (see
   below), so the two can never mix:
   ```bash
   uv run python -m eval.datasets.load_triviaqa --offset 500 --n 200 \
       --out eval/datasets/triviaqa_200.jsonl
   ```
   [`eval/datasets/adversarial.jsonl`](../../eval/datasets/adversarial.jsonl)
   holds a few false-premise and unknowable questions. They have no answer
   key, so AUROC skips them — read their rows by hand.
2. **Run the detectors.** Uses the model, detectors and threshold from
   `.env`. `--limit 10` is a good first smoke test:
   ```bash
   uv run python -m eval.run_benchmark \
       --in eval/datasets/triviaqa_200.jsonl \
       --out eval/results/triviaqa_200.jsonl --no-judge
   ```
   Each output row carries the question, answer key, greedy and final
   answers, `detector_scores` (every enabled detector), `escalated`, the
   cluster count, the raw samples, and the run's `model_id`, `sample_n` and
   `sample_temperature`. Rows are written as they finish; the run stops on
   the first error, keeping the rows already written. With a SEP probe
   enabled, the run refuses to start if any question was in the probe's
   training set.
3. **Score it.**
   ```bash
   uv run python -m eval.compute_auroc --in eval/results/triviaqa_200.jsonl
   ```
   A row counts as hallucinated when the greedy answer contains none of the
   accepted answers (case-insensitive substring — the usual TriviaQA rule).
   The output is one AUROC per detector with a bootstrap 95% CI: the
   probability that a random hallucinated row outscores a random correct
   one. With SE and PE both enabled it also reports SE's margin over PE with
   a paired-bootstrap CI, each against its target.
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
5. **Check the adversarial set.** Run step 2 on `eval/datasets/adversarial.jsonl`
   too, with the same model and settings, then:
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
sampling settings. The targets each step is checked against, and why each
needs more than a point estimate, are in the
[QA Guide](QA_GUIDE.md#what-does-success-look-like-for-the-detectors).
To compare models, rerun step 2 with a different `VLLM_MODEL_ID` and a
different `--out` file.

## How do I train and use a Semantic Entropy Probe?

A probe is trained per model: for each calibration question it computes SE
(the label) and grabs the model's hidden state at the last prompt token (the
input), then fits a ridge regression.

1. **Make a calibration set that does not overlap your benchmark.** Take the
   slice *before* the benchmark's `--offset`:
   ```bash
   uv run python -m eval.datasets.load_triviaqa --offset 0 --n 500 \
       --out data/calibration/triviaqa_calibration_500.jsonl
   ```
   The trainer reads the loader's `id` and `question` fields directly.
2. **Train.** Needs a provider that exposes hidden states (the default
   `hf_transformers` does):
   ```bash
   uv run python -m eval.train_sep_probe \
       --calibration-set data/calibration/triviaqa_calibration_500.jsonl --output-path auto
   ```
   `auto` writes `data/probes/<model-id-slug>.pt`, which is where the evaluator
   looks unless `SEP_PROBE_PATH` says otherwise. Next to it, it writes
   `<model-id-slug>.train_ids.json`, the ids of the questions it was trained
   on. Before saving, it logs the probe's 5-fold cross-validated Pearson r
   against SE (mean ± std, `--cv-folds`): how well the one-pass probe stands
   in for the ten-sample SE on questions it never saw.
3. **Enable it** in `.env` and benchmark on the disjoint slice from
   [How do I run a benchmark?](#how-do-i-run-a-benchmark):
   ```
   ENABLED_DETECTORS=["predictive_entropy","semantic_entropy_probe","semantic_entropy"]
   ```
   `run_benchmark` checks the benchmark questions against the probe's
   `.train_ids.json` and refuses to start on any overlap, or if the file is
   missing.

A probe trained on one model is meaningless on another — retrain whenever
`VLLM_MODEL_ID` changes.

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
- **N is a trade-off.** `SAMPLE_N` (2–50) sets both the resolution of SE and
  the cost of every question.
- **The judge can be wrong too.** Escalation lowers the hallucination rate; it
  does not remove it.
- **Single-turn.** Each call is independent; there is no conversation state.
- **The answer key is crude.** Substring matching against aliases mislabels
  some correct-but-differently-phrased answers; this caps measurable AUROC.
- More in [Caveats: theory vs. production](../CAVEATS.md).

## Troubleshooting

**Why does startup fail with `error parsing value for field "enabled_detectors"`?**
List settings must be JSON in `.env`:
`ENABLED_DETECTORS=["predictive_entropy","semantic_entropy"]`, not a
comma-separated list.

**Why does `SemanticEntropyProbeDetector` raise `FileNotFoundError`?**
SEP is enabled but no probe exists for the current model. Train one (above)
or set `SEP_PROBE_PATH`.

**Why do I get a connection error to `VLLM_BASE_URL`?**
`SLM_PROVIDER=vllm` but the server is not reachable. Check with
`curl $VLLM_BASE_URL/models`, or switch back to `SLM_PROVIDER=hf_transformers`.
On Vertex, add `-H "Authorization: Bearer $(gcloud auth print-access-token)"`;
a name-resolution error right after deploying usually means the dedicated
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
