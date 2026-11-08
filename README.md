# hallucinations-math-lab

A research repo built around one question: **how well do Semantic Entropy,
and a cheap single-pass probe approximating it, flag hallucinations in
self-hosted, open-weight LLMs?** It answers with two measurements — AUROC
against a predictive-entropy baseline, and an accuracy-vs-escalation-rate
curve (targeted vs. random escalation to a stronger judge) — and includes an
interactive web playground for exploring both.

Import name: `hallucinations_math_lab` · Python 3.12–3.13 · MIT license

## Table of contents

- [Summary](#summary)
- [Which model am I testing, and how do I choose it?](#which-model-am-i-testing-and-how-do-i-choose-it)
- [Quickstart: no GPU, no LLM](#quickstart-no-gpu-no-llm)
- [How do I use the web playground?](#how-do-i-use-the-web-playground)
- [What do I need before running a real model?](#what-do-i-need-before-running-a-real-model)
- [What hardware do I need, and why?](#what-hardware-do-i-need-and-why)
  - [How do I check which tier my machine fits?](#how-do-i-check-which-tier-my-machine-fits)
- [What does a single run actually do?](#what-does-a-single-run-actually-do)
- [How do the three detectors differ?](#how-do-the-three-detectors-differ)
- [How do I ask one question?](#how-do-i-ask-one-question)
- [What does a Semantic Entropy number mean?](#what-does-a-semantic-entropy-number-mean)
- [What does an AUROC number mean?](#what-does-an-auroc-number-mean)
- [When does it escalate, and how do I tune it?](#when-does-it-escalate-and-how-do-i-tune-it)
- [How do I evaluate one model?](#how-do-i-evaluate-one-model)
- [How do I re-score a saved run without the model?](#how-do-i-re-score-a-saved-run-without-the-model)
- [How do I compare several models?](#how-do-i-compare-several-models)
- [How do I train and use a Semantic Entropy Probe?](#how-do-i-train-and-use-a-semantic-entropy-probe)
  - [Which layer and token does SEP read, and are mid layers better?](#which-layer-and-token-does-sep-read-and-are-mid-layers-better)
  - [How does this SEP differ from the paper?](#how-does-this-sep-differ-from-the-paper)
  - [What changed for an existing SEP setup?](#what-changed-for-an-existing-sep-setup)
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

**The research question.** Ask a model the same question several times. If
it really knows the answer, the answers agree in meaning; if it is making
something up, they scatter. Semantic Entropy (SE) turns that scatter into a
number. This repo asks **how well that number flags hallucinations in
self-hosted, open-weight LLMs** — and whether a cheap probe can deliver the
same signal without the repeated sampling.

**The three detectors, and the role each plays:**

- **SE** is the subject: the strong but expensive signal (N generations per
  question).
- **SEP** is SE's cheap stand-in, not its rival: a probe trained to predict
  SE from one generation. Its question is how much of SE's quality survives
  at a fraction of the cost.
- **PE** is the baseline SE has to beat: a free, single-sample signal that
  reads word-level uncertainty rather than meaning.

| Detector | What it reads | Cost |
|---|---|---|
| **SE** — Semantic Entropy ([Farquhar et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07421-0)) | how many *different meanings* appear across N samples | N generations + NLI checks |
| **SEP** — Semantic Entropy Probes ([Kossen et al., 2024](https://arxiv.org/abs/2406.15927)) | p(high SE) from a logistic probe over the model's hidden states, before it answers | one extra forward pass over the prompt, plus a dot product |
| **PE** — Predictive Entropy | the token probabilities of the samples | free (a by-product of generation) |

**The two measurements this repo produces:**

1. **AUROC per detector** — how reliably each detector's score ranks the
   model's wrong answers above its right ones, across every possible
   threshold ([what an AUROC number means](#what-does-an-auroc-number-mean)). Reported with confidence intervals, and as SE's margin over PE.
2. **An accuracy-vs-escalation-rate curve** — escalate the answers a detector
   flags to a stronger judge model, and plot final accuracy against the
   share of questions escalated, next to escalating the same share **at
   random**. Escalating *any* questions to a stronger model raises accuracy;
   the gap between the targeted and random curves is what the detector
   itself earns. This is the repo's end result: what each extra escalation
   buys when a detector picks it.

The judge is any model behind `.answer(prompt) -> (answer, reasoning)`;
Gemini on Vertex is the reference. The model under test is configurable
(see [below](#which-model-am-i-testing-and-how-do-i-choose-it)).

**The web playground** puts all of this in a browser, one tab per way in:
*Score answers*, *Ask the model* and *Measure the detectors*.

There are three ways in. Each one answers a different question, and they are
ordered from cheapest to most involved. The web playground has one tab per way
in, with the same names:

| Way in | The question it answers | Does a model answer questions here? | Hardware | Start with |
|---|---|---|---|---|
| **1. Score answers** | Do these N answers to one question look like a hallucination? | No. The answers are given; only DeBERTa (a small 1.6 GB model) runs, to check which answers mean the same thing | any laptop ([tier 0](#what-hardware-do-i-need-and-why)) | [`examples/score_precollected_samples.py`](#quickstart-no-gpu-no-llm), then [`evaluate_responses(...)`](#how-do-i-score-samples-from-my-own-model); or the *Score answers* tab |
| **2. Ask the model** | Should I trust the answer the model just gave to my question, or hand it to a stronger model? | Yes: Gemma 4 answers your one question 11 times | a machine that fits the model ([tiers 1–3](#what-hardware-do-i-need-and-why)), or a remote server | [`demo.cli detect` / `demo.cli chat`](#how-do-i-ask-one-question), or the *Ask the model* tab |
| **3. Measure the detectors** | On this model, does a high score actually line up with wrong answers? | Yes: Gemma 4 answers 200 TriviaQA questions, 11 times each. Re-scoring a saved run needs no model | as for 2; re-scoring a saved run: tier 0 | [`eval.run_benchmark`](#how-do-i-evaluate-one-model), [`eval.rescore_benchmark`](#how-do-i-re-score-a-saved-run-without-the-model), [`--sweep`](#how-do-i-compare-several-models); read the results in the *Measure the detectors* tab |

**1. Score answers — you bring the answers, nothing generates them.** The
example ships three hand-written sets of ten answers, and one of them is a
**deliberate hallucination**. *Consistent* says "Paris" ten times.
*Paraphrased* says "Paris" in ten different wordings. *Divergent* answers "Who
was the first woman to walk on Mars?" with ten names that contradict each
other, which is how a model's answers scatter when it is making something up.
Because you know in advance which set is the hallucination, you can watch how
each detector reacts to it. To score your own model, pass the answers it gave
with their token probabilities.

**2. Ask the model — one question, one live decision.** You type a question.
The model answers it ten times at random plus once at its most likely, the
detectors score how much those answers disagree, and you get either the
model's answer or, if the score crosses the threshold, the stronger model's
(the judge; only `chat` calls it).

**3. Measure the detectors — do the scores mean anything on this model?**
The questions come from [TriviaQA](https://nlp.cs.washington.edu/triviaqa/),
a public trivia dataset that comes with the correct answers. The model answers
each question, and each answer is marked right or wrong against the dataset.
**Nothing wrong is planted: the wrong answers are the model's own mistakes.**
The detectors never see the correct answer. The benchmark checks whether
their score, computed blind, comes out high on exactly the answers that turned
out wrong, and reports that as one AUROC per detector.

*Why not plant wrong answers instead?* Because SE does not check whether an
answer is true; it measures whether the model contradicts itself. A planted
answer has no model behind it, so there is nothing to measure. That is what
way 1 is for: patterns you construct by hand. A run saves every answer, so
you can [re-score it later without the model](#how-do-i-re-score-a-saved-run-without-the-model):
a new threshold or detector on the same answers, on any laptop.

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

Both providers read SEP's input at the same token, the last token of the
chat-formatted prompt, and both load the registry's pinned checkpoint. But
the vLLM embed runner can serve only a probe that reads the **last layer**
(one trained with `--layers final`), and whether its vectors match the
in-process ones within tolerance is unverified until the first GPU run.
Collecting a probe's training data (`eval.collect_sep_dataset`) needs
`hf_transformers`, which exposes every layer; the fit itself runs on CPU
from the collected dataset.

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

The script scores three hand-written sets of 10 "samples" each. SE decides
which samples mean the same thing with the NLI model
`microsoft/deberta-large-mnli` (~1.6 GB, downloaded once; CPU is fine):

```
escalation: SE > 1.0 nats

scenario         PE     SE  clusters  escalate
consistent    0.050  0.000         1  False
paraphrased   0.600  0.000         1  False
divergent     0.400  1.834         7  True
```

Ten wordings of "Paris" collapse into one meaning cluster and do not
escalate, while the genuinely contradictory answers do. PE scores the
paraphrases higher than the consistent answers — it only sees token
probabilities, so varied wording looks like uncertainty. That gap is the
whole case for Semantic Entropy.

## How do I use the web playground?

Run one command and open <http://127.0.0.1:8765>:

```bash
uv sync --all-extras
uv run python -m demo.web
```

It serves a small web UI and its API from one local process, bound to
`127.0.0.1` only (`--port` picks another port). The UI is prebuilt and
committed, so you do not need Node. It has one tab per way in:

| Tab (= way in) | Needs |
|---|---|
| **1. Score answers** | nothing — DeBERTa downloads on the first score (~1.6 GB, once) |
| **2. Ask the model** | the model from `.env`, as for `demo.cli`; *Chat* also needs `GCP_PROJECT_ID` and Google credentials |
| **3. Measure the detectors** (read-only) | results files under `eval/results/`, from `eval.run_benchmark` or `eval.rescore_benchmark` |

- **Score answers** — load one of the three presets from
  `examples/score_precollected_samples.py`, or type your own answers. You
  get SE, the meaning clusters (one colour and number per meaning) and PE.
  The presets reproduce the Quickstart table exactly. Typed answers have no
  token logprobs, so PE shows "needs token logprobs" instead of a number,
  and SE weighs each meaning by how many answers say it (discrete SE) instead
  of by their likelihoods; the SE note says which one ran.
  The threshold slider flips *escalate* in the browser, with no new request.
- **Ask the model** — *Detect* and *Chat* do what `demo.cli detect` and
  `demo.cli chat` do, and show the samples grouped into clusters. A status
  strip shows the provider and model; *Chat* is disabled, with the reason,
  when the judge is not configured. Errors from the model are shown as the
  server reports them.
- **Measure the detectors** — pick a results file for per-detector AUROC with
  its 95% confidence interval (the same numbers as `eval.compute_auroc`),
  or a `--sweep` directory for the model comparison table. Benchmarks are
  run from the command line, not from the UI.

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

## What hardware do I need, and why?

A machine can run everything in its tier and the tiers below it;
[the checker](#how-do-i-check-which-tier-my-machine-fits) finds yours. Linux
and macOS only, for now.

| Tier | What you can run | Download (once) | NVIDIA GPU memory | Apple Silicon Mac memory | No usable GPU |
|---|---|---|---|---|---|
| **0** | the library, the quickstart, every scoring script (`compute_auroc`, `compare_models`, …) on existing results, and the benchmark or demo against a **remote** model server | ~1.3 GB of Python packages (macOS; larger with Linux GPU builds), +1.6 GB for the DeBERTa meaning check | not needed | any | any machine with 8 GB of memory |
| **1** | `gemma-4-E2B-it` locally — the default | 10.2 GB | 16 GB | 24 GB | 24 GB of memory; runs, but slowly |
| **2** | `gemma-4-E4B-it` | 16.0 GB | 24 GB (one L4) | 32 GB | not practical |
| **3** | `gemma-4-26B-A4B-it` / `gemma-4-31B-it` | 51.6 GB / 62.5 GB | 80 GB (one A100) / 2× 80 GB ("two A100s, eighty gig each") | 96 GB / 128 GB — estimates, untested | not practical |

The GPU and Mac columns are minimums. Disk: the download size plus a few GB
of headroom, in `~/.cache/huggingface` unless `HF_HOME` moves it. No model is
gated, so no Hugging Face account is needed.

**Why does it need this much memory?** The whole model has to sit in memory
while it answers, and every parameter takes 2 bytes, because the models run
unquantized in bf16 ("bee-eff-sixteen", a 16-bit number format). So memory ≈
2 × parameters: E2B's 5.1 billion parameters are its 10.2 GB, plus room for
the text being generated. The misunderstanding to avoid: "26B-A4B" means only
about 4 billion parameters do work for each word, but all 26 billion must
still be loaded — the "A4B" saves time, not memory.

**Why does a Mac need more than an NVIDIA GPU of the same size?** An Apple
Silicon Mac shares one pool of memory between the CPU and the GPU, and macOS
lets the GPU use only part of it (roughly two-thirds to three-quarters), while
the rest of the system needs room too. The project uses the Mac's GPU
automatically; the DeBERTa meaning check runs on its CPU. Intel Macs have no
usable GPU here.

**Why is a GPU strongly recommended, when the CPU works?** Each question
generates 11 answers (`SAMPLE_N`=10 samples plus one greedy answer), and SE
then compares those answers in pairs with DeBERTa. All of that is many
passes through the model per question, which a GPU does in parallel and a
CPU does slowly. Lowering `SAMPLE_N` speeds it up proportionally, at the cost
of a noisier SE.

**Why does bandwidth matter only once?** Models, DeBERTa and the TriviaQA
questions (~7 MB) download on first use and are cached. At 100 Mbit/s
("a hundred megabit"), 10 GB takes about 15 minutes. After that, only the
Gemini judge (`chat`, or a benchmark without `--no-judge`) and remote
servers use the network, with small requests.

**What if my machine is below tier 1?** Run the model somewhere else: a vLLM
server on a rented GPU or a Vertex AI endpoint ([DevOps
Guide](knowledge/usage_guides/DEVOPS_GUIDE.md)). This machine then only needs
tier 0.

### How do I check which tier my machine fits?

Run the checker (after `uv sync --all-extras`):

```bash
uv run python -m demo.check_hardware
```

```
OS            Darwin arm64
Memory        36 GB
NVIDIA GPU    none
PyTorch GPU   cuda=False  mps=True
Free disk     301 GB (model cache)
Download      29 Mbit/s

Tier 2: google/gemma-4-E4B-it on apple
Download      16.0 GB once, about 75 min

Set in .env:
VLLM_MODEL_ID=google/gemma-4-E4B-it
```

It reads the machine's memory, NVIDIA GPU memory, which GPU PyTorch can use,
and free disk where models are cached; times a 7 MB download from Hugging
Face; and picks the largest model that fits both memory and disk, using the
figures in the table above. Agents should add `--json` for machine-readable
output; `--no-network` skips the speed test.

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
  meaning clusters, then take the entropy over the clusters:
  `SE = -Σ_c p(c) ln p(c)` in nats. By default `p(c)` is the cluster's share
  of the samples' length-normalized sequence likelihoods, as in the paper;
  `discrete_semantic_entropy` uses `|c|/N` and needs no logprobs. One cluster
  → 0; N distinct, equally likely meanings → ln N. Two answers share a cluster
  when an NLI model says each one entails the other, with the question placed
  in front of both, so a bare "Paris" and "The capital of France is Paris." are
  judged in context.
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

- **What it measures.** p(high SE): the probability that the ten-sample SE
  would land above a split point, read by a logistic classifier from the
  model's hidden states at the final prompt token, before any answer is
  generated. Following [Kossen et al.](https://arxiv.org/abs/2406.15927), the
  label is **discrete** SE split into high and low at the best-split point,
  and the layers are chosen by a sweep, not fixed to the last one (see
  [below](#which-layer-and-token-does-sep-read-and-are-mid-layers-better)).
  The score is a probability in [0, 1], **not on SE's scale**: it is not in
  nats, and it is not the probability that the answer is wrong. It has its
  own threshold, `SEP_THRESHOLD`.
- **Assumption.** The model's internal state *before it answers* already
  encodes how scattered its answers will be, in a linearly readable way —
  a much stronger claim than SE's.
- **Fails when.** (a) Used on a different model, revision or prompt than it
  was trained on — the probe refuses to load. (b) Prompts are adversarial or
  unlike the training set. (c) The provider does not expose hidden states
  (most hosted APIs). (d) The SE labels it was trained on were themselves
  miscalibrated.
- **Cost.** One forward pass over the prompt (on vLLM, served by a second
  embed server) and a dot product with a sigmoid; no sampling and no NLI.

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
  enabled, as a percentage: `SEP P(high SE) = 72.3%`.
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

## What does an AUROC number mean?

**AUROC is the probability that, given one hallucinated answer and one
correct answer picked at random, the detector gives the hallucinated one the
higher score.** It runs from 0 to 1 and has no unit.

| AUROC | Reading |
|---|---|
| 1.0 | always ranks the hallucinated answer higher — perfect |
| 0.75 | this repo's target for SE on TriviaQA |
| 0.5 | a coin flip — the detector carries no signal |
| < 0.5 | ranks them backwards; flipping the score's sign fixes it |

- **ROC vs. AUROC.** The ROC curve plots, for every possible threshold, the
  share of hallucinations caught against the share of correct answers wrongly
  flagged — use it to *choose* a threshold. AUROC is the area under that
  curve: one number that grades the *ranking* with no threshold at all.
- **It is not accuracy.** Accuracy needs a threshold; AUROC never picks one.
  Two detectors with the same AUROC can behave very differently at the
  threshold you'd actually run.
- **Compare in absolute points.** "SE beats PE by 0.05" means 0.05 on the
  0–1 scale, not 5% of SE's value.
- **Distrust ≈ 0.99.** On a task like TriviaQA that usually means something
  leaked into the score, not a breakthrough.

Plain "AUC" (area under *a* curve) is usually understood as AUROC; this repo
writes AUROC to rule out the precision-recall area (AUPRC). Full entry:
[glossary](knowledge/GLOSSARY.md#what-is-auroc-and-how-is-it-different-from-the-roc-curve).

## When does it escalate, and how do I tune it?

It escalates when the escalation rule fires — by default
`semantic_entropy > SE_THRESHOLD` (`ESCALATION_DETECTOR`, `SE_THRESHOLD`).
The default `1.0` is a placeholder, not a measured value. The judge then receives **only the original question** (not
the samples), answers it independently, and its answer and reasoning replace
the greedy answer.

Each detector is compared against the threshold on its own scale.
`SE_THRESHOLD` (nats) serves SE, discrete SE and PE. SEP's score is a
probability, so `ESCALATION_DETECTOR=semantic_entropy_probe` uses
`SEP_THRESHOLD` (default `0.5`, the classifier's own decision boundary, a
placeholder uncalibrated for escalation); `SE_THRESHOLD` does not affect SEP.
A trained probe carries a `recommended_threshold`, chosen by Youden's J on
out-of-fold predictions over its calibration questions. It is **never loaded
implicitly**: copy it into `SEP_THRESHOLD` if you want it, and the evaluator
logs a warning when the two differ. Escalating on SEP also requires SEP in
`ENABLED_DETECTORS`; settings refuse the combination otherwise.

- **Lower threshold** → escalates more → fewer hallucinations get through,
  more judge calls (cost, latency).
- **Higher threshold** → escalates less → cheaper and faster, more
  hallucinations through.

Pick it from data, on questions you won't report results on. Benchmark a
dev slice that shares no question with your test slice, then run

```bash
uv run python -m eval.calibrate_threshold --dev <dev results> --test <test results>
```

It prints the threshold that best separates hallucinated from correct answers
on the dev slice (true-positive rate minus false-positive rate, highest
first), and refuses a dev slice that overlaps the test slice. Set it as the
setting the report names in `env_var` (`SE_THRESHOLD`, or `SEP_THRESHOLD` with
`--detector semantic_entropy_probe`), then benchmark the test slice once. If the judge's cost
matters more than that balance, read the escalation rate and the
accuracy-vs-rate curve from `eval.escalation_uplift` instead. AUROC tells
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
   cluster count, the raw samples with their token probabilities, and the
   run's `model_id`, `sample_n`, `sample_temperature` and `sep_probe` (the SEP
   probe's reference and hashes, or null without SEP). Rows are written as
   they finish; the run stops on the first error, keeping the rows already
   written.
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

   Uplift alone doesn't credit the detector: escalating *any* questions to a
   stronger judge raises accuracy. So the report also gives
   `escalation_beats_random_pp` (the run's escalation vs. escalating the
   same share at random, with a 95% CI) and `accuracy_vs_rate`, the
   accuracy-vs-escalation-rate curve for every detector next to the random
   baseline. That curve is the repo's headline result.
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

## How do I re-score a saved run without the model?

Point `eval.rescore_benchmark` at a results file. It runs the detectors again
on the answers the run saved, applies the escalation rule from `.env`, and
writes a new results file. No model is loaded, so it runs on any laptop:

```bash
# e.g. after changing SE_THRESHOLD or ENABLED_DETECTORS in .env
uv run python -m eval.rescore_benchmark \
    --in eval/results/triviaqa_200.jsonl --out eval/results/triviaqa_200_rescored.jsonl

uv run python -m eval.compute_auroc --in eval/results/triviaqa_200_rescored.jsonl
```

- **Why it works.** Every results row keeps the model's sampled answers and
  their token probabilities. That is all PE and SE need, so you generate the
  answers once (the expensive part) and try detectors, thresholds or a
  different meaning check on them as often as you like.
- **What stays the same.** The questions, the answer key, the model's answers
  and the judge's recorded answer. An escalated row takes the judge answer
  the run recorded; with a `--no-judge` run it keeps the model's own answer.
- **What it refuses.** SEP, because it reads the model's internal state,
  which a saved run does not keep. Also files from before runs saved token
  probabilities: re-run `eval.run_benchmark` for those.
- **The output is an ordinary run.** `compute_auroc`, `calibrate_threshold`,
  `escalation_uplift`, `compare_models` and the *Measure the detectors* tab all
  read it.

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

A probe is trained per model, following
[Kossen et al. 2024](https://arxiv.org/abs/2406.15927) with the deviations
listed [below](#how-does-this-sep-differ-from-the-paper). For each
calibration question, the expensive step computes SE from 10 samples and
stores the model's hidden states at every layer; the cheap step splits SE
into high and low, fits a logistic classifier, and picks which layers it
reads. **No probe has been trained or published yet**: there is no GPU
result for any Gemma 4 model so far.

A probe is a **directory**, the same on disk and on the Hugging Face Hub:

| File | Holds |
|---|---|
| `probe.safetensors` | the weights and bias (safetensors cannot carry code) |
| `probe_config.json` | the backbone (model id, Hub commit, layer count, width), the layers and token position read, the prompt (answer instruction, chat-template hash), the label and classifier settings, the recommended threshold, the training provenance and the active deviations from the paper |
| `train_ids.json` | the calibration question ids, for the leakage guard |
| `training_log.jsonl` | one line per calibration question |
| `metrics.json` | the layer sweep, the selected window with its cross-validated AUROC, the last-layer baseline and the recommended threshold |

`SEP_PROBE_PATH` takes a probe directory or a Hub reference,
`hf://<owner>/<repo>[@<revision>]`; pin `@<revision>` for reproducible runs.
Unset, the evaluator looks in `data/probes/<model slug>/`; there is no
fallback between sources. Before the first sample, it refuses a probe whose
model id, revision, layer count, width, answer instruction or chat template
differs from the model under test, naming both values.

**1. Use a published probe.** None exists yet. Once one is published, set
`SEP_PROBE_PATH=hf://<owner>/<repo>@<tag>` and enable SEP as in step 2.4
below; nothing else is needed, on any machine that runs the model with
`hf_transformers` (or with vLLM for a last-layer probe).

**2. Train your own.** Needs `SLM_PROVIDER=hf_transformers` and a GPU that
fits the model (collection runs 500 questions × 11 generations).

1. **Calibration questions come with the repo.** Without `--calibration-set`,
   collection uses the first 500 questions of the same fixed-seed shuffle,
   downloaded once to `data/questions/`. They never overlap the default
   benchmark questions. To use your own, pass any slice with answer aliases
   that does not overlap your benchmark.
2. **Collect** for the model in `VLLM_MODEL_ID` (GPU):
   ```bash
   uv run python -m eval.collect_sep_dataset
   ```
   It writes the dataset directory `data/sep_datasets/<model slug>/` (or
   `--output-dir`, which must be empty or absent): `rows.parquet` (the
   samples, their meaning clusters, discrete and likelihood-weighted SE, the
   greedy answer and its correctness), `hidden_states.safetensors` (every
   layer, at two token positions), `collection_config.json` and
   `training_log.jsonl`. If a question fails, the questions finished before it
   are still written, marked incomplete.
3. **Fit** on CPU, from the dataset alone:
   ```bash
   uv run python -m eval.train_sep_probe --dataset data/sep_datasets/<model slug>
   ```
   It writes the probe directory to `data/probes/<model slug>/` (or
   `--output-dir`). `--layers auto` (default) chooses a 5-layer window by
   cross-validation over all calibration rows and reports the nested
   cross-validation AUROC, which includes the cost of choosing;
   `--layers final` reads only the last layer, the only probe the vLLM
   embed runner can serve; `--layers <a>-<b>` fixes a range. If the fit
   does not converge, it stops and names the pre-registered fallback: refit
   with `--max-iter 1000`, then, only if that fails too, `--standardize`.
   Both are recorded in the probe.
4. **Enable it** in `.env` and benchmark on the disjoint slice:
   ```
   ENABLED_DETECTORS=["predictive_entropy","semantic_entropy_probe","semantic_entropy"]
   ```
   `run_benchmark` checks the benchmark questions against the probe's
   `train_ids.json` and refuses to start on any overlap, if the file is
   missing, or if a question has no `id`. Every row records the probe's
   reference and the SHA-256 of its weights and config.
5. **Publish** (optional): the release sequence, which merges the held-out
   benchmark into `metrics.json` and uploads the probe and its dataset, is in
   the [DevOps Guide](knowledge/usage_guides/DEVOPS_GUIDE.md#publishing-a-sep-probe-to-the-hub).

**3. Refit from a published dataset, no GPU.** The dataset repo holds every
input the fit needs, so the CPU step can be rerun anywhere: download it,
check out the commit the probe card names, and run step 2.3 on it. The
model card prints the exact commands.

A probe trained on one model is meaningless on another. To include SEP in a
model comparison, train one probe per model (rerun steps 2.2–2.3 with each
`VLLM_MODEL_ID`) and leave `SEP_PROBE_PATH` unset — `--sweep` refuses a
single explicit probe.

### Which layer and token does SEP read, and are mid layers better?

These are two separate choices. The **token position** is *which token's*
vector is read; the **layer** is *how deep* in the network it is read. "A
token in the middle layers" mixes them up.

- **Token position.** The served probe reads TBG ("token before
  generation"): the last token of the chat-formatted prompt, so it scores
  before the model answers. Collection also stores SLT ("second-last
  token", as the authors' code computes it): teacher-forcing the greedy
  answer of m tokens a[0..m-1], counted from 0, the state at answer token
  a[m-3] (the token whose output predicts a[m-2]). For answers of 1-2
  tokens this falls on the answer's first token or the last prompt token.
  SLT is the paper's main setting; it is swept and reported, but v1 serves
  only TBG probes.
- **Layer.** Layer k is the vector after k of the model's L layers (0 is the
  token embedding, L the last). The paper numbers the same vectors from 1,
  so our k is probably its k + 1; the model card prints both. The trainer
  sweeps every layer from 1 to L at both positions and picks a window of 5
  consecutive layers by cross-validation over all calibration rows; nested
  cross-validation (selection repeated inside each outer fold) gives the
  reported AUROC, which therefore includes the cost of choosing.

**The misconception to head off: "research shows mid layers are best."** In
Kossen et al., mid layers win for probes that predict *correctness* and for
*long-form* answers. For short answers, the setting used here, their SEP
AUROC rises toward the late layers, and most of their chosen windows are
the last five layers. Gemma 4 was not in the paper, and our prompt differs
from theirs, so this repo assumes neither: the sweep measures it. Read the
result from the probe's `metrics.json`: `layer_sweep` (every layer, both
positions), `selection` (the chosen window and its nested-CV AUROC) and
`final_layer_auroc` (the last layer alone, the baseline for "just use the
last layer"). No Gemma 4 sweep has run yet.

### How does this SEP differ from the paper?

Each probe lists the deviations active for it in
`probe_config.json` → `deviations_from_paper`.

| ID | Paper | Here | Why |
|---|---|---|---|
| D1 | Plain-text few-shot "Answer:" prompt, no chat template | Gemma chat template plus the brief-answer instruction, zero-shot | Our models are chat-tuned, and every detector reads the same input. TBG is therefore a different position; unverified relative to the paper. |
| D2 | Llama-2, Mistral, Phi-3 | Gemma 4 | Our backbones. The paper's layer choices do not transfer, so layers are swept. |
| D3 | γ fitted on all data before splitting | γ fitted per training fold; the final γ uses all rows | No label leakage into cross-validation estimates. |
| D4 | Layer window chosen on test-split AUROC | Chosen by inner cross-validation; the nested estimate is reported | No selection leakage. `--layers auto` only. |
| D5 | Window ranked by AUROC against model error | Ranked by AUROC against binarized SE; error AUROC still reported | Independent of the noisy correctness labeler; matches the probe's target. `--layers auto` only. |
| D6 | About 2000 questions per dataset, several datasets | 500 TriviaQA questions | GPU budget. The cross-validation spread is reported so the noise is visible. |
| D7 | Most-likely answer sampled at T = 0.1 (the text says greedy) | Greedy, T = 0 | Matches the paper's text and the repo's greedy answer. |
| D8 | SE samples at T = 1.0, top-p 0.9, top-k 50 | T = 1.0 with the checkpoint's own top-p and top-k, recorded in the probe | The project's sampler. SE values are not directly comparable with the paper's. |
| D9 | No escalation threshold | `SEP_THRESHOLD`, with a recommended value from Youden's J on out-of-fold rows | A product need; the paper reports AUROC only. |
| D10 | max_iter = 100 | 1000, only if 100 did not converge | Pre-registered fallback. |
| D11 | Best split over a 100-point grid ending at the largest SE | Exhaustive search over midpoints between distinct SE values | Exact, and no split can tie with a data value. |
| D12 | No feature scaling | A standard scaler folded into the weights, only if max_iter = 1000 did not converge either | Pre-registered fallback. |
| D13 | Layer 0 included | Layers 1..L | At TBG, layer 0 is the same vector for every prompt. |

### What changed for an existing SEP setup?

SEP was previously a ridge regression predicting SE in nats. If you used it:

| You had | Do now |
|---|---|
| `ESCALATION_DETECTOR=semantic_entropy_probe` with `SE_THRESHOLD=…` | Set `SEP_THRESHOLD` (start from the probe's `recommended_threshold`). `SE_THRESHOLD` no longer affects SEP. |
| A local `data/probes/<slug>.pt` | Retrain (`eval.collect_sep_dataset`, then `eval.train_sep_probe`); `.pt` probes are refused, not converted. |
| `SEP_PROBE_PATH=…/x.pt` | Point it at a probe directory or an `hf://` reference. |
| SEP on `SLM_PROVIDER=vllm` | Works only for a `--layers final` probe, and needs `VLLM_EMBED_BASE_URL`; otherwise use `hf_transformers`. Both are checked when the evaluator is built. |
| SEP on `SLM_PROVIDER=vllm` with a model not in `demo/model_registry.py` | Add a registry entry; the layer count is read from it. |
| Reading SEP in nats in the UI or CLI | It is now shown as a percentage, "P(high SE)". |
| A custom `SamplingProvider` | Also implement `model_revision`, `num_hidden_layers`, `hidden_size`, `chat_template_sha256` and `check_hidden_state_request(layers)`; `pooled_hidden_state` takes the layers. |
| A custom detector with `requires_hidden_state=True` | Implement `HiddenStateDetector.hidden_state_layers`; otherwise building the evaluator raises `TypeError`. |
| Results files from an earlier `eval.run_benchmark` | Re-run it: the scoring scripts refuse rows without the `sep_probe` field it now records. |
| A benchmark question without an `id`, run with SEP enabled | Give every question an `id`; a SEP run refuses it instead of skipping it. |

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
protocol (`model_id`, `model_revision`, `num_hidden_layers`, `hidden_size`,
`chat_template_sha256`, `sample`, `greedy`, `check_hidden_state_request(layers)`
and `pooled_hidden_state(prompt, layers)`) and use
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
A detector with `requires_hidden_state=True` also implements
`HiddenStateDetector.hidden_state_layers`, the layers it reads; building the
evaluator raises `TypeError` without it. See the
[Developer Guide](knowledge/usage_guides/DEVELOPER_GUIDE.md#how-do-i-add-a-detector).

## What can I experiment with?

| Knob | Where | What to watch |
|---|---|---|
| Which model, or which set of models | `VLLM_MODEL_ID`; `TARGET_MODEL_IDS` with `--sweep` | how detectors behave per model size |
| The "same meaning?" judge | pass any object with `bidirectional_entails(a, b)` to `SemanticEntropyDetector(scorer=...)`; default is DeBERTa-MNLI (`NLI_MODEL_ID`, loaded at the commit in `NLI_REVISION`; change the two together) | how cluster counts change on paraphrases |
| Number of samples N, temperature | `SAMPLE_N` (2–50), `SAMPLE_TEMPERATURE` | SE resolution: with N=10 the maximum is ln 10 ≈ 2.30 nats; every question costs N generations |
| Escalation threshold | `SE_THRESHOLD` (SE, discrete SE, PE; nats), `SEP_THRESHOLD` (SEP; probability), or `escalation_threshold=` | escalation rate vs. missed hallucinations |
| Escalation rule over several detectors | `AnyAbove`, `AllAbove`, `SingleDetector` | see [above](#when-does-it-escalate-and-how-do-i-tune-it) |
| Your own detector | the `Detector` protocol; [`examples/custom_detector.py`](examples/custom_detector.py) | its AUROC next to PE and SE |
| The question set | [`eval/datasets/`](eval/datasets/) — TriviaQA loader + an adversarial set | AUROC on easy vs. trick questions |
| SEP probe | [`eval/train_sep_probe.py`](eval/train_sep_probe.py) `--layers auto\|final\|<a>-<b>`, refit from a collected dataset on CPU | how much of SE's signal a single forward pass recovers, and at which layers |

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

**Why does SEP raise `FileNotFoundError`?**
SEP is enabled but no probe directory exists for the model being run. Train
one ([above](#how-do-i-train-and-use-a-semantic-entropy-probe)) or set
`SEP_PROBE_PATH` (single-model runs only).

**Why is my `.pt` probe refused with `ObsoleteSEPProbeError`?**
It is a pre-classifier ridge probe that predicts SE in nats; SEP now
outputs p(high SE), and an old probe cannot be converted. Retrain
(`eval.collect_sep_dataset`, then `eval.train_sep_probe`), or point
`SEP_PROBE_PATH` at a probe directory or an `hf://` probe. Any other regular
file is refused the same way: `SEP_PROBE_PATH` must name a directory.

**Why does SEP raise `SEPProbeMismatchError` before any scoring?**
The probe was trained on a different backbone or prompt than the one under
test: model id, Hub revision, layer count, hidden size, answer instruction
or chat template. The message lists every mismatch with both values. Use a
probe trained for this exact model and `ANSWER_INSTRUCTION`, or retrain.

**Why does SEP on vLLM raise `UnsupportedHiddenStateRequest`?**
The vLLM embed runner pools only the last layer, and this probe reads other
layers (or `VLLM_EMBED_BASE_URL` is unset, so no hidden states are
available at all). Use `SLM_PROVIDER=hf_transformers`, or train the probe
with `--layers final`.

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
Clustering is splitting paraphrases. Check `NLI_MODEL_ID`, and run
`examples/score_precollected_samples.py`: with a healthy NLI model, the
"paraphrased" row is one cluster with SE 0.

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
  web/                         # web playground: FastAPI app + the committed, prebuilt UI
frontend/                      # the playground UI's source (Vite, React); only for UI work
examples/                      # runnable playground scripts
eval/                          # datasets, benchmark (one model or a sweep), model comparison,
                               # success-criteria scoring, SEP dataset collection,
                               # probe training and Hub publishing
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
