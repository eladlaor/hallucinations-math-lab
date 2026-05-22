# User Guide

## Table of Contents

- [Summary](#summary)
- [What this tool actually does](#what-this-tool-actually-does)
- [About the small model — quantization](#about-the-small-model--quantization)
- [Prerequisites](#prerequisites)
- [Asking a question](#asking-a-question)
- [Interpreting the output](#interpreting-the-output)
  - [Detection block](#detection-block)
  - [Final answer](#final-answer)
  - [Judge reasoning](#judge-reasoning)
- [Reading the Semantic Entropy number](#reading-the-semantic-entropy-number)
- [SE vs PE — why two numbers?](#se-vs-pe--why-two-numbers)
- [When the system escalates](#when-the-system-escalates)
- [Tuning the threshold](#tuning-the-threshold)
- [Worked examples](#worked-examples)
- [Limitations and caveats](#limitations-and-caveats)
- [Troubleshooting](#troubleshooting)

## Summary

`seh chat "<your question>"` asks the small model (Gemma 2 9B) the same question
**10 times**, measures how much its answers **agree in meaning** (not in
wording), and then does one of two things:

- **High agreement** → returns Gemma's answer directly. It's confident.
- **Low agreement** → escalates to a stronger judge model (Vertex Gemini), which
  produces the final answer and a short justification.

The goal: catch hallucinations *before* they reach you. When Gemma is making
something up, its 10 samples tend to disagree about the meaning even if each
individual answer sounds fluent. That disagreement is the signal.

## What this tool actually does

A normal LLM gives you one answer and you have no idea whether it's confident or
guessing. This tool replaces the single-answer flow with:

1. Sample the small model N=10 times at non-zero temperature.
2. Run pairwise **bidirectional NLI** (Natural Language Inference) over the
   samples to group them into clusters of semantically equivalent answers.
3. Compute **Semantic Entropy (SE)** — Shannon entropy over those cluster sizes.
   Many small clusters → high entropy → the model can't agree with itself →
   likely hallucination.
4. If SE crosses the configured threshold, hand the question to the judge.

You see the result of that whole pipeline in a single CLI call.

## About the small model — quantization

The "small model" in step 1 is **not** raw Gemma 2 9B. It is
`hugging-quants/gemma-2-9b-it-AWQ-INT4` — Gemma 2 9B Instruct compressed with
**AWQ (Activation-aware Weight Quantization) to 4-bit integer weights**. vLLM
loads it with `--quantization awq` (see `serving/Dockerfile`).

Why this matters to you as a user:

- **It is what runs.** Every sample you see in `seh chat` comes from the
  INT4-quantized weights, not from FP16 Gemma. The SE/PE numbers and the
  hallucination behavior are properties of the quantized model.
- **It is why this fits on one L4.** A 4-bit Gemma 2 9B fits comfortably in a
  24 GB L4 with room for KV cache and N=10 parallel samples. Unquantized FP16
  would not, and the project's cost story (~$0.70/hr endpoint) depends on this.
- **Quality cost is small but not zero.** AWQ-INT4 typically loses a small
  amount of accuracy vs FP16 on factoid benchmarks. The SE detector is
  designed to catch the failures that result — including any extra ones
  introduced by quantization — and escalate them to the unquantized judge.
- **It is not a CLI knob.** The model ID is set in `.env`
  (`VLLM_MODEL_ID`) and baked into the serving container. Users do not
  switch quantization at chat time. If you want to compare against an
  unquantized run, that's a deployment-level change — see `DEVOPS_GUIDE.md`.

The **judge** model (Vertex Gemini) is not quantized by us; it runs as a
managed Google API.

## Prerequisites

Before `seh chat` will work end-to-end you need:

- A running vLLM endpoint serving Gemma 2 9B (local container or Vertex
  Prediction). See `DEVOPS_GUIDE.md` for deployment.
- `.env` populated with at least `VLLM_BASE_URL`.
- For judge escalation: `GCP_PROJECT_ID` and Application Default Credentials
  (`gcloud auth application-default login`). Without these, escalation will
  fail fast — there is no fallback to a third-party API by design.

## Asking a question

```bash
seh chat "Who wrote the novel 1984?"
```

That's the whole interface. There are no flags you normally need to pass; all
behavior (sample count, threshold, model IDs) is read from `config.py` / `.env`.

If you only want the detection signal without the judge step:

```bash
seh detect "Who wrote the novel 1984?"
```

`detect` never escalates — it just prints SE/PE and Gemma's first sample. Useful
for benchmarking and for understanding the detector in isolation.

## Interpreting the output

A `seh chat` call prints three sections.

### Detection block

```
SE: 0.18 nats  PE: 1.42 nats  escalated: false  source: gemma-2-9b
```

- **SE** — Semantic Entropy, in nats. The headline number. See
  [Reading the Semantic Entropy number](#reading-the-semantic-entropy-number).
- **PE** — Predictive Entropy, the cheap token-logprob baseline. Shown for
  comparison; the system does **not** decide based on PE.
- **escalated** — whether the judge was invoked.
- **source** — which model produced the final answer (`gemma-2-9b` or
  `gemini-...`).

### Final answer

The actual answer to your question. If `escalated: false`, this is Gemma's
most-representative sample from the cluster with the highest weight. If
`escalated: true`, this is the judge model's answer.

### Judge reasoning

Only printed when `escalated: true`. A short paragraph from the judge
explaining what it concluded and why. This is the audit trail — when the
detector flags uncertainty, you get to see the stronger model's actual
reasoning rather than a black-box override.

## Reading the Semantic Entropy number

SE is reported in **nats** (natural-log units). Rules of thumb on TriviaQA-style
factoid questions:

| SE (nats)     | Interpretation                                              |
|---------------|-------------------------------------------------------------|
| `< 0.3`       | Strong agreement. All 10 samples say essentially the same thing. |
| `0.3 – 0.8`   | Mild paraphrase variation. Usually still correct.           |
| `0.8 – 1.5`   | Real semantic disagreement. The model is uncertain.         |
| `> 1.5`       | Strong disagreement. Likely hallucination or under-specified question. |

These are rough bands, not hard cutoffs. The configured `SE_THRESHOLD` (default
in `.env.example`) is what actually controls escalation.

## SE vs PE — why two numbers?

These are **two different things** and the README/CLI deliberately shows both.

- **Predictive Entropy (PE)** is computed from token-level log-probabilities of
  a single sample. It tells you how confident the model is about each *token*.
  It is cheap (one forward pass) but weak: a model can be very confident about
  the *words* of a wrong answer.
- **Semantic Entropy (SE)** is computed across N=10 samples, after clustering
  them by meaning via NLI. It tells you how much the model agrees with itself
  about the *answer*, ignoring surface wording. This is the strong signal.

If you see `PE: low, SE: high`, the model is fluent but inconsistent — classic
hallucination shape. If you see `PE: high, SE: low`, the model is hedging
verbally but converges on the same meaning every time.

## When the system escalates

Escalation is triggered when `SE > SE_THRESHOLD`. Concretely that means:

1. The 10 Gemma samples were clustered into multiple meaning-groups of
   comparable size.
2. The judge (Vertex Gemini) is called with your original question plus,
   optionally, the disagreeing samples as context.
3. The judge's answer replaces Gemma's, and its reasoning is shown.

The escalation rate is itself useful information — if `seh chat` is escalating
most of the time, either the threshold is too tight or the questions are
genuinely outside Gemma's knowledge.

## Tuning the threshold

Set `SE_THRESHOLD` in `.env`:

```
SE_THRESHOLD=0.8
```

- **Lower** threshold → more escalation → higher accuracy, higher latency and
  cost.
- **Higher** threshold → less escalation → faster and cheaper, but more
  hallucinations slip through.

The default is picked to roughly maximize the F1 of "should have escalated" on
the TriviaQA benchmark. See `eval/` and the changelog for the exact number.

## Worked examples

**1. Confident, correct.**

```
$ seh chat "Who wrote the novel 1984?"
SE: 0.12 nats  PE: 1.30 nats  escalated: false  source: gemma-2-9b
Final Answer: George Orwell.
```

All 10 samples named Orwell. Low SE, no escalation needed.

**2. Adversarial / unknowable — caught by the detector.**

```
$ seh chat "Who was the first woman to walk on Mars?"
SE: 2.05 nats  PE: 1.85 nats  escalated: true  source: gemini-1.5-pro
Final Answer: No human has walked on Mars as of 2026. The premise of the
question is false.
Judge Reasoning: The detector flagged high semantic disagreement across
samples. On review, no crewed Mars mission has occurred; ...
```

Gemma's 10 samples invented different names. SE spiked, the judge took over,
and the judge correctly rejected the premise instead of inventing an 11th name.

**3. Real disagreement on a contested fact.**

```
$ seh chat "What is the population of Tel Aviv?"
SE: 1.10 nats  PE: 1.50 nats  escalated: true  source: gemini-1.5-pro
Final Answer: The City of Tel Aviv-Yafo has ~470,000 residents (2024); the
metropolitan area (Gush Dan) is ~4 million. ...
```

Samples disagreed because the question is ambiguous (city vs metro). The
judge disambiguates instead of picking a random number.

## Limitations and caveats

- **Open-ended generation.** SE is designed for short factoid answers. On long
  essays, NLI clustering becomes noisy and SE becomes hard to interpret.
- **N=10 is fixed.** Lower N gives noisier SE; higher N is more expensive.
  Configurable in `config.py` but not currently a CLI flag.
- **The judge is not infallible.** Escalation reduces hallucination rate, it
  does not eliminate it. Treat the final answer as a stronger guess, not as
  ground truth.
- **No multi-turn.** Each `seh chat` call is independent. Conversation context
  is out of scope for v1 (see `CLAUDE.md`).
- **Cost.** Every `seh chat` runs 10 Gemma samples. On a paid Vertex endpoint
  that adds up. Tear down the endpoint when not in use.

## Troubleshooting

**`ConnectionError` to `VLLM_BASE_URL`.**
The vLLM container isn't reachable. Verify with
`curl $VLLM_BASE_URL/models`. If that fails, the endpoint is down or the URL
in `.env` is wrong.

**Escalation requested but judge call fails.**
Check `GCP_PROJECT_ID` is set and `gcloud auth application-default
login` has been run for the account that owns the project. The system fails
fast on missing credentials by design — there is no third-party LLM fallback.

**SE is always ~0 regardless of question.**
The N=10 samples are coming back identical, which usually means temperature
is set to 0 somewhere. Check `config.py` and the vLLM launch args.

**SE is always very high.**
NLI clustering may be misconfigured (wrong NLI model, or threshold too
strict). Re-run `uv run pytest tests/unit -v` — the entropy/clustering tests
should catch this.
