# Taste

Response evaluators for LLM-native projects. Detect when a model is likely
hallucinating, route uncertain answers to a stronger judge, and place each
detector on a cost / latency / accuracy frontier you can actually plan around.

Ships as **two surfaces**:

1. **A library** — `from taste import evaluate_responses, ResponseEvaluator`.
   Drop into any LLM-native project (langrag, your own agent stack, anything
   producing N samples per prompt) and get back per-detector scores plus an
   optional judge verdict.
2. **A reference demo** — `taste chat "..."` / `taste detect "..."`. Drives
   a self-served small model (Gemma 4 family on vLLM, single-L4 deploy) so
   the SE / PE / SEP pipeline is observable end-to-end without standing up
   your own LLM first.

## Why

Most LLM-native projects ship a single answer and pray. Taste's thesis is
that the model has more to say than its first sample — and that the *shape*
of its multi-sample distribution is the signal you want for hallucination
detection. Three detectors implement that thesis at three points on the
cost/accuracy frontier. Each rests on a different assumption about *where*
hallucination signal lives — at the token level, in the meaning of the
samples, or already encoded inside the model's hidden state. Those
assumptions are what determine when each detector works and when it
silently fails.

### Detection methods: assumptions and failure modes

**Predictive Entropy (PE) — token-logprob baseline, one sample.**

- *What it measures.* The mean across samples of the length-normalized
  negative log-likelihood: `H_pred = (1/N) Σ_i ( -(1/T_i) Σ_t log p(tok_{i,t}) )`.
  In the single-sample case it collapses to the per-token NLL of the one
  sample you have.
- *Core assumption.* Low *token-level* confidence correlates with
  incorrect answers — i.e., when the model is wrong it tends to be wrong
  on tokens whose conditional probability was already low. The signal is
  purely *lexical*: it sees probabilities over surface tokens, not
  meanings.
- *What this implies.* PE only catches hallucinations that the model
  itself was lexically uncertain about. It treats every paraphrase as a
  distinct event, so two semantically identical answers with different
  wording look as uncertain as two contradictory answers.
- *Failure modes.* (a) Confident hallucinations — the model is wrong but
  high-probability on every token, and PE stays low. (b) Short answers,
  where length normalization divides by very small T and a single
  low-probability token dominates the score. (c) Tasks where the way answers are worded—what’s called "surface variation"—differs a lot, but their underlying meaning doesn’t change; for example, there are many different valid phrasings or paraphrases for the same correct answer. PE treats this diversity in wording as a sign of uncertainty, even though the responses are semantically consistent.
  even though the model is consistent.
- *Cost.* Free in marginal terms — token logprobs are a by-product of
  generation. PE is the right baseline precisely because it costs
  nothing the inference call hasn't already paid for.

**Semantic Entropy (SE) — Farquhar et al., *Nature* 2024.**

- *What it measures.* Sample N=10 generations at T>0, greedily cluster
  them by bidirectional NLI entailment (each new sample joins a cluster
  iff it entails *and* is entailed by that cluster's representative),
  then take the Shannon entropy of the cluster-size distribution:
  `SE = -Σ_c (|c|/N) log(|c|/N)` in nats. One cluster → SE=0; N
  singleton clusters → SE=log N.
- *Core assumption.* When the model is hallucinating, *independent
  samples diverge in meaning* even when each one individually sounds
  fluent. The hallucination signal lives in *semantic disagreement
  across the posterior*, not in any single sample's token
  probabilities. NLI is treated as a reliable proxy for "do these two
  answers mean the same thing?" — a strong assumption that the
  detector inherits from the underlying NLI model.
- *What this implies.* SE catches the failure mode PE misses by
  construction: the model can be lexically confident on each sample
  individually and still expose itself as uncertain by giving
  contradictory answers across samples. Conversely, SE is *insensitive*
  to paraphrase — ten different wordings of "George Orwell" cluster
  into one, and SE stays near zero.
- *Failure modes.* (a) NLI noise on long-form answers — bidirectional
  entailment becomes unreliable past a few sentences, so clusters
  fragment or merge incorrectly. (b) Consistent paraphrase that the NLI
  model fails to recognize as equivalent inflates SE even when meaning
  is stable. (c) Fixed N caps the resolution; with N=10 you cannot
  distinguish entropies finer than roughly log(10)/10 nats, and rare
  divergent meanings can be missed entirely. (d) The whole pipeline
  assumes the answer space is *clusterable* — for open-ended generative
  tasks (essays, code) the notion of "two answers mean the same thing"
  is itself ill-defined.
- *Cost.* N forward passes through the SLM plus O(N · |clusters|) NLI
  calls. Strong signal, real wall-clock cost.

**Semantic Entropy Probe (SEP) — Kossen et al. follow-up.**

- *What it measures.* A single `nn.Linear(hidden_dim, 1)` regressor
  over the SLM's pooled last-layer hidden state at the final prompt
  token, trained to predict SE (in nats) as a continuous target. Score
  units match SE's so the same escalation threshold carries over.
- *Core assumption.* Hallucination uncertainty is *linearly decodable*
  from a single forward pass over the prompt — i.e., the model's
  internal representation already "knows" how uncertain its eventual
  multi-sample distribution will be, and that knowledge lies on a
  linear subspace of the hidden state. This is a much stronger claim
  than SE's: SE only assumes the *outputs* disagree under
  hallucination; SEP assumes the *prompt-time hidden state* already
  encodes that future disagreement.
- *What this implies.* If the assumption holds, you get SE-quality
  signal for the cost of one extra matrix-vector multiply. The probe
  is *model-specific* by construction — it is trained against one
  SLM's representations and does not transfer.
- *Failure modes.* (a) Probe is SLM-specific — weights trained on
  model A do not work on model B (different hidden dimensions, but
  also different geometry of "uncertainty" in latent space). (b)
  Underperforms full SE on adversarial / out-of-distribution prompts
  where the linear approximation of SE breaks down. (c) Requires
  hidden-state access; useless against any provider that does not
  expose pooled embeddings (most hosted APIs). (d) Inherits whatever
  bias the SE training labels carried — if SE was miscalibrated on a
  subset of the training distribution, the probe learns that
  miscalibration.
- *Cost.* One linear forward pass. The cheapest of the three by orders
  of magnitude in latency.

**Pareto positioning.** PE is the floor: free, lexical-only. SE is the
ceiling on signal quality at non-trivial cost. SEP attempts to recover
most of SE's signal at PE-level cost by amortizing the work into an
offline probe-training step. The eval harness measures whether the
attempt succeeds on this dataset and this SLM.

A fourth family — LLM-as-judge (Llama 3.2 3B Instruct) — lands as an
external-judgment detector for the Pareto comparison. It rests on a
qualitatively different assumption: that a second, possibly stronger
model can recognize errors that the first model could not — a
*verification asymmetry* claim rather than an uncertainty-quantification
one.

## Library use

```python
from taste import evaluate_responses, Sample, SampleSet, TokenLogprob
from taste import PredictiveEntropyDetector, SemanticEntropyDetector, DetectorName

# Your LLM produced N=10 samples for the prompt already. Just pass them in.
samples = SampleSet(
    prompt="Who wrote 1984?",
    samples=[
        Sample(
            text="George Orwell.",
            token_logprobs=[TokenLogprob(token="George", logprob=-0.1), ...],
        )
        for _ in range(10)
    ],
)

result = evaluate_responses(
    prompt="Who wrote 1984?",
    samples=samples,
    detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector()],
    escalation_detector=DetectorName.SEMANTIC_ENTROPY,
    escalation_threshold=1.0,
    greedy_answer="George Orwell.",  # your model's deterministic answer
    judge=None,  # optional: pass a Judge to escalate uncertain answers
)

print(result.detector_scores)        # {PE: 0.42, SE: 0.12}
print(result.escalated)              # False
print(result.final_answer)           # "George Orwell."
```

The library never drives an LLM in this path. Consumers like
[langrag](../langrag) own their inference stack and pass samples in.

### Multi-detector escalation policies

The scalar `escalation_detector` / `escalation_threshold` pair encodes a
single-detector decision (`SingleDetector`). To combine detectors, pass an
`escalation_policy` instead — it **takes precedence** over the scalar pair:

```python
from taste import evaluate_responses, AnyAbove, DetectorName

result = evaluate_responses(
    prompt="Who wrote 1984?",
    samples=samples,
    detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector()],
    escalation_detector=DetectorName.SEMANTIC_ENTROPY,   # baseline, still recorded
    escalation_threshold=1.0,
    greedy_answer="George Orwell.",
    escalation_policy=AnyAbove({
        DetectorName.SEMANTIC_ENTROPY: 1.0,
        DetectorName.PREDICTIVE_ENTROPY: 2.0,
    }),
)

print(result.escalated)               # True if EITHER detector crosses
print(result.escalation_policy_desc)  # "ANY(semantic_entropy > 1.0, predictive_entropy > 2.0)"
```

When `escalation_policy` is `None`, behavior is byte-identical to the
single-detector path. When supplied, the policy decides and the
human-readable reason is recorded on `result.escalation_policy_desc`; the
scalar `escalation_detector` / `escalation_threshold` are still carried on
the result as the caller-passed baseline.

#### Detector roles in escalation policies

A false-negative-prone detector (PE / predictive entropy) **MUST NOT** be
used as a pre-filter or gate placed IN FRONT OF semantic entropy — never
`AllAbove({PE, SE})` to require PE to fire first, because a PE miss would
then suppress an SE-driven escalation. PE may only be an additive OR trigger
(`AnyAbove({SE, PE})`), so it can raise escalation but never suppress an
SE-driven one.

## Demo use

```bash
cp .env.example .env
# Set GCP_PROJECT_ID, point VLLM_BASE_URL at your serving endpoint
uv sync --extra dev
uv run taste chat "Who was the first woman to walk on Mars?"
```

The demo wires up:

- A `SamplingProvider` (`demo/slm_provider.py`) that hits a vLLM
  OpenAI-compat endpoint; optionally a sibling `--task embed` instance
  for SEP hidden states.
- The Gemma 4 model registry (`demo/model_registry.py`) selecting flags
  + GPU SKU per model.
- `ResponseEvaluator` from the library, driving the detector pipeline.
- `GeminiJudge` as the reference escalation model.

See `knowledge/usage_guides/USER_GUIDE.md` for end-to-end interpretation
of the CLI output.

## Repository layout

```
src/taste/                  # the library — no demo dependencies
  __init__.py               # public surface
  evaluator.py              # evaluate_responses + ResponseEvaluator
  constants.py              # DetectorName, DetectorCategory, AnswerSource
  config.py                 # Pydantic Settings (env-driven)
  custom_types/             # DetectionResult, Sample, SampleSet, Cluster, ...
  services/
    detector_protocol.py    # Detector, DetectionContext, DetectorScore, DetectorMetadata
    semantic_entropy.py     # SemanticEntropyDetector + bidirectional-NLI clustering
    predictive_entropy.py   # PredictiveEntropyDetector
    semantic_entropy_probe.py  # SemanticEntropyProbeDetector + SEPProbe
    nli.py                  # DeBERTa-MNLI wrapper used by SE
    sampling_provider.py    # SamplingProvider protocol
  judges/
    protocol.py             # Judge protocol
    gemini.py               # GeminiJudge reference impl (Vertex)

demo/                       # the reference demo — depends on the library
  cli.py                    # the `taste` command
  slm_provider.py           # VLLMOpenAIProvider, HFTransformersProvider
  model_registry.py         # the Gemma 4 family entries

serving/                    # vLLM container + Vertex deploy
eval/                       # benchmark + probe training scripts
tests/unit/                 # entropy math, dispatch, registry, provider
knowledge/                  # guides, plans, changelogs, obstacles
```

## Status

The library API (`evaluate_responses`, `ResponseEvaluator`, three detectors,
judge protocol, and the escalation policies `EscalationPolicy`,
`SingleDetector`, `AnyAbove`, `AllAbove`) is stable for consumer use. The Pareto-frontier extension
(LLM-judge detector, cost/latency reporting in the eval harness) is in
flight — see `knowledge/plans/PARETO_DETECTOR_FRAMEWORK.md`.

Verification: `uv run pytest tests/unit` — 44 passed, 1 GPU-gated skip.
