# Semantic Entropy — Walkthrough

> One step at a time: the **math**, the **code as it exists in this repo**, and
> **plain English** for every symbol and every line. Ends with how a number
> becomes the verdict "hallucination detected".

## Table of contents

- [Summary](#summary)
- [Notation cheat-sheet](#notation-cheat-sheet)
- [Step 0 — Shannon entropy, the one primitive](#step-0--shannon-entropy-the-one-primitive)
- [Step 1 — The baseline: predictive entropy](#step-1--the-baseline-predictive-entropy)
- [Step 2 — Sample N answers](#step-2--sample-n-answers)
- [Step 3 — Bidirectional entailment: "do these mean the same thing?"](#step-3--bidirectional-entailment-do-these-mean-the-same-thing)
- [Step 4 — Cluster the samples by meaning](#step-4--cluster-the-samples-by-meaning)
- [Step 5 — Entropy over clusters = Semantic Entropy](#step-5--entropy-over-clusters--semantic-entropy)
- [Step 6 — From a number to a verdict](#step-6--from-a-number-to-a-verdict)
- [Worked end-to-end examples](#worked-end-to-end-examples)
- [Why SE beats predictive entropy](#why-se-beats-predictive-entropy)
- [Where it breaks](#where-it-breaks)
- [Run it yourself](#run-it-yourself)

---

## Summary

Semantic Entropy (SE) asks the model the same question **N** times and measures
**how much the answers disagree in meaning** — not in wording.

1. Sample N answers at temperature > 0.
2. Group them into clusters of equivalent meaning, using a bidirectional-NLI test.
3. Compute Shannon entropy over the cluster-size distribution.

- All N answers mean the same thing → 1 cluster → **SE = 0** → confident.
- N answers scatter into many contradicting meanings → **SE ≈ log N** → confabulating.

The verdict is a threshold comparison: `SE > se_threshold` (default `1.0` nats)
→ escalate to the judge. Source: Farquhar et al., *Nature* 2024.

Code in this repo:

| Concern | File |
| --- | --- |
| SE math + clustering | `src/hallucinations_math_lab/services/semantic_entropy.py` |
| The "same meaning?" test | `src/hallucinations_math_lab/services/nli.py` |
| Baseline for comparison | `src/hallucinations_math_lab/services/predictive_entropy.py` |
| Threshold → verdict | `src/hallucinations_math_lab/escalation.py`, `src/hallucinations_math_lab/config.py:75` |
| Data shapes | `src/hallucinations_math_lab/custom_types/domain_types.py` |

---

## Notation cheat-sheet

Every symbol used below, said aloud the way an engineer says it.

| Symbol | Said aloud | Means |
| --- | --- | --- |
| `x` | "ex" | the question / prompt |
| `s` | "ess" | one answer, as a token sequence (a string) |
| `N` | "en" | how many answers we sample (default 10) |
| `s_n` | "ess sub en" | the n-th sampled answer |
| `p(s\|x)` | "pee of ess given ex" | probability the model assigns to producing answer `s` for question `x` |
| `C_k` | "see sub kay" | the k-th cluster — a bag of answers that all mean the same thing |
| `k` | "kay" | just an index: 1st meaning, 2nd meaning, … Not a quantity |
| `K` | "big kay" | the total number of distinct meanings found |
| `\|C_k\|` | "size of see sub kay" | how many of the N samples landed in cluster k |
| `Σ` | "sigma" / "sum over" | add up one term per item |
| `Π` | "pi" / "product over" | multiply one term per item |
| `H` | "aitch" | entropy — the uncertainty number we are computing |
| `log` | "log" / "natural log" | natural logarithm (base *e*). This repo uses nats, per the paper |
| `nats` | "nats" | the unit of entropy when using natural log (bits if base-2) |

---

## Step 0 — Shannon entropy, the one primitive

Everything here is one formula applied to different things. Learn it once.

### Math

```
H = - Σ_i  p_i · log(p_i)
```

- `p_i` — the probability of outcome *i*
- `Σ_i` — add up one term for every possible outcome
- `log(p_i)` — always negative, because probabilities are ≤ 1
- the leading minus sign flips it back positive, so `H ≥ 0`

### Python

```python
import numpy as np

def entropy(probs: list[float]) -> float:
    p = np.array(probs, dtype=np.float64)
    return float(-np.sum(p * np.log(p + 1e-12)))   # +1e-12 guards log(0)
```

### Plain English

Entropy is **average surprise**. An outcome you were sure about (`p = 1`)
surprises you not at all: `log(1) = 0`, so it contributes nothing. An outcome you
thought was unlikely (`p` small) has a very negative `log(p)`, so it contributes a
lot of surprise — weighted by how often it actually happens.

Two anchors worth memorizing:

```python
entropy([1.0])                      # 0.0     — certainty, no surprise
entropy([0.5, 0.5])                 # 0.693   — = log(2), a coin flip
entropy([0.1]*10)                   # 2.303   — = log(10), maximum for 10 outcomes
entropy([0.9, 0.05, 0.05])          # 0.394   — one dominant outcome, low uncertainty
```

The maximum for `K` equally-likely outcomes is `log(K)`. That gives you the ceiling
to compare any entropy value against.

---

## Step 1 — The baseline: predictive entropy

This is the cheap detector SE is measured against — and the one SE has to beat.

### Math

Over all possible token sequences:

```
H_pred = - Σ_s  p(s|x) · log p(s|x)
```

You can't enumerate every possible string, so estimate it by Monte-Carlo — average
over the N you actually sampled — and normalize by length so long answers aren't
penalized for having more tokens:

```
H_pred ≈ (1/N) · Σ_{n=1..N}  [ -(1/T_n) · Σ_{t=1..T_n}  log p(token_{n,t}) ]
                              └──────────── per-sample, length-normalized NLL ────┘
```

- `T_n` — number of tokens in sample n
- `log p(token_{n,t})` — the logprob the model reported for token t of sample n

### Python (`src/hallucinations_math_lab/services/predictive_entropy.py`)

```python
def length_normalized_predictive_entropy(sample_set: SampleSet) -> float:
    per_sample_nll: list[float] = []
    for i, s in enumerate(sample_set.samples):
        if not s.token_logprobs:
            logger.warning("Sample missing token logprobs; skipping.", ...)
            continue
        mean_lp = s.mean_token_logprob          # (1/T) * Σ_t log p(token_t)
        per_sample_nll.append(-mean_lp)         # negate → NLL
    if not per_sample_nll:
        raise ValueError("No samples had token logprobs; cannot compute predictive entropy.")
    return float(np.mean(per_sample_nll))       # (1/N) * Σ_n
```

`mean_token_logprob` lives on the `Sample` model in
`src/hallucinations_math_lab/custom_types/domain_types.py`:

```python
@property
def mean_token_logprob(self) -> float:
    if not self.token_logprobs:
        raise ValueError("Cannot compute mean token logprob: no token logprobs available.")
    return sum(t.logprob for t in self.token_logprobs) / len(self.token_logprobs)
```

### Plain English

For each answer, ask the model "on average, how confident were you in each word you
just wrote?" Flip the sign so *higher = less confident*. Average that across the N
answers.

**Why this is weak.** It measures confidence in *words*, not in *facts*. Two failure
directions:

- **False alarm:** the model says "Paris", "in Paris", "the city of Paris". Three
  different token sequences, so per-token probabilities are spread thin and
  `H_pred` rises — but the model was never uncertain about the answer.
- **Missed hallucination (the dangerous one):** the model confidently states a
  wrong fact in fluent prose. Every token is high-probability, so `H_pred` is low,
  and the detector says "confident" about a fabrication.

That second failure is why `escalation.py` documents that predictive entropy must
never gate SE via `AllAbove` — only ever add to it via `AnyAbove`.

---

## Step 2 — Sample N answers

### Math

Draw `s_1, …, s_N ~ p(· | x)` — that is, N independent draws from the model's own
answer distribution for question `x`, at temperature > 0 so the draws can differ.

### Python

Sampling happens in the vLLM client (`src/hallucinations_math_lab/services/sampling_provider.py`);
`semantic_entropy.py` receives the finished `SampleSet`. The shape it works with:

```python
class Sample(BaseModel):
    text: str
    token_logprobs: list[TokenLogprob] = Field(default_factory=list)

class SampleSet(BaseModel):
    prompt: str
    samples: list[Sample]
```

### Plain English

Ask the same question 10 times, with randomness turned on, and write down the 10
answers. Temperature 0 (greedy) would return the identical string every time and
tell you nothing — the whole signal lives in the *variation across independent
draws*.

N = 10 is the default. Larger N gives finer resolution (a meaning held by 1-in-20
samples is invisible at N=10) at linear generation cost.

---

## Step 3 — Bidirectional entailment: "do these mean the same thing?"

### Math

An NLI ("natural language inference") model maps a pair of sentences to one of three
labels:

```
NLI(premise, hypothesis) ∈ { entailment, neutral, contradiction }
```

Two answers `a` and `b` are **semantically equivalent** iff entailment holds *both
ways*:

```
equiv(a, b)  ⟺  NLI(a, b) = entailment  AND  NLI(b, a) = entailment
```

### Python (`src/hallucinations_math_lab/services/nli.py`)

```python
@torch.no_grad()
def classify(self, premise: str, hypothesis: str) -> NLILabel:
    """Return the predicted NLI label (entailment / neutral / contradiction)."""
    self._ensure_loaded()
    inputs = self._tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    ).to(self._device)
    logits = self._model(**inputs).logits
    idx = int(torch.argmax(logits, dim=-1).item())
    return DEBERTA_MNLI_LABEL_ORDER[idx]

def bidirectional_entails(self, a: str, b: str) -> bool:
    """True iff a entails b AND b entails a (the SE paper's clustering criterion)."""
    return (
        self.classify(a, b) == NLILabel.ENTAILMENT
        and self.classify(b, a) == NLILabel.ENTAILMENT
    )
```

The model is DeBERTa-MNLI, configured via `settings.nli_model_id`, loaded lazily on
first use and cached process-wide by `get_nli_scorer()`.

### Plain English

"Entailment" means: *if A is true, B must be true.* Run that test in both
directions.

Why both? One direction is too permissive:

| a | b | a ⇒ b | b ⇒ a | same meaning? |
| --- | --- | --- | --- | --- |
| "Paris" | "Paris, France" | yes | yes | **yes** — equivalent |
| "Paris" | "somewhere in Europe" | yes | no | **no** — b is strictly vaguer |
| "Paris" | "London" | no | no | **no** — contradiction |

Row 2 is the reason for the "bidirectional" requirement. "Paris" does entail
"somewhere in Europe", so a one-way test would wrongly merge a precise answer with a
vague one and understate the disagreement — hiding a hallucination.

Note the cost: each `bidirectional_entails` call is **two** forward passes through
DeBERTa.

---

## Step 4 — Cluster the samples by meaning

### Math

Partition the N samples into clusters `C_1, …, C_K` such that members of a cluster
are mutually equivalent. The paper's exact criterion is O(N²) pairwise; this repo
uses the standard greedy approximation — compare each new answer only against each
cluster's **representative** (its first member):

```
for n = 1..N:
    if ∃ k such that equiv(rep(C_k), s_n):   place s_n in that C_k
    else:                                     open a new cluster C_{K+1} with rep = s_n
```

Cost: `O(N · K)` entailment checks instead of `O(N²)`.

### Python (`src/hallucinations_math_lab/services/semantic_entropy.py`)

```python
def cluster_by_bidirectional_entailment(
    answers: list[str],
    scorer: NLIScorer | None = None,
) -> list[Cluster]:
    if not answers:
        raise ValueError("Cannot cluster empty answer list.")

    nli = scorer or get_nli_scorer()
    clusters: list[Cluster] = []

    for idx, ans in enumerate(answers):
        placed = False
        for cl in clusters:
            if nli.bidirectional_entails(cl.representative_text, ans):
                cl.member_indices.append(idx)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(member_indices=[idx], representative_text=ans))

    return clusters
```

(Error handling elided above for readability — the real function logs and re-raises
on NLI failure rather than silently starting a new cluster. Fail-fast.)

A `Cluster` is deliberately thin — indices plus the representative string:

```python
class Cluster(BaseModel):
    member_indices: list[int]
    representative_text: str

    @property
    def size(self) -> int:
        return len(self.member_indices)
```

### Plain English

Walk the 10 answers one at a time, holding a set of buckets.

For each answer, check it against the *first answer placed in each existing bucket*.
If it means the same thing as that bucket's representative, drop it in and stop
looking. If it matches no bucket, it starts a new bucket and becomes that bucket's
representative.

Concretely, for "Where is the Eiffel Tower?":

```
"Paris"           → no buckets yet          → open C1, rep="Paris"        C1=[0]
"in Paris"        → equiv to "Paris"? yes   → join C1                     C1=[0,1]
"Paris, France"   → equiv to "Paris"? yes   → join C1                     C1=[0,1,2]
"Rome"            → equiv to "Paris"? no    → open C2, rep="Rome"         C2=[3]
```

Result: `K = 2` clusters with sizes 3 and 1.

**The trade-off in the greedy approach:** comparing only against representatives can
split a cluster that a full pairwise pass would have merged (if answer 5 means the
same as answer 3 but the NLI model doesn't see it as equivalent to answer 1, the
representative). That biases SE slightly *upward* — toward over-flagging — which is
the safer direction for a hallucination detector, and it buys back a factor of
roughly N/K in NLI calls.

---

## Step 5 — Entropy over clusters = Semantic Entropy

### Math

Assign each cluster the fraction of samples it holds, then feed those probabilities
into Step 0's entropy formula:

```
p(C_k) = |C_k| / N

H_SE = - Σ_{k=1..K}  p(C_k) · log p(C_k)
```

- `|C_k| / N` — the empirical probability of meaning *k*
- The sum runs over **meanings**, not over strings. That is the entire idea.

Bounds: `0 ≤ H_SE ≤ log(N)`.

### Python (`src/hallucinations_math_lab/services/semantic_entropy.py`)

```python
def compute_semantic_entropy(sample_set: SampleSet, scorer: NLIScorer | None = None) -> SemanticEntropyResult:
    if not sample_set.samples:
        raise ValueError(
            f"Cannot compute SE: empty sample set for prompt={sample_set.prompt[:80]!r}"
        )

    answers = [s.text.strip() for s in sample_set.samples]
    clusters = cluster_by_bidirectional_entailment(answers, scorer=scorer)
    n_total = len(answers)

    probs = np.array([cl.size / n_total for cl in clusters], dtype=np.float64)
    # natural-log entropy (nats), matching the Nature paper
    se = float(-np.sum(probs * np.log(probs + 1e-12)))

    return SemanticEntropyResult(semantic_entropy=se, clusters=clusters)
```

Line by line:

| Line | Math it implements |
| --- | --- |
| `answers = [s.text.strip() ...]` | the sampled strings `s_1..s_N` |
| `clusters = cluster_by_...(answers)` | the partition `C_1..C_K` (Step 4) |
| `cl.size / n_total` | `p(C_k) = \|C_k\| / N` |
| `-np.sum(probs * np.log(probs + 1e-12))` | `-Σ p(C_k) log p(C_k)` |
| `+ 1e-12` | numerical guard; `log(0)` is `-inf` |

### Plain English

Count how many of the 10 answers fell into each meaning-bucket, turn those counts
into fractions, and measure how spread out those fractions are.

One bucket holding all 10 → the fraction is 1.0 → `log(1) = 0` → **SE = 0**. The
model gave one answer ten times over. It knows.

Ten buckets holding one each → ten fractions of 0.1 → **SE = log(10) = 2.303**. The
model produced ten different claims. It is generating from noise.

---

## Step 6 — From a number to a verdict

SE is a continuous score, not a boolean. The verdict is a threshold comparison.

### Math

```
escalate  ⟺  H_SE > τ
```

where `τ` ("tau") is `se_threshold` — default `1.0` nats (`src/hallucinations_math_lab/config.py:75`).

### Python (`src/hallucinations_math_lab/escalation.py`)

```python
@dataclass(frozen=True)
class SingleDetector:
    name: DetectorName
    threshold: float

    def should_escalate(self, scores: dict[DetectorName, float]) -> bool:
        return scores[self.name] > self.threshold
```

Two multi-detector policies exist alongside it, `AnyAbove` (OR) and `AllAbove`
(AND). The module docstring states the rule that matters:

> A false-negative-prone detector (e.g. predictive entropy) must never be used as a
> gate placed IN FRONT OF semantic entropy (do NOT require it via `AllAbove`). It may
> only be an additive OR trigger (`AnyAbove`) so it can raise escalation but never
> suppress an SE-driven escalation.

### Plain English

**"Hallucination detected"** means: `SE > 1.0` → don't trust the model's first
answer → escalate the question to the Gemini judge.

Where does 1.0 come from? It is calibrated, not derived. With N=10 the scale runs
0 → 2.303, and 1.0 nats corresponds roughly to "the samples split into about three
competing meanings, or two meanings with a meaningful minority". You pick it by
sweeping thresholds on a labeled dev set and choosing the point on the ROC curve
that matches your tolerance for false alarms versus missed hallucinations.

This is also why the project's success criteria are stated as **AUROC** (≥ 0.75 on
TriviaQA-1000, and ≥ 0.05 above predictive entropy) rather than as accuracy at one
threshold. AUROC measures whether SE *ranks* hallucinations above correct answers
across every possible threshold — a property of the detector. Accuracy at τ=1.0
measures the detector *and* one particular choice of τ.

The full record of the decision is preserved on `DetectionResult` — the score, the
threshold, and the policy description — so a reader can reconstruct why the system
escalated without re-reading config.

---

## Worked end-to-end examples

Same question, three model states. N = 10 throughout.

### A. Confident and correct — SE = 0

*"What is the capital of France?"*

```
samples  = ["Paris"] * 4 + ["Paris, France", "It's Paris", "The capital is Paris",
            "Paris.", "Paris is the capital", "paris"]
clusters = C1 = all 10                       K = 1
probs    = [10/10] = [1.0]
SE       = -(1.0 * log(1.0)) = 0.0
verdict  = 0.0 > 1.0 → False → no escalation
```

Ten different strings, one meaning. Note that **predictive entropy would be
non-zero here** — the wording varies — while SE is exactly 0. This is the false
alarm SE removes.

### B. Confabulating — SE ≈ 2.0

*"What year did the Treaty of Bruges dissolve the Hanseatic tariff union?"*
(a fabricated premise)

```
samples  = ["1358", "In 1442", "Around 1500", "1358", "The 14th century",
            "1387", "I believe 1602", "1358", "1250s", "1479"]
clusters = C1={1358, 1358, 1358}  C2={1442}  C3={~1500}  C4={14th c.}
           C5={1387}  C6={1602}  C7={1250s}  C8={1479}          K = 8
probs    = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
SE       = -(0.3·log0.3 + 7 × 0.1·log0.1)
         = -(0.3·(-1.204) + 7 × 0.1·(-2.303))
         = 0.361 + 1.612 = 1.973
verdict  = 1.973 > 1.0 → True → ESCALATE
```

Every individual answer is fluent and confident-sounding. The disagreement *across*
samples is the signal — no single answer reveals it.

### C. The borderline case — SE ≈ 0.5

```
samples  = ["Paris"] × 9 + ["Lyon"] × 1
clusters = C1 (9), C2 (1)                     K = 2
probs    = [0.9, 0.1]
SE       = -(0.9·log0.9 + 0.1·log0.1)
         = -(0.9·(-0.105) + 0.1·(-2.303))
         = 0.095 + 0.230 = 0.325
verdict  = 0.325 > 1.0 → False → no escalation
```

One dissenting sample out of ten. The model is *mostly* sure. Whether that should
escalate is exactly the judgment `τ` encodes: lowering τ to 0.3 would catch this
case and also flag many correct answers. That's the precision/recall dial.

### Reference points at N = 10

| Cluster sizes | K | SE (nats) | Reading |
| --- | --- | --- | --- |
| `[10]` | 1 | 0.000 | unanimous — trust |
| `[9, 1]` | 2 | 0.325 | one dissenter |
| `[8, 2]` | 2 | 0.500 | small minority |
| `[7, 2, 1]` | 3 | 0.802 | fraying |
| `[5, 5]` | 2 | 0.693 | a coin flip between two claims |
| `[4, 3, 2, 1]` | 4 | 1.280 | **above τ — escalate** |
| `[1]×10` | 10 | 2.303 | maximum — pure confabulation |

Note `[5,5]` = 0.693 sits *below* τ=1.0 despite being a genuine 50/50 split between
two contradicting claims. That is a real limitation of a single scalar threshold:
entropy rewards concentration, and two-way concentration still looks concentrated.
K is worth logging alongside SE for this reason.

---

## Why SE beats predictive entropy

The two detectors disagree precisely where it matters:

| Scenario | `H_pred` | `H_SE` | Who is right |
| --- | --- | --- | --- |
| Same fact, varied phrasing | **high** (false alarm) | 0 | SE |
| Confident fluent fabrication | **low** (missed) | high | SE |
| Genuine uncertainty | high | high | tie |
| Confident and correct | low | 0 | tie |

Predictive entropy is computed over **strings**; SE is computed over **meanings**.
Paraphrase variation is noise in the string space and vanishes in the meaning space.

The price is real: SE costs N generations plus `O(N·K)` NLI forward-pass pairs per
question, against a single forward pass for predictive entropy. That cost is why
**SEP** (Semantic Entropy Probes, `src/hallucinations_math_lab/services/semantic_entropy_probe.py`)
exists as the third detector — it trains a probe on the model's hidden states to
*predict* `H_SE` from one forward pass, trading some accuracy for most of SE's cost.

---

## Where it breaks

Taken from `SemanticEntropyDetector.metadata.failure_modes`:

- **NLI noise on long-form answers.** DeBERTa entailment judgments degrade past a
  few sentences. SE is reliable on short factual answers; multi-paragraph
  generations need a different equivalence test.
- **Consistent paraphrase variation inflates SE.** If the NLI model fails to see two
  phrasings as equivalent, they split into separate clusters and SE rises with no
  real disagreement. The greedy representative-based clustering makes this slightly
  more likely than full pairwise.
- **Fixed N = 10 caps resolution.** A divergent meaning appearing 1 time in 30 is
  usually invisible at N = 10.

Add to that:

- **Threshold transfer.** τ = 1.0 is calibrated for this model, this dataset, this
  N. It does not transfer unexamined to a different model or domain.
- **Two-way splits under-score.** As in the `[5,5]` row above.

See `knowledge/CAVEATS.md` for the project-level version of these.

---

## Run it yourself

The unit tests use a fake NLI scorer, so no GPU or model download is needed to watch
the math work:

```bash
uv run pytest tests/unit/test_semantic_entropy.py -v
```

To compute the reference table above by hand:

```python
import numpy as np

def se_from_cluster_sizes(sizes: list[int]) -> float:
    n = sum(sizes)
    p = np.array(sizes, dtype=np.float64) / n
    return float(-np.sum(p * np.log(p + 1e-12)))

for sizes in ([10], [9,1], [8,2], [7,2,1], [5,5], [4,3,2,1], [1]*10):
    print(f"{str(sizes):<18} K={len(sizes):<3} SE={se_from_cluster_sizes(sizes):.3f}")
```

To exercise the real pipeline with a stub NLI scorer:

```python
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropy
from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet

class FakeNLI:
    """Equivalent iff the strings share a lowercase token — stands in for DeBERTa."""
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return bool(set(a.lower().split()) & set(b.lower().split()))

samples = SampleSet(
    prompt="What is the capital of France?",
    samples=[Sample(text=t) for t in
             ["Paris", "in Paris", "Paris France", "Lyon"]],
)
result = compute_semantic_entropy(samples, scorer=FakeNLI())
print(result.semantic_entropy)          # 0.562
print([(c.representative_text, c.size) for c in result.clusters])
# [('Paris', 3), ('Lyon', 1)]

# Try it with "Paris, France" instead and you get SE=1.040 across 3 clusters:
# the naive token-overlap stub sees "paris," (comma attached) as a different
# token from "paris". A real NLI model does not make that mistake — which is
# exactly why the pipeline uses DeBERTa and not string matching.
```
