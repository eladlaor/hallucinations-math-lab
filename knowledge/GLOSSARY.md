# Glossary

## Table of contents

- [Summary](#summary)
- [What is bidirectional entailment?](#what-is-bidirectional-entailment)
- [What is disentangled attention, and how does DeBERTa use it?](#what-is-disentangled-attention-and-how-does-deberta-use-it)
- [What is AUROC, and how is it different from the ROC curve?](#what-is-auroc-and-how-is-it-different-from-the-roc-curve)

## Summary

Terms that the docs and code lean on, each explained in full: origin,
pronunciation, an analogy, the technical definition, how it differs from
look-alike concepts, where it is used here, and how it breaks in practice.

---

## What is bidirectional entailment?

Two texts bidirectionally entail each other when each one, taken as true,
forces the other to be true — the test this repo uses for "these two answers
mean the same thing".

**1. Etymology.**
"Entailment" comes from the logical/legal sense of "entail" — Old French *en-* +
*tailler* ("to cut, shape"), evolving to mean "to necessarily involve or bring about
as a consequence." In logic and later in NLP, "entailment" names the relation
where the truth of statement A guarantees the truth of statement B. "Bidirectional"
is Latin *bi-* ("two") + *directio* ("a directing, a straight line") — the relation
must hold in both directions at once.

**2. Pronunciation in casual speech between experts.**
"by-dye-REK-shun-ul en-TAIL-ment." In hallway conversation, NLP researchers often
just say "bi-entailment" or "mutual entailment," and shorten "does A entail B" to
"does A entail B" spoken plainly — no jargon contraction exists beyond that.

**3. Simple analogy explanation.**
Imagine two witnesses describing the same event in different words. If witness A's
story logically forces witness B's story to be true, and witness B's story
*also* forces witness A's to be true, they're describing the *same thing* — just
in different words. If only one direction holds ("A implies B" but not "B implies
A"), one witness said something more specific or added an extra detail — they
overlap, but aren't equivalent.

**4. Technical explanation.**
In NLI (Natural Language Inference), a model takes a premise and a hypothesis and
classifies the relation as one of `entailment`, `contradiction`, or `neutral`.
Bidirectional entailment between two texts A and B holds when:

```
entails(A, B) == entailment   AND   entails(B, A) == entailment
```

i.e., the NLI classifier says A→B is entailment *and*, run in the reverse
direction, B→A is also entailment. This is used as an operational proxy for
"A and B express the same meaning" (semantic equivalence), because true semantic
equivalence isn't directly computable — bidirectional entailment is the nearest
checkable substitute. In this repo, this is how `cluster_by_bidirectional_entailment`
(`src/hallucinations_math_lab/services/semantic_entropy.py:39`) decides whether a new sampled answer
joins an existing cluster or starts a new one, which is the input to the Semantic
Entropy calculation.

**5. Contrast with similar concept.**
- **Unidirectional entailment** — A→B holds but B→A is not tested. Useful for
  "is B a valid conclusion from A" (e.g., summarization faithfulness checks), but
  *not* sufficient for clustering by meaning: "Paris is the capital of France"
  entails "Paris is a city," but the reverse doesn't hold — they aren't the same
  claim, so unidirectional entailment alone would wrongly merge specific and
  generic answers.
- **Semantic similarity (embedding cosine distance)** — measures *how close*
  two texts are in a continuous vector space, not a binary logical relation.
  Two answers can be highly similar in embedding space (same topic, similar
  wording) while still contradicting each other factually — entailment checks
  logical consequence, not surface or topical closeness.
- **Paraphrase detection** — a related but distinct task trained to detect
  *lexical* rewording of the same statement; entailment is broader and
  logic-based, so it also merges genuinely different phrasings that a narrower
  paraphrase model might miss (or vice versa, depending on training data).

**6. Real-world use case.**
Semantic Entropy's core mechanism: sample N answers from an LLM at temperature bigger than 0, then use bidirectional entailment to group answers that mean the same thing
into clusters, even if worded completely differently ("Orwell wrote it,"
"George Orwell is the author," "1984 was written by Orwell"). The Shannon
entropy over the resulting clusters, each weighted by its answers' sequence
likelihoods (or by count alone, in discrete SE), becomes the hallucination
signal.

**7. Caveat when applying it in production.**
Bidirectional entailment is only as reliable as the underlying NLI model, which
was trained on curated sentence pairs (SNLI/MNLI-style), not on open-ended,
domain-specific QA answers. In production this means:
- **Degrades on long-form text** — entailment judgments become noisy past a
  few sentences, so clustering quality drops exactly where you'd most want a
  reliable signal (longer, more complex answers).
- **Order-sensitive in practice** — the repo's clustering is greedy (each new
  sample is compared only against existing cluster *representatives*, not
  full pairwise, for O(N·k) instead of O(N²) cost), so the final partition can
  depend on sample order, not just content.
- **Silently inherits NLI blind spots** — numeric, date, or unit-sensitive
  answers ("1984" vs. "in the year 1984" vs. "nineteen eighty-four") may be
  inconsistently judged depending on the NLI checkpoint's training
  distribution, with no visibility into when this happens unless you audit
  clusters by hand.
- **No calibrated confidence** — the entailment call is a hard label
  (entailment / neutral / contradiction), not a continuous score you can
  threshold or tune per-domain; softening this (e.g., using entailment
  probability instead of the argmax label) is a common but non-default
  extension. Note that a softmax probability is *continuous*, not *calibrated*:
  DeBERTa's probabilities are typically overconfident until measured against
  labeled pairs. Mitigation plan:
  [`plans/SOFT_NLI_EQUIVALENCE.md`](plans/SOFT_NLI_EQUIVALENCE.md).

---

## What is disentangled attention, and how does DeBERTa use it?

Disentangled attention is DeBERTa's way of handling word order: position is
never added into a token's hidden state, and is instead brought back inside
the attention score at every layer, as a separate term based on *distance*.

**1. Etymology.**
"Disentangle" is *dis-* ("undo") + *entangle* ("knot together"): to separate
two things that had been knotted into one. The term comes from the DeBERTa paper
(He et al., Microsoft, 2020). DeBERTa = **D**ecoding-**e**nhanced **BERT** with
disentangled **A**ttention. The thing being untangled is a token's *meaning*
from its *position*, which BERT mixes into one vector at the input.

**2. Pronunciation in casual speech between experts.**
"dis-en-TANG-guld uh-TEN-shun"; DeBERTa is "deh-BER-tah". People usually just
say "DeBERTa's attention" or "the disentangled thing", and name the checkpoint
when asked which model: "DeBERTa-large MNLI".

**3. Simple analogy explanation.**
You're deciding how much one word in a sentence should listen to another.
BERT hands you each word as one smoothie, with meaning and position blended
together, and you can only compare smoothies. DeBERTa hands you the word
itself, plus a separate note saying "this other word is 3 places to the
right". You can then ask "do these words relate?" and "does this word care
about something 3 places away?" as separate questions, and add up the answers.

**4. Technical explanation.**
A common misunderstanding: that DeBERTa keeps two vectors per token. It
doesn't. Like BERT, it has **one hidden-state vector per token per layer**.
What differs is what's in that vector and where position comes in.

- **BERT:** input = word embedding **+** absolute position embedding ("slot
  #7"). From then on, position is baked into the hidden state, and attention is
  a single dot product `Q_i · K_j` on the blended vectors.
- **DeBERTa:** input = word embedding only, with **no position added**.
  Position lives in a **separate table of relative-distance vectors**, one per
  offset `i − j`. That table belongs to the *pair* of tokens, not to either
  token. At every layer the attention score is the sum of three terms:

```
score(i,j) =  Qc_i · Kc_j          # content i  ↔ content j
            + Qc_i · Kr_(i−j)      # content i  ↔ distance i→j
            + Kc_j · Qr_(j−i)      # content j  ↔ distance j→i
            (scaled by 1/√(3d) instead of 1/√d)
```

  - `Qc`, `Kc` are projected from the hidden states (content).
  - `Qr`, `Kr` are projected from the distance table (relative position).
  - The **values** that get mixed into the next hidden state come only from
    content.

  The fourth possible pair, distance ↔ distance, is dropped: with relative
  positions it says nothing about the words.
- **Nuance:** hidden states still become position-aware *indirectly*, because
  the attention weights that mix them depend on distance. Position just never
  arrives as an explicitly added term. The one exception is near the output,
  where absolute positions are added once (the "enhanced mask decoder", which
  is the "decoding-enhanced" in the name; it does **not** make DeBERTa a
  decoder).

**5. Contrast with similar concept.**
- **Absolute position embeddings (BERT)** are added once at the input, so
  meaning and position are blended for every later layer.
- **Rotary position embeddings (RoPE, used by Gemma)** are also relative, but
  they rotate `Q` and `K` by an angle that depends on position. Position and
  content still meet inside **one** dot product, not three separate terms.
- **Bidirectional attention** is an unrelated property: every token can see
  tokens to its left and right (no causal mask). DeBERTa has both properties,
  but they're independent of each other. Neither is the same as the
  *bidirectional entailment* SE uses (see above).

**6. Real-world use case.**
DeBERTa is the NLI model behind this repo's SE clustering
(`microsoft/deberta-large-mnli`, the original v1 checkpoint, set by
`nli_model_id` in `src/hallucinations_math_lab/config.py`). It gets higher NLI
accuracy than BERT at a similar size, partly because word-order details like
"*not* right before the verb" are learned as separate distance-aware signals.
That's why it's the usual NLI model in the semantic-entropy literature, and
using it keeps this repo's results comparable to the paper's.

**7. Caveat when applying it in production.**
- **Length limit:** the distance table only covers offsets up to a fixed
  maximum, and the checkpoint accepts at most 512 tokens per input. The SE
  input is question + answer A + answer B, so long answers can hit that cap.
- **Cost:** three score terms per attention pair instead of one makes DeBERTa
  slower than a BERT of the same size. SE also makes two NLI calls per answer
  pair, so this overhead adds up.
- **Don't say "v3":** DeBERTa-v3 is a newer family that also changes the
  training objective. Swapping it in changes the NLI model, which becomes a
  variable you'd have to control for when comparing to the paper.

---

## What is AUROC, and how is it different from the ROC curve?

AUROC is the probability that, given one hallucinated answer and one correct
answer picked at random, the detector gives the hallucinated one the higher
score. The ROC curve is the picture behind it; AUROC is the area under that
picture.

**1. Etymology.**
"ROC" is *Receiver Operating Characteristic*, from WWII radar engineering:
how well a radar *receiver* told real aircraft from noise at each gain
setting it was *operated* at. "AUROC" is *Area Under the ROC curve*. "AUC"
alone means area under *a* curve, without saying which.

**2. Pronunciation in casual speech between experts.**
"AUROC" is "aw-rock". "ROC" is "rock". "AUC" is said letter by letter,
"A-U-C". In conversation a bare "AUC" is understood as AUROC by default; say
"AUROC" or "AUPRC" ("A-U-P-R-C") when precision-recall curves are also in
the discussion.

**3. Simple analogy explanation.**
Take the detector's scores for many answers and line the answers up from
most to least suspicious. Now pick one hallucinated answer and one correct
answer from the line at random. AUROC is how often the hallucinated one
stands ahead of the correct one. A perfect detector puts every hallucination
ahead (1.0); a detector that shuffled the line randomly gets it right half
the time (0.5).

**4. Technical explanation.**
- **The ROC curve** is built by sweeping a threshold over the detector's
  score. Each threshold gives one point: the false-positive rate (share of
  correct answers wrongly flagged) on the x-axis, the true-positive rate
  (share of hallucinations caught) on the y-axis. The points run from (0,0)
  for the strictest threshold to (1,1) for the most lenient. The false
  negatives are on the plot too: the miss rate is 1 minus the y-value.
- **AUROC** is the area under that curve, and it equals the pairwise ranking
  probability in the definition above (ties count half). Two equivalent
  readings: the ranking view (the definition), or the average share of
  hallucinations caught across every false-positive rate from 0 to 1.
- **It has no unit.** Both axes are fractions, so the area is a plain number
  between 0 and 1 — a dimensionless probability, reported as a decimal
  (0.79) or a percentage (79%).
- **The detector's output is a score, not a probability.** SE is an entropy
  in nats; SEP outputs a probability; PE is another entropy. AUROC only uses
  their *order*, which is why detectors on different scales can be compared
  on the same 0–1 scale.
- **Reading the number:** 0.5 is no signal (the diagonal of the ROC plot).
  Below 0.5 the detector ranks backwards, and flipping its sign gives
  1 − AUROC. Near 0.99 on TriviaQA is a red flag for label leakage, not a
  win; 0.75–0.85 is strong for a real hallucination detector.

**5. Contrast with similar concept.**
- **The ROC curve** — a curve, not a number. Use it to *choose* a threshold
  for how many false alarms you can tolerate. AUROC summarises the whole
  curve and picks no threshold.
- **Accuracy** — needs a threshold, and answers "how often is the
  thresholded call right". AUROC answers "how well does the score *rank*
  wrong answers above right ones". Two detectors with equal AUROC can differ
  sharply at the threshold you'd actually run, e.g. at a 5% false-alarm rate.
- **AUPRC** — the area under the precision-recall curve. It reacts to how
  rare hallucinations are in the dataset; AUROC does not.
- **Comparing AUROCs** — "SE beats PE by 0.05" means 0.05 on the 0–1 scale.
  "5% higher" is ambiguous (it could mean 5% *of* SE's value), so this repo
  uses absolute decimals.

**6. Real-world use case.**
This repo's first two success criteria are AUROC numbers: SE's AUROC on
TriviaQA must reach 0.75, and SE must beat the predictive-entropy baseline by
at least 0.05. SEP is judged by how much of SE's AUROC it keeps at a fraction
of the cost. `eval.compute_auroc` reports each detector's AUROC with a
bootstrap 95% CI, and SE's margin over PE with a paired-bootstrap CI.

**7. Caveat when applying it in production.**
- **It says nothing about the operating threshold.** A deployment runs at one
  threshold; read the ROC curve (or `eval.calibrate_threshold`) for that, and
  the accuracy-vs-escalation-rate curve (`eval.escalation_uplift`) for what
  escalation actually buys.
- **It is only as good as the labels.** Here a row is "hallucinated" when the
  greedy answer contains none of the accepted TriviaQA aliases. Matching
  errors, such as a long answer that happens to contain an alias, move AUROC
  without the detector changing at all.
- **Small samples are noisy.** On 200 questions, a 0.03 AUROC gap can sit
  inside the confidence interval. Compare intervals, not point estimates.
