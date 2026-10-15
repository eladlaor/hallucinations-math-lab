# Glossary

> Terms used in `taste`, explained for talk-prep depth: etymology, pronunciation,
> analogy, technical definition, contrast, real-world use, and production caveats.

## Table of contents

- [Bidirectional entailment](#bidirectional-entailment)

---

## Bidirectional entailment

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
checkable substitute. In `taste`, this is how `cluster_by_bidirectional_entailment`
(`src/hallucinations_math_lab/services/semantic_entropy.py:39`) decides whether a new sampled answer
joins an existing cluster or starts a new one, which is the input to the Semantic
Entropy calculation.

**5. Contrast with similar concept.**
- **Unidirectional entailment** — A→B holds but B→A doesn't tested. Useful for
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
Semantic Entropy's core mechanism: sample N answers from an LLM at temperature
> 0, then use bidirectional entailment to group answers that mean the same thing
into clusters, even if worded completely differently ("Orwell wrote it,"
"George Orwell is the author," "1984 was written by Orwell"). The Shannon
entropy of the resulting cluster-size distribution becomes the hallucination
signal — this is exactly the "second ear" in the Reversim talk outline.

**7. Caveat when applying it in production.**
Bidirectional entailment is only as reliable as the underlying NLI model, which
was trained on curated sentence pairs (SNLI/MNLI-style), not on open-ended,
domain-specific QA answers. In production this means:
- **Degrades on long-form text** — entailment judgments become noisy past a
  few sentences, so clustering quality drops exactly where you'd most want a
  reliable signal (longer, more complex answers).
- **Order-sensitive in practice** — `taste`'s clustering is greedy (each new
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
  extension.
