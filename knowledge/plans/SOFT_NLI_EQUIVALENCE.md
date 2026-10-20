# Plan: Soft NLI Equivalence for Semantic Entropy

## Table of contents

- [Summary](#summary)
- [The problem](#the-problem)
- [Terminology: continuous is not calibrated](#terminology-continuous-is-not-calibrated)
- [Options considered](#options-considered)
- [Design](#design)
- [Work items (TDD)](#work-items-tdd)
- [Success criteria](#success-criteria)
- [Out of scope](#out-of-scope)
- [Risks](#risks)

## Summary

- **The problem:** SE decides whether two answers mean the same thing using
  DeBERTa's single most likely label. That decision has no knob, so it can't be
  tuned for a new domain.
- **The plan:** keep the paper's rule as the default, and add a second rule that
  merges two answers when the **entailment probability in both directions is at
  least `t`**. `t` is picked on a dev slice, never on the test slice.
- **Groundwork:** expose the label probabilities from `NLIScorer`. That step also
  fixes the hardcoded label order (a B2 item in [`TODO.md`](TODO.md)).
- **Cost of the sweep:** cache the directed-pair probabilities per question, so
  sweeping `t` costs no extra NLI calls.
- **Outcome:** ship it only if the measured result supports it. If it's no better
  than the default, it's reported as a negative result, not a promoted feature.

## The problem

`NLIScorer.classify` (`services/nli.py`) takes the top label from DeBERTa's three
outputs and throws the probabilities away. `bidirectional_entails` then requires
*entailment* in both directions. The consequences:

- **No operating point.** A pair at 51% entailment and a pair at 99% entailment are
  treated the same. When a domain makes DeBERTa hedge, for example by putting more
  probability on neutral for numbers, dates or units, the only response available
  is to swap the checkpoint.
- **An all-or-nothing rule.** A pair that leans just below the top label in one
  direction is split, even when the evidence for equivalence is strong overall.
- **Nothing to sweep.** Without a continuous score there's no curve to show and no
  sensitivity analysis to report.

This is what `knowledge/GLOSSARY.md` lists as "No calibrated confidence".

## Terminology: continuous is not calibrated

The glossary bullet uses "calibrated" loosely, and the plan depends on the
distinction:

- **Continuous score:** any number you can threshold, such as DeBERTa's softmax
  P(entailment). That's what this plan adds.
- **Calibrated probability:** a score that matches the real frequency. Of the pairs
  scored 0.8, about 80% really are equivalent. DeBERTa's softmax is **not**
  calibrated out of the box, because fine-tuned classifiers are typically
  overconfident.

A threshold tuned on a dev slice only needs the score to **rank** pairs well, not to
be calibrated. Calibration (for example, temperature scaling fitted on labeled
pairs) matters only if `t` has to carry over to another domain or checkpoint, or be
read as a probability. That's an optional extra, not part of the main path.

**Don't confuse the two thresholds:**

- **`t`** is the NLI equivalence threshold, on the 0–1 probability scale. It
  decides the clusters.
- **`τ` (`se_threshold`)** is the escalation threshold, measured in nats. It
  decides whether a question gets escalated.

## Options considered

| Option | What it does | Verdict |
|---|---|---|
| A. Keep the argmax rule, strict | The current behavior and the paper's default. | **Keep as default** for comparability with the paper. |
| B. Non-strict rule from the paper | No contradiction in either direction, and not both neutral. The reference code has it behind `--no-strict_entailment`. | **Add.** It's cheap, it comes from the paper, and it's a second reference point. |
| C. Probability threshold `t` | Equivalent iff P(ent \| a→b) ≥ `t` and P(ent \| b→a) ≥ `t`. | **Recommended addition.** It's the tunable knob this plan exists for. |
| D. Soft clustering / kernel entropy | Use pairwise probabilities as a similarity matrix and compute entropy without hard clusters (e.g., Kernel Language Entropy, Nikitin et al. 2024). | **Out of scope.** It's a new detector, and those are v2 per `CLAUDE.md`. |

Option C contains A as a near special case, but not exactly: the argmax can be
entailment at 0.40 when the other two labels are both lower. So A stays as its own
rule, and the default output doesn't change.

## Design

1. **`NLIScorer.label_probs(premise, hypothesis) -> dict[NLILabel, float]`.**
   - Returns the softmax over the logits.
   - Label names come from `model.config.id2label`, mapped onto `NLILabel`. If the
     model has a label that doesn't map, fail fast with the model id in the error.
   - `classify` becomes the argmax over `label_probs`.
   - `DEBERTA_MNLI_LABEL_ORDER` is deleted.
2. **`EquivalenceRule(StrEnum)`** in `constants.py`, with three members:
   - `STRICT_ARGMAX`, the default.
   - `NON_STRICT_ARGMAX`.
   - `ENTAILMENT_PROB_THRESHOLD`.
3. **Config** (`config.py`):
   - `se_equivalence_rule: EquivalenceRule = STRICT_ARGMAX`.
   - `se_entailment_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)`, which is
     read only by the threshold rule.
4. **`NLIScorer.bidirectional_entails(a, b)`** dispatches on the rule. The
   `EntailmentScorer` protocol is unchanged (`bidirectional_entails -> bool`), so
   custom scorers and `examples/` keep working.
5. **The offline sweep tool, `eval/sweep_nli_threshold.py`:**
   - The benchmark runner stores, per question, the full N×(N−1) directed
     `label_probs` matrix: 90 pairs at N=10, batched.
   - The sweep script re-clusters greedily from that cache for each `t` in a grid,
     recomputes SE, and reports the dev-slice AUROC per `t` and per rule.
   - Without the full matrix, the pairs that greedy clustering touches would depend
     on `t`, and each `t` would need new NLI calls.
6. **Splitting the data:**
   - `t*` is chosen on a dev slice that doesn't overlap the test slice, using the same
     split as the τ calibration in [`TODO.md`](TODO.md).
   - The test AUROC is reported once, at `t*`.

## Work items (TDD)

Each item starts with a failing test.

1. **Label probabilities from the model config.**
   - *Test:* a fake model config with labels in an unusual order still gets the right
     label for each probability, and the probabilities sum to 1.
   - *Test:* an unmapped label raises an error that names the model id.
2. **Equivalence rules.**
   - *Test:* given fixed fake probabilities, each rule merges or splits exactly as
     specified. Include a case where the argmax is entailment at 0.40.
   - *Test:* the default `STRICT_ARGMAX` reproduces today's clusters on the existing
     SE fixtures, so the default output doesn't change.
3. **Config wiring.**
   - *Test:* `se_equivalence_rule` and `se_entailment_threshold` reach `NLIScorer`.
   - *Test:* out-of-range thresholds fail validation.
4. **Probability cache in benchmark rows.**
   - *Test:* a row round-trips the directed-pair matrix.
   - *Test:* re-clustering from the cache equals clustering through live calls with a
     fake scorer.
5. **The sweep script.**
   - *Test:* on a synthetic cache where the right `t` is known, the script picks it.
   - *Test:* it refuses to run if the dev and test question ids overlap.
6. **Pair audit set (optional but recommended).**
   - Hand-label about 200 question-prefixed answer pairs from a dev run as
     equivalent or not.
   - Report precision and recall at the pair level for each rule. This is the direct
     check of the clustering decision, not just a proxy through AUROC.
   - Temperature scaling on this set is the place to add calibration, if it's
     needed.
7. **Docs.**
   - Walkthrough Step 3: a new question, "Why not just use the top label?"
   - Update the glossary bullet.
   - CHANGELOG `Added`.
   - QA guide: how to run the sweep.

## Success criteria

- **No regression:** with the default rule, every existing test passes and SE
  values are bit-identical on the fixtures.
- **Engineering:** the sweep runs from cached probabilities with zero extra NLI
  calls, and it's covered by tests.
- **Empirical,** on the first real GPU run, reported with `eval/inference_stats.py`
  (`auroc`, `paired_bootstrap`):
  - The dev-slice curve of AUROC against `t` for the threshold rule, with both
    argmax rules marked on it.
  - The test-slice SE AUROC at `t*` minus the strict-argmax AUROC, with a paired
    bootstrap 95% CI.
  - **Ship as recommended** only if the CI's lower bound is ≥ 0, meaning it's no
    worse. Otherwise keep it as a documented option and report the negative result.
- **If item 6 is done:** report pair-level F1 per rule, and note whether the most
  common failures are numbers, dates or units, the blind spot the glossary predicts.

## Out of scope

- Soft or kernel-based entropy (option D) and any other new detector.
- Swapping the NLI checkpoint, or using an LLM as the equivalence judge.
- Calibration that carries over across domains. Temperature scaling is offered only
  on the audit set.

## Risks

- **Overfitting `t` to a small dev slice.** Mitigated by the disjoint split and by
  reporting the full curve, not just `t*`. A flat curve is itself a useful finding.
- **The full-matrix cache costs more NLI calls** than greedy clustering: about 90 per
  question instead of about 10·K. It's batched and paid once, and the sweep
  depends on it.
- **The greedy order effect stays.** Changing `t` changes which answers become
  cluster representatives, so the clusters still depend on sample order (CAVEATS
  section 2). The plan doesn't fix that, and it's reported, not hidden.
