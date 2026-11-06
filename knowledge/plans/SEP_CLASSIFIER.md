# Plan: SEP as the paper's classifier, with a layer sweep and a publishable probe

## Table of contents

- [Summary](#summary)
- [Which layer and token does the probe read, and are mid layers better?](#which-layer-and-token-does-the-probe-read-and-are-mid-layers-better)
- [Decisions and why](#decisions-and-why)
  - [Why binarize SE instead of regressing it?](#why-binarize-se-instead-of-regressing-it)
  - [Which SE goes into the label?](#which-se-goes-into-the-label)
  - [Where exactly is the high/low split?](#where-exactly-is-the-highlow-split)
  - [Which classifier, and why no feature scaling?](#which-classifier-and-why-no-feature-scaling)
  - [How do we number layers?](#how-do-we-number-layers)
  - [How is the layer window chosen without peeking at the test rows?](#how-is-the-layer-window-chosen-without-peeking-at-the-test-rows)
  - [Which token positions are collected, and which one is served?](#which-token-positions-are-collected-and-which-one-is-served)
  - [How does each provider hand over hidden states?](#how-does-each-provider-hand-over-hidden-states)
  - [How is the probe tied to one exact backbone and prompt?](#how-is-the-probe-tied-to-one-exact-backbone-and-prompt)
- [Deviations from the paper](#deviations-from-the-paper)
- [Data flow](#data-flow)
- [Artifact format](#artifact-format)
- [Config and environment changes](#config-and-environment-changes)
- [Threshold semantics and migration for users](#threshold-semantics-and-migration-for-users)
- [Backward compatibility for old probes](#backward-compatibility-for-old-probes)
- [Docs to update](#docs-to-update)
- [Ordered TDD work items](#ordered-tdd-work-items)
- [Changes to PUBLISHED_SEP_PROBE.md](#changes-to-published_sep_probemd)
- [What needs a real GPU run](#what-needs-a-real-gpu-run)
- [Deferred follow-ups](#deferred-follow-ups)
- [Open decisions for the owner](#open-decisions-for-the-owner)

## Summary

**What changes.** The Semantic Entropy Probe (SEP) moves from Ridge regression on raw SE
(in nats) to the recipe in Kossen et al. 2024 (arXiv 2406.15927 v1; official code
`github.com/OATML/semantic-entropy-probes` at commit `02e2167`):

1. Compute **discrete SE**, the share of samples in each meaning cluster.
2. Split it into high/low with the paper's **best-split** rule.
3. Fit sklearn `LogisticRegression()` with the paper's defaults.
4. **Sweep every layer**, at two token positions, and report AUROC.
5. Pick a **5-layer window** by nested cross-validation.
6. Output **p(high SE)**, a probability, with its own escalation threshold, `SEP_THRESHOLD`.

The trained probe is a directory: safetensors weights plus a config that pins the backbone,
revision, layers, token position and prompt. The detector loads it from disk or from
`hf://<repo>@<rev>` and refuses any mismatch before scoring. That directory is exactly what
[`PUBLISHED_SEP_PROBE.md`](PUBLISHED_SEP_PROBE.md) publishes. This plan carries out that plan's
items 1–6 and 8, and changes it as listed in
[Changes to PUBLISHED_SEP_PROBE.md](#changes-to-published_sep_probemd).

**Actionable now (no GPU):**

- Work items 1–18, each with its failing test written first.
- Item 1 fixes a real bug. Today, `escalation_detector=semantic_entropy_probe` compares SEP's
  score against `SE_THRESHOLD` (`evaluator.py:271`, `demo/web/app.py:183,227`). Once SEP outputs
  a probability, that comparison against a nats threshold would never escalate.

**Needs a GPU:** collecting the 500-row calibration set, the actual best layer for Gemma 4,
convergence at `max_iter=100`, and HF-vs-vLLM agreement. Until measured, all of these are
assumptions. See [What needs a real GPU run](#what-needs-a-real-gpu-run).

**Owner decisions:** the Hub namespace, the Gemma license treatment, which token position v1
publishes (after the sweep), and how to serve a non-last-layer probe on vLLM.

**Verify every item with:**

```
uv sync --locked --all-extras && uv run pytest -q && uv run mypy \
  && uvx ruff@0.17.0 check . && uvx ruff@0.17.0 format --check .
```

Also run the frontend's `npm test` for item 17.

## Which layer and token does the probe read, and are mid layers better?

**Short answer.** "A token in the more mid-layers" mixes up two separate choices that vary
independently:

- **Token position** is *which token's* vector we read. TBG ("token before generation") is
  the last token of the prompt. SLT ("second-last token") is the second-last token of the
  generated answer.
- **Layer** is *how deep* in the network we read that vector.

The README's sentence "the hidden state at the last token of the chat-formatted prompt"
(`README.md:102`) is about **position**, and it is correct. What it leaves out is the
**layer**: today the code always reads the last one (`README.md:333`,
`demo/slm_provider.py:399-401`).

**The misconception to head off: "research shows mid layers are best."** In Kossen et al.,
that holds for two settings that are not ours:

- **The accuracy-probe baseline.** These probes are trained to predict correctness. Their
  chosen windows are Llama-2-7B [18-22], Mistral-7B [12-16] and Llama-3-70B long [31-35]
  (App. B.3, Table 4).
- **Long-form answers.** Here mid-layer peaks appear "more often" (Sec. 6, Fig. 4).

For **short answers**, which is our TriviaQA setting, SEP AUROC rises toward the **late**
layers (Sec. 6, Figs. 2–3). The chosen SEP windows are the last five layers for Llama-2-7B,
Mistral-7B and Llama-2-70B. Phi-3-3.8B is the one mid-to-late case, at [21-25]/32 (Table 4).
The "mid layers" belief may come from other truthfulness-probe work; that source was not
identified here.

**What the owner's instinct gets right.** "Last layer" is an assumption, not a measurement.
Gemma was never tested in the paper. So this plan **measures** every layer, at both positions,
and lets the data choose. Nothing about "mid vs late" for Gemma 4 will be claimed until the
first GPU sweep exists.

**A second correction for the code.** The comment at `demo/slm_provider.py:399` says "Matches
the Kossen et al. SEP setup". It is wrong on two counts:

- **Layer:** the paper sweeps layers; it does not fix the last one.
- **Position:** the paper's TBG is the last token of a **plain-text** `...Answer:` prompt. Its
  code never calls `apply_chat_template`. Ours is the last token of the **Gemma chat template**,
  which is the opening of the model's turn. This is a different position, and its behaviour is
  unverified.

## Decisions and why

Evidence tags:

- **verified** means checked against the paper text or the official code.
- **likely** means inferred from code reading, not observed.
- **unverified** means an assumption for the first GPU run.

### Why binarize SE instead of regressing it?

**Answer.** Because that is the paper's method (Sec. 4, "Binarization"). It also matches how SEP
is used: we only need to know "is SE high enough to escalate?", not the exact value in nats.

**The misconception to head off:** that a regressor is more informative and therefore better.
Discrete SE with N=10 samples takes only a handful of distinct values and is noisy near the
middle. The paper's own comparison metric is AUROC of a classifier (Figs. 2–3, Tables 1–2).

**Consequence.** The score is no longer on SE's scale. It is a probability in [0, 1]. Pearson r
is dropped and AUROC replaces it.

### Which SE goes into the label?

**Answer: discrete SE.** Each cluster's probability is its count divided by N, and
SE = −Σ p log p, in nats. This is the code's `cluster_assignment_entropy` (verified).

Our SE detector defaults to the likelihood-weighted estimator. The collector therefore clusters
**once** and applies both estimators to the same clusters (`_cluster_probs` already supports
`SemanticEntropyEstimator.DISCRETE`). It stores both values and trains on the discrete one.
No extra NLI calls are made.

### Where exactly is the high/low split?

**Answer: the best-split rule.** γ\* is the threshold that minimizes the total within-group
squared error of SE (Sec. 4, Eq. 5; verified). The label is 1 when SE > γ\*.

**Our search differs from the paper's, on purpose (deviation D11).** The code searches a
100-point grid, `np.linspace(1e-10, max SE, 100)`. That grid **ends exactly at max(SE)**.
For a small model on TriviaQA, many rows will sit at SE = ln 10 ≈ 2.303, where all 10 answers
are distinct. A split at that endpoint ties with real data, and the paper's code labels ties −1.
The paper's rule is also internally inconsistent: the groups are SE < γ and SE ≥ γ, but the
label is SE > γ.

We instead search **exhaustively over the midpoints between consecutive distinct SE values**:

- No sample can sit on γ, so the −1 case disappears.
- The search is exact and deterministic.
- When several γ give equal error, the lower γ wins.
- SE is rounded to `SE_SPLIT_DECIMALS` (10) before the search and before labeling. The
  collector sums cluster entropies in discovery order, so one partition (e.g. [7, 2, 1]) or
  two partitions with equal entropy (e.g. [4, 2, 1, 1, 1, 1] and [2] × 5) give SE values a
  few ulp apart. Without rounding they would count as distinct levels that γ could fall
  between, and values exactly 1 ulp apart would have no midpoint at all. For N = 10 (the collector's fixed N), distinct discrete levels are at least
  2.9 × 10⁻³ apart; even pooled over N ≤ 10 they are at least 10⁻⁴ apart, so rounding never
  merges genuinely different levels.
- If two distinct rounded SE values are still adjacent floats (only possible far beyond any
  entropy of N samples), fitting raises instead of returning a γ equal to a data value.

A test checks that on data where the paper's grid has no tie, our split's error is **≤** the
paper's (`best_split` is copied from notebook cell 16 into the test).

**Where γ is fitted.** Only on training rows: inside every inner and outer fold, γ comes from
that fold's training part (D3). The final probe uses γ fitted on all rows, stored in
`probe_config.json`. If there are fewer than two distinct SE values, or a class has fewer rows
than `cv_folds`, fitting raises with the class counts.

**Not adopted** (the paper uses these only as ablations or for long answers): the median split,
soft labels, and the 55–80% quantile filtering (App. A, App. B.3).

### Which classifier, and why no feature scaling?

**Answer.** `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=100,
random_state=RANDOM_SEED)`, with **no** standardization. That is the notebook's bare
`LogisticRegression()` (cells 14 and 32; a grep finds no scaler; verified).

*Implementation note (item 3):* scikit-learn 1.8 deprecated `penalty` (removed in 1.10), so
the code spells pure L2 as `l1_ratio=0.0` and leaves `penalty` at its default. The fit is
identical (verified: max |Δcoef| = 0). The `scikit-learn` floor in the `eval` extra rises to
`>=1.8`, where `l1_ratio=0` means L2. `probe_config.classifier.penalty` still records `"l2"`.

**The misconception to head off:** that we *know* this fails on Gemma's large unscaled
hidden states. We don't; it has never been run. So we start with the paper's settings and use
a **pre-registered fallback**:

1. A `ConvergenceWarning` is turned into an error whose message names `--max-iter`. It never
   silently ships an unconverged probe.
2. Fallback step 1: refit **on CPU** from the stored dataset with `--max-iter 1000`. This needs
   no new GPU run. It is recorded in `classifier.max_iter` and as deviation D10.
3. Fallback step 2, only if step 1 still fails: `--standardize`. The scaler is **folded into the
   linear weights** (w' = w/σ, b' = b − Σ w·μ/σ), so the artifact stays one linear layer. It
   is recorded as D12. A column with σ = 0 scales to near-zeros rather than exact zeros, since
   its mean carries rounding error; the L2 fit leaves its w' at (near-)0, and no mask is applied.

The same `--standardize` flag can also be run as a **reported ablation** on the CPU refit
(`metrics.json["ablations"]`). It is never the default.

Constants (in `constants.py`): `SEP_LOGREG_C = 1.0`, `SEP_LOGREG_L1_RATIO = 0.0`,
`SEP_LOGREG_SOLVER = "lbfgs"`, `SEP_LOGREG_MAX_ITER = 100`, `SEP_LOGREG_FALLBACK_MAX_ITER = 1000`,
the `SEPStandardization` StrEnum (`none`, `folded_into_weights`),
`SEP_LAYER_WINDOW = 5`, `SEP_CV_FOLDS = 5`.

### How do we number layers?

**Answer: layer k is the vector after k decoder layers.** k = 0 is the token embeddings and
k = L is the last layer. This is exactly the index into the tuple that HF returns with the
**boolean** `output_hidden_states=True` (verified in transformers 5.19,
`utils/output_capturing.py:115-124`).

Two traps:

- **Index L is already normalized.** HF overwrites index L with `last_hidden_state`, so it has
  the final norm applied. Indices k < L are raw. `probe_config.json` records
  `final_norm_applied` for each selected layer.
- **The list form of `output_hidden_states` shifts the index by one.** It omits the embeddings,
  so its index k means "after k+1 layers". We never use it, and a test pins this.

**Paper numbering.** The paper stacks the same HF tuple and labels it 1-based, so our k is
probably the paper's "layer k+1" (likely, not verified). The README and the model card print
both numbers.

**Layer 0 is excluded from the search.** At TBG, the last chat-template token is the same for
every prompt, so its embedding is a constant vector and carries no information. For simplicity
the same range, 1..L, is used for SLT.

### How is the layer window chosen without peeking at the test rows?

**Answer: nested cross-validation (nested CV).** Cross-validation splits the rows into k parts,
called folds; it trains on all but one fold and scores on the one left out. "Nested" means the
layer choice is itself made by an inner CV that only ever sees the outer training rows. The
reported score therefore includes the cost of choosing.

**The misconception to head off: "the best layer's CV score is the probe's real score."** The
maximum of many CV scores is optimistic. The paper's notebook picks layers on **test-split**
AUROC (cell 31; verified), which is a mild leak. We fix this (D4).

`metrics.json` gets two outputs.

**(a) Per-layer sweep (descriptive only, never used to choose).** For each position and each
k in 1..L, we record the 5-fold CV AUROC (mean ± std) of a single-layer probe:

- against binarized SE; and
- against model error (1 − correct), when correctness labels exist.

These numbers are the comparison point for Figs. 2–3. `final_layer_auroc`, the single layer L
at TBG, is picked out as the **baseline for the old "last layer" behaviour**. It answers the
owner's question with data.

**(b) Selected probe**, following App. B.3's concatenation:

- **Candidates:** every window [k, k+4] inside 1..L. Features are the window's vectors
  **concatenated in increasing k, not averaged** (cell 32; verified).
- **Outer loop:** `StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)`.
- **Inner loop:** on the outer-training rows only, an inner 5-fold CV scores each window by
  mean AUROC against binarized SE, with γ refit per inner fold. The highest score wins. On a
  tie, the deeper window wins, since that is the paper's typical pick.
- **Outer scoring:** the chosen window is fitted on outer-train and scored on outer-test.
- **Reported:** the outer-fold mean ± std, plus the window chosen in each fold as a stability
  check. AUROC against error is also reported for the selected window.
- **Final probe:** the same inner procedure is run on all rows to pick the window, then the
  probe is fitted on all rows.

**What stratification uses.** Folds need class labels before γ exists. Fold *assignment* uses
labels from a γ fitted on the rows being split (all rows for the outer loop, outer-train for
the inner loop). This uses only SE, never features, and the **training labels are still
recomputed from each fold's own γ**. If a test fold still ends up single-class, fitting raises
with a "need a spread of easy and hard questions" hint.

**Why rank windows by binarized SE and not by error (D5).** The paper ranks by AUROC against
model error (cell 31). We rank by the probe's own training target for two reasons:

- Our correctness labeler is alias string-match, a known-noisy B2 item. Selection should not
  depend on it.
- SEP's claim is "predicts SE". Whether that also buys error detection is a separate question,
  answered on the held-out benchmark slice.

**Layer modes.** One trainer flag covers them: `--layers auto|final|<a>-<b>`, default `auto`.

- `final` means (L,), the only layer the vLLM embed runner can serve today.
- With a fixed choice, the outer CV still runs, but there is no inner selection.
- Recorded as `classifier.layer_selection = "nested_cv" | "fixed"`.

**Cost (unverified).** All on CPU. For E2B, per position: 31 windows × 5 inner folds × 5 outer
folds (775 fits) on 500 rows × 7680 features, plus the 5 outer fits, plus 31 × 5 for the final
window choice on all rows, plus 35 × 5 single-layer sweep fits. This is estimated at minutes to
tens of minutes. It must be measured; a `--n-jobs` flag is added in item 10 only if that
measurement calls for it.

### Which token positions are collected, and which one is served?

**Answer: both TBG and SLT are collected and swept; only TBG is served in v1.** The owner's
decision ([`TODO.md`](TODO.md)) is to sweep layers **and** token position. SLT is the paper's
main setting (Sec. 4, "Probing Locations").

- **TBG** = the last token of the chat-formatted prompt with `add_generation_prompt=True`. The
  input is unchanged.
- **SLT** reproduces `huggingface_models.py`, which takes generation step n_gen − 2 (EOS
  included). Our teacher-forced equivalent:
  1. Run one forward pass over the prompt plus the greedy answer tokens plus EOS.
  2. Let m be the number of non-EOS answer tokens. Read the position of answer token a[m−2],
     whose output predicts the last content token.
  3. If m = 1, read the last prompt token.

  That this matches generate-time states is likely, not verified (a GPU check).
- **Answer used for SLT:** our greedy answer (T = 0), which is what the paper's text says. The
  paper's code samples at T = 0.1 (D7).
- **Served position:** v1 loads only TBG probes. `TokenPosition` is a StrEnum with
  `TBG` and `SLT`, and the config schema already carries `token_position`. The loader raises on
  an SLT probe with "SLT serving is a follow-up". **Why:** TBG keeps SEP's selling point, one
  forward pass *before* answering, and it is the only position vLLM can serve. Serving SLT is
  deferred.

### How does each provider hand over hidden states?

**Answer: through one minimal protocol change.**

- `pooled_hidden_state(prompt, layers: tuple[int, ...]) -> Tensor[len(layers), hidden]`
- a new `model_revision` property
- an HF-only `hidden_states_all_layers(prompt, position, answer=None) -> Tensor[L+1, hidden]`,
  used by the collector

The detector exposes `hidden_state_layers`. `DetectionContext.pooled_hidden_state` keeps its
name, now with shape [len(layers), d], and the detector asserts the first dimension.

**HF (`HFTransformersProvider`).**

- One forward pass with boolean `output_hidden_states=True`.
- Assert `len == L+1`.
- Stack row `[0, pos]` of the requested layers; return float32 on CPU.
- L and d come from `config.get_text_config()`.

**vLLM (`VLLMOpenAIProvider`).**

- Supports only `layers == (L,)` at TBG. That is the embed runner's final-normed LAST pooling,
  which corresponds to HF index L.
- Anything else raises `UnsupportedHiddenStateRequest`. The message reads: "this probe reads
  layers [a..b]; the vLLM embed runner can only pool the last layer; use
  SLM_PROVIDER=hf_transformers for SEP, or train with --layers final".
- The evaluator triggers this check at **build time**, before any scoring.

**Rejected options:**

- `--hf-overrides num_hidden_layers` truncation. It breaks the E-series per-layer embeddings
  and KV-sharing layout, and it still applies the final norm (likely; from vLLM source reading).
- A vLLM model plugin or `extract_hidden_states`. Both are GPU-only and unverified. They are
  deferred to an owner decision.

**Training** requires `SLM_PROVIDER=hf_transformers`, because the sweep needs every layer.
The vLLM provider raises "training requires hf_transformers".

**Agreement across providers is not claimed.** Results are not bit-identical (different
kernels, bf16 summation order). The README line "a probe trained through one works through
the other" (`README.md:102-103`) becomes: "vLLM can serve last-layer probes only; whether its
vectors match HF's within tolerance is unverified until the first GPU run."

### How is the probe tied to one exact backbone and prompt?

**Answer: a pinned registry revision, plus a compatibility check at load.**

**The registry is the single source of truth.** `ModelEntry` gains:

- `revision`: a 40-hex commit sha, read from the Hub once per entry. This needs network access,
  not a GPU.
- `num_hidden_layers`: verified E2B = 35 and E4B = 42; larger models are read from their
  config.json.
- `hidden_size`: verified E2B = 1536 and E4B = 2560.

**Both providers use it.** HF calls `from_pretrained(revision=entry.revision)` for the
tokenizer and the model, then asserts `model.config._commit_hash == entry.revision`.
`serving/model_flags.py` adds `--revision`. The vLLM server echoes nothing back, so the launch
flag is what we trust.

**`detector.check_compatible(model_id, revision, num_hidden_layers, hidden_size,
answer_instruction, chat_template_sha256)`** raises on any mismatch, naming **both** values.
`ResponseEvaluator` calls it in its constructor, so it fails before any `sample()` call. The
prompt fields matter because a changed instruction or template moves the TBG state.

**Cleanup.** The unused `ModelEntry.sep_probe_path` (`demo/model_registry.py:84`, referenced
nowhere else) is removed.

## Deviations from the paper

This table goes into the README, the model card, and `probe_config.deviations_from_paper`.
A test asserts that the list is present. No other deviation is allowed silently.

| ID | Paper | Ours | Why |
|---|---|---|---|
| D1 | Plain-text few-shot "Answer:" prompt, no chat template | Gemma chat template plus our brief-answer instruction, zero-shot | Our models are chat-tuned, and every detector must read the same input. TBG is therefore a different position (the start of the model's turn). Unverified relative to the paper. |
| D2 | Llama-2, Mistral, Phi-3 | Gemma 4 E2B/E4B | Our backbones. Table 4's layers do not transfer, so we sweep. |
| D3 | γ fitted on all data before splitting | γ fitted per training fold; the final γ uses all rows | No label leakage into CV estimates. |
| D4 | Window chosen on test-split AUROC | Chosen by inner CV; nested-CV estimate reported | No selection leakage. |
| D5 | Window ranked by AUROC against model error | Ranked by AUROC against binarized SE; error AUROC still reported | Independent of our noisy correctness labeler; matches the probe's target. |
| D6 | n = 2000 per dataset, 4–5 datasets | n = 500 TriviaQA rows (calibration slice) | GPU budget. CV std is reported so the noise is visible. |
| D7 | Most-likely answer sampled at T = 0.1 (the text says greedy) | Greedy, T = 0 | Matches the paper's text and our existing greedy answer. |
| D8 | SE samples at T = 1.0, top-p 0.9, top-k 50 | T = 1.0, **no top-p or top-k** (`config.py:86`; the provider sends neither) | Unchanged project sampler. Pure T = 1 sampling spreads more, which may raise SE. |
| D9 | No escalation threshold | `SEP_THRESHOLD`, with a recommended value from Youden's J on out-of-fold rows | A product need. The paper reports AUROC only. |
| D10 | max_iter = 100 | Same, **unless** non-convergence forces 1000 (recorded) | Pre-registered fallback; see the classifier section. |
| D11 | Best split over a 100-point grid ending at max(SE) | Exhaustive search over midpoints of distinct SE values | Exact; no ties with data, so no −1 labels. |
| D12 | No feature scaling | Same, **unless** fallback step 2 is needed (scaler folded into the weights, recorded) | Pre-registered fallback only. |
| D13 | Layer 0 included | Search range 1..L | At TBG, layer 0 is a constant vector. |

Active deviations are listed in each probe's config. D10 and D12 appear only if they were used.

## Data flow

```
COLLECT (GPU, SLM_PROVIDER=hf_transformers)      eval/collect_sep_dataset.py
  calibration question
   ├─ sample N=10 at T=1 ─ cluster once (NLI) ─ se_discrete, se_likelihood_weighted, cluster_ids
   ├─ greedy answer (T=0) ─ greedy_correct (alias match)
   └─ hidden_states_all_layers: TBG [L+1,d], SLT [L+1,d]   (one extra forward pass for SLT)
          │  streamed per row
          ▼
  <dataset dir>/ rows.parquet + hidden_states.safetensors + collection_config.json
                 + training_log.jsonl

FIT (CPU only; reads only the dataset dir)       eval/train_sep_probe.py --dataset DIR
  per position: single-layer sweep ─────────────────────────────► metrics.layer_sweep
  nested CV (γ per fold, window by inner CV) ───────────────────► metrics.selection
  out-of-fold probabilities → Youden's J vs greedy_correct ─────► recommended_threshold
  final window + γ on all rows → LogisticRegression ─► probe.safetensors, probe_config.json,
                                                       train_ids.json, metrics.json

USE
  SEP_PROBE_PATH (dir | hf://repo@rev | unset → data/probes/<slug>/)
   → resolve_probe_source → load_probe_dir → check_compatible   [evaluator constructor]
   → provider capability check for (position, layers)            [evaluator constructor]
   → provider.pooled_hidden_state(prompt, layers) → sigmoid(w·x+b) = p(high SE)
   → escalate if p > SEP_THRESHOLD   (when ESCALATION_DETECTOR=semantic_entropy_probe)

HELD-OUT + PUBLISH
  run_benchmark on the disjoint test slice (records the probe sha256 per row)
   → compute_auroc --json-out → publish_sep_probe (checks the sha, merges heldout, renders
     the card) → Hub model repo + dataset repo, tag v1
```

## Artifact format

### The probe directory

The probe lives at `data/probes/<model slug>/` (or `--output-dir`). It is **the same layout on
disk and on the Hub**.

| File | Content |
|---|---|
| `probe.safetensors` | `weight` float32 [F] and `bias` float32 [1], where F = len(layers) × hidden_size. Header metadata: `{"format": "sep-logistic", "schema_version": "1"}`. Cannot carry code. |
| `probe_config.json` | Pydantic `SEPProbeConfig` with `extra="forbid"`; an unknown `schema_version` fails. Contents below. |
| `train_ids.json` | Today's sidecar content (`model_id`, `calibration_set`, `question_ids`), now at a fixed name inside the directory. |
| `training_log.jsonl` | One line per question: id, se_discrete, se_likelihood_weighted, n_clusters, answer word counts, greedy_correct, elapsed_s. |
| `metrics.json` | Contents below. |

**`probe_config.json` blocks:**

- **`schema_version`**: 1.
- **`backbone`**: model_id, revision, num_hidden_layers, hidden_size, dtype.
- **`features`**:
  - token_position
  - layers (e.g. `[27, 28, 29, 30, 31]`; illustrative only, the real value is measured)
  - layer_convention: `"after_k_decoder_layers"`
  - final_norm_applied: one bool per layer
  - concat: `"increasing_layer"`
- **`prompt`**: chat_template_sha256, add_generation_prompt: true, answer_instruction.
- **`labels`**:
  - se_estimator: `"discrete"`
  - sample_n, temperature, top_p: null, top_k: null
  - nli_model_id, nli_revision
  - binarization: `"best_split_midpoints"`
  - gamma_nats, n_high, n_low
- **`classifier`**:
  - type: `"logistic_regression"`
  - C, penalty, solver, max_iter
  - standardization: `"none" | "folded_into_weights"`
  - layer_selection, cv_folds, seed, sklearn_version
- **`output`**: semantics: `"p_high_se"`, recommended_threshold.
- **`training`**:
  - provider, transformers_version
  - n_rows, calibration_set_sha256
  - repo_commit, repo_dirty
  - created_at (UTC ISO)
- **`deviations_from_paper`**: IDs from the table above.

**`metrics.json` contents:**

- **`layer_sweep[position][k]`**: `{auroc_vs_binarized_se: {mean, std}, auroc_vs_error: {mean, std} | null}`
- **`final_layer_auroc[position]`**: the old-behaviour baseline.
- **`selection[position]`**: `{window, inner_cv_auroc, nested_cv_auroc_vs_binarized_se: {mean, std, per_fold, window_per_fold}, nested_cv_auroc_vs_error}`
- **`threshold`**: `{recommended, tpr, fpr, escalation_rate, source: "oof_youden"}`
- **`converged`**, **`n_iter`**
- **`ablations`**: optional.
- **`heldout_benchmark`**: added at publish time; holds compute_auroc's output for SEP, PE, SE and discrete SE with bootstrap CIs, the run config, and the probe sha256.

### The dataset directory (companion Hub dataset repo)

| File | Content |
|---|---|
| `rows.parquet` | id, question, ground_truth_aliases, sampled_answers (list[str], N=10), cluster_ids (list[int]), se_discrete, se_likelihood_weighted, n_clusters, greedy_answer, greedy_correct |
| `hidden_states.safetensors` | `tbg` and `slt`, each float32 [rows, L+1, d]. The header holds the ordered id list, which must equal the parquet row order; this is checked on load. About 110 MB per position for E2B (arithmetic, not measured). |
| `collection_config.json` | The `backbone`, `prompt` and `labels` blocks (without γ). |
| `training_log.jsonl` | As above. |

Tensors go in safetensors, not parquet list columns, because lists of 55k floats bloat parquet.
Splitting collection from fitting makes "a refit from the export reproduces the weights" true
by construction.

**Dependencies (`pyproject.toml`):**

- `safetensors` and `huggingface-hub` move into core. Both are already locked as transitive
  dependencies and are now imported directly.
- `pyarrow` and `scikit-learn` go in the `eval` extra.
- The library module does **not** import sklearn.

## Config and environment changes

| Setting / env | Change |
|---|---|
| `sep_threshold` / `SEP_THRESHOLD` | **New.** `float = Field(0.5, ge=0, le=1)`. The classifier's own decision boundary, labelled a **placeholder, uncalibrated for escalation**. |
| `Settings.escalation_threshold_for(detector)` | **New** pure method. SEP → `sep_threshold`. SE, discrete SE or PE → `se_threshold` (PE behaviour unchanged). Anything else raises `ValueError`. |
| `Settings` model validator | **New.** Rejects `escalation_detector=semantic_entropy_probe` when SEP is not in `enabled_detectors`. The check moves earlier, out of the evaluator. |
| `SEP_PROBE_PATH` | Now a **directory** or `hf://<owner>/<repo>[@<revision>]`. Unset → `data/probes/<slug>/`. No silent fallback to the Hub. |
| `ModelEntry.revision`, `.num_hidden_layers`, `.hidden_size` | **New**, required. |
| `ModelEntry.sep_probe_path` | **Removed** (unused). |
| `constants.py` | Adds `TokenPosition` StrEnum (TBG, SLT), the recipe constants, file-name constants (`SEP_PROBE_WEIGHTS_FILE`, `SEP_PROBE_CONFIG_FILE`, `SEP_TRAIN_IDS_FILE`, `SEP_TRAINING_LOG_FILE`, `SEP_METRICS_FILE`, `SEP_DATASET_ROWS_FILE`, `SEP_DATASET_TENSORS_FILE`, `SEP_COLLECTION_CONFIG_FILE`), `HF_PROBE_URI_PREFIX = "hf://"`, `SEP_PROBE_SCHEMA_VERSION = 1` and `LEGACY_SEP_PROBE_SUFFIX = ".pt"`. Removes `SEP_PROBE_SUFFIX`. |
| `SamplingProvider` protocol | `pooled_hidden_state(prompt, layers)` and `model_revision`. A **breaking change** for anyone implementing the protocol (v0.x); goes in the CHANGELOG. |
| `vllm_flags` | Adds `--revision`. |

## Threshold semantics and migration for users

**What the number means now.** SEP's score is **p(high SE)**: the probability that the 10-sample
discrete SE would exceed γ\*. Higher means more likely a hallucination. Its form is
sigmoid(w·x + b), the paper's `predict_proba[:, 1]` (Sec. 4; verified).

- It is **not** a "hallucination probability", and it is **not** in nats.
- The paper gives **no** deployment threshold (verified), so any threshold is our design
  choice.
- The rule is unchanged: escalate when `score > τ`.

**How to set `SEP_THRESHOLD`:**

1. **At training time**, `recommended_threshold` is Youden's J (through the existing
   `eval/calibrate_threshold.youden_threshold`). It is computed on the selected procedure's
   **out-of-fold** probabilities against `greedy_correct`. The calibration slice is disjoint
   from the test slice, so it is a legitimate development set. TPR, FPR and escalation rate are
   reported with it.
2. **Unchanged path:** `uv run python -m eval.calibrate_threshold --dev <run> --detector
   semantic_entropy_probe`. It works on any score scale; its output now adds `"env_var":
   "SEP_THRESHOLD"`.

**The threshold is never loaded implicitly from the probe.** The env value is the threshold.
The evaluator logs both values and warns when they differ.

**Migration checklist (goes in the CHANGELOG and the README):**

| You had | Do now |
|---|---|
| `ESCALATION_DETECTOR=semantic_entropy_probe` with `SE_THRESHOLD=…` | Set `SEP_THRESHOLD` (start from the probe's `recommended_threshold`). `SE_THRESHOLD` no longer affects SEP. |
| A local `data/probes/<slug>.pt` | Retrain (`eval.collect_sep_dataset`, then `eval.train_sep_probe`), or use the published probe. |
| `SEP_PROBE_PATH=…/x.pt` | Point it at a probe directory or an `hf://` reference. |
| SEP on `SLM_PROVIDER=vllm_openai` | Works only for `--layers final` probes. Otherwise use `hf_transformers`. |
| Reading SEP in nats in the UI or CLI | It is now shown as a percentage, "P(high SE)". |

## Backward compatibility for old probes

**Old Ridge `.pt` probes are refused, not converted.**

**Why not convert.** A Ridge probe predicts nats. Mapping it to p(high SE) would need a γ and a
calibrated slope, so any "conversion" would be an invented classifier. No `.pt` probe was ever
trained or published ([`PUBLISHED_SEP_PROBE.md`](PUBLISHED_SEP_PROBE.md), [`TODO.md`](TODO.md)),
so only local experimenters can be affected.

**Behaviour.** The loader and `run_benchmark`'s sweep preflight both raise
`ObsoleteSEPProbeError` in two cases:

- `SEP_PROBE_PATH` ends in `.pt` or points at a regular file;
- the default directory is missing but `data/probes/<slug>.pt` exists.

The message reads: "…is a pre-classifier Ridge probe (predicts SE in nats). SEP now outputs
p(high SE); retrain with `uv run python -m eval.collect_sep_dataset` then `eval.train_sep_probe`,
or use the published probe `SEP_PROBE_PATH=hf://…`."

**CHANGELOG** lists this under Changed and Removed, marked breaking.

## Docs to update

All of these are done in work item 18.

- **README**
  - Detector table row: "p(high SE) from a logistic probe over hidden states".
  - SEP section (`README.md:331-343`): the label, the classifier, the layer and position sweep,
    "not on SE's scale".
  - Lines 102-103: replace the provider sentence.
  - New Q&A: **"Which layer and token does SEP read, and are mid layers better?"**, carrying the
    terminology correction and the Table 4 evidence above, and pointing to `metrics.json`.
  - "How do I train and use a SEP?":
    1. Use the published probe.
    2. Train your own (collect, then fit, then the release commands).
    3. Refit from the published dataset, no GPU.
  - The deviations table.
  - Escalation section: `SEP_THRESHOLD`.
  - Troubleshooting: the `.pt` refusal, a backbone or prompt mismatch, "vLLM can't serve this
    layer".
- **`knowledge/GLOSSARY.md`**: token position (TBG, SLT) vs layer; the layer-index convention;
  best-split binarization; linear probe vs linear classifier; nested CV; p(high SE).
- **`knowledge/CAVEATS.md` §6**:
  - SEP imitates *binarized discrete* SE;
  - the probe is tied to a backbone **and revision** and to the prompt;
  - chat-template TBG ≠ the paper's TBG;
  - vLLM parity is unverified;
  - `SEP_THRESHOLD=0.5` is a placeholder.
- **`knowledge/theory/SEMANTIC_ENTROPY_WALKTHROUGH.md`** (around line 665): rewrite the SEP
  section in Q&A form:
  - "Why a classifier and not a regressor?"
  - "How does SEP turn SE into a yes/no target?", with a worked best-split example on 8 SE values
  - "Why sweep layers instead of using the last one?"
  - "Isn't the middle layer always best?"
- **`knowledge/usage_guides/QA_GUIDE.md`**: replace the Pearson-r criterion with nested-CV AUROC
  against binarized SE plus held-out AUROC against error. Update the test table. Explain why
  selection cannot see the test fold.
- **`DEVELOPER_GUIDE.md`**: the new eval files and the protocol change.
- **`DEVOPS_GUIDE.md`**: the embed runner serves last-layer probes only; `--revision`; the
  publish runbook (the command sequence in item 15).
- **User-facing docs live in the README** (the user guide was folded into it on purpose; do not
  recreate `USER_GUIDE.md`): the published-probe path and the `SEP_THRESHOLD` migration.
- **`knowledge/plans/WEB_PLAYGROUND.md`**: SEP is shown as a probability; status shows
  `sep_threshold`.
- **[`PUBLISHED_SEP_PROBE.md`](PUBLISHED_SEP_PROBE.md)**: the edits listed in
  [Changes to PUBLISHED_SEP_PROBE.md](#changes-to-published_sep_probemd).
- **[`TODO.md`](TODO.md)**: tick the SEP item and add the [deferred follow-ups](#deferred-follow-ups).
- **CHANGELOG `[Unreleased]`**
  - Changed (breaking): SEP outputs p(high SE); probes are directories; provider protocol.
  - Added: `SEP_THRESHOLD`, the layer and position sweep, the dataset export, `hf://` probes,
    the publish script, `compute_auroc --json-out`, registry revisions.
  - Removed: Ridge `.pt` probes and `--ridge-alpha`.
  - Fixed: escalation with SEP compared its score against `SE_THRESHOLD`.
- **Code comments**:
  - `demo/slm_provider.py:399` (the false "Matches Kossen" comment);
  - `demo/model_registry.py`'s `_EMBED_POOLER_CONFIG` comment: "serves layer L only";
  - the evaluator's "last-layer" error text.

## Ordered TDD work items

Each item: write the failing test first, then the code. "PSP n" refers to item n of
[`PUBLISHED_SEP_PROBE.md`](PUBLISHED_SEP_PROBE.md). That plan's items 1–6 and 8 are carried out
here and are not repeated there; its item 7 (the real run) stays in that plan.

**1. Fix the threshold routing bug; add `SEP_THRESHOLD`.**

- Failing tests first:
  - `test_escalation_threshold_for_routes_per_detector`
  - `test_response_evaluator_sep_escalates_above_sep_threshold`: SE_THRESHOLD=1.0, SEP score
    0.8, SEP_THRESHOLD=0.5 → escalates. This fails today.
  - a web `/ask` equivalent (TestClient with fakes)
  - `test_settings_rejects_sep_escalation_without_sep_enabled`
  - `sep_threshold` bounds
- Files: `config.py`, `evaluator.py`, `demo/web/app.py`, `demo/web/api_schemas.py` (`/status`
  gains `escalation_threshold` and `sep_threshold`), `escalation.py` (doc note),
  `tests/unit/test_evaluator.py`, `tests/unit/test_escalation.py`, `tests/unit/web/`.
- Done when: all three call sites use `escalation_threshold_for`, and the tests pass.

**2. Best-split binarization.**

- Failing tests first:
  - `test_best_split_minimizes_sse_over_all_midpoints` (brute force)
  - `test_best_split_never_ties_with_data`, including many rows at ln 10
  - `test_best_split_sse_not_worse_than_paper_grid`, against a copy of notebook cell 16 (renamed, plotting line dropped)
  - `test_binarize_strictly_greater`
  - `test_best_split_raises_on_fewer_than_two_distinct_values`
- Files: `eval/sep_recipe.py` (new; pure numpy/sklearn), `tests/unit/test_sep_recipe.py`.
- Done when: the tests pass, and the module has no I/O.

**3. Logistic fit with the paper's defaults, fail-fast convergence, optional folded scaler.**

- Failing tests first:
  - `test_fit_logistic_uses_paper_defaults` (asserts the params dict, no scaler)
  - `test_convergence_warning_raises_naming_max_iter`
  - `test_folded_standardize_matches_pipeline_probs` (allclose 1e-6)
  - `test_zero_variance_column_gets_zero_weight`
- Files: `eval/sep_recipe.py`, `constants.py`.
- Done when: the tests pass.

**4. Layer sweep, nested-CV window selection, out-of-fold threshold.**

- Synthetic data: signal only in layer 7.
- Failing tests first:
  - `test_single_layer_sweep_peaks_at_signal_layer`
  - `test_window_selection_contains_signal_layer`
  - `test_nested_cv_never_fits_on_outer_test_rows` (a recording estimator)
  - `test_gamma_fitted_on_train_fold_only`
  - `test_stratification_uses_no_features`
  - `test_single_class_test_fold_raises_with_hint`
  - `test_concat_order_increasing_layer`
  - `test_layer_zero_excluded`
  - `test_ties_prefer_deeper_window`
  - `test_fixed_layers_skip_inner_selection`
  - `test_pure_noise_nested_auroc_near_half`
  - `test_recommended_threshold_is_youden_on_oof_probs`, which must give output identical to
    `youden_threshold`
- Files: `eval/sep_layer_selection.py` (new; pure, built on `eval/sep_recipe.py`),
  `tests/unit/test_sep_layer_selection.py`, `constants.py` (`SEP_LAYER_WINDOW`,
  `SEP_CV_FOLDS`, the `SEPLayerSelection` and `SEPThresholdSource` StrEnums).
- *Implementation note:* the selection code lives in its own module rather than in
  `sep_recipe.py`, which would otherwise pass 500 lines; `sep_recipe.py` keeps binarization
  and the logistic fit. The module works on one position's `[rows, L+1, d]` array; item 10
  keys its outputs by position (`layer_sweep`, `nested_cv`, `choose_window`,
  `fit_window_probe`, `recommend_threshold`). It is sequential: `--n-jobs` is left to item 10
  if the measured CPU cost calls for it.
- *Schema note:* the module returns dataclasses, not the `metrics.json` shape; item 10
  serializes them. The mapping: `NestedCVResult.auroc_vs_binarized_se` and `.auroc_vs_error`
  become `selection[position].nested_cv_auroc_vs_binarized_se` and
  `.nested_cv_auroc_vs_error`; `window_per_fold` moves under the binarized-SE AUROC block;
  `WindowChoice.window` and `.inner_cv_auroc` fill `selection[position].window` and
  `.inner_cv_auroc`; `layer_sweep` entries drop `per_fold`, keeping `{mean, std}`.
- Done when: the tests pass, and the outputs carry every field the `metrics.json` schema
  needs (item 10 maps them, per the schema note).

**5. Discrete and weighted SE from one clustering.**

- Failing test first: `test_collector_row_has_both_estimators_from_one_clustering`. A fake
  scorer counts NLI calls, which must equal one clustering's worth.
- Files: `eval/collect_sep_dataset.py` (new), `services/semantic_entropy.py` (expose a
  "cluster once, score both" helper if one is missing).
- Done when: the test passes.

**6. Layer convention and HF all-layer extraction at TBG (protocol change).**

- A fake HF model returns L+1 distinct constant tensors.
- Failing tests first:
  - `test_hf_hidden_states_index_k_is_after_k_layers`
  - `test_layer_L_flagged_final_norm`
  - `test_output_hidden_states_is_boolean_true` (spy on kwargs)
  - `test_wrong_tuple_length_raises`
  - `test_pooled_hidden_state_stacks_requested_layers`
  - `test_tbg_is_last_prompt_token`
- Files: `services/sampling_provider.py`, `services/detector_protocol.py`,
  `demo/slm_provider.py` (fix the line-399 comment), `tests/unit/test_slm_provider.py`.
- Done when: the tests pass, and no caller uses the old one-argument signature.

**7. Pin the backbone revision (PSP 2), registry dimensions, vLLM capability.**

- Failing tests first:
  - `test_every_entry_has_40_hex_revision_layers_and_hidden_size`
  - `test_vllm_flags_include_revision`
  - `test_hf_from_pretrained_called_with_revision`
  - `test_commit_hash_mismatch_raises`
  - `test_missing_commit_hash_raises`
  - `test_model_revision_from_registry`
  - `test_vllm_final_layer_posts_same_body_as_today`
  - `test_vllm_rejects_non_final_layers_or_slt_with_remedy`
- Files: `demo/model_registry.py` (add the fields; remove `sep_probe_path`),
  `serving/model_flags.py`, `demo/slm_provider.py`, `tests/unit/test_model_registry.py`,
  `tests/unit/test_serving_launch.py`, `tests/unit/test_slm_provider.py`.
- Done when: the tests pass, and the shas are read from the Hub and committed.

**8. SLT extraction, teacher-forced (HF, collection only).**

- A fake tokenizer and model echo the position index as the hidden value.
- Failing tests first:
  - `test_slt_is_position_of_a_m_minus_2`
  - `test_slt_single_token_answer_falls_back_to_last_prompt_token`
  - `test_slt_token_construction_includes_eos`
- Files: `demo/slm_provider.py`, `tests/unit/test_slm_provider.py`.
- Done when: the tests pass.

**9. Dataset collection and export (PSP 5, changed: always written).**

- Failing tests first:
  - `test_dataset_rows_and_columns`
  - `test_tensor_ids_match_parquet_order`, with a load-time check that raises on a mismatch
  - `test_collect_requires_hf_provider`
  - `test_collect_fails_fast_on_missing_hidden_states_or_revision`
  - `test_training_log_one_line_per_question`
- Files: `eval/collect_sep_dataset.py`, `eval/sep_dataset_io.py` (new; read and write only),
  `pyproject.toml`.
- Done when: the tests pass with fake provider, sampler and NLI.

**10. Trainer writes the probe directory (PSP 1 + 3).**

- Failing tests first:
  - `test_train_writes_all_five_files`
  - `test_config_holds_backbone_revision_layers_position_gamma_classifier_prompt`
  - `test_probe_config_lists_deviations`
  - `test_safetensors_roundtrip_identical_scores`
  - `test_probe_loader_uses_safetensors_only` (no pickle)
  - `test_refit_from_export_reproduces_weights` (allclose, same seed)
  - `test_layers_final_gives_L_and_fixed_selection`
  - `test_records_repo_commit_and_dirty`
  - `test_metrics_json_matches_schema` (the item-4 schema note's field mapping)
- Files: `eval/train_sep_probe.py` (rewritten as orchestration; Ridge, Pearson and
  `--ridge-alpha` removed), `custom_types/sep_probe_types.py` (new).
- Done when: the tests pass. PSP 3's open question ("the `.pt` path keeps working or is
  removed") is decided: **removed**.

**11. Detector: load the directory or Hub, check compatibility, refuse `.pt` (PSP 4).**

- Failing tests first:
  - `test_hf_uri_parses_repo_and_revision` (accepts `hf://a/b` and `hf://a/b@v1`, rejects `hf://a`)
  - `test_hub_download_mocked_loads_matching_probe` (asserts the repo_id, revision and
    allow_patterns passed to `snapshot_download`)
  - `test_mismatch_{model_id,revision,layers,hidden_size,answer_instruction,template}_raises_naming_both`
  - `test_unknown_schema_or_extra_field_raises`
  - `test_slt_probe_refused_in_v1`
  - `test_pt_file_raises_obsolete_with_retrain_hint`
  - `test_missing_default_dir_with_legacy_pt_raises_obsolete`
  - `test_score_is_sigmoid_in_unit_interval`
- Files: `services/semantic_entropy_probe.py` (rewritten; no sklearn import),
  `tests/unit/test_semantic_entropy_probe.py`.
- Done when: the tests pass.

**12. Evaluator wiring.**

- Failing tests first:
  - `test_provider_asked_for_probe_layers`
  - `test_backbone_mismatch_raises_before_any_sample`
  - `test_vllm_capability_checked_at_build_time`
  - `test_warns_when_sep_threshold_differs_from_recommended`
- Files: `evaluator.py`, `tests/unit/test_evaluator.py`.
- Done when: the tests pass.

**13. Provenance and the benchmark run config.**

- Failing tests first:
  - `test_train_ids_inside_probe_dir`
  - `test_run_config_records_probe_ref_revision_and_sha256`
  - `test_preflight_expects_probe_dir_and_gives_pt_message`
  - `test_run_benchmark_preflight_fails_before_scoring_on_mismatch`
  - the existing overlap tests, ported
- Files: `eval/sep_probe_provenance.py`, `eval/run_benchmark.py`,
  `tests/unit/test_benchmark_pipeline.py`.
- Done when: the tests pass.

**14. compute_auroc JSON output and calibrate_threshold env hint.**

- Failing tests first:
  - `test_compute_auroc_json_out_matches_stdout`
  - `test_calibrate_reports_env_var_for_sep`
- Files: `eval/compute_auroc.py` (add `--json-out`; today it only prints),
  `eval/calibrate_threshold.py`.
- Done when: the tests pass.

**15. Publish script (PSP 6), with the held-out merge.**

- Failing tests first:
  - `test_model_card_snapshot` (golden file `tests/unit/fixtures/sep_model_card.golden.md`)
  - `test_publish_refuses_without_heldout`
  - `test_heldout_refuses_probe_sha_mismatch`
  - `test_publish_refuses_model_id_mismatch`
  - `test_publish_refuses_repo_dirty`
  - `test_publish_has_no_token_arg`
  - `test_upload_calls_mocked_hfapi_with_both_repos_and_tag`
  - `test_dry_run_never_uploads`
- Files: `eval/publish_sep_probe.py` (new), `eval/templates/SEP_MODEL_CARD.md` (new;
  `string.Template`).
- The card carries: the deviations table, both layer numberings, the limits, citations and the
  commit link.
- Done when: the tests pass. DEVOPS_GUIDE documents the GPU-run sequence:
  1. `eval.collect_sep_dataset`
  2. `eval.train_sep_probe --dataset …`
  3. `eval.run_benchmark` on the disjoint slice
  4. `eval.compute_auroc --json-out`
  5. `eval.publish_sep_probe --dry-run`, then for real

**16. Web and frontend.**

- Failing tests first:
  - an API schema test for `sep_threshold`
  - vitest for SEP rendered as a percentage, "P(high SE)", with no nats unit, its meaning text,
    and the threshold line using `sep_threshold`
- Files: `frontend/src/lib/scores.ts`, `api.ts`, `App.tsx`, `test/fixtures.ts`,
  `frontend/src/lib/lib.test.ts`, `demo/cli.py`; rebuild the committed static bundle.
- Done when: `npm test` and pytest pass.

**17. Full gate.**

- Run the verify command from the Summary plus `npm test`.
- Done when: everything is green on a clean `uv sync --locked --all-extras`.

**18. Docs (PSP 8).**

- Everything in [Docs to update](#docs-to-update).
- No test beyond the existing link checks.
- Done when: it has been reviewed against that list.

## Changes to PUBLISHED_SEP_PROBE.md

Apply these edits to that plan in item 18. They change it as follows:

- **Prerequisite 1** points here and is marked done once items 1–17 land.
- **Artifact table:**
  - `probe_config.json` content is replaced by a link to [Artifact format](#artifact-format).
  - `metrics.json` adds the layer sweep, the selection, `final_layer_auroc` and the
    recommended threshold.
  - The dataset repo becomes `rows.parquet` plus `hidden_states.safetensors` (all layers, TBG
    and SLT) plus `collection_config.json`, instead of one "pooled hidden state" per row.
- **"The 11 answers"** → "the 10 sampled answers plus the greedy answer".
- **Item 3:** `.pt` is **removed**, not kept.
- **Item 5:** the export is **always** written, not optional. It is the fit step's only input.
- **Items 1–6 and 8:** "implemented by SEP_CLASSIFIER.md items 2–18".
- **Item 7** stays, and adds a step: the owner picks the v1 position after reading the sweep.
- **Success criteria:** add "the README states the measured layer window and the
  `final_layer_auroc` baseline".

## What needs a real GPU run

All of these are stated as **assumptions** until the first run.

1. Collect the 500 E2B calibration rows (TBG and SLT, all layers), then fit, benchmark the
   disjoint 200-question slice, and publish `v1`. This is PSP 7, and it needs the owner's
   namespace and license decisions first.
2. The actual best window and position for Gemma 4 E2B/E4B, and whether chat-template TBG
   reaches useful AUROC at all. **Publish the result even if it is weak.**
3. lbfgs convergence at `max_iter=100` on unscaled Gemma hidden states. If it fails, run the
   pre-registered fallback (D10, then D12) on CPU from the export.
4. Gemma 4 under transformers 5.19: `len(hidden_states) == L+1`, and index L is final-normed.
   Both were read from code, not observed.
5. `model.config._commit_hash` is populated for Gemma 4 loaded through `AutoModelForCausalLM`.
6. Teacher-forced SLT agrees with generate-time SLT within tolerance.
7. HF index L vs the vLLM embed runner at TBG: cosine similarity, and agreement of probe
   probabilities and AUROC. Placeholder bars are cosine ≥ 0.999 and |Δp| ≤ 0.02. Record the
   result in `metrics.json["provider_parity"]` and CAVEATS. This is a manual check for now;
   the script is deferred.
8. vLLM `--revision` works together with `--language-model-only` on the v0.31 image.
9. CPU time of the nested sweep, and disk use (about 110 MB per position for E2B).
10. Whether the discrete-SE best split gives a usable class balance on E2B.

## Deferred follow-ups

Add these to [`TODO.md`](TODO.md):

- **Serving SLT probes.** The collection and sweep exist; loading needs the greedy answer
  passed through.
- **`eval/check_hidden_state_parity.py`**, automating GPU item 7.
- **A resumable one-command release orchestrator** (`eval/release_sep_probe.py`).
- **A vLLM path for non-final layers.** Either an out-of-tree model plugin returning aux hidden
  states, or an HF-transformers sidecar. Pending an owner decision.

## Open decisions for the owner

1. **Hub namespace:** personal account or an org. Needed before PSP 7.
2. **Gemma license:** whether the model card must carry the Gemma terms. Needed before PSP 7.
3. **v1 token position:** TBG (recommended, because the product is pre-generation detection)
   or SLT. Decide after seeing the sweep. Both sets of numbers are published either way, and
   only TBG is servable in v1.
4. **Serving a non-last-layer probe on the deployed E4B/vLLM endpoint:** a vLLM plugin, an HF
   sidecar, or train with `--layers final` and accept that sweep's measured cost. Decide after
   the sweep. If the chosen window is at or next to L, `final` may cost almost nothing.
