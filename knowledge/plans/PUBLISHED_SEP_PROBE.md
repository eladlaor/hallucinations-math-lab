# Plan: Publish a Trained SEP Probe on Hugging Face

> **Priority: HIGH — a release requirement, not a nice-to-have.** Today the repo
> ships the *recipe* for a Semantic Entropy Probe (SEP) but no trained probe:
> nobody has run training, and nothing is published. A newcomer who wants to see
> SEP work must first rent a GPU and run 500 questions × 11 samples. Until a
> trained probe, its training record and its measured results are one download
> away, the SEP part of the project is a claim the reader cannot check.

## Table of contents

- [Summary](#summary)
- [Why is this important?](#why-is-this-important)
- [Terminology: is the probe a "linear classifier"?](#terminology-is-the-probe-a-linear-classifier)
- [What gets published](#what-gets-published)
- [Prerequisites (ordering)](#prerequisites-ordering)
- [Work items (TDD)](#work-items-tdd)
- [Success criteria](#success-criteria)
- [Out of scope](#out-of-scope)
- [Risks and decisions](#risks-and-decisions)

## Summary

- **Deliverable:** one Hugging Face model repo per backbone, starting with
  `google/gemma-4-E2B-it` (the smallest model, the one most users can run),
  holding the probe weights, a config that pins the exact backbone it belongs to,
  the full training record, the measured metrics, and a model card.
- **Companion dataset repo:** the calibration rows (question, the 10 sampled
  answers plus the greedy answer, their meaning clusters, the SE values, and
  every layer's hidden states at two token positions). With it, anyone can
  refit or re-examine the probe on a laptop, no GPU.
- **Status:** the code (items 1–6 and 8) is implemented by
  [`SEP_CLASSIFIER.md`](SEP_CLASSIFIER.md) items 2–18. Item 7, the real run,
  is open. The Hub namespace and license decisions were made on 2026-11-08 (see
  [Risks and decisions](#risks-and-decisions)), and both cards now carry the licenses.
- **Code:** the trainer writes its record to files (not only stdout), a
  publish script uploads, and the detector loads a probe straight from the Hub,
  refusing a probe whose backbone doesn't match the model under test.
- **Do it after** switching SEP to the classifier recipe and the B2 label fixes in
  [`TODO.md`](TODO.md) — otherwise the first published probe is obsolete on arrival.
- **Done when:** a fresh clone, with no training run, can enable SEP by pointing at
  the Hub repo and reproduce the published held-out numbers.

## Why is this important?

Because SEP's whole selling point is "most of SE's signal at one forward pass",
and the only way a reader can test that claim today is to pay for the expensive
part first. Concretely, a published probe:

- **Lets people play immediately.** Way 3 (ask the model) and way 2 (benchmark)
  gain SEP with one `.env` line, on any machine that can run E2B.
- **Makes the claim checkable.** The training record and held-out metrics show
  how well the probe tracks SE — including if it tracks it badly. An honest weak
  result is still a result; a recipe with no result is not.
- **Is the reference point for contributors.** Anyone retraining (another
  backbone, the paper's recipe) compares against a fixed, versioned baseline.
- **Closes part of the "No results" item** in [`TODO.md`](TODO.md).

## Terminology: is the probe a "linear classifier"?

**Yes, now.** The original trainer fit ridge regression, a linear
*regressor*: it predicted the SE value itself (a number in nats). A
*classifier* predicts a category — "high SE" vs "low SE". The paper (Kossen
et al., 2024) does the classifier version: it splits SE into high/low and fits
logistic regression. **Decided 2026-11-05 and implemented by
[`SEP_CLASSIFIER.md`](SEP_CLASSIFIER.md):** the probe is a logistic classifier
whose score is p(high SE), so the published probe is a linear classifier. The
general term covering both is **linear probe**.

**"Training log" — what to expect.** A linear probe is fitted in one shot, not
over many epochs, so there is no falling loss curve to show. The meaningful
record is (a) the per-question data collection (SE label, cluster count,
answers) and (b) the cross-validated agreement between probe and SE. That is
what gets published as the training record.

## What gets published

**Model repo** `eladlaor/sep-probe-gemma-4-E2B-it`:

| File | Content |
|---|---|
| `probe.safetensors` | The linear layer's weights and bias. `safetensors`, the Hub's standard weight format, which cannot carry executable code. |
| `probe_config.json` | See [Artifact format](SEP_CLASSIFIER.md#artifact-format): the backbone (id, commit, depth, width), the layers and token position read, the prompt, the label and classifier settings, the recommended threshold, the training provenance and the active deviations from the paper. |
| `train_ids.json` | The training question ids, so `run_benchmark` can keep refusing overlap. |
| `training_log.jsonl` | One line per calibration question: id, SE (both estimators), cluster count, answer lengths, greedy correctness, elapsed time. |
| `metrics.json` | The per-layer sweep at both token positions, the layer-window selection with its nested-CV AUROC against binarized SE (mean ± std per fold), the last-layer baseline (`final_layer_auroc`), the recommended threshold with its TPR, FPR and escalation rate, plus the held-out benchmark AUROC of SEP next to PE and SE on the disjoint 200-question slice, with confidence intervals. |
| `README.md` | Model card: what it is, how to use it with this repo, the numbers, the limits (works only for this exact backbone; TriviaQA-style questions only; not a correctness oracle), the license, and citations. |

**Dataset repo** `eladlaor/sep-calibration-gemma-4-E2B-it`: `rows.parquet`
(one row per calibration question: the question, the 10 sampled answers plus
the greedy answer, their cluster ids, both SE values, greedy correctness),
`hidden_states.safetensors` (every layer at two token positions, TBG and SLT)
and `collection_config.json` (see
[the dataset directory](SEP_CLASSIFIER.md#the-dataset-directory-companion-hub-dataset-repo)),
plus `training_log.jsonl` and a dataset card (`README.md`, rendered from
`eval/templates/SEP_DATASET_CARD.md`). The card covers the files and columns, the refit
command, provenance, and the license and TriviaQA attribution.
500 rows, and it is what turns "trust our probe" into "refit it yourself".

## Prerequisites (ordering)

1. ~~**Switch SEP to the classifier recipe.**~~ Done: implemented by
   [`SEP_CLASSIFIER.md`](SEP_CLASSIFIER.md) items 1–17.
2. **B2 label fixes** that change SE values: brief-answer prompt, correctness
   matching (for the benchmark AUROC), NLI label order.
3. **B1 history cleanup / repo going public** — the model card links to the repo.
   Decided 2026-11-08 (the owner delegated it): rewrite the existing repo with
   `git filter-repo`, which keeps its URL and history. This runs **last**, after every
   branch is merged to `main`, the GPU run is done, and the probe has been verified
   loading from the Hub (both Hub repos are uploaded private first). Before the rewrite,
   take a full mirror backup. After it, run gitleaks and trufflehog on the result, then
   force-push the rewritten `main` and tags (this needs the owner's explicit go-ahead at
   execution time), then delete the remote `v0.1.0` tag and the
   `claude/load-bearing-words-*` branch. The GitHub repo and both Hub repos then go
   public together.
4. **A GPU run** that fits E2B (README hardware tiers). This is also the
   project's first real-hardware run.

## Work items (TDD)

Each item: failing test first, then the code.

**Items 1–6 and 8 are implemented by [`SEP_CLASSIFIER.md`](SEP_CLASSIFIER.md)
items 2–18**, with the changes noted in each item below. Item 7 stays here.

1. **Trainer writes its record to files.** `train_sep_probe` writes
   `training_log.jsonl`, `metrics.json` and `probe_config.json` next to the probe.
   *Test:* with a fake provider, all three files exist, the log has one line per
   question, and the config holds the backbone id and hidden size.
2. **Pin the backbone revision.** The provider reports the model's commit hash;
   training records it. *Test:* the config contains a non-empty revision; a
   missing one raises.
3. **Save as `safetensors`.** *Test:* round-trip save → load gives identical
   scores. The `.pt` path is **removed**, not kept: a `.pt` probe is refused
   with a retrain hint (`ObsoleteSEPProbeError`).
4. **Load from the Hub.** `SEP_PROBE_PATH=hf://<repo>[@revision]` downloads and
   loads. Fail fast if the probe's backbone id, revision or hidden size differs
   from the model under test. *Tests:* mismatch raises with both ids in the
   message; a matching local fixture loads (Hub download mocked).
5. **Calibration dataset export.** The export is **always** written, not
   optional: `eval.collect_sep_dataset` writes the dataset directory, and it is
   the fit step's only input (`eval.train_sep_probe --dataset DIR`). *Test:* row
   count and columns; refitting from the export reproduces the saved weights.
6. **Publish script.** `eval/publish_sep_probe.py --probe-dir … --repo …` renders
   the model card from `metrics.json` + `probe_config.json` and uploads both repos
   (`huggingface_hub`, token from `hf auth login`, never from the CLI args).
   *Test:* card rendering is pure and snapshot-tested; upload is mocked. Refuses
   to publish without `metrics.json` containing the held-out benchmark AUROC.
7. **Run it for real.** Collect and train on E2B; the owner then reads the
   layer and position sweep in `metrics.json` and picks the v1 token position
   (only TBG is servable in v1); benchmark on the disjoint slice; publish with
   a version tag (`v1`). The command sequence is in the
   [DevOps Guide](../usage_guides/DEVOPS_GUIDE.md#publishing-a-sep-probe-to-the-hub).
   Afterwards, replace the README's and the `.pt` refusal's generic `hf://`
   hint with the published reference.
8. **Docs.** README *How do I train and use a SEP?* gets a "Use the published
   probe" path first and "Train your own" second; the web playground plan
   ([`WEB_PLAYGROUND.md`](WEB_PLAYGROUND.md)) points at it; CHANGELOG entry.
   (The user guide is folded into the README.)

## Success criteria

- [ ] A fresh clone, no `data/probes/`, with `SEP_PROBE_PATH=hf://…` and
      `VLLM_MODEL_ID=google/gemma-4-E2B-it`, runs `run_benchmark` with SEP enabled.
- [ ] That run reproduces the SEP AUROC in the published `metrics.json` within
      its confidence interval.
- [ ] Pointing the probe at a different backbone fails before any scoring.
- [ ] Refitting from the published dataset reproduces the published weights.
- [ ] The model card states the numbers, the limits and the license, and links
      back to the exact repo commit that trained it.
- [ ] The README states the measured layer window and the `final_layer_auroc`
      baseline.

## Out of scope

- Probes for the larger Gemma 4 models — follow-ups using the same pipeline,
  once E2B is published.
- A Hub-hosted inference demo (Space). The local web playground covers that.
- Probes for backbones outside the model registry.

## Risks and decisions

| Item | Note |
|---|---|
| **Gemma license** | **Decided 2026-11-08: the model card carries the base model's license terms.** The Hub lists `google/gemma-4-E2B-it` (and the rest of the Gemma 4 family) as `apache-2.0`, ungated, with `license_link` https://ai.google.dev/gemma/docs/gemma_4_license, which redirects to Google's Apache 2.0 page. So the probe weights and config are released under Apache 2.0, the base model's license. The model card states this (`license: apache-2.0` and `license_link` in its front matter, plus a license section), and this repo's code stays MIT. The dataset card releases the derived columns under Apache 2.0 and credits TriviaQA (Joshi et al. 2017) for the questions. The Hub lists TriviaQA's license as unknown, so users must check its terms before redistributing the questions. |
| **Silent backbone drift** | A re-uploaded or updated Gemma checkpoint changes hidden states and quietly breaks the probe. Mitigated by pinning the revision (item 2) and refusing a mismatch (item 4). |
| **The result may be weak** | Publish it anyway, with the numbers. The value is that it is checkable. |
| **Hub namespace** | **Decided 2026-11-08: the personal account `eladlaor`.** The repos are `eladlaor/sep-probe-gemma-4-E2B-it` and `eladlaor/sep-calibration-gemma-4-E2B-it`. |
