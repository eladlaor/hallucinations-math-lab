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
- **Companion dataset repo:** the calibration rows (question, the 11 answers,
  their meaning clusters, the SE label, and the hidden state). With it, anyone
  can refit or re-examine the probe on a laptop, no GPU.
- **Code:** the trainer writes its record to files (not only stdout), a
  publish script uploads, and the detector loads a probe straight from the Hub,
  refusing a probe whose backbone doesn't match the model under test.
- **Do it after** the SEP recipe decision and the B2 label fixes in
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

**Not today — it is a linear *regressor*.** The current trainer fits ridge
regression: it predicts the SE value itself (a number in nats). A *classifier*
predicts a category — "high SE" vs "low SE". The paper (Kossen et al., 2024)
does the classifier version: it splits SE into high/low and fits logistic
regression. Whether to switch is an open decision in [`TODO.md`](TODO.md); the
published probe will be whichever recipe that decision picks. The general term
covering both is **linear probe**.

**"Training log" — what to expect.** A linear probe is fitted in one shot, not
over many epochs, so there is no falling loss curve to show. The meaningful
record is (a) the per-question data collection (SE label, cluster count,
answers) and (b) the cross-validated agreement between probe and SE. That is
what gets published as the training record.

## What gets published

**Model repo** `<hf-user>/sep-probe-gemma-4-E2B-it`:

| File | Content |
|---|---|
| `probe.safetensors` | The linear layer's weights and bias. `safetensors`, the Hub's standard weight format, which cannot carry executable code. |
| `probe_config.json` | Backbone model id **and commit hash**, hidden size, which layer and token the hidden state is taken from, the chat template used, recipe (ridge / logistic), its hyperparameters, the SE settings (N samples, temperature, NLI model), and — for a classifier — the high/low split point and the recommended threshold. |
| `train_ids.json` | The existing sidecar: the training question ids, so `run_benchmark` can keep refusing overlap. |
| `training_log.jsonl` | One line per calibration question: id, SE, cluster count, answer lengths, elapsed time. |
| `metrics.json` | Cross-validated agreement (Pearson r, or AUROC for a classifier) with mean ± std per fold, plus the held-out benchmark AUROC of SEP next to PE and SE on the disjoint 200-question slice, with confidence intervals. |
| `README.md` | Model card: what it is, how to use it with this repo, the numbers, the limits (works only for this exact backbone; TriviaQA-style questions only; not a correctness oracle), the license, and citations. |

**Dataset repo** `<hf-user>/sep-calibration-gemma-4-E2B-it`: one row per
calibration question — question, the 11 sampled answers, their cluster ids,
SE, and the pooled hidden state. Small (500 rows), and it is what turns "trust
our probe" into "refit it yourself".

## Prerequisites (ordering)

1. **SEP recipe decision** (TODO, *Open decisions*): Kossen logistic recipe or
   ridge with the claim dropped. Publishing first means republishing.
2. **B2 label fixes** that change SE values: brief-answer prompt, correctness
   matching (for the benchmark AUROC), NLI label order.
3. **B1 history cleanup / repo going public** — the model card links to the repo.
4. **A GPU run** that fits E2B (README hardware tiers). This is also the
   project's first real-hardware run.

## Work items (TDD)

Each item: failing test first, then the code.

1. **Trainer writes its record to files.** `train_sep_probe` writes
   `training_log.jsonl`, `metrics.json` and `probe_config.json` next to the probe.
   *Test:* with a fake provider, all three files exist, the log has one line per
   question, and the config holds the backbone id and hidden size.
2. **Pin the backbone revision.** The provider reports the model's commit hash;
   training records it. *Test:* the config contains a non-empty revision; a
   missing one raises.
3. **Save as `safetensors`.** *Test:* round-trip save → load gives identical
   scores; the `.pt` path keeps working for local probes or is removed (decide in item).
4. **Load from the Hub.** `SEP_PROBE_PATH=hf://<repo>[@revision]` downloads and
   loads. Fail fast if the probe's backbone id, revision or hidden size differs
   from the model under test. *Tests:* mismatch raises with both ids in the
   message; a matching local fixture loads (Hub download mocked).
5. **Calibration dataset export.** Training optionally writes the rows above as
   Parquet. *Test:* row count and columns; refitting from the export reproduces
   the saved weights.
6. **Publish script.** `eval/publish_sep_probe.py --probe-dir … --repo …` renders
   the model card from `metrics.json` + `probe_config.json` and uploads both repos
   (`huggingface_hub`, token from `hf auth login`, never from the CLI args).
   *Test:* card rendering is pure and snapshot-tested; upload is mocked. Refuses
   to publish without `metrics.json` containing the held-out benchmark AUROC.
7. **Run it for real.** Train on E2B, benchmark on the disjoint slice, publish
   with a version tag (`v1`).
8. **Docs.** README *How do I train and use a SEP?* gets a "Use the published
   probe" path first and "Train your own" second; USER_GUIDE and the web
   playground plan ([`WEB_PLAYGROUND.md`](WEB_PLAYGROUND.md)) point at it;
   CHANGELOG entry.

## Success criteria

- [ ] A fresh clone, no `data/probes/`, with `SEP_PROBE_PATH=hf://…` and
      `VLLM_MODEL_ID=google/gemma-4-E2B-it`, runs `run_benchmark` with SEP enabled.
- [ ] That run reproduces the SEP AUROC in the published `metrics.json` within
      its confidence interval.
- [ ] Pointing the probe at a different backbone fails before any scoring.
- [ ] Refitting from the published dataset reproduces the published weights.
- [ ] The model card states the numbers, the limits and the license, and links
      back to the exact repo commit that trained it.

## Out of scope

- Probes for the larger Gemma 4 models — follow-ups using the same pipeline,
  once E2B is published.
- A Hub-hosted inference demo (Space). The local web playground covers that.
- Probes for backbones outside the model registry.

## Risks and decisions

| Item | Note |
|---|---|
| **Gemma license** | The probe is trained on Gemma's internal states. Check whether the Gemma terms treat the probe as a derivative that must carry them; if so, the model card ships the terms. Decide before item 7. |
| **Silent backbone drift** | A re-uploaded or updated Gemma checkpoint changes hidden states and quietly breaks the probe. Mitigated by pinning the revision (item 2) and refusing a mismatch (item 4). |
| **The result may be weak** | Publish it anyway, with the numbers. The value is that it is checkable. |
| **Hub namespace** | Personal account vs an org. Needs the owner's choice before item 7. |
