# QA Guide

## Table of Contents

- [Summary](#summary)
- [What does success look like for the detectors?](#what-does-success-look-like-for-the-detectors)
  - [Why can't the layer choice see the test fold?](#why-cant-the-layer-choice-see-the-test-fold)
  - [Why AUROC and not accuracy at a threshold?](#why-auroc-and-not-accuracy-at-a-threshold)
  - [Why do the comparative criteria need a significance test as well as a bar?](#why-do-the-comparative-criteria-need-a-significance-test-as-well-as-a-bar)
  - [Why does criterion 4 need an effect size and not just p < 0.01?](#why-does-criterion-4-need-an-effect-size-and-not-just-p--001)
  - [What keeps two runs from being compared by mistake?](#what-keeps-two-runs-from-being-compared-by-mistake)
  - [What keeps the SEP probe from being scored on its own training data?](#what-keeps-the-sep-probe-from-being-scored-on-its-own-training-data)
  - [Why isn't an accuracy uplift enough to credit the detector?](#why-isnt-an-accuracy-uplift-enough-to-credit-the-detector)
  - [Why might the escalation uplift be overstated?](#why-might-the-escalation-uplift-be-overstated)
- [Test strategy](#test-strategy)
- [What does each test file cover?](#what-does-each-test-file-cover)
- [Running the suites](#running-the-suites)
- [GPU-gated tests](#gpu-gated-tests)
- [Fixtures and test doubles](#fixtures-and-test-doubles)
- [Continuous integration](#continuous-integration)
- [Manual checks before a release](#manual-checks-before-a-release)
- [Bug triage and reporting](#bug-triage-and-reporting)

## Summary

```bash
uv sync --all-extras
uv run pytest -q             # unit suite, CPU only, ~1 minute
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # type check (paths configured in pyproject.toml)
```

Every test runs on CPU without a model download, network or cloud account:
the language model, the NLI model and the judge are replaced by small
in-memory doubles. One smoke test needs a CUDA GPU and skips itself
otherwise. CI runs the same commands plus `terraform fmt` and `validate`;
tests run on Python 3.12 and 3.13.

Unit tests prove the code computes what it claims. Whether the detectors
*work* — whether high uncertainty actually means hallucination — is a
research question answered by the criteria below, on real models.

## What does success look like for the detectors?

Four measurable targets, each checked by a script over
`eval/run_benchmark.py` output. Each script prints its value with a 95%
interval, the target, and whether it was met. The pass bars live in
`eval/success_criteria.py`.

| # | Criterion | Met when | Measured by |
|---|---|---|---|
| 1 | SE separates hallucinated from correct TriviaQA answers | SE AUROC ≥ 0.75 on TriviaQA-1000 (bootstrap 95% CI reported alongside) | `python -m eval.compute_auroc --in <triviaqa results>` |
| 2 | Clustering by meaning beats token-level uncertainty | SE AUROC − PE AUROC ≥ 0.05 **and** the paired-bootstrap 95% CI of the gap excludes 0 | same command, `criteria.se_minus_pe_auroc` |
| 3 | Escalating to the judge pays off | final − greedy accuracy ≥ 5 percentage points **and** McNemar's exact test p < 0.05 | `python -m eval.escalation_uplift --in <triviaqa results>` (benchmark run with the judge) |
| 3b | The detector picks which questions to escalate better than chance | final accuracy − the expected accuracy of escalating the same number of questions at random > 0, with a paired-bootstrap 95% CI that excludes 0 | same command, `criteria.escalation_beats_random_pp` |
| 4 | SE flags adversarial prompts | one-sided Mann-Whitney U p < 0.01 **and** effect size AUC ≥ 0.75 | `python -m eval.adversarial_separation --adversarial <adversarial results> --reference <triviaqa results>` |

The Semantic Entropy Probe (SEP) adds two diagnostics rather than pass bars,
both AUROCs of its p(high SE) score:

- **Against binarized SE, by nested cross-validation** (mean ± std over the
  outer folds, `metrics.json` → `selection`), written by
  `python -m eval.train_sep_probe`: how well the one-pass probe predicts
  whether the ten-sample SE would be high, on calibration questions it was
  not fitted to.
- **Against model error, on the held-out benchmark slice**: SEP's AUROC from
  `eval.compute_auroc` next to PE and SE, on questions the probe never saw.

**Status: not yet measured on real hardware.** The scripts are unit-tested
on hand-built rows; no GPU benchmark run has been recorded. Run
[the benchmark](../../README.md#how-do-i-evaluate-one-model) to produce the
numbers.

### Why can't the layer choice see the test fold?

Because the best of many cross-validated scores is optimistic: pick the
window of layers that scored highest and report that same score, and part
of it is luck. So the trainer nests the choice: in each outer fold, an inner
cross-validation picks the window on the outer-training rows only, and the
outer test rows then score that pick. The high/low split point γ is also
refitted from each fold's training rows. The reported score therefore
includes the cost of choosing, and the window chosen in each outer fold is
recorded as a stability check. The per-layer sweep in `metrics.json` is
descriptive only; nothing is chosen from it.

### Why AUROC and not accuracy at a threshold?

AUROC scores the *ranking* across every possible threshold, so it measures
the detector itself; the threshold is a separate, deployment-specific choice
(see the [README](../../README.md#when-does-it-escalate-and-how-do-i-tune-it)).

### Why do the comparative criteria need a significance test as well as a bar?

Because a point estimate that clears a bar can be sampling noise. On a few
hundred rows, an AUROC moves by several hundredths between resamples, about
the size of criterion 2's 0.05 margin. So criterion 2 also needs the 95% CI
of the SE − PE gap to exclude zero. The interval comes from a *paired*
bootstrap: both AUROCs are recomputed on the same resampled rows, which keeps
their correlation (the same questions are hard for both detectors) and gives
a much tighter interval for the difference than two separate intervals would.
Criterion 3 compares two accuracies on the same questions, so it uses
McNemar's exact test, which looks only at the questions where escalation
changed correctness (fixed vs broken), plus a paired bootstrap CI.

### Why does criterion 4 need an effect size and not just p < 0.01?

A p-value measures how sure we are that there is *some* difference, which
grows with sample size; it does not measure how *large* the difference is.
With a thousand reference rows, two heavily overlapping score distributions
still reach p < 0.01. The effect size AUC = U / (n_adversarial × n_reference)
is the probability that a random adversarial prompt outscores a random
correctly answered one. The bar of 0.75 matches criterion 1 and sits above
Vargha and Delaney's 0.71 threshold for a "large" effect. The Mann-Whitney
test compares ranks, so it assumes nothing about the score distribution.
Entropies are bounded, skewed, and often tied at 0, which rules out a
t-test. The committed adversarial set has only 7 prompts, so p < 0.01 needs
nearly perfect separation; add prompts to
`eval/datasets/adversarial.jsonl` for a more sensitive test.

### What keeps two runs from being compared by mistake?

Every benchmark row records `model_id`, `sample_n` and `sample_temperature`.
`compute_auroc` and `escalation_uplift` refuse a results file whose rows
disagree on them, and `adversarial_separation` refuses two runs that differ.
Rows written before these fields existed are refused too; re-run the
benchmark.

### What keeps the SEP probe from being scored on its own training data?

Two things. `load_triviaqa --offset` builds non-overlapping slices of one
fixed shuffle of the *unique* questions (TriviaQA lists most questions once
per evidence source, so slicing rows would repeat questions across slices),
so the calibration and benchmark sets are disjoint by construction; the
default slices (questions 0–499 for calibration, 500–699 for the benchmark)
are. And `train_sep_probe` writes `train_ids.json` inside the probe
directory; `run_benchmark` refuses to start if any benchmark question id is
in it, if a probe in use has no such file, or if a benchmark question has no
`id` to check. `eval.publish_sep_probe` repeats the check on the held-out
report's question ids before anything is uploaded. A probe evaluated on the
questions it was fit to would report an optimistic AUROC.

### Why isn't an accuracy uplift enough to credit the detector?

Because escalating *any* share of questions to a stronger judge raises
accuracy, so criterion 3 alone would pass with a detector that outputs
noise. Criterion 3b asks the counterfactual question: had the same number
of questions been escalated at random, what accuracy would the system have
reached? That expectation is exact,
`greedy accuracy + (k / n) · mean(judge correct − greedy correct)`, so no
simulation is needed. It requires the judge's answer to every question, which
`run_benchmark` records as `judge_answer` whenever the judge is enabled, at
one judge call per question.

The report also gives `accuracy_vs_rate`: accuracy at escalation rates 0 to
100% for random escalation and for each recorded detector, escalating the
top-k questions by that detector's score. Ties at the cutoff share the
remaining slots, so a detector with many identical scores, such as discrete
SE at 0, gets no credit for an arbitrary tie-break.

### Why might the escalation uplift be overstated?

Correctness is a whole-token match of any accepted answer after SQuAD-style
normalization (`eval/benchmark_results.py::is_correct`). That still favours
longer answers: "It could be Paris or Lyon" contains more candidates than
"Lyon". If the judge
answers more verbosely than the small model, part of the uplift is
verbosity. `escalation_uplift` reports the mean answer length on both sides
and sets `length_warning` (and logs a warning) when the final answers are
more than 1.5 times longer. Treat a met criterion 3 with a length warning as
unconfirmed until the answers are checked by hand or by a stricter matcher.

## Test strategy

- **Math is tested against hand-computed values.** Entropy formulas get
  inputs whose answers you can check on paper (e.g. 10 identical answers →
  SE = 0; 10 distinct answers → ln 10).
- **Contracts are tested at the boundary.** The evaluator must fail fast when
  a detector's inputs are missing, when the escalation detector is not
  enabled, or when a policy references an unscored detector.
- **External models are faked, not mocked deep.** A fake SLM, a
  string-equality NLI and a fake judge implement the same protocols as the
  real ones, so tests exercise the real dispatch code.
- **The eval loop is tested end to end.** Benchmark → AUROC runs on a
  four-question set with a fake model whose correct ranking is known.
- **Real-model behavior is a manual check**, not a unit test — see
  [Manual checks](#manual-checks-before-a-release).

New features follow test-first: write the failing test that defines
"done", then the code.

## What does each test file cover?

| File | Covers |
|---|---|
| `test_predictive_entropy.py` | PE formula against a hand computation, confidence ordering, empty and missing-logprob errors |
| `test_semantic_entropy.py` | clustering by bidirectional entailment, question-conditioned NLI input, likelihood-weighted and discrete SE values against hand computations, length normalization, fail-fast on missing or non-finite logprobs, empty input |
| `test_nli.py` | NLI label names read from the checkpoint's `id2label` in any order, softmax `label_probs`, argmax `classify`, fail-fast on unknown labels (fake model, no download) |
| `test_semantic_entropy_probe.py` | SEP probe directory and `hf://` loading (Hub download mocked), the compatibility check naming both values for each backbone and prompt field, `.pt` and other files refused as obsolete, SLT probes refused in v1, the score as a sigmoid over the probe's layers in increasing order, the probe bytes hashed as loaded |
| `test_sep_recipe.py` | best-split binarization (exhaustive midpoints, lower γ on ties, never on a data value, no worse than the paper's grid, rounding of summation-order noise), the logistic fit's paper defaults, convergence failure naming the next fallback, the folded scaler matching a scaler pipeline, p(high SE) as a sigmoid |
| `test_sep_layer_selection.py` | single-layer sweep, window selection by nested CV that never fits on outer test rows, γ fitted per training fold, stratification without features, layer 0 excluded, deeper window on ties, fixed layers, a pure-noise AUROC near 0.5, out-of-fold probabilities and the Youden threshold |
| `test_collect_sep_dataset.py` | the collector's rows (both SE estimators from one clustering), the HF-only and registry-revision requirements, every layer's hidden states, the finished questions written before a failure or Ctrl-C, one training-log line per question, calibration-set validation |
| `test_sep_dataset_io.py` | the dataset directory round trip, tensor ids in parquet order, shapes checked against the backbone, unknown config fields and schema versions refused, an incomplete dataset refused unless allowed |
| `test_train_sep_probe.py` | the five probe-directory files, the probe config's blocks and deviations, safetensors-only loading, a refit from the dataset reproducing the weights, `--layers final`, the pre-registered `--max-iter` values, repo commit and dirty state, the `metrics.json` schema |
| `test_publish_sep_probe.py` | the model card (golden file), every publish refusal (no held-out report, probe hashes, model, dataset, dirty repo, training questions in the held-out set, bad repo id or tag), no token argument, the mocked upload and tagging of both repos, `--dry-run` uploading nothing |
| `test_demo_cli.py` | SEP printed as "P(high SE)" with a percentage threshold, nats detectors as plain numbers |
| `test_escalation.py` | `SingleDetector`, `AnyAbove`, `AllAbove` decisions and validation |
| `test_evaluator.py` | `evaluate_responses` / `ResponseEvaluator` dispatch, prerequisites, judge escalation |
| `test_model_registry.py` | registry entries, single local/deployed default, vLLM flags per runner (raw logprobs for generate, unnormalized last-token pooling for embed), tensor parallelism per GPU count, L4 context cap, every machine type able to carry its accelerator type and count |
| `test_slm_provider.py` | vLLM provider against a mocked OpenAI client (sampling, chat-shaped embedding request, `VLLM_AUTH` modes), HF logprob extraction (raw distribution, first EOS kept, padding dropped), provider selection, sweep model-id validation |
| `test_google_auth.py` | `GoogleAccessToken` reuse and refresh with fake credentials, plus a contract test against the **real** OpenAI client (fake HTTP transport) proving the token is refreshed on chat and embedding requests |
| `test_serving_launch.py` | the container's `vllm serve` argv: default runner, Vertex port, fail-fast on missing model, unknown runner or unregistered model |
| `test_benchmark_pipeline.py` | `eval/run_benchmark.py` → `eval/compute_auroc.py` end to end; run config (incl. answer instruction) recorded per row; judge answer recorded on every row; SEP probe leakage guard (overlap, missing provenance and id-less questions refused); the probe's reference and hashes recorded per row |
| `test_eval_criteria.py` | statistics primitives (rank AUROC vs sklearn, deterministic bootstrap, exact McNemar); whole-token SQuAD-normalized correctness; each criterion met only with its significance or effect-size condition; escalation vs random escalation at the same rate, tie-averaged top-k, accuracy-vs-rate curve; answer-length distribution; run comparability; slice bounds; probe provenance |
| `test_model_comparison.py` | `run_benchmark --sweep` (one file per model, one model at a time, registry and per-model SEP probe checked before any scoring); SEP's default probe follows the evaluated model; `compare_models` accuracy and AUROC per model, refusing different questions, different sampling or a repeated model |
| `test_calibrate_threshold.py` | τ by Youden's J under the strict `score > τ` rule, ties to the higher τ, dev/test overlap refused, the setting to use (`SE_THRESHOLD` or `SEP_THRESHOLD`) reported |
| `test_gemini_judge.py` | `GeminiJudge` request shape and verdict parsing against a fake `google-genai` client |
| `web/test_web_shell.py` | playground status (provider, model, judge readiness with its reason), presets equal the example's `SCENARIOS`, the UI served at `/`, JSON 404 for unknown API routes, fail-fast without a built UI, `127.0.0.1:8765` binding |
| `web/test_web_score.py` | `/api/score`: the three presets give the example's PE / SE / cluster counts, clusters and SE for typed answers, PE null unless every answer carries logprobs, 422 on empty or blank input, NLI failures surfaced with their message |
| `web/test_web_ask.py` | `/api/ask`: detect vs chat, the judge called only by chat and only on escalation, chat refused (400) without `GCP_PROJECT_ID`, provider errors surfaced (500 with the message), the provider built once |
| `web/test_web_results.py` | `/api/results` listing (runs and sweep directories), run report equal to `auroc_report`, sweep report equal to `comparison_report`, paths outside `eval/results/` refused (400), missing (404) and unscorable (422) files |
| `web/test_web_static_build.py` | the Vite build is committed under `demo/web/static/` and every asset `index.html` loads exists; the bundle's tab list reads Score answers, Ask the model, Measure the detectors in that order, with the retired "Benchmark results" label absent; the README's numbered ways in follow the same order |

The playground UI has its own suite under `frontend/` (Vitest + React
Testing Library, the API client mocked), covering the logger and API client,
the Score tab (presets, cluster chips, the threshold slider flipping
*escalate* with no request, the "needs token logprobs" state), the Ask tab
(status strip, Chat disabled with its reason, progress, errors shown
verbatim) and the *Measure the detectors* tab (empty state, AUROC with CIs,
comparison table).

## Running the suites

```bash
uv run pytest -q                                   # everything
uv run pytest tests/unit/test_semantic_entropy.py -v
uv run pytest -k escalation                        # by name
uv run pytest -x --lf                              # stop at first failure, rerun last failures
```

`pyproject.toml` puts both `src/` and the repo root on the import path, so
tests can import `hallucinations_math_lab`, `demo` and `eval` without
installing them. The playground UI's checks run from `frontend/` (needs Node):

```bash
cd frontend && npm ci
npm run typecheck && npm run lint && npm test   # tsc --noEmit, eslint, vitest run
```

`tests/integration/` is reserved for tests that need real
services and is currently empty.

## GPU-gated tests

Tests marked `@pytest.mark.requires_gpu` call `pytest.skip` when no CUDA
device is present, so a plain run shows them as `s`. To run them on a GPU
machine:

```bash
uv run pytest -m requires_gpu -v
```

Today this is one smoke test that loads the local default model through
`HFTransformersProvider` and generates once.

## Fixtures and test doubles

- **`FakeSLM`** (`test_evaluator.py`) and **`ScatteringSLM`**
  (`test_benchmark_pipeline.py`) implement `SamplingProvider` with fixed
  answers.
- **`FakeNLI` / `EqualityNLI`** decide "same meaning" by string equality,
  standing in for DeBERTa.
- **`_ConstScoreDetector`** returns a fixed score and declares chosen
  prerequisites, for testing dispatch and escalation without real math.
- **Settings** are isolated with `monkeypatch.setenv(...)` plus
  `get_settings.cache_clear()` before and after — settings are cached, so
  forgetting the clear leaks configuration between tests.
- Temporary files use pytest's `tmp_path`. Nothing writes into the repo.
- Real data for manual runs: `eval/datasets/adversarial.jsonl` (committed)
  and TriviaQA slices built by `eval/datasets/load_triviaqa.py` — the two
  defaults are built on first use into `data/questions/`
  (`eval/datasets/default_question_sets.py`).
  Benchmark output goes under `eval/results/` and probes under `data/`,
  both gitignored.

## Continuous integration

`.github/workflows/ci.yml`, on every push to `main` and every pull request:

| Job | Does |
|---|---|
| `lint` | `ruff check .` and `ruff format --check .`, ruff pinned to 0.17.0 |
| `terraform` | `terraform fmt -check -recursive`, `init -backend=false`, `validate` (Terraform 1.16.6, no credentials) |
| `typecheck` | `uv sync --locked --all-extras`, then `mypy` (`check_untyped_defs` on) |
| `test` (3.12, 3.13) | `uv sync --locked --all-extras`, then `pytest -q` |

A red `test` job that fails during install usually means `uv.lock` is stale
— run `uv lock` and commit it.

## Manual checks before a release

1. `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`
   and `uv run mypy` pass, and `terraform validate` passes in `terraform/`.
2. No known vulnerabilities in the locked dependencies:
   ```bash
   uv export --all-extras --no-hashes --no-emit-project --format requirements-txt -o reqs.txt \
     && uvx pip-audit -r reqs.txt --no-deps --disable-pip
   ```
   Fix findings with `uv lock --upgrade-package <name>` (or `uv lock --upgrade`)
   and rerun the suite.
3. `uv build` succeeds and the wheel contains only `hallucinations_math_lab/`
   plus `LICENSE`.
4. Both playground scripts run and print the tables shown in the README:
   `examples/score_precollected_samples.py` and `examples/custom_detector.py`.
5. With a model available: `python -m demo.cli detect "Who wrote 1984?"`
   returns SE near 0, and a 10-question `python -m eval.run_benchmark --limit 10 --no-judge`
   followed by `python -m eval.compute_auroc` completes. With two small models
   in `TARGET_MODEL_IDS`, the same with `--sweep` followed by
   `python -m eval.compare_models` prints one row per model.
6. The web playground: `npm run build` in `frontend/` leaves no diff in
   `demo/web/static/` (the committed build is current); `uv run python -m demo.web`
   serves `http://127.0.0.1:8765`; on the Score tab the *paraphrased* preset
   gives PE 0.600, SE 0.000, 1 cluster and *divergent* PE 0.400, SE 1.834,
   7 clusters; moving the threshold slider sends no request; and
   `curl "http://127.0.0.1:8765/api/results/report?path=../../etc/passwd"` returns 400.
7. Every command in the README and the usage guides still matches the code.
8. Before publishing a SEP probe: the GPU-only assumptions in
   [`SEP_CLASSIFIER.md`](../plans/SEP_CLASSIFIER.md#what-needs-a-real-gpu-run)
   (hidden-state layer count, convergence at the paper's iteration limit,
   class balance of the split, provider agreement) are checked on the real
   run, and the README states the measured layer window and the
   `final_layer_auroc` baseline.

## Bug triage and reporting

Open a GitHub issue with:

- **What you ran** — the exact command or a minimal script.
- **What you expected and what happened** — full traceback; the structured
  log context (`ctx` fields) usually names the prompt, detector and setting
  involved.
- **Environment** — OS, Python version, `SLM_PROVIDER`, `VLLM_MODEL_ID`,
  `ENABLED_DETECTORS`, GPU or CPU.

Triage order:

1. **Wrong numbers** (a detector score that contradicts its formula) — top
   priority; reproduce with a hand-built `SampleSet` and add it as a
   regression test.
2. **Crashes on valid configuration** — reproduce with `.env.example`
   defaults, then add a failing test.
3. **Model / infrastructure issues** (downloads, vLLM, Vertex) — check the
   [DevOps Guide caveats](DEVOPS_GUIDE.md#caveats-before-you-deploy) first.
4. **Docs drift** — fix the doc in the same pull request as the code it
   describes.
