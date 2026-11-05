# QA Guide

## Table of Contents

- [Summary](#summary)
- [What does success look like for the detectors?](#what-does-success-look-like-for-the-detectors)
  - [Why AUROC and not accuracy at a threshold?](#why-auroc-and-not-accuracy-at-a-threshold)
  - [Why do the comparative criteria need a significance test as well as a bar?](#why-do-the-comparative-criteria-need-a-significance-test-as-well-as-a-bar)
  - [Why does criterion 4 need an effect size and not just p < 0.01?](#why-does-criterion-4-need-an-effect-size-and-not-just-p--001)
  - [What keeps two runs from being compared by mistake?](#what-keeps-two-runs-from-being-compared-by-mistake)
  - [What keeps the SEP probe from being scored on its own training data?](#what-keeps-the-sep-probe-from-being-scored-on-its-own-training-data)
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
| 4 | SE flags adversarial prompts | one-sided Mann-Whitney U p < 0.01 **and** effect size AUC ≥ 0.75 | `python -m eval.adversarial_separation --adversarial <adversarial results> --reference <triviaqa results>` |

A fifth number is a diagnostic rather than a pass bar: how closely the
Semantic Entropy Probe tracks SE on prompts it never saw, as a 5-fold
cross-validated Pearson r (mean ± std), logged by
`python -m eval.train_sep_probe` before it saves the probe.

**Status: not yet measured on real hardware.** The scripts are unit-tested
on hand-built rows; no GPU benchmark run has been recorded. Run
[the benchmark](../../README.md#how-do-i-evaluate-one-model) to produce the
numbers.

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
are. And `train_sep_probe` writes `<probe>.train_ids.json` next to
the probe; `run_benchmark` refuses to start if any benchmark question id is
in it, or if a probe in use has no such file. A probe evaluated on the
questions it was fit to would report an optimistic AUROC.

### Why might the escalation uplift be overstated?

Correctness is a case-insensitive substring match against the accepted
answers, the usual TriviaQA convention. That favours longer answers: "It
could be Paris or Lyon" contains more candidates than "Lyon". If the judge
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
| `test_semantic_entropy.py` | clustering by bidirectional entailment, SE values, empty input |
| `test_semantic_entropy_probe.py` | SEP probe loading and forward pass, missing-file, missing-hidden-state and hidden-size-mismatch errors |
| `test_escalation.py` | `SingleDetector`, `AnyAbove`, `AllAbove` decisions and validation |
| `test_evaluator.py` | `evaluate_responses` / `ResponseEvaluator` dispatch, prerequisites, judge escalation |
| `test_model_registry.py` | registry entries, single local/deployed default, vLLM flags per runner (raw logprobs for generate, unnormalized last-token pooling for embed), tensor parallelism per GPU count, L4 context cap |
| `test_slm_provider.py` | vLLM provider against a mocked OpenAI client (sampling, chat-shaped embedding request, `VLLM_AUTH` modes), HF logprob extraction (raw distribution, first EOS kept, padding dropped), provider selection, sweep model-id validation |
| `test_google_auth.py` | `GoogleAccessToken` reuse and refresh with fake credentials, plus a contract test against the **real** OpenAI client (fake HTTP transport) proving the token is refreshed on chat and embedding requests |
| `test_serving_launch.py` | the container's `vllm serve` argv: default runner, Vertex port, fail-fast on missing model, unknown runner or unregistered model |
| `test_benchmark_pipeline.py` | `eval/run_benchmark.py` → `eval/compute_auroc.py` end to end; run config recorded per row; SEP probe leakage guard (overlap and missing provenance refused) |
| `test_eval_criteria.py` | statistics primitives (rank AUROC vs sklearn, deterministic bootstrap, exact McNemar); each criterion met only with its significance or effect-size condition; run comparability; cross-validated SEP Pearson r; slice bounds; probe provenance |
| `test_model_comparison.py` | `run_benchmark --sweep` (one file per model, one model at a time, registry and per-model SEP probe checked before any scoring); SEP's default probe follows the evaluated model; `compare_models` accuracy and AUROC per model, refusing different questions, different sampling or a repeated model |
| `test_gemini_judge.py` | `GeminiJudge` request shape and verdict parsing against a fake `google-genai` client |

## Running the suites

```bash
uv run pytest -q                                   # everything
uv run pytest tests/unit/test_semantic_entropy.py -v
uv run pytest -k escalation                        # by name
uv run pytest -x --lf                              # stop at first failure, rerun last failures
```

`pyproject.toml` puts both `src/` and the repo root on the import path, so
tests can import `hallucinations_math_lab`, `demo` and `eval` without
installing them. `tests/integration/` is reserved for tests that need real
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
   `examples/score_precollected_samples.py` (with and without
   `--nli deberta`) and `examples/custom_detector.py`.
5. With a model available: `python -m demo.cli detect "Who wrote 1984?"`
   returns SE near 0, and a 10-question `python -m eval.run_benchmark --limit 10 --no-judge`
   followed by `python -m eval.compute_auroc` completes. With two small models
   in `TARGET_MODEL_IDS`, the same with `--sweep` followed by
   `python -m eval.compare_models` prints one row per model.
6. Every command in the README and the usage guides still matches the code.

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
