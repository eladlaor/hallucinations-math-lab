# QA Guide

## Table of Contents

- [Summary](#summary)
- [What does success look like for the detectors?](#what-does-success-look-like-for-the-detectors)
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
`eval/run_benchmark.py` output. Each script prints its value, the target and
whether it was met. The pass bars live in `eval/success_criteria.py`.

| # | Criterion | Target | Measured by |
|---|---|---|---|
| 1 | SE separates hallucinated from correct TriviaQA answers | SE AUROC ≥ 0.75 on TriviaQA-1000 | `python -m eval.compute_auroc --in <triviaqa results>` |
| 2 | Clustering by meaning beats token-level uncertainty | SE AUROC − PE AUROC ≥ 0.05 | same command, `criteria.se_minus_pe_auroc` |
| 3 | Escalating to the judge pays off | final accuracy − greedy accuracy ≥ 5 percentage points | `python -m eval.escalation_uplift --in <triviaqa results>` (benchmark run with the judge) |
| 4 | SE flags adversarial prompts | adversarial SE > SE on correctly answered TriviaQA, one-sided Mann-Whitney U, p < 0.01 | `python -m eval.adversarial_separation --adversarial <adversarial results> --reference <triviaqa results>` |

A fifth number is a diagnostic rather than a pass bar: how closely the
Semantic Entropy Probe tracks SE on held-out prompts (Pearson r), logged by
`python -m eval.train_sep_probe` before it saves the probe.

**Status: not yet measured on real hardware.** The scripts are unit-tested
on hand-built rows; no GPU benchmark run has been recorded. Run
[the benchmark](USER_GUIDE.md#how-do-i-run-a-benchmark) to produce the
numbers.

**Why AUROC and not accuracy at a threshold?** AUROC scores the *ranking*
across every possible threshold, so it measures the detector itself; the
threshold is a separate, deployment-specific choice (see the
[User Guide](USER_GUIDE.md#how-do-i-tune-the-threshold)).

**Why Mann-Whitney for criterion 4?** It compares ranks, so it assumes
nothing about the score distribution. Entropies are bounded, skewed, and
often tied at 0, which rules out a t-test. Note that the committed
adversarial set has 7 prompts, so p < 0.01 needs nearly perfect separation;
add prompts to `eval/datasets/adversarial.jsonl` for a more sensitive test.

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
| `test_slm_provider.py` | vLLM provider against a mocked OpenAI client (sampling, chat-shaped embedding request, `VLLM_AUTH` modes), HF logprob extraction (raw distribution, first EOS kept, padding dropped), provider selection, sweep validation |
| `test_google_auth.py` | `GoogleAccessToken` reuse and refresh with fake credentials, plus a contract test against the **real** OpenAI client (fake HTTP transport) proving the token is refreshed on chat and embedding requests |
| `test_serving_launch.py` | the container's `vllm serve` argv: default runner, Vertex port, fail-fast on missing model, unknown runner or unregistered model |
| `test_benchmark_pipeline.py` | `eval/run_benchmark.py` → `eval/compute_auroc.py` end to end |
| `test_eval_criteria.py` | success-criteria scripts: AUROC gap, escalation uplift, Mann-Whitney separation, held-out SEP Pearson r, and their fail-fast cases |
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
  and TriviaQA subsets generated by `eval/datasets/load_triviaqa.py`.
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
   followed by `python -m eval.compute_auroc` completes.
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
