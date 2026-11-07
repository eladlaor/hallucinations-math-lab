# Developer Guide

## Table of Contents

- [Summary](#summary)
- [Local setup](#local-setup)
- [How is the code organized?](#how-is-the-code-organized)
- [Why is the library separate from the demo?](#why-is-the-library-separate-from-the-demo)
- [How does one evaluation flow through the code?](#how-does-one-evaluation-flow-through-the-code)
- [How do I add a detector?](#how-do-i-add-a-detector)
- [How do I add a model to the registry?](#how-do-i-add-a-model-to-the-registry)
- [How do I plug in a different judge?](#how-do-i-plug-in-a-different-judge)
- [How do I work on the web playground?](#how-do-i-work-on-the-web-playground)
- [Coding conventions](#coding-conventions)
- [Contribution workflow](#contribution-workflow)

## Summary

```bash
uv sync --all-extras && cp .env.example .env
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run python examples/score_precollected_samples.py
```

If these pass you have a working setup — no GPU, model or cloud account
needed. The library lives in `src/hallucinations_math_lab/` and must never
import `demo/`. New detectors implement the `Detector` protocol and are
picked up by the benchmark, AUROC and comparison scripts automatically.
Using the project (models, demo, benchmark) is documented in the
[README](../../README.md).

## Local setup

1. Install [uv](https://docs.astral.sh/uv/) and clone the repo.
2. `uv sync --all-extras` — creates `.venv` with every optional dependency
   group. The test suite imports the demo provider, the judge and the deploy
   script, so the `dev` extra alone is not enough:

   | Extra | For |
   |---|---|
   | `demo` | the vLLM / OpenAI client, Google access tokens, the demo CLI |
   | `judge` | `GeminiJudge` (`google-genai`) |
   | `eval` | the benchmark and scoring scripts (`datasets`, `scikit-learn`, `scipy`) |
   | `deploy` | `serving/vertex_deploy.py` (`google-cloud-aiplatform`), run by Terraform |
   | `web` | the web playground server (`fastapi`, `uvicorn`) |
   | `dev` | `pytest`, `ruff`, `mypy` |
3. `cp .env.example .env` — every setting is documented inline. List-valued
   settings are JSON (`ENABLED_DETECTORS=["predictive_entropy"]`).
4. `uv run pytest -q` — about a minute on a laptop CPU.

Python 3.12 or 3.13. Use `uv add` / `uv lock` to change dependencies, never
`pip`; CI installs with `--locked`.

## How is the code organized?

```
src/hallucinations_math_lab/
  __init__.py              # public surface — everything a consumer imports
  evaluator.py             # evaluate_responses (pure scoring) + ResponseEvaluator (drives a provider)
  escalation.py            # EscalationPolicy protocol; SingleDetector, AnyAbove, AllAbove
  constants.py             # DetectorName, DetectorCategory, AnswerSource, NLILabel, VLLMAuthMode
  config.py                # Pydantic Settings, read from env / .env
  custom_types/
    domain_types.py        # Sample, SampleSet, TokenLogprob, Cluster, DetectionResult
    llm_schemas.py         # JudgeVerdict (the judge's structured output)
  services/
    detector_protocol.py   # Detector, DetectionContext, DetectorScore, DetectorMetadata
    predictive_entropy.py  # PE
    semantic_entropy.py    # SE + bidirectional-entailment clustering
    semantic_entropy_probe.py  # SEP + the SEPProbe linear module
    nli.py                 # EntailmentScorer protocol + DeBERTa-MNLI NLIScorer (lazy-loaded)
    sampling_provider.py   # SamplingProvider protocol
  judges/                  # Judge protocol + GeminiJudge (google-genai)
demo/
  cli.py                   # `python -m demo.cli detect|chat`
  slm_provider.py          # VLLMOpenAIProvider, HFTransformersProvider, sweep builder
  google_auth.py           # refreshing Google access token for VLLM_AUTH=google
  model_registry.py        # per-model serving settings + vLLM flags per runner (Gemma 4 family)
  web/                     # `python -m demo.web`: FastAPI app (app.py), presets, results reports,
                           # and static/ — the committed UI build
frontend/                  # the playground UI source: Vite, React, TypeScript, Tailwind, shadcn/ui
eval/
  datasets/                # TriviaQA loader, adversarial prompts
  run_benchmark.py         # questions -> JSONL of scores and answers; --sweep: one file per model
  compare_models.py        # several models' runs side by side: accuracy + AUROC per detector
  benchmark_results.py     # shared row schema, loading, correctness rule
  compute_auroc.py         # criteria 1-2: per-detector AUROC, SE-PE gap
  escalation_uplift.py     # criterion 3: accuracy gained by escalating
  adversarial_separation.py  # criterion 4: Mann-Whitney U, adversarial vs correct
  success_criteria.py      # the pass bars, as constants
  train_sep_probe.py       # fit an SEP probe, report held-out Pearson r vs SE
examples/                  # runnable playground scripts
serving/
  Dockerfile               # vllm/vllm-openai:v0.31.0-cu129 + registry + launcher
  launch.py                # container entrypoint: registry -> `vllm serve` argv
  model_flags.py           # registry CLI: paste-able `vllm serve`, Terraform tfvars
  vertex_deploy.py         # upload + deploy to Vertex with the invoke route
tests/unit/                # see the QA Guide
terraform/                 # see the DevOps Guide
```

## Why is the library separate from the demo?

So that someone who already has an LLM stack can install the scorers
without pulling in vLLM clients, Gemma settings or Google Cloud. The package
is not on PyPI; install it from Git
(`uv add git+https://github.com/eladlaor/hallucinations-math-lab`). The wheel
ships only `src/hallucinations_math_lab/`; `demo/`, `eval/` and `examples/`
exist only in the repository.

The rule that keeps this true: **nothing under `src/` imports `demo`** at
module import time. The single exception is `ResponseEvaluator` built with
no provider, which lazily falls back to `demo.slm_provider` — that path only
makes sense inside the repo.

## How does one evaluation flow through the code?

1. `ResponseEvaluator.evaluate(prompt)` asks the provider for N samples (if
   any detector `requires_samples`), a greedy answer, and a pooled hidden
   state — the last layer's vector at the final prompt token — (if any
   detector `requires_hidden_state`).
2. It calls `evaluate_responses(...)`, which validates that every detector's
   inputs are present (fail fast), builds one `DetectionContext`, and calls
   each detector's `score(ctx)`.
3. The escalation policy reads the score dictionary; if it fires and a judge
   is set, `judge.answer(prompt)` replaces the final answer.
4. Everything lands in a `DetectionResult`.

## How do I add a detector?

1. Add an id to `DetectorName` in `constants.py`.
2. Create `services/<your_detector>.py` with a class that has `name`,
   `metadata` (signal, assumption, failure modes, category),
   `requires_samples`, `requires_hidden_state`, and
   `score(ctx) -> DetectorScore`. Higher score must mean "more likely
   hallucinating" — invert confidence-style signals. Fill `cost_usd` and
   `latency_ms` honestly; they feed the planned cost/accuracy comparison.
3. Register it in `_build_default_detectors` in `evaluator.py` so it can be
   turned on with `ENABLED_DETECTORS`.
4. Export it from `__init__.py` if consumers should import it.
5. Write the unit test first (see `tests/unit/test_semantic_entropy.py`):
   known inputs → known score, plus the missing-prerequisite error.

`examples/custom_detector.py` is a working sketch of steps 2–3 without
touching the library.

## How do I add a model to the registry?

Add a `ModelEntry` to `_REGISTRY` in `demo/model_registry.py` with its vLLM
settings (dtype, max length, quantization) and hardware (machine type,
accelerator type and count). `vllm_flags(runner)` turns the entry into
`vllm serve` flags for the `generate` or `embed` runner, including
`--tensor-parallel-size` equal to the accelerator count. Size
`max_model_len` to what the GPU's memory can hold next to the weights, not to
the model's maximum context, or vLLM refuses to start. Pin the checkpoint
too: set `revision` to the repo's current full 40-hex commit sha from the Hub
API (not a branch or tag), and copy `num_hidden_layers` and `hidden_size`
from `text_config` in that commit's `config.json`. Both providers load exactly
that commit, and the HF provider refuses a checkpoint whose depth or width
disagrees with the entry. Exactly one entry may
be `is_local_default` and one `is_deployed_default`; `tests/unit/test_model_registry.py` enforces this. A
model must be in the registry before it can appear in `VLLM_MODEL_ID` or
`TARGET_MODEL_IDS`. Its SEP probe and its sweep results file are named by
`model_slug(model_id)` (`services/semantic_entropy_probe.py`).

## How do I plug in a different judge?

Any object with `answer(prompt: str) -> tuple[str, str]` (answer,
reasoning) satisfies the `Judge` protocol. Pass it as `judge=` to
`evaluate_responses` or `ResponseEvaluator`. `judges/gemini.py` is the
reference; it calls Gemini on Vertex AI through `google-genai`, asks for JSON
validated against `JudgeVerdict`, and imports the SDK lazily so the library
loads without it.

## How do I work on the web playground?

The playground is two halves. `demo/web/` is a thin FastAPI layer: every
endpoint calls existing code (`evaluate_responses`, `ResponseEvaluator`,
`auroc_report`, `comparison_report`) and adds no scoring logic of its own.
`create_app(...)` takes the NLI scorer, the provider and judge factories, the
results directory and the settings as arguments, so the tests in
`tests/unit/web/` pass fakes. `frontend/` is the UI; its production build is
committed to `demo/web/static/` so users never need Node.

The dev loop needs Node 22.12+ and two terminals:

```bash
uv run python -m demo.web                 # terminal 1: the API on 127.0.0.1:8765
cd frontend && npm ci && npm run dev      # terminal 2: the UI with hot reload; /api is proxied to 8765
```

Before committing a UI change, run the checks and rebuild — the build is part
of the commit:

```bash
cd frontend
npm run typecheck && npm run lint && npm test   # tsc --noEmit, eslint, vitest run
npm run build                                   # writes ../demo/web/static/
```

UI conventions: every API call goes through `src/lib/api.ts`, which logs
start, success and failure with the structured `logger` (`src/lib/logger.ts`)
and throws the backend's error message unchanged; components log mount,
unmount and user actions with context; route paths, modes and limits live in
`src/lib/constants.ts`. The Score tab's presets come from
`examples/score_precollected_samples.py`, imported at runtime, so edit the
example, not the UI, to change them.

## Coding conventions

- `src/` layout, `uv` for everything, Python 3.12+ idioms.
- Immutable values and enums in `constants.py` (`StrEnum` for closed string
  sets); anything configurable in `config.py`. No inline magic strings in
  logic.
- Fail fast: validate inputs at the boundary, raise with context (ids,
  parameters, the function's intent), never swallow an exception.
- Log with structured context: `logger.info("...", extra={LOG_EXTRA_KEY: {...}})`.
- One responsibility per module; name modules after their domain, no
  `utils.py`.
- Heavy dependencies (`transformers` models, Google SDKs) load lazily.
- Type hints everywhere; `mypy` runs in CI with `check_untyped_defs`.
  Protocol members that implementations only expose (never assign) are
  read-only properties, so frozen dataclasses satisfy them.

## Contribution workflow

1. Write or update a failing test that captures the change.
2. Make it pass, then run the same gates as CI:
   ```bash
   uv run pytest -q
   uv run ruff check . && uv run ruff format --check .
   uv run mypy
   terraform -chdir=terraform fmt -check -recursive   # if you touched terraform/
   terraform -chdir=terraform init -backend=false && terraform -chdir=terraform validate
   ```
3. Update the docs the change touches and add a line under `[Unreleased]`
   in `CHANGELOG.md` (Keep a Changelog format).
4. Open a pull request; CI must be green.
