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
- [Coding conventions](#coding-conventions)
- [Contribution workflow](#contribution-workflow)

## Summary

```bash
uv sync --all-extras && cp .env.example .env
uv run pytest -q && uv run ruff check .
uv run python examples/score_precollected_samples.py
```

If all three pass you have a working setup — no GPU, model or cloud account
needed. The library lives in `src/hallucinations_math_lab/` and must never
import `demo/`. New detectors implement the `Detector` protocol and are
picked up by the benchmark and AUROC scripts automatically.

## Local setup

1. Install [uv](https://docs.astral.sh/uv/) and clone the repo.
2. `uv sync --all-extras` — creates `.venv` with every optional dependency
   group (`demo`, `judge`, `eval`, `dev`). The test suite imports the demo
   provider, so the `dev` extra alone is not enough.
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
  constants.py             # DetectorName, DetectorCategory, AnswerSource, NLILabel
  config.py                # Pydantic Settings, read from env / .env
  custom_types/            # Sample, SampleSet, TokenLogprob, Cluster, DetectionResult
  services/
    detector_protocol.py   # Detector, DetectionContext, DetectorScore, DetectorMetadata
    predictive_entropy.py  # PE
    semantic_entropy.py    # SE + bidirectional-entailment clustering
    semantic_entropy_probe.py  # SEP + the SEPProbe linear module
    nli.py                 # DeBERTa-MNLI wrapper (lazy-loaded)
    sampling_provider.py   # SamplingProvider protocol
  judges/                  # Judge protocol + GeminiJudge
demo/
  cli.py                   # `python -m demo.cli detect|chat`
  slm_provider.py          # VLLMOpenAIProvider, HFTransformersProvider, sweep builder
  model_registry.py        # per-model serving settings (Gemma 4 family)
eval/                      # datasets, run_benchmark, compute_auroc, train_sep_probe
examples/                  # runnable playground scripts
tests/unit/                # see the QA Guide
serving/, terraform/       # see the DevOps Guide
```

## Why is the library separate from the demo?

So that someone who already has an LLM stack can `pip install` the scorers
without pulling in vLLM clients, Gemma settings or Google Cloud. The wheel
ships only `src/hallucinations_math_lab/`; `demo/`, `eval/` and `examples/`
exist only in the repository.

The rule that keeps this true: **nothing under `src/` imports `demo`** at
module import time. The single exception is `ResponseEvaluator` built with
no provider, which lazily falls back to `demo.slm_provider` — that path only
makes sense inside the repo.

## How does one evaluation flow through the code?

1. `ResponseEvaluator.evaluate(prompt)` asks the provider for N samples (if
   any detector `requires_samples`), a greedy answer, and a pooled hidden
   state (if any detector `requires_hidden_state`).
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
accelerator). Exactly one entry may be `is_local_default` and one
`is_deployed_default`; `tests/unit/test_model_registry.py` enforces this. A
model must be in the registry before it can appear in `VLLM_MODEL_ID` or
`TARGET_MODEL_IDS`.

## How do I plug in a different judge?

Any object with `answer(prompt: str) -> tuple[str, str]` (answer,
reasoning) satisfies the `Judge` protocol. Pass it as `judge=` to
`evaluate_responses` or `ResponseEvaluator`. `judges/gemini.py` is the
reference; it imports the Vertex SDK lazily so the library loads without it.

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
- Heavy dependencies (`transformers` models, Vertex SDK) load lazily.

## Contribution workflow

1. Write or update a failing test that captures the change.
2. Make it pass; keep `uv run ruff check .` clean.
3. Update the docs the change touches and add a line under `[Unreleased]`
   in `CHANGELOG.md` (Keep a Changelog format).
4. Open a pull request; CI must be green.
