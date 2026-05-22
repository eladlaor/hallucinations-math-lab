# Changelog — 2026-05-21

## Table of contents
- [Summary](#summary)
- [1. Strategic framing — FDE interview exemplar](#1-strategic-framing--fde-interview-exemplar)
- [2. Target-model sweep as a first-class config axis](#2-target-model-sweep-as-a-first-class-config-axis)
- [3. Model registry — replaced with Gemma 4 family](#3-model-registry--replaced-with-gemma-4-family)
- [4. SLMProvider — per-model construction](#4-slmprovider--per-model-construction)
- [5. Tests](#5-tests)
- [Verification](#verification)
- [Deferred / out of scope for this changeset](#deferred--out-of-scope-for-this-changeset)

## Summary

Reframed the project around the **Gemma 4 family** (released 2026-04-02) and
introduced a **target-model sweep** as a first-class configuration axis, so
the SE / predictive-entropy / judge pipeline runs once per candidate SLM and
results are directly comparable across models. Defaults updated to all four
Gemma 4 SKUs (E2B, E4B, 26B-A4B MoE, 31B dense). Also marked this repo as
the exemplar artifact for the Google Forward Deployed Engineer interview.

## 1. Strategic framing — FDE interview exemplar

**What.** Added a paragraph under "Strategic frame" in `.claude/CLAUDE.md`
linking to the FDE interview workspace at
`/home/elad.l/Code/fun_projects/joyb/google/forward_deployed_engineer` and
declaring this repo the exemplar artifact for that interview.

**Why.** Forces design decisions here to be optimized for what an FDE
panel would scrutinize (clarity of the SLM-evaluation thesis, SE vs
predictive-entropy distinction, defensible Vertex serving choices,
detector / judge / evaluation separation). Constraint kept explicit: this
repo must stay self-contained and runnable without the FDE workspace.

**When.** 2026-05-21.

## 2. Target-model sweep as a first-class config axis

**What.** In `src/seh_handler/config.py`:

- Added `target_model_ids: list[str]`. Default = all four Gemma 4 SKUs.
  Pydantic `field_validator` enforces:
  - at least 2 ids (sweep needs comparison),
  - uniqueness,
  - every id must exist in `services.model_registry`.
- Added `target_model_base_urls: dict[str, str]`. Optional per-model
  base-url override. Missing keys fall back to `vllm_base_url`. Lets the
  same harness drive either (a) one shared endpoint reconfigured between
  runs (sequential sweep, cheap) or (b) N parallel endpoints, one per
  model (expensive, true side-by-side). `model_validator` rejects stray
  keys not present in `target_model_ids`.
- `vllm_model_id` retained as the **active model for single-model
  commands** (`seh chat`), separated from the sweep axis. Default
  switched to `google/gemma-4-E2B-it`.

**Why.** "Models we are evaluating" and "model we're chatting with" are
two different concepts; conflating them into a single string was the
shape blocking a sweep. Splitting them keeps single-model commands working
unchanged while making the sweep first-class. The list-of-ids + sibling
dict shape (vs. a list of pydantic objects) was chosen because env-var
encoding stays trivial — JSON-in-env for nested objects is unfriendly to
author.

**When.** 2026-05-21.

## 3. Model registry — replaced with Gemma 4 family

**What.** In `src/seh_handler/services/model_registry.py`:

| HF id                          | Params               | Ctx   | Quant | dtype | GPU            |
|--------------------------------|----------------------|-------|-------|-------|----------------|
| `google/gemma-4-E2B-it`        | 2.3B eff / 5.1B emb  | 128K  | none  | bf16  | 1× L4          |
| `google/gemma-4-E4B-it`        | 4.5B eff / 8B emb    | 128K  | none  | bf16  | 1× L4          |
| `google/gemma-4-26B-A4B-it`    | 4B active / 26B total| 32K\* | none  | bf16  | 1× A100-80GB   |
| `google/gemma-4-31B-it`        | 31B dense            | 32K\* | none  | bf16  | 2× A100-80GB   |

\* Per the official vLLM recipe, the 26B/31B are launched with
`--max-model-len 32768`; native context is 256K but the recipe caps it
for serving sanity. E2B/E4B keep the native 128K.

- `is_deployed_default = True` now on **E4B** — keeps the `$0.70/hr`
  single-L4 cost story in `CLAUDE.md` honest.
- `is_local_default = True` now on **E2B** — small enough for laptop dev.
- Docstring updated to make explicit that the **sweep is orthogonal** to
  both default flags: every sweep id must be in the registry, but the
  sweep is not derived from the defaults.

**Why.** The previous registry pointed at Gemma 2 9B (deployed default)
and Llama-3.1-70B (local default). Llama-70B was never an SLM and
contradicted the project's framing. Gemma 4 is current-gen, native
multimodal, and gives a clean spread across the SLM size band (2B / 4B /
26B-MoE / 31B-dense) that produces a meaningful comparison.

**Notes on serving fidelity.** vLLM has a known **MXFP4/NVFP4 weight-load
bug** in the fused MoE layer for `26B-A4B`; we therefore keep that
variant unquantized on A100-80GB until upstream fixes ship. NVIDIA's
official NVFP4 checkpoint currently exists only for the 31B dense. The
31B dense official vLLM recipe assumes TP=2.

**When.** 2026-05-21.

## 4. SLMProvider — per-model construction

**What.** In `src/seh_handler/services/slm_provider.py`:

- `VLLMOpenAIProvider.__init__` now accepts optional `model_id` and
  `base_url`. Defaults preserved (`None` → fall back to
  `config.vllm_model_id` / `config.vllm_base_url`).
- `HFTransformersProvider.__init__` now accepts optional `model_id`.
- `build_default_provider(model_id=None)` looks up the per-model
  `base_url` override from `config.target_model_base_urls` when a
  `model_id` is passed.
- **New** `build_sweep_providers() -> list[SLMProvider]` constructs one
  provider per id in `config.target_model_ids`. This is the entry point
  the evaluation harness will use.
- Header docstring updated: removed stale Llama-3.1-70B reference,
  replaced with the four Gemma 4 SKUs.

**Why.** Both providers used to hard-bind to `settings.vllm_model_id` at
construction time, which made a sweep impossible without a singleton
hack. Making `model_id` an explicit constructor argument preserves all
single-model call sites (`DetectorAgent()` still works unchanged via
`main.py`'s `chat` and `detect` commands) while unblocking sweep
construction.

**When.** 2026-05-21.

## 5. Tests

**What.** In `tests/unit/test_model_registry.py`:

- Rewrote tests to validate the same **invariants** against the new
  registry contents:
  - unknown id raises with a helpful message,
  - E4B entry exposes the expected vLLM flags (bf16, no quantization,
    128K context),
  - 31B dense omits `--quantization`, runs bf16, and pairs with 2×
    A100-80GB,
  - `get_deployed_default()` returns a single-L4 Gemma 4 SKU (load-
    bearing for the cost story; will fail loudly if changed),
  - `get_local_default()` returns one of the laptop-runnable SKUs
    (E2B or E4B),
  - `list_model_ids()` returns all four Gemma 4 SKUs,
  - parametrized test: every unquantized entry must be bf16 native.
- Test file's docstring updated to reflect the new defaults.

**Why.** The previous tests pinned the old Gemma 2 9B / Llama-3.1-70B
strings and would have failed misleadingly. Re-grounded them on
invariants instead of specific id strings where possible.

**When.** 2026-05-21.

## Verification

- `uv run pytest tests/unit -q` → **40 passed, 1 skipped**.
- `uv run mypy src/seh_handler/config.py
  src/seh_handler/services/model_registry.py` → clean.
- `uv run ruff check` on edited files → clean.
- `Settings()` instantiates with the new defaults; the
  `target_model_ids` validator round-trips against the registry without
  import cycles.

The package-wide `mypy` baseline still emits 20 errors, all in pre-
existing lazy-load (`_ensure_loaded`-gated) `None`-attribute access in
`HFTransformersProvider`, `NLIScorer`, and `JudgeAgent`. None of them
were introduced by this changeset; addressing them is out of scope.

## Deferred / out of scope for this changeset

- **Judge model.** Still `gemini-2.5-flash` (online) / `gemini-2.5-pro`
  (offline eval). The earlier conversation considered switching the
  Judge to a self-served Llama-70B, but that conflicts with the "no
  third-party APIs; Judge uses Vertex Gemini" rule in `CLAUDE.md` and
  changes the cost profile materially. Decision pending.
- **`_resolve_sep_probe_path()` in `detector_agent.py`** still reads
  `settings.vllm_model_id` directly, so during a sweep every model's
  SEP probe would resolve to the same slug. Not blocking — SEP is not
  in the default `enabled_detectors`. Fix when SEP rejoins the sweep.
- **`DetectionResult.model_id`** is missing. Any sweep harness that
  writes results needs it to label rows. Trivial addition; deferred
  pending the harness design.
- **Sweep harness itself.** `build_sweep_providers()` is the entry
  point, but no harness yet iterates it and writes per-model results.
  Next step.
