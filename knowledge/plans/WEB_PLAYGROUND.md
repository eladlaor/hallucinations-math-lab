# Plan: Local Web Playground

## Table of contents

- [Summary](#summary)
- [Why](#why)
- [Scope: three tabs, one per way in](#scope-three-tabs-one-per-way-in)
- [Architecture](#architecture)
- [API](#api)
- [Design direction](#design-direction)
- [Work items (TDD)](#work-items-tdd)
- [Success criteria](#success-criteria)
- [Out of scope](#out-of-scope)
- [Risks and decisions](#risks-and-decisions)

## Summary

- **What:** a minimal, elegant web UI on `localhost` that covers the README's
  three ways in, so a user can try the project without the command line.
- **One command, no Node for users:** `uv run python -m demo.web` serves the
  UI and its API from one local Python process, bound to `127.0.0.1` only. The
  built frontend is committed, so only frontend *developers* need Node.
- **Stack:** FastAPI backend (thin layer over existing library and eval
  functions — no logic duplicated) + Vite, React, TypeScript, Tailwind,
  shadcn/ui frontend.
- **Tabs:** named and ordered as the README's three ways in, cheapest first:
  *Score answers* (way 1, works on any laptop), *Ask the model* (way 2, needs
  a model), *Measure the detectors* (way 3, read-only viewer of existing
  results files).
- **Done when:** the success criteria below pass, including the "paraphrased"
  preset reproducing the example script's numbers exactly.

## Why

Every way in today starts with a CLI command and ends with a printed table.
The core idea — answers that *mean* the same thing form one cluster, and SE is
the spread across clusters — is visual. Seeing ten answers grouped by colour,
and a threshold slider flipping "escalate", teaches it faster than any table.

## Scope: three tabs, one per way in

### 1. Score answers (way 1 — tier 0, no model needed)

- A question field and an editable list of answers (add, remove, edit).
- Presets: the three scenarios from `examples/score_precollected_samples.py`
  (`consistent`, `paraphrased`, `divergent`), loaded from the backend — the
  example's `SCENARIOS` is the single source.
- On *Score*: SE, the meaning clusters, and PE.
  - **Clusters are the centrepiece:** each answer shown as a chip coloured by
    its cluster, clusters grouped visually, with each cluster's size.
  - **PE needs token logprobs.** Presets carry them. Typed answers do not, so
    PE shows "needs token logprobs" with a one-line explanation instead of a
    number — never a fabricated value.
- An escalation threshold slider (default `SE_THRESHOLD` from settings).
  "Escalate yes/no" recomputes in the browser as the slider moves (it is just
  `SE > threshold`), with no new request.
- First request loads DeBERTa (~1.6 GB, first time only): the UI says so while
  it waits.

### 2. Ask the model (way 2 — tier 1+ or a remote server)

- A question field; *Detect* (scores only) and *Chat* (scores + judge on
  escalation), mirroring `demo.cli detect` / `chat`.
- Shows the greedy answer, the N samples grouped into clusters (same component
  as tab 1), the detector scores, escalated yes/no, the answer source, and the
  judge's reasoning when it answered. PE and SE are shown in nats; SEP, when
  enabled, as a percentage, "P(high SE)", since its score is a probability.
  The threshold line uses the run's own escalation threshold, so a
  SEP-escalated run reads its threshold as a percentage too.
- A status strip from `GET /api/status`: provider, model id, whether Google
  credentials are configured (for *Chat*). Missing prerequisites disable the
  action with the reason shown, linking to the README's hardware section.
- Generation can take minutes on a CPU: a clear in-progress state with elapsed
  time; errors from the backend shown verbatim (fail-fast, no silent retry).

### 3. Measure the detectors (way 3 — read-only)

- Lists results files under `eval/results/` (single runs and `--sweep`
  directories).
- A single run: per-detector AUROC with its 95% CI as an interval plot,
  plus the run config and question counts — from `auroc_report`.
- A sweep directory: the model-comparison table from `comparison_report`.
- Empty state: shows the command to produce results
  (`uv run python -m eval.run_benchmark --out eval/results/triviaqa_200.jsonl --no-judge`).
- Running a benchmark from the UI is out of scope (long jobs).

## Architecture

```
demo/web/
  __init__.py
  __main__.py        # `python -m demo.web`: starts uvicorn on 127.0.0.1
  app.py             # create_app(...) factory; dependencies injected (testable)
  api_schemas.py     # Pydantic request/response models
  constants.py       # routes, default port, results dir — no inline literals
  static/            # committed Vite build output, served at /
frontend/            # Vite + React + TS source (dev only)
  src/lib/logger.ts  # structured JSON logger (debug/info/warn/error)
  src/lib/api.ts     # typed API client; logs start/success/failure per call
```

- `create_app(nli_scorer, provider_factory, judge_factory, results_dir)`: tests
  pass fakes; `__main__` passes the real ones. DeBERTa and the model load
  lazily, once, on first use.
- New `web` optional extra in `pyproject.toml` (`fastapi`, `uvicorn`), added to
  `all`.
- Backend follows the repo's Python standards: try/except with context,
  JSON-structured logs via `LOG_EXTRA_KEY`, `StrEnum` / constants, fail fast.
- Frontend follows the global frontend-logging rule: mount/unmount, every API
  call, and user actions logged with structured context.

## API

| Method & path | Body / params | Returns | Backed by |
|---|---|---|---|
| `GET /api/status` | — | provider, model id, judge configured, SE threshold (nats), SEP threshold (`sep_threshold`, p(high SE)), the escalation threshold `/ask` applies, NLI model id | `get_settings()` |
| `GET /api/presets` | — | the three scenarios (question, answers, logprobs) | `examples/score_precollected_samples.SCENARIOS` |
| `POST /api/score` | question, answers (+ optional per-answer token logprobs) | SE, PE or null, clusters (member indices + representative) | `evaluate_responses` with `NLIScorer` |
| `POST /api/ask` | question, `mode: detect \| chat` | the `DetectionResult` fields the CLI prints | the same `ResponseEvaluator` path as `demo/cli.py` |
| `GET /api/results` | — | results files and sweep dirs under `eval/results/` | filesystem |
| `GET /api/results/report` | `path` (relative) | AUROC report, or comparison report for a dir | `auroc_report`, `comparison_report` |

`path` is resolved and must stay inside `eval/results/`; anything else is a
400. The server binds `127.0.0.1` only.

## Design direction

- Minimal and calm: one column of content, generous spacing, system font
  stack, light and dark mode, no decorative chrome.
- Plain-language labels; every score has a one-line "what this means" note
  (e.g. SE: "how many different meanings the answers spread across").
- Cluster colours from one accessible categorical palette; colour is never the
  only signal (cluster number on each chip).
- Keyboard-navigable, visible focus, ARIA on the slider and tabs.

## Work items (TDD)

Each item: write the failing test first, then the code.

1. **Backend skeleton** — `create_app`, `/api/status`, `/api/presets`, static
   mount, `web` extra. Tests: FastAPI `TestClient`; presets equal `SCENARIOS`.
2. **`/api/score`** — tests with a fake NLI scorer: clusters and SE for the
   three presets; PE null when logprobs absent; 422 on empty answers.
3. **`/api/ask`** — tests with a fake provider and judge: detect vs chat,
   escalation, error surfaced with message when the provider fails.
4. **Results endpoints** — tests with fixture JSONL files: listing, AUROC
   report equals `auroc_report`, sweep dir equals `comparison_report`, path
   traversal (`../`) rejected with 400.
5. **Frontend shell** — Vite/React/TS/Tailwind/shadcn, logger, API client,
   tabs. Vitest + React Testing Library.
6. **Score tab** — cluster chips, threshold slider (escalate recomputed
   client-side), presets, PE "needs logprobs" state. Component tests.
7. **Ask tab** — status strip, disabled-with-reason states, progress, result
   view reusing the cluster component. Component tests with a mocked API.
8. **Measure the detectors tab** — list, AUROC interval plot, comparison table, empty state.
9. **Build & serve** — `npm run build` writes `demo/web/static/`; committed.
10. **Docs** — README: a "Web playground" entry in the ways-in table's
    *Start with* column and a short "How do I use the web playground?"
    section; DEVELOPER_GUIDE (frontend dev loop), QA_GUIDE (new suites),
    CHANGELOG `[Unreleased]` → Added.

## Success criteria

1. On a clean clone, `uv sync --all-extras && uv run python -m demo.web` serves
   the UI at `http://127.0.0.1:8765` with **no Node installed**.
2. Score tab, *paraphrased* preset, real DeBERTa: SE `0.000`, 1 cluster, PE
   `0.600`; *divergent*: SE `1.834`, 7 clusters — identical to
   `examples/score_precollected_samples.py`.
3. Typed answers without logprobs score SE; PE shows the "needs token
   logprobs" state.
4. Moving the threshold slider flips *escalate* without a network request.
5. The Measure the detectors tab shows the same AUROC and CI as
   `eval.compute_auroc` for the same file.
6. `GET /api/results/report?path=../../etc/passwd` returns 400.
7. Ask tab with no usable model shows the backend's error message; it does not
   hang silently.
8. `uv run pytest`, `ruff check`, `ruff format --check`, `mypy`, and the
   frontend's `tsc --noEmit`, `eslint`, `vitest run` all pass.

## Out of scope

- Running benchmarks or training SEP probes from the UI.
- Any deployment beyond localhost; authentication.
- Streaming tokens as they generate.
- Windows support (Linux and macOS only, matching the README).

## Risks and decisions

- **Committed build output** keeps users Node-free at the cost of a generated
  folder in git. Mitigation: one build command, and a test that the served
  `index.html` exists.
- **Importing `SCENARIOS` from `examples/`**: the backend imports it rather
  than copying it, so the presets and the example never drift.
- **Long CPU generations** on the Ask tab: a plain request with a generous
  timeout and a visible elapsed timer; streaming is deferred.
