# Session Handoff: Start Here

> Rewritten **2026-10-17 11:33 IDT** by session **`unnamed session`**.

## Table of contents

- [Summary](#summary)
- [First three actions](#first-three-actions)
- [Where everything is](#where-everything-is)
- [State as of 2026-10-17](#state-as-of-2026-10-17)
- [Pre-publication review: what must be fixed](#pre-publication-review-what-must-be-fixed)
- [Decisions already made](#decisions-already-made)
- [Open decisions](#open-decisions)
- [Traps that will bite](#traps-that-will-bite)
- [How to update this file](#how-to-update-this-file)

## Summary

`hallucinations-math-lab` is a library plus an experimentation playground for
hallucination detectors that work from the model's own outputs, with no LLM judge in
the scoring loop. It ships three detectors: predictive entropy (PE), semantic entropy
(SE), and Semantic Entropy Probes (SEP). An LLM judge (Gemini) is used only for
escalation. Small models (Gemma 4) run on vLLM, locally or on Vertex AI.

The repo is **private** on GitHub (`eladlaor/hallucinations-math-lab`) and is being
prepared for public release. Code, docs, CI, and the serving and Terraform paths are
built, and CI is green. Two things are missing:

- **No experiment has run on real hardware.** There is no GPU run, no Vertex deploy,
  and no benchmark result.
- **A pre-publication review on 2026-10-17 blocks release.** It found 3 blockers and
  about a dozen major issues, listed
  [below](#pre-publication-review-what-must-be-fixed). That list is the current work
  queue.

## First three actions

1. **Fix the method blockers (B2 below), test-first.** These are the brief-answer
   prompt, normalized correctness matching, NLI conditioned on the question, and the
   discrete-SE and SEP wording.
   - Write a failing test for each before changing code.
   - Run from the repo root:
     `uv sync --locked --all-extras && uv run pytest -q && uv run mypy && uvx ruff@0.17.0 check . && uvx ruff@0.17.0 format --check .`
2. **Fix the registry blocker (B3).** In `demo/model_registry.py`, the
   `google/gemma-4-26B-A4B-it` entry uses `machine_type="a2-highgpu-1g"`. Change it to
   `a2-ultragpu-1g`. Add a unit test that checks each accelerator type against its
   valid machine families.
3. **Get the owner's decision on history cleanup (B1)**, and do nothing destructive
   before that.
   - The options are `git filter-repo` plus a force-push, or a fresh single-commit
     repo. Either way the remote `v0.1.0` tag and the remote `claude/load-bearing-words-…`
     branch must go.
   - This is blocked on the owner's explicit approval, because it is a history rewrite.

## Where everything is

| Path | What lives there |
|---|---|
| `src/hallucinations_math_lab/` | The library (the only package in the wheel). Detectors live in `services/`. Settings are in `config.py`, constants in `constants.py`, and the public API in `__init__.py`. |
| `demo/` | Not shipped. It holds the SLM providers (`slm_provider.py`: vLLM and HF), `google_auth.py`, `model_registry.py` (**source of truth** for per-model serving and hardware), and `cli.py`. |
| `eval/` | The benchmark pipeline. `run_benchmark.py` feeds four scoring scripts: `compute_auroc.py`, `escalation_uplift.py`, `adversarial_separation.py` and `train_sep_probe.py`. The statistics live in `inference_stats.py`, the criteria thresholds in `success_criteria.py`, and the dataset loaders in `datasets/`. |
| `serving/` | The vLLM image (`Dockerfile`, `launch.py` entrypoint), `vertex_deploy.py` (upload, deploy, undeploy) and `model_flags.py`. |
| `terraform/` | GCP resources. Vertex endpoints are gated by `deploy_vertex_endpoint` (default `false`). State is local; the GCS backend is commented out in `backend.tf`. |
| `examples/` | No-GPU playground scripts. `score_precollected_samples.py` is the 5-minute quickstart. |
| `knowledge/usage_guides/` | USER, DEVELOPER, DEVOPS and QA guides. **The QA guide is the reference for the success criteria.** |
| `knowledge/setup/` | `VERTEX_DEPLOY.md`, `LOCAL_DEV.md`. |
| `CHANGELOG.md` | The only release-notes file (Keep a Changelog). Breaking changes are listed under `[Unreleased]`. |
| `.github/workflows/ci.yml` | Five CI jobs: lint, terraform, typecheck, and tests on Python 3.12 and 3.13. |

## State as of 2026-10-17

| Fact | Value |
|---|---|
| Repo position | Run `git log --oneline -5`. The last code commit is "Make the success criteria statistically sound". |
| Branch model | Single-owner repo, so commits land on `main` directly. |
| GitHub visibility | **PRIVATE.** Do not make it public until the blockers are fixed and the owner says so. |
| CI on the last code commit | All five jobs green. |
| Local suite (fresh `uv sync --locked --all-extras`) | 114 passed, 1 skipped (`requires_gpu`). Coverage is about 70%. |
| Packaging | `pyproject` version `0.1.0`, `requires-python >=3.12,<3.14`. The wheel and sdist are clean, with no private files. |
| Remote refs | `main`, the tag `v0.1.0` (points at an old pre-rename commit), and an unmerged branch `claude/load-bearing-words-hallucinations-v0a34d`. |
| Serving image base | `vllm/vllm-openai:v0.31.0-cu129`, pinned by tag only. |
| Deployed or costing money | Nothing. No GCP resources were applied from this repo in these sessions. |
| PyPI | Not published under any name. |

## Pre-publication review: what must be fixed

The review ran on 2026-10-17. Four independent reviewers covered the ML and
statistics method, code correctness and packaging, infrastructure and CI, and docs,
hygiene and history. The blockers were then re-checked against the code.

### Blockers

**B1. Git history would leak private material.**
- **What is in history:**
  - Files that are now gitignored but were committed in older commits: `.claude/`
    (agent notes), `knowledge/lectures/` (talk scripts), `knowledge/submissions/`
    (a conference submission, which includes the organizers' email).
  - The original GCP project id, in old `terraform/backend.tf`, `terraform.tfvars.example`
    and the DEVOPS guide.
  - `Co-Authored-By` and `Claude-Session` trailers in commit messages, which the
    owner's convention forbids.
  - Inconsistent author names.
- **Remote refs to remove:**
  - The tag `v0.1.0` points at a commit that contains private files.
  - The branch `claude/load-bearing-words-…` lists private claude.ai artifact URLs.
- **Secrets:** none found. Only regex scans were run; gitleaks and trufflehog are not
  installed.
- **Fix:** rewrite history (`git filter-repo --path .claude --path knowledge/lectures
  --path knowledge/submissions --invert-paths`, plus scrubbing the project id and the
  trailers), or publish a fresh single-commit repo. Delete the remote tag and branch.
  This needs owner approval.

**B2. The evaluation method has flaws an expert would reject.**
- **No brief-answer prompt.** The raw TriviaQA question goes to chat-tuned Gemma
  (`demo/slm_provider.py`), and `MAX_TOKENS=64` cuts off the long answers. This breaks
  SE (NLI over cut-off text), PE (dominated by style tokens) and the correctness labels.
  - Fix: add a fixed answer-format instruction, record it in the run config, and
    report the answer-length distribution.
- **Correctness is a raw substring match** (`eval/benchmark_results.py::is_correct`).
  - Examples: `"It was 1921."` matches alias `"2"`; `"Paris Hilton"` matches `"Paris"`;
    `"the U.K."` misses `"UK"`. The loader uses `aliases`, not `normalized_aliases`.
  - Fix: SQuAD-style normalization with whole-token matching against
    `normalized_aliases`, or an LLM correctness judge. Hand-check about 100 labels and
    report the error rate.
- **NLI never sees the question** (`services/semantic_entropy.py`, `services/nli.py`).
  The reference implementation prepends the question.
  - Fix: score `f"{question} {answer}"` pairs. Read the label order from
    `model.config.id2label`, not the hardcoded `DEBERTA_MNLI_LABEL_ORDER`.
- **It is discrete SE, but the docs claim the Nature paper's estimator.**
  `p(c)=|c|/N` ignores sequence likelihoods.
  - Fix: call it "discrete SE" everywhere, and add probability-weighted SE from the
    logprobs already collected.
- **The SEP probe does not match Kossen et al., but claims to.**
  - The repo uses unstandardized Ridge (alpha=1.0, untuned) on raw SE from the last
    layer, reported as Pearson r. The paper trains a logistic classifier on binarized
    SE with a layer sweep and reports AUROC.
  - The "same τ works for SEP" claim is wrong, because ridge pulls predictions toward
    the mean.
  - Fix: implement the paper's recipe or drop the claim. Add a layer sweep and CV
    AUROC, and give SEP its own calibrated threshold.
- **τ=1.0 is called "calibrated" in the walkthrough, but nothing was measured.** The
  success bars in `eval/success_criteria.py` cite no sources.
  - Fix: label τ a placeholder, calibrate it on a disjoint dev slice, and justify each
    bar.
- **The escalation-uplift criterion doesn't test the detector.** It compares Gemini
  against the small model, which a stronger model wins at any escalation rate.
  - Fix: compare against random escalation and PE-triggered escalation at the same
    rate, report the accuracy-vs-rate curve, and control for answer length.
- **The adversarial criterion is weak.**
  - It uses 7 prompts (`eval/datasets/adversarial.jsonl`), and the comparison group is
    chosen by outcome.
  - A model that refuses consistently gets low SE, so the criterion rewards the wrong
    behavior.
  - Fix: use a few hundred items from an established set (e.g. SelfAware or FalseQA),
    with labeled refusals.

**B3. The 26B model cannot deploy.** `demo/model_registry.py` pairs `NVIDIA_A100_80GB`
with `a2-highgpu-1g`, which only carries A100 40GB.
- Fix: `a2-ultragpu-1g`, plus a registry test.

### Major

- **Library settings leak into consumers.**
  - `config.py` has no `env_prefix`. A consumer's `.env` with `MAX_TOKENS=4096` makes
    `NLIScorer()` raise `ValidationError`.
  - Fix: `env_prefix="HML_"`, and move the demo and sweep fields into a separate demo
    settings class.
- **`ResponseEvaluator()` with no `slm` imports `demo`**, which is not in the wheel,
  so it raises `ModuleNotFoundError`. Fix: make `slm` required.
- **The extras install dependencies for unshipped code.** `demo`, `eval`, `deploy` and
  `dev` are package extras. Fix: move them to `[dependency-groups]`.
- **Errors are flattened.**
  - `ResponseEvaluator.evaluate` wraps everything as `RuntimeError`, so the documented
    `DetectorPrerequisiteError` can't be caught. A judge error is wrapped twice.
  - Fix: `e.add_note(...)` and re-raise the original.
- **The vLLM client has no timeouts and retries blindly.**
  - The OpenAI default 600 s timeout × 2 SDK retries × 3 tenacity attempts adds up to
    about 9 sends and about 90 minutes worst case. 401 and 400 errors are retried too,
    while `greedy` and `pooled_hidden_state` have no retry at all.
  - Fix: pass `timeout` and `max_retries` from config, and retry only connection, rate
    limit and 5xx errors.
- **Silent fallbacks break fail-fast.**
  - PE skips samples that have no logprobs (`predictive_entropy.py`), and NaN logprobs
    pass through, so escalation silently becomes `False`.
  - Fix: raise on a partial sample set or non-finite values.
- **Sampling parameters are unverified.** Gemma's `generation_config` top-k and top-p
  defaults may apply in both HF and vLLM, which biases PE and SE. Fix: set them
  explicitly and log the effective values.
- **Cold start.**
  - Each replica downloads 16–62 GB from Hugging Face anonymously
    (`HF_HOME=/models` is ephemeral, and there is no `HF_TOKEN`).
  - There is no `serving_container_deployment_timeout` and no startup probe, so a
    restart loop is possible.
  - Fix: stage the weights in GCS (`artifact_uri`), add a startup probe, set the
    deployment timeout to 7200, and pass an optional `HF_TOKEN` from Secret Manager.
- **Teardown gaps.**
  - An interrupted `terraform apply` after `model.deploy` leaves a GPU deployed that
    Terraform doesn't track.
  - Fix: a teardown script that lists and undeploys models by endpoint, plus a
    documented `gcloud ai endpoints undeploy-model` recovery step.
- **No cost guardrails.**
  - There is no budget resource or alert.
  - The "$0.70/hr" figure in `variables.tf`, `vertex.tf` and the DEVOPS guide is a GCE
    price. Vertex with an L4 costs about $1.0–1.2/hr.
  - Fix: add `google_billing_budget` and correct the figure.
- **IAM is over-broad.**
  - The runtime service account has `roles/aiplatform.user` and project-wide
    storage and artifact readers it doesn't use.
  - Fix: keep only the log and metric writers. Document that the deployer needs
    `iam.serviceAccountUser`.
- **Supply chain.**
  - The base image is pinned by tag, not digest, and actions are pinned by tag, not
    SHA.
  - There is no `permissions:` block in CI and no Dependabot.
- **Packaging and metadata.**
  - The version is still `0.1.0` despite breaking changes, so bump it to 0.2.0 or fold
    everything into a clean 0.1.0.
  - Python 3.14 is excluded.
  - There is no `py.typed`, `[project.urls]` or classifiers, and the relative README
    links break on PyPI.
- **Leftover old names.**
  - The Terraform `name_prefix` defaults to `"taste"`
    (`variables.tf`, `terraform.tfvars.example`, DEVOPS guide).
  - `services/detector_protocol.py` mentions the removed `DetectorAgent`.
- **Missing OSS files.** There is no CONTRIBUTING, SECURITY.md, CODE_OF_CONDUCT, issue
  or PR templates (under `.github/`), `CITATION.cff`, or README badges. The original
  SE paper (Kuhn, Gal & Farquhar, ICLR 2023) is not cited.
- **No results.** Run one PE-vs-SE AUROC table on TriviaQA after the B2 fixes, or say
  plainly that the repo is a toolkit, not a study.

### Minor

- SE adds `+1e-12` inside the log, so a single cluster scores slightly negative.
  Remove it.
- The log message in `semantic_entropy.py` says "fallback" and then re-raises.
- `evaluate_responses` doesn't check that `prompt` matches `samples.prompt`.
- Duplicate detector names silently overwrite each other's scores.
- `escalation_detector` is still required when `escalation_policy` is given.
- The `AnyAbove` and `AllAbove` frozen dataclasses hold a mutable dict. Clearing it
  makes `AllAbove` escalate on everything. Use `MappingProxyType`.
- Several docs contradict the code:
  - They say a missing hidden state degrades to "SEP: n/a", but it raises.
  - They call SEP probe loading lazy, but it is eager.
  - `model_registry.py` says the Dockerfile uses `model_flags.py`; it uses `launch.py`.
- `HFTransformersProvider._device` is dead code.
- `NLIScorer._ensure_loaded` has no lock, so the model could be loaded twice.
- `serving/launch.py --help` crashes when `MODEL_ID` is unset.
- `SEP_PROBE_PATH` is missing from `.env.example`.
- `cloudbuild.googleapis.com` is missing from `apis.tf`, and `cloudbuild.yaml` uses
  the default 10-minute timeout.
- The fixed default `vertex_endpoint_id` (1000000001) can collide between stacks.
- `torch` and `transformers` are hard dependencies even for scoring pre-collected
  samples.
- Coverage gaps: `nli.py` is at 40%. `demo/cli.py`, `serving/vertex_deploy.py` and
  `serving/model_flags.py` are at 0%. NaN, partial-logprob and empty-sample cases are
  untested.
- The CHANGELOG `[0.1.0]` block says "Initial public surface" although the repo was
  never public.

### Checked and found sound

- The quickstart runs on CPU and matches the README.
- The wheel installs clean.
- No broken links.
- No secrets in history (regex scan).
- The greedy answer is kept separate from the high-temperature samples.
- The probe/benchmark leakage guard holds.
- Bootstrap pairing and McNemar alignment are correct.
- AUROC ties and direction are correct.
- The invoke-route URL form and the `/health` route on port 8080 match the Vertex docs.

## Decisions already made

| Date | Decision | Reason |
|---|---|---|
| 2026-10 | Keep the repo private for now. | The owner is preparing the release, not publishing yet. |
| 2026-10 | Private files stay gitignored on disk and are never deleted. | They are the owner's working material, just not public material. |
| 2026-10 | Import package `hallucinations_math_lab`, MIT license. | It matches the final distribution name. |
| 2026-10 | Serve Gemma 4 on `vllm/vllm-openai:v0.31.0-cu129`, one runner per container. | Gemma 4 support starts at v0.19. The cu129 build runs on Vertex's R535+ drivers. |
| 2026-10 | Deploy to Vertex through the invoke route (`"/*"`) on dedicated endpoints, with numeric endpoint ids. | The OpenAI-compatible paths reach vLLM unchanged, and Vertex rejects non-numeric ids. |
| 2026-10 | The judge is `gemini-3.5-flash-lite` via `google-genai`, controlled by thinking level. | Gemini 2.5 retires on 2026-10-20, and 3.5 on Vertex ignores sampling parameters. |
| 2026-10 | Deleted `eval/vertex_genai_eval.py`. | It scored groundedness with no context to ground against, and pulled `litellm`, which has an open advisory. |
| 2026-10 | Ruff pinned to 0.17.0 in CI. | An unpinned `uvx ruff` broke CI when new defaults shipped. |

## Open decisions

| Decision | What it blocks |
|---|---|
| History cleanup: `filter-repo` plus force-push, or a fresh single-commit repo. Also delete the remote `v0.1.0` tag and the `claude/…` branch. | Going public (B1). Needs explicit owner approval. |
| SEP: implement the Kossen recipe, or keep ridge regression and drop the "matches" claim. | The SEP docs and success criterion 1. |
| Correctness labeling: normalized token match, or an LLM correctness judge. | Every AUROC number. |
| Version: 0.2.0, or a re-cut clean 0.1.0. | CHANGELOG release block and tag. |
| Whether to run a real benchmark (L4 GPU) before going public. | The README "results" story. |
| Whether this handoff file and `knowledge/plans/` ship publicly. | Going public. This file names internal review findings. |

## Traps that will bite

- **Symptom:** one `test_google_auth` test fails with an `httpx2.Client` error.
  **Cause:** a stale `.venv` holding openai 2.x. Fix with `uv sync --locked --all-extras`.
- **Symptom:** after a commit, a fresh checkout is missing `.claude/`,
  `knowledge/lectures/` or `knowledge/submissions/`.
  **Cause:** they are gitignored by design. They exist only in the owner's original
  checkout at `~/code/hallucinations-math-lab`. Never `git rm` them, and never
  `git add -A` blindly.
- **Symptom:** the repo root has untracked `sep_presentation.pdf` and `.coverage`.
  **Cause:** local artifacts that are not ignored. `git add -A` would publish them, so
  stage paths explicitly.
- **Symptom:** CI lint fails on code that passed locally.
  **Cause:** a different ruff version. Always use `uvx ruff@0.17.0`.
- **Symptom:** `git push` says "Repository not found".
  **Cause:** the wrong `gh` account is active. Run `gh auth switch --user eladlaor`.
- **Symptom:** mypy errors around transformers `generate`.
  **Cause:** in transformers 5.x, `PreTrainedModel` is no longer a `GenerationMixin`.
  The provider narrows at runtime; keep that pattern.
- **Symptom:** a Vertex endpoint delete fails with a model still deployed.
  **Cause:** the deploy is tracked only by a Terraform `null_resource`. Undeploy with
  `uv run --extra deploy python serving/vertex_deploy.py undeploy …` first.
- **Invisible state in a fresh clone:**
  - **No `.env`.** Copy `.env.example`. The keys that need real values are
    `GCP_PROJECT_ID`, `GCP_REGION`, `VLLM_BASE_URL`, `VLLM_API_KEY` (or
    `VLLM_AUTH=google` with ADC from `gcloud auth application-default login`), and
    `SEP_PROBE_PATH` if SEP is enabled.
  - **No `.venv`.** Rebuild it with `uv sync --locked --all-extras`.
  - **No local Terraform state** (`terraform.tfstate` is gitignored, and the backend
    is local until the first apply).
  - **No `dist/`.**

## How to update this file

Rewrite it; never append. It describes the present. Remove whatever is no longer true,
because history lives in `CHANGELOG.md` and git. As review items get fixed, delete them
from the review section and record the change in the `CHANGELOG.md` `[Unreleased]`
section.
