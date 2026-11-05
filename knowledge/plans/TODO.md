# TODO

## Table of contents

- [Summary](#summary)
- [Recently done](#recently-done)
- [Blockers](#blockers)
- [Major](#major)
- [Minor](#minor)
- [Open decisions](#open-decisions)
- [Decisions already made](#decisions-already-made)
- [Traps that will bite](#traps-that-will-bite)
- [How to update this file](#how-to-update-this-file)

## Summary

The work queue for `hallucinations-math-lab`. It replaces `SESSION_HANDOFF.md`.

- **The repo is private** and is being prepared for public release. CI is green.
- **No experiment has run on real hardware.** There has been no GPU run, no Vertex
  deploy and no benchmark result, so no stored numbers are invalidated by the fixes
  below.
- **A pre-publication review on 2026-10-17 blocks release.** Its findings are the
  open items below.
- **Next actions, in order:**
  1. Finish the method blockers (B2), test-first.
  2. Fix B3.
  3. Get the owner's decision on B1.

  Verify with `uv sync --locked --all-extras && uv run pytest -q && uv run mypy && uvx ruff@0.17.0 check . && uvx ruff@0.17.0 format --check .`

## Recently done

- [x] **SE's NLI check now sees the question** (2026-10-19, B2).
  - **The bug:** `services/semantic_entropy.py` passed bare answers to DeBERTa. The
    reference code (`jlko/semantic_uncertainty`, `condition_on_question=True` by
    default) uses `f"{question} {answer}"`.
  - **The effect, on our checkpoint:** "Paris" vs. "The capital of France is Paris."
    scored neutral/entailment, which gave two clusters for one correct answer. With
    the question prefixed it scores entailment/entailment, which gives one cluster.
    The bug inflated SE on correct answers and depressed AUROC.
  - **The fix:** `SE_NLI_INPUT_TEMPLATE` in `constants.py`, and a `question` argument on
    `cluster_by_bidirectional_entailment`. `compute_semantic_entropy` passes
    `sample_set.prompt`.
  - **Tests:** `test_nli_inputs_are_conditioned_on_the_question` and
    `test_cluster_representative_text_stays_the_bare_answer`.
  - **Docs:** walkthrough Step 3, CHANGELOG `[Unreleased] / Fixed`.
  - **Also verified to match the reference code:** both directions must be
    entailment (strict), each answer is compared only to each cluster's first
    member, and the label is the argmax.
  - **Follow-up:** the first SEP probe must be trained after this fix. That's
    automatic, since none has been trained yet.

## Blockers

**B1. Git history would leak private material.** Needs owner approval; do nothing
destructive before it.
- **What is in history:**
  - Files that are now gitignored: `.claude/`, `knowledge/lectures/`,
    `knowledge/submissions/` (which includes the organizers' email).
  - The original GCP project id, in old `terraform/backend.tf`,
    `terraform.tfvars.example` and the DEVOPS guide.
  - `Co-Authored-By` and `Claude-Session` trailers.
  - Inconsistent author names.
- **Remote refs to remove:**
  - The tag `v0.1.0` points at a commit with private files.
  - The branch `claude/load-bearing-words-…` lists private artifact URLs.
- **Fix:** `git filter-repo --path .claude --path knowledge/lectures --path
  knowledge/submissions --invert-paths`, plus scrubbing the project id and the
  trailers. The alternative is a fresh single-commit repo. Either way, delete the
  remote tag and branch.
- **Secrets:** none found by regex scan. gitleaks and trufflehog were not run.

**B2. Evaluation-method flaws.** Remaining items:
- [ ] **Brief-answer prompt.**
  - The raw TriviaQA question goes to chat-tuned Gemma (`demo/slm_provider.py`), and
    `MAX_TOKENS=64` cuts off long answers.
  - Add a fixed answer-format instruction, record it in the run config, and report
    the answer-length distribution.
- [ ] **Correctness matching.**
  - `eval/benchmark_results.py::is_correct` is a raw substring match on `aliases`
    (`"It was 1921."` matches alias `"2"`).
  - Use SQuAD-style normalization with whole-token matching on `normalized_aliases`,
    or an LLM correctness judge.
  - Hand-check about 100 labels and report the error rate.
- [ ] **NLI label order.** Read it from `model.config.id2label`, not the hardcoded
  `DEBERTA_MNLI_LABEL_ORDER`. This is folded into step 1 of
  [`SOFT_NLI_EQUIVALENCE.md`](SOFT_NLI_EQUIVALENCE.md).
- [ ] **"Discrete SE" naming.** `p(c)=|c|/N` ignores sequence likelihoods.
  - Call it "discrete SE" everywhere.
  - Add probability-weighted SE from the logprobs already collected.
- [ ] **The SEP probe does not match Kossen et al.**
  - It uses unstandardized Ridge regression on raw last-layer SE, reported as Pearson
    r. The paper trains a logistic classifier on binarized SE, with a layer sweep,
    and reports AUROC.
  - Implement the paper's recipe or drop the claim, and give SEP its own threshold.
- [ ] **τ=1.0 is called "calibrated", but nothing was measured.**
  - Label it a placeholder and calibrate it on a disjoint dev slice.
  - Justify each bar in `eval/success_criteria.py`.
- [ ] **The escalation-uplift criterion doesn't test the detector.** Compare against
  random and PE-triggered escalation at the same rate, report the accuracy-vs-rate
  curve, and control for answer length.
- [ ] **The adversarial criterion is weak.**
  - It uses 7 prompts, and the comparison group is chosen by outcome. A model that
    refuses consistently gets low SE.
  - Use a few hundred items from SelfAware or FalseQA, with labeled refusals.
- [ ] **NLI hard labels (argmax)** cannot be thresholded or tuned. See
  [`SOFT_NLI_EQUIVALENCE.md`](SOFT_NLI_EQUIVALENCE.md).

**B3. The 26B model cannot deploy.**
- [ ] `demo/model_registry.py` pairs `NVIDIA_A100_80GB` with `a2-highgpu-1g`, which
  only carries the 40GB A100. Change it to `a2-ultragpu-1g`.
- [ ] Add a test that checks each accelerator type against its valid machine families.

## Major

- [ ] **Library settings leak into consumers.** `config.py` has no `env_prefix`, so a
  consumer's `MAX_TOKENS=4096` breaks `NLIScorer()`.
  - Use `env_prefix="HML_"`.
  - Move the demo and sweep fields into a separate demo settings class.
- [ ] **`ResponseEvaluator()` with no `slm` imports `demo`**, which is not in the
  wheel. Make `slm` required.
- [ ] **The extras install dependencies for unshipped code.** Move `demo`, `eval`,
  `deploy` and `dev` from extras to `[dependency-groups]`.
- [ ] **Errors are flattened.** `ResponseEvaluator.evaluate` wraps everything as
  `RuntimeError`. Use `e.add_note(...)` and re-raise the original.
- [ ] **The vLLM client has no timeouts and retries blindly**: about 9 sends and about
  90 minutes in the worst case.
  - Pass `timeout` and `max_retries` from config.
  - Retry only connection, rate-limit and 5xx errors.
- [ ] **Silent fallbacks.** PE skips samples with no logprobs, and NaN logprobs pass
  through. Raise on a partial sample set or non-finite values.
- [ ] **Sampling parameters are unverified.** Gemma's `generation_config` top-k and
  top-p defaults may apply. Set them explicitly and log the effective values.
- [ ] **Cold start.**
  - Stage the weights in GCS (`artifact_uri`) and add a startup probe.
  - Set `serving_container_deployment_timeout=7200`.
  - Pass an optional `HF_TOKEN` from Secret Manager.
- [ ] **Teardown gaps.** An interrupted `terraform apply` can leave an untracked GPU
  deployed. Add a teardown script and document the `gcloud ai endpoints
  undeploy-model` recovery step.
- [ ] **No cost guardrails.**
  - Add `google_billing_budget`.
  - Correct "$0.70/hr" to about $1.0–1.2/hr for Vertex with an L4.
- [ ] **IAM is over-broad.**
  - Keep only the log and metric writers on the runtime service account.
  - Document that the deployer needs `iam.serviceAccountUser`.
- [ ] **Supply chain.**
  - Pin the base image by digest and the actions by SHA.
  - Add a CI `permissions:` block and Dependabot.
- [ ] **Packaging.**
  - Version bump (see the open decisions).
  - Python 3.14 is excluded.
  - Add `py.typed`, `[project.urls]` and classifiers.
  - Fix the README links so they work on PyPI.
- [x] **Leftover old names.** The Terraform `name_prefix` now defaults to `"hml"`;
  `detector_protocol.py` no longer mentions `DetectorAgent`.
- [ ] **Missing OSS files.**
  - CONTRIBUTING, SECURITY.md, CODE_OF_CONDUCT, issue and PR templates,
    `CITATION.cff`, badges.
  - Cite Kuhn, Gal & Farquhar (ICLR 2023).
- [ ] **No results.** After B2, run one PE-vs-SE AUROC table on TriviaQA, or state
  that the repo is a toolkit, not a study.

## Minor

- [ ] SE adds `+1e-12` inside the log, so a single cluster scores slightly negative.
- [ ] The log message in `semantic_entropy.py` says "fallback" and then re-raises.
- [ ] `evaluate_responses` doesn't check that `prompt == samples.prompt`. This
  matters more now that SE reads `samples.prompt` as the NLI question.
- [ ] Duplicate detector names silently overwrite each other's scores.
- [ ] `escalation_detector` is still required when `escalation_policy` is given.
- [ ] The `AnyAbove` and `AllAbove` frozen dataclasses hold a mutable dict. Use
  `MappingProxyType`.
- [ ] Docs that contradict the code:
  - A missing hidden state is said to degrade to "SEP: n/a", but it raises.
  - SEP probe loading is called lazy, but it is eager.
  - The Dockerfile is said to use `model_flags.py`, but it uses `launch.py`.
- [ ] `HFTransformersProvider._device` is dead code.
- [ ] `NLIScorer._ensure_loaded` has no lock.
- [ ] `serving/launch.py --help` crashes when `MODEL_ID` is unset.
- [ ] `SEP_PROBE_PATH` is missing from `.env.example`.
- [ ] `cloudbuild.googleapis.com` is missing from `apis.tf`, and `cloudbuild.yaml` uses
  the default 10-minute timeout.
- [ ] The fixed default `vertex_endpoint_id` (1000000001) can collide between stacks.
- [ ] `torch` and `transformers` are hard dependencies even for scoring pre-collected
  samples.
- [ ] Coverage gaps: `nli.py` at 40%, and `demo/cli.py`, `serving/vertex_deploy.py`
  and `serving/model_flags.py` at 0%.
- [ ] The CHANGELOG `[0.1.0]` block says "Initial public surface" although the repo
  was never public.

## Open decisions

| Decision | What it blocks |
|---|---|
| History cleanup: `filter-repo` plus force-push, or a fresh single-commit repo. | Going public (B1). Needs explicit owner approval. |
| SEP: the Kossen recipe, or keep ridge regression and drop the claim. | The SEP docs and criterion 1. |
| Correctness labeling: normalized token match, or an LLM judge. | Every AUROC number. |
| Version: 0.2.0, or a re-cut clean 0.1.0. | The CHANGELOG release block and tag. |
| Whether to run a real L4 benchmark before going public. | The README results story. |
| Whether `knowledge/plans/` ships publicly. | Going public; it names internal review findings. |

## Decisions already made

| Date | Decision | Reason |
|---|---|---|
| 2026-10 | Keep the repo private for now. | The release is being prepared, not published. |
| 2026-10 | Private files stay gitignored on disk, never deleted. | They are the owner's working material. |
| 2026-10 | Package `hallucinations_math_lab`, MIT license. | Matches the distribution name. |
| 2026-10 | Serve Gemma 4 on `vllm/vllm-openai:v0.31.0-cu129`. | Gemma 4 support starts at v0.19, and cu129 runs on Vertex's R535+ drivers. |
| 2026-10 | Vertex invoke route (`"/*"`), dedicated endpoints, numeric ids. | The OpenAI paths pass through unchanged, and Vertex rejects non-numeric ids. |
| 2026-10 | Judge is `gemini-3.5-flash-lite`, controlled by thinking level. | Gemini 2.5 retires on 2026-10-20, and 3.5 ignores sampling parameters. |
| 2026-10 | Deleted `eval/vertex_genai_eval.py`. | Groundedness with no context, and `litellm` has an open advisory. |
| 2026-10 | Ruff pinned to 0.17.0 in CI. | An unpinned ruff broke CI. |
| 2026-10-19 | SE conditions NLI on the question. | Matches the reference implementation; see [Recently done](#recently-done). |
| 2026-10-19 | `SESSION_HANDOFF.md` replaced by this file. | One work queue instead of a narrative handoff. |

## Traps that will bite

- **`test_google_auth::test_openai_client_sends_a_refreshed_token…` fails.**
  - **Cause:** a stale `.venv` holding openai 2.x.
  - **Fix:** `uv sync --locked --all-extras`.
- **A fresh checkout is missing `.claude/`, `knowledge/lectures/` or
  `knowledge/submissions/`.**
  - They are gitignored by design.
  - Never `git rm` them, and never `git add -A`.
- **Untracked `sep_presentation.pdf` and `.coverage` in the root.** These are local
  artifacts, so stage paths explicitly.
- **CI lint fails on code that passed locally.** Always use `uvx ruff@0.17.0`.
- **`git push` says "Repository not found".** Run `gh auth switch --user eladlaor`.
- **mypy errors around transformers `generate`.** In 5.x, `PreTrainedModel` is not a
  `GenerationMixin`. Keep the runtime narrowing.
- **A Vertex endpoint delete fails.** Undeploy first with `uv run --extra deploy
  python serving/vertex_deploy.py undeploy …`.
- **A fresh clone has no `.env`, `.venv`, Terraform state or `dist/`.**
  - Copy `.env.example` and fill in `GCP_PROJECT_ID`, `GCP_REGION`, `VLLM_BASE_URL`,
    and `VLLM_API_KEY` (or `VLLM_AUTH=google`), plus `SEP_PROBE_PATH` if SEP is
    enabled.

## How to update this file

- Check items off in place when they're done. Move notable ones to
  [Recently done](#recently-done) with a one-line why.
- Record behavior changes in `CHANGELOG.md` `[Unreleased]`.
- Prune [Recently done](#recently-done) once a release is cut, because history lives
  in the CHANGELOG and git.
