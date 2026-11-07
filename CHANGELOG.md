# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Local web playground (`uv run python -m demo.web`, http://127.0.0.1:8765): a
  FastAPI app with a committed React UI, one tab per way in — score answers
  with their meaning clusters, PE and a client-side escalation threshold; ask
  the model (`detect` / `chat`); and view benchmark AUROCs with CIs or a
  model comparison. New `web` optional extra (`fastapi`, `uvicorn`), included
  in `all`.
- README hardware tiers: what each model needs in GPU memory, Mac memory,
  download and disk, and why.
- `python -m demo.check_hardware` (Linux, macOS): measures memory, GPU, free
  disk and download speed, and recommends the tier and `VLLM_MODEL_ID`;
  `--json` for agents.
- Model comparison: `eval.run_benchmark --sweep` runs the question set once per
  model in `TARGET_MODEL_IDS` (one results file per model, one model loaded at a
  time, registry and per-model SEP probes checked before any scoring), and
  `eval.compare_models` reports accuracy and per-detector AUROC with CIs side by
  side, refusing runs over different questions or sampling settings.
- `eval/calibrate_threshold.py`: chooses `SE_THRESHOLD` on a dev slice by
  Youden's J and refuses a dev slice that overlaps the test slice. The
  default `1.0` is now documented as a placeholder.
- Criterion 3b in `eval/escalation_uplift.py`: the run's escalation is compared
  with escalating the same number of questions at random (exact expectation,
  paired-bootstrap CI), and `accuracy_vs_rate` gives accuracy at every
  escalation rate for random and for each detector's top-k.
- `run_benchmark` records the judge's answer to every question
  (`judge_answer`), not only escalated ones; `ResponseEvaluator.judge`
  read-only property.
- `DiscreteSemanticEntropyDetector` (`discrete_semantic_entropy`): Semantic
  Entropy with p(c) = |c| / N, for samples without token logprobs.
- Benchmark rows record `max_tokens` and `answer_instruction`, and the
  scoring scripts refuse to mix or compare runs that differ in them.
- `eval/compute_auroc.py` reports the greedy answer-length distribution
  (median, p90, max words).
- Terraform configuration under `terraform/` for GCP resources (APIs, GCS,
  Artifact Registry, IAM, Vertex). The Vertex AI endpoints are gated behind
  `deploy_vertex_endpoint` (default `false`) so GPU nodes run only during
  active benchmark runs.
- `examples/langrag_integration.py` reference integration.
- Runnable no-GPU playground scripts: `examples/score_precollected_samples.py`
  (PE and SE on hand-written samples, with DeBERTa NLI) and
  `examples/custom_detector.py` (a user-defined detector next to PE and SE).
- Measurable research success criteria (QA Guide) with a script for each:
  `eval/compute_auroc.py` now reports the SE AUROC and SE-PE gap,
  `eval/escalation_uplift.py`, `eval/adversarial_separation.py`, and a
  held-out SEP-vs-SE Pearson r in `eval/train_sep_probe.py`.
- Statistical inference for the success criteria (`eval/inference_stats.py`):
  bootstrap 95% CIs for every AUROC, a paired-bootstrap CI for the SE-PE gap,
  McNemar's exact test and a paired CI for the escalation uplift, and the AUC
  effect size for adversarial separation.
- `load_triviaqa --offset` for disjoint dataset slices; `train_sep_probe`
  writes a `<probe>.train_ids.json` sidecar and `run_benchmark` refuses to
  score a probe on its own training questions, or a probe without a sidecar.
- `ResponseEvaluator.slm` and `.detectors` read-only properties;
  `SemanticEntropyProbeDetector.probe_path`.
- `VLLM_AUTH=google`: the vLLM client sends a refreshing Google access token,
  so it can call a Vertex AI endpoint directly (`demo/google_auth.py`).
- `EntailmentScorer` protocol: Semantic Entropy accepts any equivalence
  judge; `EntailmentScorer` and `NLIScorer` are exported.
- `serving/launch.py` container entrypoint and `serving/vertex_deploy.py`
  (upload with the invoke route, deploy, undeploy); `deploy` extra.
- Terraform outputs `vllm_base_url` / `vllm_embed_base_url`, ready to use as
  OpenAI base URLs; optional embed-runner endpoint (`with_embed_endpoint`).
- `JUDGE_THINKING_LEVEL` setting (default `LOW`). The judge sends no sampling
  parameters, which Gemini 3.5 on Vertex ignores, and its schema puts
  `reasoning` before `answer`.
- GitHub Actions CI with four jobs: ruff lint and format check (ruff pinned),
  Terraform fmt/validate, mypy, and the test suite on Python 3.12 and 3.13.
- `MIT` `LICENSE` file.
- QA Guide, an end-to-end test of the benchmark -> AUROC loop, and unit tests
  for the launcher, the judge, token refresh (including a contract test
  against the real OpenAI client) and the HF logprob extraction.

### Changed
- The "Computed semantic entropy." log record replaces the scalar `se_nats`
  and `estimator` fields with `se_nats_by_estimator`, a map from estimator to
  SE in nats.
- `eval.run_benchmark --in` and `eval.train_sep_probe --calibration-set` are
  optional: without them, fixed non-overlapping TriviaQA slices (200
  benchmark, 500 calibration questions) are downloaded once to
  `data/questions/`.
- The README is now the user guide; `knowledge/usage_guides/USER_GUIDE.md` is
  folded into it, and it states up front which model is under test and how to
  run one model or compare several.
- `demo.slm_provider.build_sweep_providers` is replaced by `sweep_model_ids`,
  which validates the ids so the sweep can build one provider at a time.
- The playground's Score answers tab reports which SE estimator ran
  (`se_estimator`): likelihood-weighted when every answer carries token
  logprobs, discrete for typed or edited answers.
- `semantic_entropy` now uses Farquhar et al.'s likelihood-weighted
  estimator: each cluster's probability is its samples' share of
  length-normalized sequence likelihood. The previous count-based estimator
  is `discrete_semantic_entropy`. Likelihood-weighted SE fails fast on samples
  without finite token logprobs.
- **Breaking:** the import package is renamed from `taste` to
  `hallucinations_math_lab`, matching the `hallucinations-math-lab` distribution.
- The Terraform `name_prefix` default changes from `taste` to `hml`, renaming the
  service account, Artifact Registry repo and Vertex display names.
- **Breaking:** `GeminiJudge` uses the Google Gen AI SDK (`google-genai`)
  with schema-validated output instead of the deprecated
  `vertexai.generative_models`; the `judge` extra changed accordingly.
- **Breaking:** default judge model is `gemini-3.5-flash-lite`; the Gemini 2.5
  models retire on 2026-10-20.
- **Breaking:** `ModelEntry.vllm_flags()` takes a `ServingRunner` and no longer
  emits `--model`; `serving/model_flags.py <id> vllm` takes a runner and prints
  a full `vllm serve` command.
- **Breaking:** the serving image runs one runner per container
  (`VLLM_RUNNER=generate|embed`) on `vllm/vllm-openai:v0.31.0-cu129`;
  `VLLM_EMBED_PORT` and the sibling embed process are gone.
- Vertex deployment uses the invoke route on dedicated endpoints, one per
  vLLM runner, with numeric endpoint ids (`vertex_endpoint_id`). Google
  provider `~> 8.0`; the provider lock file is committed.
- vLLM generation reports raw logprobs (`--logprobs-mode raw_logprobs`);
  embeddings are last-token, unnormalized, over the chat-templated prompt,
  matching `HFTransformersProvider` so SEP probes transfer between providers.
- **Breaking:** success criteria 2-4 now also need significance or effect
  size: the SE-PE gap's CI must exclude zero, the escalation uplift must pass
  McNemar's test (p < 0.05), and adversarial separation needs an AUC effect
  size of at least 0.75 besides p < 0.01.
- **Breaking:** `compute_auroc` reports `detectors.<name>.{auroc, ci}` instead
  of a flat `auroc` map, and the SE-PE gap is computed from unrounded AUROCs.
- **Breaking:** benchmark rows record `model_id`, `sample_n` and
  `sample_temperature`; the scoring scripts refuse files that mix runs, rows
  written without these fields, and adversarial/reference runs that differ.
- **Breaking:** `train_sep_probe` reads `load_triviaqa` output (`id`,
  `question`) instead of a `prompt` field, and reports a 5-fold
  cross-validated Pearson r (`--cv-folds`) instead of one held-out split
  (`--holdout-fraction` removed).
- `escalation_uplift` reports how many answers escalation fixed and broke and
  the mean answer length on both sides, warning when final answers are much
  longer (substring matching rewards length).
- `EscalationPolicy.detectors_referenced` and `SamplingProvider.model_id` are
  read-only protocol members, so frozen dataclasses satisfy them.
- Public API carved into response-shaped (`evaluate_responses`) and
  provider-shaped (`ResponseEvaluator`) surfaces; `DetectorAgent` retired.
- Demo code carved out of `src/` into a top-level `demo/` package, run with
  `python -m demo.cli`.
- Detector pipeline made pluggable; SLM provider is now swappable. SEP
  promoted to a first-class detector alongside predictive entropy and
  Semantic Entropy.
- `eval/run_benchmark.py` records every enabled detector under
  `detector_scores` and stops on the first failed row instead of skipping it;
  `eval/compute_auroc.py` reports one AUROC per detector.
- Model-registry validation of `TARGET_MODEL_IDS` moved from library settings
  into `demo.slm_provider.build_sweep_providers`.
- Machine type, accelerator type/count, and `model_id` are sourced from the
  Python model registry via `serving/model_flags.py <id> terraform`.
- Dependencies upgraded to the latest compatible releases (openai 3,
  google-genai 2, transformers 5.19, torch 2.14); `mypy` with
  `check_untyped_defs` and ruff 0.17's default rules are enforced.
- README and all usage guides rewritten around using the repo as an
  experimentation playground; theory and caveat docs restructured as Q&A.
- Private talk-preparation and agent notes are no longer tracked.

### Removed
- `eval/vertex_genai_eval.py` and `EVAL_JUDGE_MODEL_ID`: it scored
  groundedness without any context to ground against, and its dependency
  pulled a `litellm` release with an open advisory.
- `autoawq` from the serving image (unmaintained, and unused by the registry's
  unquantized models).
- `serving/deploy_vertex.sh` — replaced by the Terraform path.

### Fixed
- `eval.datasets.load_triviaqa` loads `mandarjoshi/trivia_qa`; the bare
  `trivia_qa` name is rejected by current `datasets` releases.
- TriviaQA slices are cut from unique questions: the validation split repeats
  most questions once per evidence source, which put the same question in
  both the calibration and benchmark slices and twice within a slice.
- The default SEP probe is resolved from the model being evaluated, not from
  `VLLM_MODEL_ID`, so a provider built for another model no longer loads the
  wrong model's probe.
- Semantic Entropy added `1e-12` inside the log, so one cluster scored
  slightly below zero; one cluster now scores exactly 0.
- Answer correctness used a raw substring match, so alias "2" matched "It was
  1921."; it now matches whole tokens after SQuAD-style normalization of both
  sides (lowercase, no punctuation, no articles).
- The SLM received the bare TriviaQA question and answered in long sentences
  cut off by `MAX_TOKENS`; every provider path now prepends a brief-answer
  instruction (`ANSWER_INSTRUCTION`), while `SampleSet.prompt` and SE's NLI
  input stay the bare question.
- `NLIScorer` read labels in a hardcoded DeBERTa-MNLI index order; it now
  names each output from the checkpoint's `config.id2label` and fails fast on
  an unrecognized label. New `NLIScorer.label_probs` returns the softmax.
- The 26B registry entry paired the 80 GB A100 with `a2-highgpu-1g`, which
  carries only the 40 GB A100; it now uses `a2-ultragpu-1g`.
- Semantic Entropy now prefixes each sampled answer with the question before
  the DeBERTa NLI check, matching the reference implementation; bare short
  answers ("Paris" vs. "The capital of France is Paris.") were judged neutral
  and split into separate clusters, inflating SE on correct answers.
- `HFTransformersProvider` computed logprobs from temperature/top-p processed
  scores, counted batch padding after EOS, and skipped the chat template;
  it now uses raw logits, keeps the first EOS, drops padding, and applies the
  chat template like the vLLM path.
- The 31B registry entry requested two GPUs without `--tensor-parallel-size 2`.
- E2B/E4B asked vLLM for a 131,072-token context that cannot fit next to the
  weights on a 24 GB L4; now 8,192.
- Vertex endpoint names were non-numeric, which the API rejects.
- The serving image's vLLM pin predated Gemma 4, and `--task embed` no
  longer exists in current vLLM.
- `eval/run_benchmark.py` imported the removed `DetectorAgent` and could not run.
- The library imported the unshipped `demo` package when loading settings,
  so `NLIScorer`, `GeminiJudge` and `ResponseEvaluator` failed outside the repo.
- `.env.example` set `ENABLED_DETECTORS` as a comma-separated list, which
  failed settings parsing at startup; it is now a JSON list.
- `HFTransformersProvider` loaded with `device_map="auto"` without the
  `accelerate` package it requires; `accelerate` is now a `demo` dependency.
- A sweep could pair a generation model with an embed endpoint serving a
  different model, feeding SEP the wrong hidden states; this now fails fast.
- A failed Vertex deploy left the uploaded model orphaned; it is now deleted.
- Applying Terraform from a different checkout path forced a redeploy.
- Removed unused `torch.nn` import in `eval/train_sep_probe.py`.

### Security
- Locked dependencies upgraded until `pip-audit` reports no known
  vulnerabilities; `litellm` (open advisory GHSA-3cv6-jpf6-8222) is no longer
  pulled in.

## [0.1.0] - 2026-05-21

### Added
- Initial public surface: predictive entropy, Semantic Entropy (SE), and
  Semantic Entropy Probes (SEP) detectors.
- Gemini-based judge with judge protocol for LLM-as-a-judge escalation.
- vLLM-on-Vertex serving reference under `serving/` (Dockerfile,
  `cloudbuild.yaml`, `deploy_vertex.sh`, model flags).
- Offline benchmarking pipeline under `eval/` (TriviaQA loader, adversarial
  set, AUROC computation, Vertex Gen AI Evaluation Service integration, SEP
  trainer).
- Unit test suite covering detectors, SEP, evaluator, SLM provider, and
  model registry.
- User guide and developer documentation under `knowledge/`.

[Unreleased]: https://github.com/eladlaor/hallucinations-math-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eladlaor/hallucinations-math-lab/releases/tag/v0.1.0
