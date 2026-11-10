# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SEP provenance records the torch build that produced the hidden states:
  `torch_version` and `torch_cuda_version` (None for CPU-only) in the dataset's
  collection config and the probe's training record, shown on both Hub cards;
  the DevOps guide documents swapping in the CUDA 12.6 build on older-driver pods.
- Playground clustering replay: the *Score answers* and *Ask the model* tabs
  step through the real SE clustering run, one NLI comparison at a time
  (the answer, the meaning cluster's representative, both directional
  verdicts with the question in front, and whether it joins or opens a new
  meaning cluster), then fill in SE = −Σ p(c) ln p(c) with the p(c) the run
  used. Play, pause, step back or forward, at speeds from slow motion to real
  time. The Ask tab records it unless *Record the clustering* is unticked.
  `/api/status` reports `se_trace_unavailable_reason` (Score) and
  `ask_se_trace_unavailable_reason` (Ask, which also needs an SE detector in
  `ENABLED_DETECTORS`); both tabs score without the replay, saying why, when
  it cannot be recorded or the status has not loaded.
- Opt-in SE trace for a step-by-step replay of the clustering: `se_trace: true`
  on the playground's `/api/score` and `/api/ask` returns every NLI
  comparison in the order it ran (each direction's label), where each answer
  ended up, and the p(c) SE was computed from. In the library,
  `cluster_with_trace`, `trace=True` on `compute_semantic_entropy`, and
  `trace_semantic_entropy` on `evaluate_responses` / `ResponseEvaluator.evaluate`
  produce it; per-direction labels come from the new
  `DirectionalEntailmentScorer` protocol, which `NLIScorer` satisfies. Without
  the flag, clusters, NLI calls and responses are unchanged.
- `SEP_THRESHOLD` (default `0.5`, a placeholder): SEP's own escalation
  threshold on its probability scale. A trained probe records a
  `recommended_threshold` (Youden's J on out-of-fold predictions), which is
  logged next to the threshold in use but never applied implicitly.
- SEP layer and token-position sweep: `metrics.json` reports a single-layer
  AUROC for every layer at the prompt's last token and near the answer's end,
  the last-layer baseline, and the selected window's nested cross-validated
  AUROC.
- `eval.collect_sep_dataset`: the SEP calibration dataset (sampled answers,
  meaning clusters, discrete and likelihood-weighted SE, greedy correctness,
  and every layer's hidden states) as `rows.parquet`,
  `hidden_states.safetensors`, `collection_config.json` and
  `training_log.jsonl`.
- SEP probes load from the Hugging Face Hub with
  `SEP_PROBE_PATH=hf://<owner>/<repo>[@<revision>]`.
- `eval.publish_sep_probe`: checks a probe, its dataset and a held-out
  benchmark report against each other, renders a model card, and uploads the
  probe and dataset repos under one tag (`--dry-run` uploads nothing).
- The SEP model card states its license: `license: apache-2.0` with the Gemma 4
  `license_link`, plus a license section covering the base model's terms, the
  repo's MIT code license at the fitting commit, and the backbone restriction.
  TriviaQA question ids keep TriviaQA's terms wherever they appear
  (`train_ids.json`, `training_log.jsonl`, and `heldout_benchmark` in
  `metrics.json`). A backbone with no recorded license is refused.
- A dataset card (`README.md`) for the SEP calibration dataset repo. It covers
  the files and columns, the refit command, provenance, and the license and
  attribution: derived columns are Apache 2.0, and the questions are credited to
  TriviaQA (Joshi et al., 2017), whose terms users must check.
- `eval.compute_auroc --json-out` writes the report, plus every scored
  question's id, to a file.
- `eval.rescore_benchmark`: re-runs PE, SE and discrete SE on a saved
  results file and re-applies the escalation rule, with no model loaded;
  refuses SEP and files without saved token logprobs.
- Benchmark rows save each sample's token logprobs
  (`sample_token_logprobs`), and `DetectionResult` carries them.
- Local web playground (`uv run python -m demo.web`, http://127.0.0.1:8765): a
  FastAPI app with a committed React UI, one tab per way in, in the README's
  order — *Score answers* (meaning clusters, PE and a client-side escalation
  threshold); *Ask the model* (`detect` / `chat`); and *Measure the detectors*
  (benchmark AUROCs with CIs or a model comparison). New `web` optional extra
  (`fastapi`, `uvicorn`), included in `all`.
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
- `eval/calibrate_threshold.py`: chooses an escalation threshold on a dev
  slice by Youden's J, refuses a dev slice that overlaps the test slice, and
  reports the setting to use in `env_var` (`SE_THRESHOLD`, or `SEP_THRESHOLD`
  for SEP). `--detector` accepts only detectors with an escalation threshold
  (LLM judges are refused), and τ is printed unrounded. The default `1.0` is
  now documented as a placeholder.
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
  `eval/escalation_uplift.py` and `eval/adversarial_separation.py`.
- Statistical inference for the success criteria (`eval/inference_stats.py`):
  bootstrap 95% CIs for every AUROC, a paired-bootstrap CI for the SE-PE gap,
  McNemar's exact test and a paired CI for the escalation uplift, and the AUC
  effect size for adversarial separation.
- `load_triviaqa --offset` for disjoint dataset slices; the SEP probe
  directory holds `train_ids.json`, and `run_benchmark` refuses to score a
  probe on its own training questions, a probe without that file, or (with
  SEP enabled) a question without an `id`.
- `ResponseEvaluator.slm` and `.detectors` read-only properties;
  `SemanticEntropyProbeDetector.probe_dir`, `.source` and `.config`.
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
- The minimum pydantic is now 2.12 (was 2.7): untraced results leave
  `se_trace` out of their JSON through `Field(exclude_if=...)`, which older
  pydantic ignores.
- The playground's Ask tab runs discrete SE on the app's shared NLI scorer,
  as it already did for likelihood-weighted SE.
- The "Computed semantic entropy." log record replaces the scalar `se_nats`
  and `estimator` fields with `se_nats_by_estimator`, a map from estimator to
  SE in nats.
- README: the three ways in are renamed to one question each (*Score
  answers*, *Ask the model*, *Measure the detectors*), reordered from cheapest
  to most involved, and explain that way 1 contains a deliberate hallucination
  while way 3's wrong answers are the model's own mistakes.
- `eval.run_benchmark --in` and `eval.collect_sep_dataset --calibration-set` are
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
  the same position `HFTransformersProvider` reads; the embed runner serves
  last-layer SEP probes only, and agreement with the HF provider's vectors is
  unverified.
- **Breaking:** success criteria 2-4 now also need significance or effect
  size: the SE-PE gap's CI must exclude zero, the escalation uplift must pass
  McNemar's test (p < 0.05), and adversarial separation needs an AUC effect
  size of at least 0.75 besides p < 0.01.
- **Breaking:** `compute_auroc` reports `detectors.<name>.{auroc, ci}` instead
  of a flat `auroc` map, and the SE-PE gap is computed from unrounded AUROCs.
- **Breaking:** benchmark rows record `model_id`, `sample_n` and
  `sample_temperature`; the scoring scripts refuse files that mix runs, rows
  written without these fields, and adversarial/reference runs that differ.
- **Breaking:** SEP follows Kossen et al.'s classifier recipe: the probe is a
  logistic classifier on discrete SE split at the best-split point, and its
  score is p(high SE) in [0, 1], not SE in nats. Layers are chosen by a sweep
  (by default a 5-layer window chosen by cross-validation over all calibration
  rows, with a nested cross-validation AUROC that includes the cost of
  choosing) instead of fixed to the last one; deviations from the paper are
  listed in the README and in each probe's config.
- **Breaking:** a SEP probe is a directory (`probe.safetensors`,
  `probe_config.json`, `train_ids.json`, `training_log.jsonl`, `metrics.json`)
  that pins the backbone's model id, Hub revision, layer count, hidden size,
  answer instruction and chat template; `SEP_PROBE_PATH` takes a directory or
  `hf://<owner>/<repo>[@<revision>]`, and a mismatching probe is refused
  (`SEPProbeMismatchError`) before any sample.
- **Breaking:** training is two steps: `eval.collect_sep_dataset` (GPU) writes
  the calibration dataset, and `eval.train_sep_probe --dataset DIR` (CPU) fits
  from it alone, with `--layers auto|final|<a>-<b>` and the pre-registered
  `--max-iter 1000` / `--standardize` fallbacks for non-convergence.
- **Breaking:** the `SamplingProvider` protocol adds `model_revision`,
  `num_hidden_layers`, `hidden_size`, `chat_template_sha256` and
  `check_hidden_state_request(layers)`, and
  `pooled_hidden_state(prompt, layers)` returns one row per requested layer. A
  detector with `requires_hidden_state=True` must implement
  `HiddenStateDetector.hidden_state_layers`, or building the evaluator raises
  `TypeError`.
- **Breaking:** SEP on `SLM_PROVIDER=vllm` serves only last-layer probes and
  needs `VLLM_EMBED_BASE_URL`; both are checked when the evaluator is built
  (`UnsupportedHiddenStateRequest`), not at the first request.
- **Breaking:** benchmark rows record `sep_probe` (the probe's reference,
  resolved Hub revision and the SHA-256 of its weights and config), and the
  scoring scripts refuse results files without it; re-run `eval.run_benchmark`.
  A SEP run refuses a question without an `id` instead of skipping it.
- The web playground and `demo.cli` show SEP as a percentage, "P(high SE)",
  and `/api/status` reports `sep_threshold` and the escalation threshold in
  use.
- Registry entries pin each model's Hub revision, layer count and hidden
  size; both providers load that revision, and vLLM is launched with
  `--revision` and `--tokenizer-revision`.
- The NLI model is loaded at a pinned commit (`NLI_REVISION`); an explicit
  `NLIScorer(model_id=...)` must pass its revision.
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
- **Breaking:** Ridge `.pt` SEP probes, `--ridge-alpha` and the Pearson-r
  report. A `.pt` probe (or any file given as `SEP_PROBE_PATH`) is refused
  with `ObsoleteSEPProbeError` and a retrain hint; it is not converted.
- `ModelEntry.sep_probe_path`, which nothing read.
- `eval/vertex_genai_eval.py` and `EVAL_JUDGE_MODEL_ID`: it scored
  groundedness without any context to ground against, and its dependency
  pulled a `litellm` release with an open advisory.
- `autoawq` from the serving image (unmaintained, and unused by the registry's
  unquantized models).
- `serving/deploy_vertex.sh` — replaced by the Terraform path.

### Fixed
- Escalation with `ESCALATION_DETECTOR=semantic_entropy_probe` compared SEP's
  score against `SE_THRESHOLD`; it now uses `SEP_THRESHOLD`, and settings
  refuse escalating on SEP when SEP is not in `ENABLED_DETECTORS`.
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
