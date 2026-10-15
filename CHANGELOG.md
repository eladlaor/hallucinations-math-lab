# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Terraform configuration under `terraform/` for GCP resources (APIs, GCS,
  Artifact Registry, IAM, Vertex). The Vertex AI endpoint is gated behind the
  `deploy_vertex_endpoint` variable (default `false`) so the GPU node is only
  provisioned during active benchmark runs.
- `examples/langrag_integration.py` reference integration.
- Runnable no-GPU playground scripts: `examples/score_precollected_samples.py`
  (PE and SE on hand-written samples, string-match vs. DeBERTa NLI) and
  `examples/custom_detector.py` (a user-defined detector next to PE and SE).
- GitHub Actions CI running `ruff check` and the test suite on Python 3.12 and 3.13.
- `MIT` `LICENSE` file.
- QA Guide (`knowledge/usage_guides/QA_GUIDE.md`) and an end-to-end test of the
  benchmark → AUROC loop.

### Changed
- **Breaking:** the import package is renamed from `taste` to
  `hallucinations_math_lab`, matching the `hallucinations-math-lab` distribution.
- License changed from Apache-2.0 to MIT.
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
- Vertex serving is now end-to-end Terraform-managed: `vertex.tf` uploads the
  model, creates the endpoint, and deploys it through a `null_resource` whose
  destroy step undeploys cleanly.
- Machine type, accelerator type/count, and `model_id` are sourced from the
  Python model registry via `serving/model_flags.py <id> terraform`.
- README and all usage guides rewritten around using the repo as an
  experimentation playground; theory and caveat docs restructured as Q&A.
- Private talk-preparation and agent notes are no longer tracked.

### Removed
- `serving/deploy_vertex.sh` — replaced by the Terraform path.

### Fixed
- `eval/run_benchmark.py` imported the removed `DetectorAgent` and could not run.
- The library imported the unshipped `demo` package when loading settings,
  so `NLIScorer`, `GeminiJudge` and `ResponseEvaluator` failed outside the repo.
- `.env.example` set `ENABLED_DETECTORS` as a comma-separated list, which
  failed settings parsing at startup; it is now a JSON list.
- Predict route in `terraform/vertex.tf` corrected to `/v1/chat/completions`.
- Removed unused `torch.nn` import in `eval/train_sep_probe.py`.

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
