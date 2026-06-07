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

### Changed
- Public API carved into response-shaped and provider-shaped surfaces;
  `DetectorAgent` retired in favor of the new entry points.
- Project renamed to `taste`; demo code carved out of `src/` into a top-level
  `demo/` package, exposed via the `taste` CLI script.
- Detector pipeline made pluggable; SLM provider is now swappable. SEP
  promoted to a first-class in-scope detector alongside predictive entropy
  and Semantic Entropy.

### Changed
- Vertex serving is now end-to-end Terraform-managed. The `vertex.tf` stack
  uploads the model, creates the endpoint, and uses a `null_resource` to call
  `gcloud ai endpoints deploy-model` (closing a gap in the Google provider
  for custom-container deployed models). The destroy provisioner undeploys
  cleanly so a single `terraform apply -var deploy_vertex_endpoint=false`
  tears the stack down.
- Machine type, accelerator type/count, and `model_id` are sourced from the
  Python model registry via `serving/model_flags.py <id> terraform`, which
  writes `terraform/model_registry.auto.tfvars.json`. The Python registry
  remains the single source of truth.
- Predict route in `terraform/vertex.tf` corrected to `/v1/chat/completions`
  to match the serving container.

### Removed
- `serving/deploy_vertex.sh` — replaced by the Terraform path; image build
  remains a separate `gcloud builds submit` step using
  `serving/cloudbuild.yaml`.

### Fixed
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

[Unreleased]: https://github.com/eladlaor/taste/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eladlaor/taste/releases/tag/v0.1.0
