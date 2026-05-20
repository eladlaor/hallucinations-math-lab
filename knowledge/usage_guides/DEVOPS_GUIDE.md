# DevOps Guide

## Table of Contents

- [Summary](#summary)
- [GCP setup](#gcp-setup)
- [Deploying the vLLM endpoint](#deploying-the-vllm-endpoint)
- [Running the benchmark](#running-the-benchmark)
- [Tearing down](#tearing-down)

## Summary

Deploy `gemma-2-9b-it-AWQ-INT4` to a Vertex AI Endpoint on a single L4, point the
detector at the endpoint, run the benchmark, then tear the endpoint down (L4 on
Vertex is ~$0.70/hr; keep it on only when needed).

## GCP setup

```bash
gcloud auth login
gcloud config set project ${GCP_PROJECT_ID}
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

## Deploying the vLLM endpoint

See `serving/README.md`. Headline command:

```bash
export GCP_PROJECT_ID=...
bash serving/deploy_vertex.sh
```

Then set `VLLM_BASE_URL` in `.env` to the endpoint's predict URL (or front it with
a small proxy that translates the Vertex predict request shape to OpenAI-chat).

## Running the benchmark

```bash
uv run python eval/datasets/load_triviaqa.py --n 1000
uv run python eval/run_benchmark.py --in eval/datasets/triviaqa_subset.jsonl \
                                    --out eval/results/triviaqa.jsonl
uv run python eval/compute_auroc.py --in eval/results/triviaqa.jsonl
uv run python eval/vertex_genai_eval.py --in eval/results/triviaqa.jsonl
```

## Tearing down

```bash
gcloud ai endpoints undeploy-model "${ENDPOINT_ID}" --region="${GCP_REGION}" --deployed-model-id=<id>
gcloud ai endpoints delete "${ENDPOINT_ID}" --region="${GCP_REGION}"
```

Skipping teardown leaves an L4 burning ~$17/day.
