# Serving

## Table of contents

- [Summary](#summary)
- [Files](#files)
- [Quick test (local, before pushing to Vertex)](#quick-test-local-before-pushing-to-vertex)
- [Deploy](#deploy)

## Summary

A vLLM OpenAI-compatible server for any model in the registry
(`demo/model_registry.py`), packaged as a custom container for Vertex AI
Prediction or any GPU host. **Before building:** the image pins
`vllm>=0.5.4,<0.7`, which predates Gemma 4 — raise it to a release that
supports Gemma 4 (see the
[DevOps Guide known gaps](../knowledge/usage_guides/DEVOPS_GUIDE.md#known-gaps-before-you-deploy)).

## Files

- `Dockerfile` — CUDA 12.4 + vLLM. Serves the OpenAI-compatible API on
  `:8080`. Reads `MODEL_ID` at start and gets the vLLM launch flags from the
  registry via `model_flags.py`. When `VLLM_EMBED_PORT` is set it also starts
  a second vLLM process for embeddings, which SEP uses as its hidden state.
  Files are copied under `/opt/taste` inside the image.
- `cloudbuild.yaml` — Cloud Build config that builds and pushes the image to
  Artifact Registry.
- `model_flags.py` — registry shim. `vllm` mode prints the launch flags for
  the container; `terraform` mode prints the JSON written to
  `terraform/model_registry.auto.tfvars.json`.

## Quick test (local, before pushing to Vertex)

```bash
docker build -t vllm-gemma -f serving/Dockerfile .
docker run --gpus all -p 8080:8080 -e MODEL_ID=google/gemma-4-E4B-it vllm-gemma

# In another shell — 10 samples with logprobs, what the detectors consume:
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E4B-it",
       "messages": [{"role": "user", "content": "Who wrote 1984?"}],
       "n": 10, "temperature": 1.0, "logprobs": true, "top_logprobs": 5}'
```

## Deploy

Deployment is Terraform-driven. Follow the
[DevOps Guide](../knowledge/usage_guides/DEVOPS_GUIDE.md) (build → push →
`terraform apply -var deploy_vertex_endpoint=true`).
