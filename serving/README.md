# Serving

vLLM OpenAI-compatible server hosting a registry-selected Gemma SLM, deployed
to Vertex AI Prediction as a custom container.

## Files

- `Dockerfile` — CUDA 12.4 + vLLM ≥0.5.4. Exposes the OpenAI-compatible API on
  `:8080`. Reads `MODEL_ID` at start and resolves the vLLM launch flags from
  the model registry via `model_flags.py`. Optionally launches a sibling
  embed-task vLLM when `VLLM_EMBED_PORT` is set (SEP support).
- `cloudbuild.yaml` — Cloud Build config used to build and push the image to
  Artifact Registry.
- `model_flags.py` — registry shim. `vllm` mode prints the launch flags for
  the container CMD; `terraform` mode prints a JSON blob the Terraform stack
  consumes via `terraform/model_registry.auto.tfvars.json`.

## Quick test (local, before pushing to Vertex)

```bash
docker build -t vllm-gemma -f serving/Dockerfile .
docker run --gpus all -p 8080:8080 -e MODEL_ID=google/gemma-4-E4B-it vllm-gemma

# In another shell:
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E4B-it",
       "messages": [{"role": "user", "content": "Who wrote 1984?"}],
       "n": 10, "temperature": 1.0, "logprobs": true, "top_logprobs": 5}'
```

## Deploy

Deployment is fully Terraform-driven. See
`knowledge/usage_guides/DEVOPS_GUIDE.md` for the end-to-end runbook
(build → push → `terraform apply -var deploy_vertex_endpoint=true`).
