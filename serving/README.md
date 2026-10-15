# Serving

## Table of contents

- [Summary](#summary)
- [Files](#files)
- [Quick test (local, before pushing to Vertex)](#quick-test-local-before-pushing-to-vertex)
- [Deploy](#deploy)

## Summary

A vLLM OpenAI-compatible server for any model in the registry
(`demo/model_registry.py`), packaged as a custom container for Vertex AI
Prediction or any GPU host. The image is built on
`vllm/vllm-openai:v0.31.0-cu129` (vLLM 0.19 was the first release with Gemma 4
support). Each container serves one model (`MODEL_ID`) with one vLLM runner
(`VLLM_RUNNER`): `generate` for chat completions with raw logprobs (PE, SE),
or `embed` for unnormalized last-token hidden states (SEP).

## Files

- `Dockerfile` — the image. Copies only the model registry and the launcher
  under `/opt/app`; the library itself is not installed in it.
- `launch.py` — container entrypoint. Builds the `vllm serve` argument list
  from the registry for `MODEL_ID` and `VLLM_RUNNER`, binds `$AIP_HTTP_PORT`
  (default 8080) with `VLLM_GPU_MEMORY_UTILIZATION` (default 0.90), and
  `exec`s vLLM. No shell quoting is involved, so the embed runner's JSON
  pooler config passes through intact.
- `model_flags.py` — registry CLI. `<id> vllm {generate|embed}` prints a
  paste-able `vllm serve ...` command for your own GPU box; `<id> terraform`
  prints the JSON written to `terraform/model_registry.auto.tfvars.json`.
- `vertex_deploy.py` — uploads the image to Vertex with invoke route prefix
  `/*` and deploys it to a dedicated endpoint (or undeploys and deletes it).
  Terraform runs it via `uv run --extra deploy`.
- `cloudbuild.yaml` — Cloud Build config that builds and pushes the image to
  Artifact Registry.

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

The embed runner (`-e VLLM_RUNNER=embed`) answers the request SEP sends —
the chat-templated prompt, pooled at its last token, unnormalized:

```bash
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E4B-it",
       "messages": [{"role": "user", "content": "Who wrote 1984?"}],
       "add_generation_prompt": true, "use_activation": false}'
```

## Deploy

Deployment is Terraform-driven. Follow the
[DevOps Guide](../knowledge/usage_guides/DEVOPS_GUIDE.md) (build → push →
`terraform apply -var deploy_vertex_endpoint=true`), and the
[Vertex deploy notes](../knowledge/setup/VERTEX_DEPLOY.md) for why it is
shaped this way.
