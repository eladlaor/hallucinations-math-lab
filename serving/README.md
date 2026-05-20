# Serving

vLLM OpenAI-compatible server hosting `hugging-quants/gemma-2-9b-it-AWQ-INT4`,
deployed to Vertex AI Prediction as a custom container.

## Files

- `Dockerfile` — CUDA 12.4 + vLLM ≥0.5.4 image. Exposes the OpenAI-compatible API on `:8080`.
- `deploy_vertex.sh` — builds the image, uploads to Vertex Model Registry, deploys to an Endpoint on a single L4 GPU.

## Quick test (local, before pushing to Vertex)

```bash
docker build -t vllm-gemma2 serving/
docker run --gpus all -p 8080:8080 vllm-gemma2

# In another shell:
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "hugging-quants/gemma-2-9b-it-AWQ-INT4",
       "messages": [{"role": "user", "content": "Who wrote 1984?"}],
       "n": 10, "temperature": 1.0, "logprobs": true, "top_logprobs": 5}'
```

## Deploy

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
bash serving/deploy_vertex.sh
```
