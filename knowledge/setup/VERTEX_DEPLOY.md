# Vertex AI Deploy Notes

## TOC

- [Summary](#summary)
- [Why a custom container on Vertex Prediction, not Model Garden?](#why-a-custom-container-on-vertex-prediction-not-model-garden)
- [What does Vertex require of the container?](#what-does-vertex-require-of-the-container)
- [How long does a cold start take?](#how-long-does-a-cold-start-take)

## Summary

The small model is served as a **custom vLLM container on Vertex AI
Prediction** because Vertex Model Garden's managed open-model endpoints do
not return token-level logprobs, and Predictive Entropy needs them. The
step-by-step deploy runbook is in the
[DevOps Guide](../usage_guides/DEVOPS_GUIDE.md); this page explains the
choices.

## Why a custom container on Vertex Prediction, not Model Garden?

Because it is the cheapest managed option that still returns per-token
logprobs.

| Option | Logprobs | Effort | Notes |
|---|---|---|---|
| Model Garden one-click Gemma | **No** | lowest | logprobs are only on the Gemini API, not on open-model endpoints |
| GKE + vLLM | yes | high | Google's documented path; you run the cluster |
| **Vertex Prediction custom container** | **yes** | medium | what this repo uses |
| Plain VM + vLLM | yes | low | cheapest, but nothing managed |

## What does Vertex require of the container?

Three things, all satisfied by vLLM's OpenAI-compatible server once the
Dockerfile binds it to the right port:

- listen on `$AIP_HTTP_PORT` (default 8080);
- return 200 on `GET $AIP_HEALTH_ROUTE` (default `/health`);
- serve predictions at `$AIP_PREDICT_ROUTE` — the Dockerfile points it at
  `/v1/chat/completions`.

## How long does a cold start take?

Expect several minutes for Vertex to pull the image, download the model
weights and warm vLLM; the full deploy step takes 10–20 minutes. The
Terraform deploy sets a minimum of one replica so the endpoint stays warm
while it is up — which is also why it must be torn down after a run.
