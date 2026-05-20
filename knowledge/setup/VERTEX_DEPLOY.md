# Vertex AI Deploy Notes

## TOC
- [Summary](#summary)
- [Why Vertex Prediction (not Model Garden)](#why-vertex-prediction-not-model-garden)
- [Custom container contract](#custom-container-contract)
- [Cold-start mitigation](#cold-start-mitigation)

## Summary

We use a **custom-container deployment on Vertex AI Prediction** because Vertex AI
Model Garden's managed Gemma endpoints **do not expose token-level logprobs**,
which the Semantic Entropy detector requires.

## Why Vertex Prediction (not Model Garden)

| Option | Logprobs | Effort | Notes |
|---|---|---|---|
| Model Garden one-click Gemma | **No** | Lowest | Logprobs feature is only on the Gemini API, not on open-model endpoints |
| GKE + vLLM | Yes | High | Official Google tutorial; idiomatic GKE story |
| **Vertex AI Prediction custom container** | **Yes** | Medium | What we chose |
| GCE VM + vLLM | Yes | Lowest | Cheapest, but not a "Vertex" story |

## Custom container contract

Vertex AI Prediction expects the container to:
- Listen on `$AIP_HTTP_PORT` (default 8080).
- Respond 200 on `GET $AIP_HEALTH_ROUTE` (default `/health`).
- Serve predictions at `$AIP_PREDICT_ROUTE`.

vLLM's OpenAI-compatible server satisfies all three out of the box on port 8000;
the Dockerfile rebinds to 8080 and points `AIP_PREDICT_ROUTE` at
`/v1/chat/completions`.

## Cold-start mitigation

A Gemma 2 9B AWQ-INT4 image pulls + warms in ~5 minutes on L4. For interactive
demos set `--min-replica-count=1` (already done in `deploy_vertex.sh`).
