# Vertex AI Deploy Notes

## TOC

- [Summary](#summary)
- [Why a custom container on Vertex Prediction, not Model Garden?](#why-a-custom-container-on-vertex-prediction-not-model-garden)
- [What does Vertex require of the container?](#what-does-vertex-require-of-the-container)
- [Why the invoke route and dedicated endpoints?](#why-the-invoke-route-and-dedicated-endpoints)
- [Why one endpoint per vLLM runner?](#why-one-endpoint-per-vllm-runner)
- [How long does a cold start take?](#how-long-does-a-cold-start-take)

## Summary

The small model is served as a **custom vLLM container on Vertex AI
Prediction** because Vertex Model Garden's managed open-model endpoints do
not return token-level logprobs, and Predictive Entropy needs them. Each
model is deployed with the **invoke route** to a **dedicated endpoint**, so
the stock OpenAI client reaches vLLM's own paths. The step-by-step runbook is
in the [DevOps Guide](../usage_guides/DEVOPS_GUIDE.md); this page explains
the choices.

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

Two things, both satisfied by vLLM's OpenAI-compatible server as
`serving/launch.py` starts it:

- listen on `$AIP_HTTP_PORT` (default 8080);
- return 200 on the health route, `/health`.

There is no predict route: with the invoke route (below) Vertex forwards
requests by path instead.

## Why the invoke route and dedicated endpoints?

Because it is the only Vertex mode in which the client can talk to vLLM's
OpenAI API unchanged. The classic `predict` / `rawPredict` calls send every
request to a single container route, so one deployment could answer chat
completions *or* embeddings, and the OpenAI client could not call it
directly. A model uploaded with invoke route prefix `/*` instead has
`https://<dedicated DNS>/v1/projects/<number>/locations/<region>/endpoints/<id>/invoke/<path>`
forwarded to the container's `/<path>`, so `.../invoke/v1` works as an
OpenAI base URL.

The trade-offs: the invoke route is in Public Preview, it can only be used
with dedicated endpoints, and only the Vertex SDK or REST API can set it
(not `gcloud`, not Terraform) — hence `serving/vertex_deploy.py`.

## Why one endpoint per vLLM runner?

Because vLLM serves one runner per process, and a Vertex endpoint divides
traffic between its deployed models by percentage, not by path. The
`generate` runner (chat completions with raw logprobs) and the `embed` runner
(pooled last-token hidden states for SEP) therefore each get their own
container and their own endpoint. The embed endpoint is optional
(`with_embed_endpoint`) because it doubles the GPU cost.

## How long does a cold start take?

Expect several minutes for Vertex to pull the image, download the model
weights and warm vLLM; the full deploy step takes 10–20 minutes. A new
dedicated endpoint's DNS name can take longer still to resolve. The deploy
sets a minimum of one replica so the endpoint stays warm while it is up —
which is also why it must be torn down after a run.
