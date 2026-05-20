# semantic-entropy-hallucination-handler

A chatbot that detects when its small model is likely hallucinating, using
**Semantic Entropy** (Farquhar et al., *Nature* 2024), and escalates uncertain
questions to a larger judge model.

Stack:

- **Small model**: `hugging-quants/gemma-2-9b-it-AWQ-INT4`, served via **vLLM** as
  a custom container on **Vertex AI Prediction** (single L4 GPU).
- **Detector**: Semantic Entropy over N=10 samples (T=1.0), clustered by
  bidirectional NLI entailment using `microsoft/deberta-large-mnli`. A predictive
  (token-logprob) entropy baseline runs alongside for comparison.
- **Judge**: **Gemini 2.5 Flash** via Vertex AI, invoked only when SE exceeds the
  threshold.
- **Orchestration**: **Google ADK** — Detector agent + Judge agent.
- **Evaluation**: **Vertex Gen AI Evaluation Service** for groundedness and
  pairwise comparison; AUROC on TriviaQA-1000 + a small adversarial set.

## Why both predictive entropy and Semantic Entropy?

They measure different things:

- *Predictive entropy* — variance in the model's per-token probabilities for a
  single sample. Cheap, but only catches lexical uncertainty.
- *Semantic Entropy* — N samples at T>0, clustered by **meaning** (bidirectional
  NLI), then Shannon entropy over cluster sizes. Catches the case where the model
  produces 10 confidently-worded contradictory answers.

The repo computes both, and the eval harness reports AUROC for each — the headline
result is the gap between them.

## Status

Phase 1 scaffold. SE math, vLLM client, ADK agents, eval harness, and Vertex
deploy script are in place. Cloud deployment and end-to-end benchmark pending.

## Layout

See `knowledge/usage_guides/DEVELOPER_GUIDE.md`.
