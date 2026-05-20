# Developer Guide

## Table of Contents

- [Summary](#summary)
- [Local Setup](#local-setup)
- [Running the SE pipeline locally](#running-the-se-pipeline-locally)
- [Layout](#layout)
- [Coding conventions](#coding-conventions)

## Summary

Get from zero to a passing `pytest` and a `seh detect` call against a local vLLM
server in 10 minutes. End-to-end (with Vertex) is in `DEVOPS_GUIDE.md`.

## Local Setup

```bash
uv sync --extra dev
cp .env.example .env
# edit .env: set GCP_PROJECT_ID (for judge), or leave blank to skip the judge step
uv run pytest tests/unit -v
```

## Running the SE pipeline locally

1. **Bring up vLLM** (separate machine with an L4 / RunPod / etc; not the dev laptop):
   ```bash
   docker build -t vllm-gemma2 serving/
   docker run --gpus all -p 8080:8080 vllm-gemma2
   ```
2. **Point the client at it** — set `VLLM_BASE_URL=http://<host>:8080/v1` in `.env`.
3. **Detect**:
   ```bash
   uv run seh detect "Who wrote 1984?"
   uv run seh chat "Who was the first woman to walk on Mars?"
   ```

## Layout

```
src/seh_handler/
  constants.py             # frozen enums + literals
  config.py                # Pydantic Settings
  custom_types/            # api_schemas.py, domain_types.py
  services/
    vllm_client.py         # OpenAI-compat vLLM client (n=10, logprobs)
    predictive_entropy.py  # token-logprob baseline
    semantic_entropy.py    # NLI clustering + SE
    nli.py                 # DeBERTa-MNLI wrapper
    judge.py               # Vertex Gemini judge
  agents/
    detector_agent.py
    judge_agent.py
  main.py                  # Typer CLI
serving/                   # vLLM Dockerfile + Vertex deploy script
eval/                      # benchmark scripts, datasets
tests/unit/                # entropy + clustering math tests
```

## Coding conventions

- Python 3.12, `uv` for deps, src layout.
- Constants in `constants.py` (use `StrEnum` for closed string sets).
- All configurable values in `config.py` (Pydantic Settings, env-driven).
- Fail-fast on unexpected errors; log with `extra={"ctx": {...}}`.
- One responsibility per module; no "utils.py".
