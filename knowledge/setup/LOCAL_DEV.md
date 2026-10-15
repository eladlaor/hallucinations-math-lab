# Local Development

## TOC

- [Summary](#summary)
- [Without a GPU](#without-a-gpu)
- [With a local model](#with-a-local-model)
- [With a GPU on a remote box](#with-a-gpu-on-a-remote-box)

## Summary

You do not need a GPU to develop here: the math, the clustering, the
examples and the whole test suite run on CPU. A GPU only matters when you
want a real small model to generate answers, either in-process
(`SLM_PROVIDER=hf_transformers`) or through a vLLM server
(`SLM_PROVIDER=vllm`).

## Without a GPU

```bash
uv sync --all-extras
cp .env.example .env
uv run pytest -q
uv run python examples/score_precollected_samples.py              # string-equality NLI, instant
uv run python examples/score_precollected_samples.py --nli deberta # real NLI on CPU, ~1.6 GB download once
```

## With a local model

The default `.env` loads `google/gemma-4-E2B-it` in-process through
`transformers` (on CUDA if available, otherwise CPU):

```bash
uv run python -m demo.cli detect "Who wrote 1984?"
```

The first call downloads the model. On CPU expect a slow first answer; use
`SAMPLE_N` and `MAX_TOKENS` in `.env` to keep experiments small.

## With a GPU on a remote box

On any machine with an NVIDIA GPU and the NVIDIA container runtime:

```bash
docker build -t vllm-gemma -f serving/Dockerfile .
docker run --gpus all -p 8080:8080 -e MODEL_ID=google/gemma-4-E4B-it vllm-gemma
```

Then point your local `.env` at it: `SLM_PROVIDER=vllm`,
`VLLM_AUTH=api_key`, `VLLM_BASE_URL=http://<host>:8080/v1`,
`VLLM_MODEL_ID=google/gemma-4-E4B-it`. For SEP's hidden states, run a second
container with `-e VLLM_RUNNER=embed` and set `VLLM_EMBED_BASE_URL` — see
[Running vLLM on your own GPU box](../usage_guides/DEVOPS_GUIDE.md#running-vllm-on-your-own-gpu-box-instead).
