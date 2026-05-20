# Local Development

## TOC
- [Summary](#summary)
- [Without a GPU](#without-a-gpu)
- [With a GPU on a remote box](#with-a-gpu-on-a-remote-box)

## Summary

You don't need a local GPU to develop on this repo. The SE math, NLI clustering,
and unit tests all run on CPU. To exercise the end-to-end path you need vLLM
running somewhere reachable.

## Without a GPU

```bash
uv sync --extra dev
uv run pytest tests/unit -v
```

That validates the entropy math and clustering logic with a fake NLI scorer.
For NLI on real text against `deberta-large-mnli`, CPU works (slow): you can do
short manual smokes by writing a small script that calls `NLIScorer().classify`.

## With a GPU on a remote box

Option A — RunPod L4 (cheapest, ad-hoc):
```bash
# in the orchestrator repo, use the gpu-training-run skill to provision an L4
# then on the pod:
docker run --gpus all -p 8080:8080 vllm-gemma2
```

Option B — local CUDA workstation: same `docker run` command.

In either case point `VLLM_BASE_URL` at the host and run `seh detect ...`.
