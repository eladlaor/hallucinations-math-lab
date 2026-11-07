"""Model registry — single source of truth for SLM-specific serving and provider config.

Each entry pins everything that varies per SLM:
  - vLLM launch flags: quantization, dtype, max_model_len, tensor
    parallelism, and the per-runner flags (generate vs embed).
  - Deployment hints: required GPU SKU, accelerator count, recommended
    `accelerator_type` value for Vertex (NVIDIA_L4 / NVIDIA_A100 / etc).
  - Default provider for the SLMProvider abstraction (vllm vs hf_transformers).
  - SEP probe weight path under data/probes/, when one has been trained.

Two distinct concepts both deserve to be defaults here, and they're not the
same default:

  - `is_deployed_default = True` on one entry. That's what the Vertex
    deployment uses if nothing overrides it. Today: Gemma-4-E4B on L4 — the
    cost figures in the DevOps Guide (~$0.70/hr) depend on staying single-L4.

  - `is_local_default = True` on one entry. That's what single-model
    commands (`python -m demo.cli chat`) pick up when no .env override is present, for
    local development. Today: Gemma-4-E2B via HF transformers — small
    enough for laptop dev.

Mixing these into one "default" field would force one of them into a lie.
Keeping them separate is the SRP-correct split.

The *sweep* (which models the evaluation harness compares against each
other) is a separate concept, lives in `config.target_model_ids`, and is
orthogonal to both defaults above. Every id in the sweep must be present
in this registry, but the sweep is not derived from the default flags.

The registry is consumed:
  - In Python by `VLLMOpenAIProvider` and `HFTransformersProvider` to pick
    runtime parameters by `model_id`.
  - In shell by `serving/Dockerfile` via the `serving/model_flags.py` CLI
    shim, which prints the vLLM engine flags for a given `MODEL_ID` and
    runner.
  - In Terraform by `serving/model_flags.py <id> terraform`, which writes
    `terraform/model_registry.auto.tfvars.json` so the deployed-model machine
    type, accelerator, and `model_id` come from this registry rather than
    drifting in HCL.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ProviderKind(StrEnum):
    VLLM = "vllm"
    HF_TRANSFORMERS = "hf_transformers"


class ServingRunner(StrEnum):
    """What a vLLM server process does with the model.

    vLLM serves one runner per process: `generate` answers chat completions
    (samples + logprobs for PE / SE), `embed` answers `/v1/embeddings` with a
    pooled hidden state (for SEP). Each runs in its own container.
    """

    GENERATE = "generate"
    EMBED = "embed"


# The embed runner pools the final-layer output at the final prompt token,
# meant to match what `HFTransformersProvider.pooled_hidden_state` returns for
# layers=(L,). It is expected to carry the final norm, as HF's index L does;
# that, and whether its vectors match HF's within tolerance, is unverified.
# It L2-normalizes by default; `use_activation: false` turns that off.
_EMBED_POOLER_CONFIG: Final[str] = '{"seq_pooling_type":"LAST","use_activation":false}'


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    quantization: str | None  # vLLM --quantization flag value; None = unquantized
    dtype: str  # vLLM --dtype flag value (float16, bfloat16, auto, ...)
    max_model_len: int  # vLLM --max-model-len flag value
    accelerator_type: str  # Vertex accelerator type (NVIDIA_L4, NVIDIA_A100_80GB, ...)
    accelerator_count: int  # how many of them
    machine_type: str  # GCE machine type that pairs with the accelerator
    default_provider: ProviderKind
    # Decoder depth L (text_config.num_hidden_layers in the checkpoint's
    # config.json): the index of the last layer under the layer convention in
    # hallucinations_math_lab.services.sampling_provider.
    num_hidden_layers: int
    is_deployed_default: bool = False
    is_local_default: bool = False
    sep_probe_path: str | None = None  # relative to repo root, once a probe is trained

    def vllm_flags(self, runner: ServingRunner) -> list[str]:
        """`vllm serve <model_id>` engine flags for this entry and runner.

        The model id is positional to `vllm serve` and host/port belong to
        the container, so neither is included here.
        """
        flags = [
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--tensor-parallel-size",
            str(self.accelerator_count),
            # Gemma 4 is multimodal; this lab is text-only. Skipping the
            # vision/audio towers frees their memory and startup profiling.
            "--language-model-only",
        ]
        if self.quantization:
            flags += ["--quantization", self.quantization]
        if runner is ServingRunner.GENERATE:
            # Logprobs of the model's own distribution, before temperature /
            # top-p. Predictive entropy must not depend on sampler settings.
            flags += ["--logprobs-mode", "raw_logprobs"]
        else:
            flags += [
                "--runner",
                "pooling",
                "--convert",
                "embed",
                "--pooler-config",
                _EMBED_POOLER_CONFIG,
            ]
        return flags


# Gemma 4 family (released 2026-04-02). Values cross-checked against the
# official vLLM recipe at docs.vllm.ai/projects/recipes/.../Gemma4.html and
# the HF blog post huggingface.co/blog/gemma4. Native dtype is bf16 across
# the family. The 26B-A4B MoE has a known vLLM MXFP4/NVFP4 weight-loading
# bug in the fused MoE layer; we therefore keep it unquantized on A100-80GB
# until that ships. The 31B dense official recipe uses TP=2.
#
# max_model_len is sized to the GPU, not to the model's 128K maximum: on a
# 24 GB L4 the bf16 weights leave room for only a few GB of KV cache, and vLLM
# refuses to start if the configured context cannot fit. Short-answer QA
# prompts plus SAMPLE_N completions need far less than 8K tokens.
_L4_MAX_MODEL_LEN: Final[int] = 8192
_REGISTRY: Final[dict[str, ModelEntry]] = {
    "google/gemma-4-E2B-it": ModelEntry(
        model_id="google/gemma-4-E2B-it",
        quantization=None,
        dtype="bfloat16",
        max_model_len=_L4_MAX_MODEL_LEN,
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        machine_type="g2-standard-12",
        default_provider=ProviderKind.HF_TRANSFORMERS,
        num_hidden_layers=35,
        is_local_default=True,
    ),
    "google/gemma-4-E4B-it": ModelEntry(
        model_id="google/gemma-4-E4B-it",
        quantization=None,
        dtype="bfloat16",
        max_model_len=_L4_MAX_MODEL_LEN,
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        machine_type="g2-standard-12",
        default_provider=ProviderKind.VLLM,
        num_hidden_layers=42,
        is_deployed_default=True,
    ),
    "google/gemma-4-26B-A4B-it": ModelEntry(
        model_id="google/gemma-4-26B-A4B-it",
        quantization=None,
        dtype="bfloat16",
        max_model_len=32768,
        accelerator_type="NVIDIA_A100_80GB",
        accelerator_count=1,
        machine_type="a2-ultragpu-1g",
        default_provider=ProviderKind.VLLM,
        num_hidden_layers=30,
    ),
    "google/gemma-4-31B-it": ModelEntry(
        model_id="google/gemma-4-31B-it",
        quantization=None,
        dtype="bfloat16",
        max_model_len=32768,
        accelerator_type="NVIDIA_A100_80GB",
        accelerator_count=2,
        machine_type="a2-ultragpu-2g",
        default_provider=ProviderKind.VLLM,
        num_hidden_layers=60,
    ),
}


def get_model_entry(model_id: str) -> ModelEntry:
    """Fail fast on an unregistered model id rather than guessing flags."""
    if model_id not in _REGISTRY:
        raise KeyError(
            f"Model {model_id!r} is not in the model registry. Known: {sorted(_REGISTRY)!r}. "
            "Add an entry in model_registry.py before using a new SLM."
        )
    return _REGISTRY[model_id]


def list_model_ids() -> list[str]:
    return sorted(_REGISTRY)


def get_deployed_default() -> ModelEntry:
    matches = [e for e in _REGISTRY.values() if e.is_deployed_default]
    if len(matches) != 1:
        raise RuntimeError(
            f"Model registry must have exactly one is_deployed_default; found {len(matches)}."
        )
    return matches[0]


def get_local_default() -> ModelEntry:
    matches = [e for e in _REGISTRY.values() if e.is_local_default]
    if len(matches) != 1:
        raise RuntimeError(
            f"Model registry must have exactly one is_local_default; found {len(matches)}."
        )
    return matches[0]
