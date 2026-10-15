"""Model registry contract tests.

Verify:
  - Unknown model id raises with a helpful message.
  - Known entries return the expected vLLM CLI flags.
  - Exactly one deployed default and one local default exist (registry
    invariants — multiple defaults would silently let one win, which is
    exactly the kind of ambiguity the registry exists to prevent).
  - The deployed default stays a single-L4 Gemma 4 SKU (load-bearing for
    the DevOps Guide cost figures; flag any change loudly via this test).
  - vLLM flags are correct per runner (generate vs embed) and per GPU count.
"""

import json

import pytest

from demo.model_registry import (
    ProviderKind,
    ServingRunner,
    get_deployed_default,
    get_local_default,
    get_model_entry,
    list_model_ids,
)


_GEMMA4_IDS = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]


def test_unknown_model_id_raises_with_helpful_message():
    with pytest.raises(KeyError, match="not in the model registry"):
        get_model_entry("acme/imaginary-model")


def _flag_value(flags: list[str], name: str) -> str:
    assert name in flags, f"{name} missing from {flags}"
    return flags[flags.index(name) + 1]


def test_gemma_4_e4b_generate_flags():
    flags = get_model_entry("google/gemma-4-E4B-it").vllm_flags(ServingRunner.GENERATE)
    assert "--quantization" not in flags
    assert _flag_value(flags, "--dtype") == "bfloat16"
    assert _flag_value(flags, "--tensor-parallel-size") == "1"
    assert "--language-model-only" in flags
    # PE must see the model's own distribution, not the temperature-scaled one.
    assert _flag_value(flags, "--logprobs-mode") == "raw_logprobs"
    assert "--runner" not in flags


def test_embed_runner_pools_last_token_without_normalization():
    """SEP probes are trained on raw last-token hidden states (HF provider);
    the vLLM embed runner must return the same thing, not a unit vector."""
    flags = get_model_entry("google/gemma-4-E4B-it").vllm_flags(ServingRunner.EMBED)
    assert _flag_value(flags, "--runner") == "pooling"
    assert _flag_value(flags, "--convert") == "embed"
    pooler = json.loads(_flag_value(flags, "--pooler-config"))
    assert pooler == {"seq_pooling_type": "LAST", "use_activation": False}
    assert "--logprobs-mode" not in flags


def test_model_id_is_not_a_flag():
    """`vllm serve` takes the model positionally; the Dockerfile supplies it."""
    for runner in ServingRunner:
        assert "--model" not in get_model_entry("google/gemma-4-E4B-it").vllm_flags(runner)


@pytest.mark.parametrize("model_id", _GEMMA4_IDS)
def test_tensor_parallelism_matches_gpu_count(model_id):
    """A multi-GPU entry that doesn't shard the model leaves GPUs idle and the
    model may not fit on one of them."""
    entry = get_model_entry(model_id)
    flags = entry.vllm_flags(ServingRunner.GENERATE)
    assert _flag_value(flags, "--tensor-parallel-size") == str(entry.accelerator_count)


@pytest.mark.parametrize("model_id", _GEMMA4_IDS)
def test_l4_entries_cap_context_to_fit_the_kv_cache(model_id):
    entry = get_model_entry(model_id)
    if entry.accelerator_type == "NVIDIA_L4":
        assert entry.max_model_len <= 8192


def test_gemma_4_31b_dense_omits_quantization_and_runs_bf16():
    entry = get_model_entry("google/gemma-4-31B-it")
    flags = entry.vllm_flags(ServingRunner.GENERATE)
    assert "--quantization" not in flags
    assert entry.dtype == "bfloat16"
    # Official vLLM recipe pairs the 31B dense with 2x A100-80GB (TP=2).
    assert entry.accelerator_type == "NVIDIA_A100_80GB"
    assert entry.accelerator_count == 2


def test_deployed_default_is_single_l4_gemma_4():
    """Load-bearing invariant for the cost figures in the DevOps Guide
    (~$0.70/hr on one L4). Change both together."""
    entry = get_deployed_default()
    assert entry.model_id.startswith("google/gemma-4-")
    assert entry.accelerator_type == "NVIDIA_L4"
    assert entry.accelerator_count == 1


def test_local_default_fits_on_a_laptop():
    """The local default is the one `python -m demo.cli chat` uses for dev. Must be one of
    the small Gemma 4 SKUs (E2B or E4B) so it actually runs on developer
    hardware."""
    entry = get_local_default()
    assert entry.model_id in {"google/gemma-4-E2B-it", "google/gemma-4-E4B-it"}


def test_registry_lists_all_four_gemma_4_skus():
    ids = list_model_ids()
    for mid in _GEMMA4_IDS:
        assert mid in ids, f"missing {mid} from registry"


@pytest.mark.parametrize("model_id", _GEMMA4_IDS)
def test_every_gemma_4_entry_runs_bf16_native(model_id):
    """Gemma 4 ships natively in bf16. If a registry entry ever flips to
    fp16/auto without an explicit quantization, that's a misconfiguration
    worth catching at test time rather than at first vLLM launch."""
    entry = get_model_entry(model_id)
    if entry.quantization is None:
        assert entry.dtype == "bfloat16", (
            f"{model_id} is unquantized but dtype={entry.dtype!r}; Gemma 4 is natively bf16."
        )


def test_provider_kind_enum_round_trips():
    entry = get_model_entry("google/gemma-4-E4B-it")
    assert isinstance(entry.default_provider, ProviderKind)
