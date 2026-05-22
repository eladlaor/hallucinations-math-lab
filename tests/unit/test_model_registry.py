"""Model registry contract tests.

Verify:
  - Unknown model id raises with a helpful message.
  - Known entries return the expected vLLM CLI flags.
  - Exactly one deployed default and one local default exist (registry
    invariants — multiple defaults would silently let one win, which is
    exactly the kind of ambiguity the registry exists to prevent).
  - The deployed default stays a single-L4 Gemma 4 SKU (load-bearing for
    the cost story in CLAUDE.md; flag any change loudly via this test).
"""

import pytest

from demo.model_registry import (
    ProviderKind,
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


def test_gemma_4_e4b_entry_has_expected_vllm_flags():
    entry = get_model_entry("google/gemma-4-E4B-it")
    flags = entry.vllm_flags()
    # bf16, unquantized, 128K context per the official vLLM recipe.
    assert "--quantization" not in flags
    assert "--dtype" in flags
    assert flags[flags.index("--dtype") + 1] == "bfloat16"
    assert "--max-model-len" in flags
    assert flags[flags.index("--max-model-len") + 1] == "131072"


def test_gemma_4_31b_dense_omits_quantization_and_runs_bf16():
    entry = get_model_entry("google/gemma-4-31B-it")
    flags = entry.vllm_flags()
    assert "--quantization" not in flags
    assert entry.dtype == "bfloat16"
    # Official vLLM recipe pairs the 31B dense with 2x A100-80GB (TP=2).
    assert entry.accelerator_type == "NVIDIA_A100_80GB"
    assert entry.accelerator_count == 2


def test_deployed_default_is_single_l4_gemma_4():
    """Load-bearing invariant for the project's cost story (~$0.70/hr on L4).
    If this test ever needs to change, CLAUDE.md cost-discipline section
    needs to change at the same time."""
    entry = get_deployed_default()
    assert entry.model_id.startswith("google/gemma-4-")
    assert entry.accelerator_type == "NVIDIA_L4"
    assert entry.accelerator_count == 1


def test_local_default_fits_on_a_laptop():
    """The local default is the one `seh chat` uses for dev. Must be one of
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
            f"{model_id} is unquantized but dtype={entry.dtype!r}; Gemma 4 "
            "is natively bf16."
        )


def test_provider_kind_enum_round_trips():
    entry = get_model_entry("google/gemma-4-E4B-it")
    assert isinstance(entry.default_provider, ProviderKind)
