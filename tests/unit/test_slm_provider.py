"""SLMProvider tests.

VLLMOpenAIProvider is exercised with a mocked OpenAI client so the tests run
without a live vLLM server. HFTransformersProvider needs an actual model
checkpoint + a GPU and is marked `requires_gpu`, skipped by default.

Contract pinned here:
  - `pooled_hidden_state` returns None when VLLM_EMBED_BASE_URL is unset.
  - When the embed endpoint is configured, the provider hits /v1/embeddings
    and returns a float32 tensor of shape [1, hidden]; it serves only the
    final layer and refuses any other layer request.
  - The HF provider follows the layer convention: row k of its all-layer
    extraction is HF's boolean `output_hidden_states` entry k (0 = embeddings,
    L = last layer, final norm applied), read at the last prompt token (TBG).
  - `sample` extracts text + token logprobs from chat.completions response.
  - The model sees the brief-answer instruction plus the question on every
    path (sample, greedy, embed); `SampleSet.prompt` stays the bare question,
    which is what SE's NLI step is conditioned on.
  - `build_default_provider` selects the right class from config.slm_provider.
  - `raw_token_logprobs` (the HF provider's logprob math) uses the raw
    distribution, cuts each sample at its first EOS/PAD, and is model-free so
    it runs on CPU.
"""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BatchEncoding, GenerationConfig, GenerationMixin

from demo.model_registry import get_model_entry
from demo.slm_provider import (
    HFTransformersProvider,
    VLLMOpenAIProvider,
    build_answer_message,
    build_default_provider,
    raw_token_logprobs,
    stop_token_ids,
    sweep_model_ids,
)
from hallucinations_math_lab.services.sampling_provider import (
    UnsupportedHiddenStateRequest,
    final_norm_applied,
)

EOS_ID = 0
# google/gemma-4-E4B-it's decoder depth; the vLLM fixtures select that model.
E4B_NUM_LAYERS = 42
VOCAB = 4


def _step(rows: list[list[float]]) -> torch.Tensor:
    """One generation step's logits, shape [n, vocab]."""
    return torch.tensor(rows, dtype=torch.float32)


def test_raw_token_logprobs_matches_log_softmax_of_raw_logits():
    logits = (_step([[0.0, 2.0, 1.0, 0.5]]), _step([[0.0, 0.1, 0.2, 3.0]]))
    generated = torch.tensor([[1, 3]])

    [pairs] = raw_token_logprobs(logits, generated, stop_token_ids={EOS_ID})

    expected = [
        (1, torch.log_softmax(logits[0][0], dim=-1)[1].item()),
        (3, torch.log_softmax(logits[1][0], dim=-1)[3].item()),
    ]
    assert [tok for tok, _ in pairs] == [tok for tok, _ in expected]
    for (_, got), (_, want) in zip(pairs, expected, strict=True):
        assert math.isclose(got, want, rel_tol=1e-6)


def test_raw_token_logprobs_keeps_first_stop_token_and_drops_padding():
    """Sample 0 ends early and is padded with EOS. Its own EOS is part of the
    sequence likelihood; the padding after it must not count."""
    logits = tuple(_step([[1.0] * VOCAB, [1.0] * VOCAB]) for _ in range(4))
    generated = torch.tensor([[2, EOS_ID, EOS_ID, EOS_ID], [1, 2, 3, 3]])

    short, full = raw_token_logprobs(logits, generated, stop_token_ids={EOS_ID})

    assert [tok for tok, _ in short] == [2, EOS_ID]
    assert [tok for tok, _ in full] == [1, 2, 3, 3]


def test_raw_token_logprobs_rejects_step_count_mismatch():
    with pytest.raises(ValueError, match="must match one-to-one"):
        raw_token_logprobs((_step([[1.0] * VOCAB]),), torch.tensor([[1, 2]]), {EOS_ID})


def test_stop_token_ids_collects_eos_list_and_pad():
    config = GenerationConfig(eos_token_id=[1, 106], pad_token_id=0)
    assert stop_token_ids(config) == {0, 1, 106}


def test_stop_token_ids_fails_fast_without_eos_or_pad():
    config = GenerationConfig(eos_token_id=None, pad_token_id=None)
    with pytest.raises(ValueError, match="neither eos_token_id nor pad_token_id"):
        stop_token_ids(config)


@pytest.fixture
def gemma_settings(monkeypatch):
    """Force settings to a known shape: vLLM provider, Gemma 4 E4B, no embed."""
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", "google/gemma-4-E4B-it")
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", "")
    monkeypatch.setenv("VLLM_EMBED_MODEL_ID", "")
    yield
    get_settings.cache_clear()


@pytest.fixture
def gemma_settings_with_embed(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", "google/gemma-4-E4B-it")
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", "http://localhost:8081/v1")
    monkeypatch.setenv("VLLM_EMBED_MODEL_ID", "")
    yield
    get_settings.cache_clear()


def test_vllm_provider_returns_none_hidden_state_without_embed_endpoint(gemma_settings):
    with patch("demo.slm_provider.OpenAI"):
        provider = VLLMOpenAIProvider()
    assert provider.pooled_hidden_state("anything", layers=(E4B_NUM_LAYERS,)) is None


@pytest.mark.parametrize("layers", [(E4B_NUM_LAYERS - 1,), (E4B_NUM_LAYERS - 1, E4B_NUM_LAYERS)])
def test_vllm_provider_returns_none_without_embed_endpoint_for_any_layers(gemma_settings, layers):
    """No embed endpoint means no hidden states at all: None, as the protocol
    says, not a layer refusal."""
    with patch("demo.slm_provider.OpenAI"):
        provider = VLLMOpenAIProvider()
    assert provider.pooled_hidden_state("anything", layers=layers) is None


def test_vllm_provider_returns_tensor_when_embed_endpoint_configured(
    gemma_settings_with_embed,
):
    fake_embedding_data = MagicMock()
    fake_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4]
    fake_resp = MagicMock(data=[fake_embedding_data])

    with patch("demo.slm_provider.OpenAI") as openai_cls:
        # Two OpenAI clients are created (generation + embed). The embed one
        # is returned by the second call.
        generation_client = MagicMock()
        embed_client = MagicMock()
        embed_client.post.return_value = fake_resp
        openai_cls.side_effect = [generation_client, embed_client]

        provider = VLLMOpenAIProvider()
        result = provider.pooled_hidden_state("anything", layers=(E4B_NUM_LAYERS,))

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.tolist() == [[pytest.approx(x) for x in [0.1, 0.2, 0.3, 0.4]]]
    embed_client.post.assert_called_once()


def test_vllm_embedding_request_is_chat_templated_last_token_unnormalized(
    gemma_settings_with_embed,
):
    """Chat template + generation prompt, last token, unnormalized: the request
    that corresponds to HF's layer index L at TBG. The body carries no revision:
    the pin is in the server's launch flags."""
    with patch("demo.slm_provider.OpenAI") as openai_cls:
        embed_client = MagicMock()
        embed_client.post.return_value = MagicMock(data=[MagicMock(embedding=[1.0])])
        openai_cls.side_effect = [MagicMock(), embed_client]
        VLLMOpenAIProvider().pooled_hidden_state("Who wrote Hamlet?", layers=(E4B_NUM_LAYERS,))

    path = embed_client.post.call_args.args[0]
    kwargs = embed_client.post.call_args.kwargs
    assert path == "/embeddings"
    assert kwargs["body"] == {
        "model": "google/gemma-4-E4B-it",
        "messages": [{"role": "user", "content": build_answer_message("Who wrote Hamlet?")}],
        "add_generation_prompt": True,
        "use_activation": False,
    }
    # Without this the client would skip refreshing a Google access token.
    assert kwargs["options"] == {"security": {"bearer_auth": True}}


def test_vllm_provider_reads_depth_from_the_registry(gemma_settings):
    with patch("demo.slm_provider.OpenAI"):
        assert VLLMOpenAIProvider().num_hidden_layers == E4B_NUM_LAYERS


@pytest.mark.parametrize("embed_base_url", ["", "http://localhost:8081/v1"])
def test_vllm_depth_of_an_unregistered_model_fails_fast_naming_the_registry(
    monkeypatch, embed_base_url
):
    """L comes from the registry, so SEP on vLLM needs a registry entry, with or
    without an embed endpoint: the evaluator reads L before asking for states."""
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", "acme/unregistered-slm")
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", embed_base_url)
    monkeypatch.setenv("VLLM_EMBED_MODEL_ID", "")
    try:
        with patch("demo.slm_provider.OpenAI"):
            provider = VLLMOpenAIProvider()
        with pytest.raises(KeyError, match="not in the model registry"):
            _ = provider.num_hidden_layers
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("layers", [(E4B_NUM_LAYERS - 1,), (E4B_NUM_LAYERS - 1, E4B_NUM_LAYERS)])
def test_vllm_refuses_layers_other_than_the_final_one(gemma_settings_with_embed, layers):
    """The embed runner pools only the last layer; any other request must fail
    before a network call, naming the remedy."""
    with patch("demo.slm_provider.OpenAI") as openai_cls:
        embed_client = MagicMock()
        openai_cls.side_effect = [MagicMock(), embed_client]
        provider = VLLMOpenAIProvider()
        with pytest.raises(UnsupportedHiddenStateRequest, match="SLM_PROVIDER=hf_transformers"):
            provider.pooled_hidden_state("q", layers=layers)
    embed_client.post.assert_not_called()


HF_NUM_LAYERS = 3
HF_HIDDEN_SIZE = 2
PROMPT_TOKENS = 4
# The fake model writes `layer + position * POSITION_SCALE` into every hidden
# value, so a read-back shows both which layer and which token it came from.
POSITION_SCALE = 0.01
TBG_POSITION = PROMPT_TOKENS - 1


class _FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> BatchEncoding:
        return BatchEncoding(
            {
                "input_ids": torch.arange(PROMPT_TOKENS).unsqueeze(0),
                "attention_mask": torch.ones(1, PROMPT_TOKENS, dtype=torch.long),
            }
        )


class _FakeHFModel:
    """Stands in for a loaded causal LM's forward pass.

    Returns `HF_NUM_LAYERS + 1 + extra_states` hidden-state tensors, entry k
    filled with `k + position * POSITION_SCALE`; `calls` records the kwargs."""

    def __init__(self, extra_states: int = 0) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(num_hidden_layers=HF_NUM_LAYERS)
        )
        self.calls: list[dict[str, Any]] = []
        self._n_states = HF_NUM_LAYERS + 1 + extra_states

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        seq_len = kwargs["input_ids"].shape[1]
        positions = torch.arange(seq_len, dtype=torch.bfloat16).view(1, seq_len, 1)
        states = tuple(
            (k + positions * POSITION_SCALE).expand(1, seq_len, HF_HIDDEN_SIZE).clone()
            for k in range(self._n_states)
        )
        return SimpleNamespace(hidden_states=states)


def _hf_provider(model: _FakeHFModel) -> HFTransformersProvider:
    provider = HFTransformersProvider(model_id="google/gemma-4-E2B-it")
    fakes: Any = provider
    fakes._tokenizer, fakes._model = _FakeTokenizer(), model
    return provider


def _expected_row(layer: int, position: int = TBG_POSITION) -> list[float]:
    """The fake model's value for (layer, position), with the same bf16 rounding."""
    value = (layer + torch.tensor(position, dtype=torch.bfloat16) * POSITION_SCALE).item()
    return [value] * HF_HIDDEN_SIZE


def test_hf_hidden_states_index_k_is_after_k_layers():
    """Row k is HF's boolean `output_hidden_states` entry k: 0 = embeddings,
    L = after the last decoder layer."""
    rows = _hf_provider(_FakeHFModel()).hidden_states_all_layers("q")

    assert rows.shape == (HF_NUM_LAYERS + 1, HF_HIDDEN_SIZE)
    assert rows.dtype == torch.float32
    assert rows.device.type == "cpu"
    assert rows.tolist() == [_expected_row(k) for k in range(HF_NUM_LAYERS + 1)]


def test_hf_rows_move_to_cpu_float32_before_stacking(monkeypatch):
    """A model sharded with device_map="auto" keeps each layer's state on its
    own device; stacking them before the move would raise."""
    real_stack = torch.stack
    stacked: list[list[torch.Tensor]] = []

    def spy_stack(tensors: list[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        stacked.append(list(tensors))
        return real_stack(tensors, *args, **kwargs)

    model = _FakeHFModel()
    provider = _hf_provider(model)
    monkeypatch.setattr(torch, "stack", spy_stack)
    provider.hidden_states_all_layers("q")

    [rows] = stacked
    assert all(row.device.type == "cpu" and row.dtype == torch.float32 for row in rows)


def test_layer_L_flagged_final_norm():
    flags = [final_norm_applied(k, HF_NUM_LAYERS) for k in range(HF_NUM_LAYERS + 1)]
    assert flags == [False] * HF_NUM_LAYERS + [True]


@pytest.mark.parametrize("layer", [-1, HF_NUM_LAYERS + 1])
def test_final_norm_applied_rejects_out_of_range_layer(layer):
    with pytest.raises(ValueError, match="outside"):
        final_norm_applied(layer, HF_NUM_LAYERS)


def test_output_hidden_states_is_boolean_true():
    """The boolean form includes the embeddings at index 0; the list form
    would shift every index by one."""
    model = _FakeHFModel()
    _hf_provider(model).hidden_states_all_layers("q")

    [kwargs] = model.calls
    assert kwargs["output_hidden_states"] is True


def test_wrong_tuple_length_raises():
    provider = _hf_provider(_FakeHFModel(extra_states=1))
    with pytest.raises(RuntimeError, match=f"expected {HF_NUM_LAYERS + 1}"):
        provider.hidden_states_all_layers("q")


def test_pooled_hidden_state_stacks_requested_layers():
    pooled = _hf_provider(_FakeHFModel()).pooled_hidden_state("q", layers=(1, 3))

    assert pooled.shape == (2, HF_HIDDEN_SIZE)
    assert pooled.dtype == torch.float32
    assert pooled.tolist() == [_expected_row(1), _expected_row(3)]


@pytest.mark.parametrize("layers", [(), (HF_NUM_LAYERS + 1,), (-1,), (2, 1), (1, 1)])
def test_pooled_hidden_state_rejects_invalid_layers(layers):
    with pytest.raises(ValueError, match="layers"):
        _hf_provider(_FakeHFModel()).pooled_hidden_state("q", layers=layers)


def test_tbg_is_last_prompt_token():
    """TBG reads the last token of the chat-formatted prompt with the
    generation prompt appended: no answer token is involved."""
    last_layer = _hf_provider(_FakeHFModel()).hidden_states_all_layers("q")[HF_NUM_LAYERS]
    assert last_layer.tolist() == _expected_row(HF_NUM_LAYERS, PROMPT_TOKENS - 1)
    assert last_layer.tolist() != _expected_row(HF_NUM_LAYERS, PROMPT_TOKENS - 2)


def test_hf_provider_reads_depth_from_the_loaded_config():
    assert _hf_provider(_FakeHFModel()).num_hidden_layers == HF_NUM_LAYERS


E2B_ID = "google/gemma-4-E2B-it"


class _LoadedLM(GenerationMixin):
    """What `AutoModelForCausalLM.from_pretrained` hands back, reduced to the
    attributes the loader reads."""

    def __init__(self, num_hidden_layers: int, hidden_size: int) -> None:
        self.device = torch.device("cpu")
        text_config = SimpleNamespace(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.config = SimpleNamespace(get_text_config=lambda: text_config)

    def eval(self) -> "_LoadedLM":
        return self


def _load(monkeypatch, model: _LoadedLM) -> tuple[MagicMock, MagicMock]:
    """Run the HF provider's loader against spies; return (tokenizer, model) spies."""
    import transformers

    tokenizer_spy = MagicMock(return_value=_FakeTokenizer())
    model_spy = MagicMock(return_value=model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_spy)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_spy)
    _ = HFTransformersProvider(model_id=E2B_ID).num_hidden_layers  # triggers the load
    return tokenizer_spy, model_spy


def _registry_sized_model() -> _LoadedLM:
    entry = get_model_entry(E2B_ID)
    return _LoadedLM(entry.num_hidden_layers, entry.hidden_size)


def test_hf_from_pretrained_called_with_revision(monkeypatch):
    """Weights and tokenizer (chat template included) come from the pinned commit."""
    tokenizer_spy, model_spy = _load(monkeypatch, _registry_sized_model())

    revision = get_model_entry(E2B_ID).revision
    assert tokenizer_spy.call_args.args == (E2B_ID,)
    assert tokenizer_spy.call_args.kwargs["revision"] == revision
    assert model_spy.call_args.args == (E2B_ID,)
    assert model_spy.call_args.kwargs["revision"] == revision


@pytest.mark.parametrize(("extra_layers", "extra_width"), [(1, 0), (0, 1)])
def test_loaded_dimensions_must_match_registry(monkeypatch, extra_layers, extra_width):
    """The registry's depth and width are what probes are checked against, so
    a checkpoint that disagrees with them must not load."""
    entry = get_model_entry(E2B_ID)
    model = _LoadedLM(entry.num_hidden_layers + extra_layers, entry.hidden_size + extra_width)
    with pytest.raises(ValueError, match="model registry"):
        _load(monkeypatch, model)


def test_model_revision_from_registry(gemma_settings):
    with patch("demo.slm_provider.OpenAI"):
        vllm_provider = VLLMOpenAIProvider()
    hf_provider = HFTransformersProvider(model_id=E2B_ID)

    assert vllm_provider.model_revision == get_model_entry(vllm_provider.model_id).revision
    # Read without loading the model.
    assert hf_provider.model_revision == get_model_entry(E2B_ID).revision
    assert hf_provider._model is None


def test_embed_endpoint_for_another_model_fails_fast(gemma_settings_with_embed):
    """One embed endpoint serves one model; a sweep entry for a different
    model must not read that model's hidden states."""
    with (
        patch("demo.slm_provider.OpenAI"),
        pytest.raises(ValueError, match="SEP needs hidden states from the same model"),
    ):
        VLLMOpenAIProvider(model_id="google/gemma-4-31B-it")


def test_google_auth_mode_passes_a_token_provider_to_both_clients(monkeypatch):
    from demo.google_auth import GoogleAccessToken
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_AUTH", "google")
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", "https://embed.example/v1")
    try:
        with patch("demo.slm_provider.OpenAI") as openai_cls:
            VLLMOpenAIProvider()
        keys = [call.kwargs["api_key"] for call in openai_cls.call_args_list]
    finally:
        get_settings.cache_clear()

    assert len(keys) == 2
    assert all(isinstance(key, GoogleAccessToken) for key in keys)


def test_api_key_auth_mode_sends_the_static_key(gemma_settings, monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("VLLM_API_KEY", "secret-for-test")
    try:
        with patch("demo.slm_provider.OpenAI") as openai_cls:
            VLLMOpenAIProvider()
    finally:
        get_settings.cache_clear()
    assert openai_cls.call_args.kwargs["api_key"] == "secret-for-test"


def test_vllm_provider_sample_extracts_text_and_logprobs(gemma_settings):
    """The provider should map OpenAI-compat response choices to Samples
    with per-token logprobs preserved."""
    fake_token = MagicMock(token="Paris", logprob=-0.1)
    fake_logprobs = MagicMock(content=[fake_token])
    fake_message = MagicMock(content="Paris.")
    fake_choice = MagicMock(message=fake_message, logprobs=fake_logprobs)
    fake_resp = MagicMock(choices=[fake_choice, fake_choice])

    with patch("demo.slm_provider.OpenAI") as openai_cls:
        generation_client = MagicMock()
        generation_client.chat.completions.create.return_value = fake_resp
        openai_cls.return_value = generation_client

        provider = VLLMOpenAIProvider()
        sample_set = provider.sample("What is the capital of France?")

    assert len(sample_set.samples) == 2
    assert sample_set.samples[0].text == "Paris."
    assert sample_set.samples[0].token_logprobs[0].token == "Paris"


def test_build_default_provider_selects_vllm(gemma_settings):
    with patch("demo.slm_provider.OpenAI"):
        provider = build_default_provider()
    assert isinstance(provider, VLLMOpenAIProvider)


def test_build_default_provider_selects_hf_transformers(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "hf_transformers")
    monkeypatch.setenv("VLLM_MODEL_ID", "google/gemma-4-E2B-it")
    try:
        provider = build_default_provider()
        assert isinstance(provider, HFTransformersProvider)
        # Construction must be lazy — model id is set, but no weights loaded yet.
        assert provider._model is None
    finally:
        get_settings.cache_clear()


def test_build_default_provider_rejects_unknown_kind(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "imaginary")
    try:
        with pytest.raises(ValueError, match="Unknown SLM_PROVIDER"):
            build_default_provider()
    finally:
        get_settings.cache_clear()


def test_sweep_model_ids_rejects_unregistered_model(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("TARGET_MODEL_IDS", '["google/gemma-4-E4B-it", "acme/not-a-real-model"]')
    try:
        with pytest.raises(ValueError, match="not in the model registry"):
            sweep_model_ids()
    finally:
        get_settings.cache_clear()


@pytest.mark.requires_gpu
def test_hf_provider_end_to_end_on_tiny_model():
    """Sanity check that HFTransformersProvider runs end-to-end on a small
    real model. Skipped unless a CUDA device is present — this test exists
    to catch shape/decode bugs that mocks can't, not for routine CI."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA-capable device.")
    # Intentionally not asserting specifics — this is a smoke test. Run
    # manually with: uv run pytest -m requires_gpu tests/unit/test_slm_provider.py
    provider = HFTransformersProvider()
    provider.sample("hello")
    provider.pooled_hidden_state("hello", layers=(provider.num_hidden_layers,))


def test_answer_message_puts_the_instruction_before_the_question(monkeypatch):
    from hallucinations_math_lab.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ANSWER_INSTRUCTION", "Answer in one word.")
    try:
        message = build_answer_message("Who wrote Hamlet?")
    finally:
        get_settings.cache_clear()
    assert message == "Answer in one word.\nQuestion: Who wrote Hamlet?"


def test_answer_instruction_must_not_be_empty(monkeypatch):
    from pydantic import ValidationError

    from hallucinations_math_lab.config import Settings

    monkeypatch.setenv("ANSWER_INSTRUCTION", "")
    with pytest.raises(ValidationError):
        Settings()


def test_vllm_sample_and_greedy_send_the_instruction_but_keep_the_bare_prompt(gemma_settings):
    fake_choice = MagicMock(message=MagicMock(content="Paris"), logprobs=None)
    with patch("demo.slm_provider.OpenAI") as openai_cls:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(choices=[fake_choice])
        openai_cls.return_value = client
        provider = VLLMOpenAIProvider()
        sample_set = provider.sample("Capital of France?")
        provider.greedy("Capital of France?")

    expected = [{"role": "user", "content": build_answer_message("Capital of France?")}]
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["messages"] == expected
    assert sample_set.prompt == "Capital of France?"
