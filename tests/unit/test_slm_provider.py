"""SLMProvider tests.

VLLMOpenAIProvider is exercised with a mocked OpenAI client so the tests run
without a live vLLM server. HFTransformersProvider needs an actual model
checkpoint + a GPU and is marked `requires_gpu`, skipped by default.

Contract pinned here:
  - `pooled_hidden_state` returns None when VLLM_EMBED_BASE_URL is unset.
  - When the embed endpoint is configured, the provider hits /v1/embeddings
    and returns a float32 tensor.
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
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import GenerationConfig

from demo.slm_provider import (
    HFTransformersProvider,
    VLLMOpenAIProvider,
    build_answer_message,
    build_default_provider,
    raw_token_logprobs,
    stop_token_ids,
    sweep_model_ids,
)

EOS_ID = 0
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
    assert provider.pooled_hidden_state("anything") is None


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
        result = provider.pooled_hidden_state("anything")

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.tolist() == [pytest.approx(x) for x in [0.1, 0.2, 0.3, 0.4]]
    embed_client.post.assert_called_once()


def test_vllm_embedding_request_matches_the_hf_hidden_state(gemma_settings_with_embed):
    """Chat template + generation prompt, unnormalized: the same feature
    HFTransformersProvider.pooled_hidden_state returns, so SEP probes transfer."""
    with patch("demo.slm_provider.OpenAI") as openai_cls:
        embed_client = MagicMock()
        embed_client.post.return_value = MagicMock(data=[MagicMock(embedding=[1.0])])
        openai_cls.side_effect = [MagicMock(), embed_client]
        VLLMOpenAIProvider().pooled_hidden_state("Who wrote Hamlet?")

    path = embed_client.post.call_args.args[0]
    kwargs = embed_client.post.call_args.kwargs
    assert path == "/embeddings"
    assert kwargs["body"]["messages"] == [
        {"role": "user", "content": build_answer_message("Who wrote Hamlet?")}
    ]
    assert kwargs["body"]["add_generation_prompt"] is True
    assert kwargs["body"]["use_activation"] is False
    # Without this the client would skip refreshing a Google access token.
    assert kwargs["options"] == {"security": {"bearer_auth": True}}


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
    provider.pooled_hidden_state("hello")


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
