"""SLMProvider tests.

VLLMOpenAIProvider is exercised with a mocked OpenAI client so the tests run
without a live vLLM server. HFTransformersProvider needs an actual model
checkpoint + a GPU and is marked `requires_gpu`, skipped by default.

Contract pinned here:
  - `pooled_hidden_state` returns None when VLLM_EMBED_BASE_URL is unset.
  - When the embed endpoint is configured, the provider hits /v1/embeddings
    and returns a float32 tensor.
  - `sample` extracts text + token logprobs from chat.completions response.
  - `build_default_provider` selects the right class from config.slm_provider.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from demo.slm_provider import (
    HFTransformersProvider,
    VLLMOpenAIProvider,
    build_default_provider,
)


@pytest.fixture
def gemma_settings(monkeypatch):
    """Force settings to a known shape: vLLM provider, Gemma 4 E4B, no embed."""
    from taste.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL_ID", "google/gemma-4-E4B-it")
    monkeypatch.setenv("VLLM_EMBED_BASE_URL", "")
    monkeypatch.setenv("VLLM_EMBED_MODEL_ID", "")
    yield
    get_settings.cache_clear()


@pytest.fixture
def gemma_settings_with_embed(monkeypatch):
    from taste.config import get_settings

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
        embed_client.embeddings.create.return_value = fake_resp
        openai_cls.side_effect = [generation_client, embed_client]

        provider = VLLMOpenAIProvider()
        result = provider.pooled_hidden_state("anything")

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.tolist() == [pytest.approx(x) for x in [0.1, 0.2, 0.3, 0.4]]
    embed_client.embeddings.create.assert_called_once()


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
    from taste.config import get_settings

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
    from taste.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("SLM_PROVIDER", "imaginary")
    try:
        with pytest.raises(ValueError, match="Unknown SLM_PROVIDER"):
            build_default_provider()
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
