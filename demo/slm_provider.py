"""SLMProvider protocol.

The "small language model" — the four Gemma 4 SKUs (E2B, E4B, 26B A4B,
31B dense) by default, others by adding registry entries — is consumed
only through this protocol. The detector orchestrator never touches vLLM
or HuggingFace directly.

Two responsibilities, kept separate by SRP:
  - `sample(...)` returns N stochastic completions with per-token logprobs.
    Feeds SE (semantic clustering) and PE (token-level entropy).
  - `pooled_hidden_state(...)` returns a single pooled hidden-state vector for
    the prompt. Feeds SEP (linear probe). Returns None when the provider does
    not expose hidden states — for example, plain vLLM serving in
    generation-only mode without a sibling embed endpoint. In that case SEP
    cannot run; the orchestrator surfaces it as `SEP: n/a` rather than
    failing.

Concrete implementations live in this module:
  - `VLLMOpenAIProvider` — OpenAI-compat HTTP client. Hidden states come from
    a sibling vLLM instance launched with `--task embed`, accessed via the
    `/v1/embeddings` endpoint when `VLLM_EMBED_BASE_URL` is configured.
  - `HFTransformersProvider` — local HuggingFace `transformers` load. Lands
    in Chunk 4; declared here only as a forward reference.
"""

import logging
from typing import Protocol, runtime_checkable

import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from taste.config import get_settings
from taste.constants import LOG_EXTRA_KEY
from taste.custom_types.domain_types import Sample, SampleSet, TokenLogprob

logger = logging.getLogger(__name__)


@runtime_checkable
class SLMProvider(Protocol):
    """Capability protocol for the small language model.

    Implementations advertise the prompt-time generation backbone the
    detector pipeline runs against. Hidden-state availability is provider-
    specific: a provider that cannot expose hidden states returns `None`
    from `pooled_hidden_state`, and SEP gracefully reports `n/a`."""

    model_id: str

    def sample(self, prompt: str) -> SampleSet: ...
    def greedy(self, prompt: str) -> str: ...
    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None: ...


class VLLMOpenAIProvider:
    """vLLM behind the OpenAI-compat HTTP API.

    Two endpoints are addressed:
      - The generation endpoint at `VLLM_BASE_URL` (chat/completions).
      - Optionally, a sibling embed endpoint at `VLLM_EMBED_BASE_URL`. When
        set, that endpoint is a separate vLLM process launched with
        `--task embed` and is hit via `/v1/embeddings` to return a pooled
        hidden state per prompt. When unset, `pooled_hidden_state` returns
        `None`."""

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._generation = OpenAI(
            base_url=base_url or settings.vllm_base_url,
            api_key=settings.vllm_api_key,
        )
        self.model_id: str = model_id or settings.vllm_model_id
        self._n = settings.sample_n
        self._temperature = settings.sample_temperature
        self._top_logprobs = settings.top_logprobs
        self._max_tokens = settings.max_tokens

        self._embed_client: OpenAI | None
        if settings.vllm_embed_base_url:
            self._embed_client = OpenAI(
                base_url=settings.vllm_embed_base_url,
                api_key=settings.vllm_api_key,
            )
            self._embed_model_id = settings.vllm_embed_model_id or settings.vllm_model_id
        else:
            self._embed_client = None
            self._embed_model_id = ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def sample(self, prompt: str) -> SampleSet:
        try:
            resp = self._generation.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                n=self._n,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                logprobs=True,
                top_logprobs=self._top_logprobs,
            )
        except Exception as e:
            raise RuntimeError(
                f"VLLMOpenAIProvider.sample failed for prompt={prompt[:80]!r} "
                f"model={self.model_id!r} n={self._n}: {e!s}"
            ) from e

        samples: list[Sample] = []
        for ch in resp.choices:
            text = ch.message.content or ""
            token_logprobs: list[TokenLogprob] = []
            if ch.logprobs and ch.logprobs.content:
                for tlp in ch.logprobs.content:
                    token_logprobs.append(TokenLogprob(token=tlp.token, logprob=tlp.logprob))
            samples.append(Sample(text=text, token_logprobs=token_logprobs))

        logger.info(
            "Sampled from vLLM.",
            extra={
                LOG_EXTRA_KEY: {
                    "n": len(samples),
                    "model": self.model_id,
                    "prompt_preview": prompt[:80],
                }
            },
        )
        return SampleSet(prompt=prompt, samples=samples)

    def greedy(self, prompt: str) -> str:
        try:
            resp = self._generation.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(
                f"VLLMOpenAIProvider.greedy failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e

    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None:
        if self._embed_client is None:
            return None
        try:
            resp = self._embed_client.embeddings.create(
                model=self._embed_model_id,
                input=prompt,
            )
        except Exception as e:
            raise RuntimeError(
                f"VLLMOpenAIProvider.pooled_hidden_state failed for prompt={prompt[:80]!r} "
                f"embed_model={self._embed_model_id!r}: {e!s}"
            ) from e

        if not resp.data:
            raise RuntimeError(
                "vLLM embed endpoint returned no data; cannot extract pooled hidden state."
            )
        vector = resp.data[0].embedding
        return torch.tensor(vector, dtype=torch.float32)


class HFTransformersProvider:
    """Local HuggingFace `transformers` provider.

    Hosts the SLM in-process. Supports both generation (with per-token
    logprobs reconstructed from `output_scores`) and pooled hidden state
    (separate forward pass with `output_hidden_states=True`, take the
    last-layer hidden state at the final prompt token).

    Two forward passes per query (one for sampling, one for the pooled
    hidden state) is intentional — keeps responsibilities separate. The
    HF provider exists for dev and research; production throughput goes
    through `VLLMOpenAIProvider`. If you need single-pass HF dispatch
    later, a `Detector`-aware fast path can read hidden states from the
    sampling forward via a forward hook.

    Construction is lazy on the model so importing this module does not
    pull a multi-billion-param checkpoint into memory."""

    def __init__(self, model_id: str | None = None) -> None:
        settings = get_settings()
        self.model_id: str = model_id or settings.vllm_model_id
        self._n = settings.sample_n
        self._temperature = settings.sample_temperature
        self._max_tokens = settings.max_tokens
        self._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so test suites that never instantiate this provider
        # don't pay the transformers import cost.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from demo.model_registry import get_model_entry

        entry = get_model_entry(self.model_id)
        torch_dtype = (
            torch.bfloat16 if entry.dtype == "bfloat16"
            else torch.float16 if entry.dtype == "float16"
            else "auto"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._model.eval()

    def sample(self, prompt: str) -> SampleSet:
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self._temperature,
                    num_return_sequences=self._n,
                    max_new_tokens=self._max_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.sample failed for prompt={prompt[:80]!r} "
                f"model={self.model_id!r} n={self._n}: {e!s}"
            ) from e

        prompt_len = inputs["input_ids"].shape[1]
        sequences = out.sequences  # [n, prompt_len + new_tokens]
        # scores is a tuple of length new_tokens, each tensor [n, vocab].
        # Reshape to [n, new_tokens, vocab] for vectorized logprob extraction.
        stacked_scores = torch.stack(out.scores, dim=1)  # [n, new, vocab]
        log_probs = torch.log_softmax(stacked_scores, dim=-1)
        chosen_token_ids = sequences[:, prompt_len:]  # [n, new]
        chosen_logprobs = log_probs.gather(
            dim=-1, index=chosen_token_ids.unsqueeze(-1)
        ).squeeze(-1)  # [n, new]

        samples: list[Sample] = []
        for i in range(sequences.shape[0]):
            new_token_ids = chosen_token_ids[i].tolist()
            token_logprobs: list[TokenLogprob] = []
            for tok_id, lp in zip(new_token_ids, chosen_logprobs[i].tolist()):
                token = self._tokenizer.decode([tok_id], skip_special_tokens=True)
                token_logprobs.append(TokenLogprob(token=token, logprob=float(lp)))
            text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
            samples.append(Sample(text=text, token_logprobs=token_logprobs))

        logger.info(
            "Sampled from HF transformers.",
            extra={
                LOG_EXTRA_KEY: {
                    "n": len(samples),
                    "model": self.model_id,
                    "prompt_preview": prompt[:80],
                }
            },
        )
        return SampleSet(prompt=prompt, samples=samples)

    def greedy(self, prompt: str) -> str:
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    do_sample=False,
                    num_return_sequences=1,
                    max_new_tokens=self._max_tokens,
                )
            prompt_len = inputs["input_ids"].shape[1]
            new_token_ids = out[0, prompt_len:].tolist()
            return self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.greedy failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e

    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None:
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model(**inputs, output_hidden_states=True)
            # Last-layer hidden state at the final prompt token. Matches the
            # Kossen et al. SEP setup.
            last_layer = out.hidden_states[-1]  # [1, seq_len, hidden_dim]
            return last_layer[0, -1].detach().to(dtype=torch.float32, device="cpu")
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.pooled_hidden_state failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e


def build_default_provider(model_id: str | None = None) -> SLMProvider:
    """Construct the SLMProvider selected by config.slm_provider.

    When `model_id` is None the active model from config.vllm_model_id is
    used (single-model commands like `seh chat`). When provided, the same
    config.slm_provider kind is built but bound to the requested model id —
    this is the entry point the evaluation sweep uses to iterate over
    config.target_model_ids.

    For VLLMOpenAIProvider, the per-model base_url override (if any) is
    pulled from config.target_model_base_urls; missing -> falls back to
    config.vllm_base_url.
    """
    settings = get_settings()
    kind = settings.slm_provider
    if kind == "vllm":
        base_url_override: str | None = None
        if model_id is not None:
            base_url_override = settings.target_model_base_urls.get(model_id)
        return VLLMOpenAIProvider(model_id=model_id, base_url=base_url_override)
    if kind == "hf_transformers":
        return HFTransformersProvider(model_id=model_id)
    raise ValueError(
        f"Unknown SLM_PROVIDER={kind!r}. Expected one of: vllm, hf_transformers."
    )


def build_sweep_providers() -> list[SLMProvider]:
    """Construct one SLMProvider per id in config.target_model_ids.

    The evaluation harness iterates the returned list, running the full
    detector pipeline once per provider so results are directly comparable
    across the target SLMs.
    """
    settings = get_settings()
    return [build_default_provider(model_id=mid) for mid in settings.target_model_ids]
