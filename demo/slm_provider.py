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
    a second vLLM server running the embed runner, accessed via
    `/v1/embeddings` when `VLLM_EMBED_BASE_URL` is configured.
  - `HFTransformersProvider` — in-process HuggingFace `transformers` model.
    Exposes hidden states directly, so SEP needs no extra endpoint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from demo.google_auth import GoogleAccessToken
from demo.model_registry import ProviderKind, list_model_ids
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, VLLMAuthMode
from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet, TokenLogprob

if TYPE_CHECKING:
    from transformers import (
        BatchEncoding,
        GenerationConfig,
        GenerationMixin,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    class CausalLM(PreTrainedModel, GenerationMixin):
        """Static type for what `AutoModelForCausalLM` returns. transformers 5
        no longer makes every `PreTrainedModel` a `GenerationMixin`, so the
        loader checks that at runtime before narrowing to this type."""


logger = logging.getLogger(__name__)


@runtime_checkable
class SLMProvider(Protocol):
    """Capability protocol for the small language model.

    Implementations advertise the prompt-time generation backbone the
    detector pipeline runs against. Hidden-state availability is provider-
    specific: a provider that cannot expose hidden states returns `None`
    from `pooled_hidden_state`, and SEP gracefully reports `n/a`."""

    @property
    def model_id(self) -> str: ...

    def sample(self, prompt: str) -> SampleSet: ...
    def greedy(self, prompt: str) -> str: ...
    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None: ...


class VLLMOpenAIProvider:
    """vLLM behind the OpenAI-compat HTTP API.

    Two endpoints are addressed:
      - The generation endpoint at `VLLM_BASE_URL` (chat/completions).
      - Optionally, an embed endpoint at `VLLM_EMBED_BASE_URL`: a second vLLM
        server running the `embed` runner, hit via `/v1/embeddings` to return
        a pooled hidden state per prompt. When unset, `pooled_hidden_state`
        returns `None`.

    Both authenticate per `VLLM_AUTH`: a static key for a server you run, or
    a refreshing Google access token for a Vertex AI endpoint."""

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        credential: str | GoogleAccessToken = (
            GoogleAccessToken()
            if settings.vllm_auth is VLLMAuthMode.GOOGLE
            else settings.vllm_api_key
        )
        self._generation = OpenAI(
            base_url=base_url or settings.vllm_base_url,
            api_key=credential,
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
                api_key=credential,
            )
            self._embed_model_id = settings.vllm_embed_model_id or settings.vllm_model_id
            # There is one embed endpoint, serving one model. Pairing it with a
            # different generation model would feed SEP another model's hidden
            # states: a dimension error at best, silently wrong scores at worst.
            if self._embed_model_id != self.model_id:
                raise ValueError(
                    f"VLLM_EMBED_BASE_URL serves {self._embed_model_id!r} but this provider "
                    f"generates with {self.model_id!r}; SEP needs hidden states from the "
                    "same model. Unset VLLM_EMBED_BASE_URL for this model, or sweep one "
                    "model at a time."
                )
        else:
            self._embed_client = None
            self._embed_model_id = ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
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
        # Embed the prompt exactly as generation sees it: chat template plus
        # the generation prompt, pooled at its last token, unnormalized. This
        # matches HFTransformersProvider.pooled_hidden_state, so a SEP probe
        # trained on either provider reads the same feature.
        body = {
            "model": self._embed_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "use_activation": False,
        }
        try:
            resp = self._embed_client.post(
                "/embeddings",
                body=body,
                cast_to=CreateEmbeddingResponse,
                options={"security": {"bearer_auth": True}},
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
    logprobs from the raw `output_logits`, see `raw_token_logprobs`) and
    pooled hidden state
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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: CausalLM | None = None

    def _ensure_loaded(self) -> tuple[PreTrainedTokenizerBase, CausalLM]:
        """Load tokenizer and model on first call; return both, never None."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        # Imported lazily so test suites that never instantiate this provider
        # don't pay the transformers import cost.
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin

        from demo.model_registry import get_model_entry

        entry = get_model_entry(self.model_id)
        dtype = (
            torch.bfloat16
            if entry.dtype == "bfloat16"
            else torch.float16
            if entry.dtype == "float16"
            else "auto"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, device_map="auto"
            )
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model {self.model_id!r}: {e!s}") from e
        if not isinstance(model, GenerationMixin):
            raise TypeError(
                f"{self.model_id!r} loaded as {type(model).__name__}, which cannot generate."
            )
        causal_lm = cast("CausalLM", model)
        self._tokenizer, self._model = tokenizer, causal_lm
        return tokenizer, causal_lm

    def _encode(
        self, tokenizer: PreTrainedTokenizerBase, model: CausalLM, prompt: str
    ) -> BatchEncoding:
        """Tokenize `prompt` exactly as the vLLM path sees it.

        `VLLMOpenAIProvider` calls the chat-completions endpoint, which wraps
        the prompt in the model's chat template server-side. Applying the
        same template here keeps both providers scoring the same input, so
        PE / SE / SEP numbers are comparable across backends.
        """
        from transformers import BatchEncoding

        if tokenizer.chat_template:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
        if not isinstance(encoded, BatchEncoding):
            raise TypeError(f"Tokenizer returned {type(encoded).__name__}, expected BatchEncoding.")
        return encoded.to(model.device)

    def sample(self, prompt: str) -> SampleSet:
        tokenizer, model = self._ensure_loaded()
        try:
            inputs = self._encode(tokenizer, model, prompt)
            with torch.no_grad():
                out = _generate(
                    model,
                    inputs,
                    do_sample=True,
                    temperature=self._temperature,
                    num_return_sequences=self._n,
                    max_new_tokens=self._max_tokens,
                    output_logits=True,
                )
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.sample failed for prompt={prompt[:80]!r} "
                f"model={self.model_id!r} n={self._n}: {e!s}"
            ) from e

        prompt_len = inputs["input_ids"].shape[1]
        per_sample = raw_token_logprobs(
            logits=_require_logits(out),
            generated_ids=out.sequences[:, prompt_len:],
            stop_token_ids=stop_token_ids(model.generation_config),
        )
        samples: list[Sample] = []
        for pairs in per_sample:
            token_logprobs = [
                TokenLogprob(token=_decode(tokenizer, [tok_id]), logprob=lp) for tok_id, lp in pairs
            ]
            text = _decode(tokenizer, [tok_id for tok_id, _ in pairs], skip_special_tokens=True)
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
        tokenizer, model = self._ensure_loaded()
        try:
            inputs = self._encode(tokenizer, model, prompt)
            with torch.no_grad():
                out = _generate(
                    model,
                    inputs,
                    do_sample=False,
                    num_return_sequences=1,
                    max_new_tokens=self._max_tokens,
                )
            prompt_len = inputs["input_ids"].shape[1]
            new_token_ids = out.sequences[0, prompt_len:].tolist()
            return _decode(tokenizer, new_token_ids, skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.greedy failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e

    def pooled_hidden_state(self, prompt: str) -> torch.Tensor | None:
        tokenizer, model = self._ensure_loaded()
        try:
            inputs = self._encode(tokenizer, model, prompt)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # Last-layer hidden state at the final prompt token. Matches the
            # Kossen et al. SEP setup.
            last_layer = out.hidden_states[-1]  # [1, seq_len, hidden_dim]
            return last_layer[0, -1].detach().to(dtype=torch.float32, device="cpu")
        except Exception as e:
            raise RuntimeError(
                f"HFTransformersProvider.pooled_hidden_state failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e


def _generate(model: CausalLM, inputs: BatchEncoding, **kwargs: Any) -> GenerateDecoderOnlyOutput:
    """`model.generate` with a dict-shaped, decoder-only output guaranteed."""
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    # transformers annotates `generate`'s `self` with a private structural
    # protocol that concrete PreTrainedModel subclasses do not satisfy for
    # mypy; the runtime contract is checked on the return value instead.
    out = model.generate(**inputs, return_dict_in_generate=True, **kwargs)  # type: ignore[misc]
    if not isinstance(out, GenerateDecoderOnlyOutput):
        raise TypeError(
            f"generate returned {type(out).__name__}, expected GenerateDecoderOnlyOutput."
        )
    return out


def _require_logits(out: GenerateDecoderOnlyOutput) -> tuple[torch.Tensor, ...]:
    if out.logits is None:
        raise RuntimeError("generate was asked for output_logits but returned none.")
    return out.logits


def _decode(
    tokenizer: PreTrainedTokenizerBase, token_ids: list[int], skip_special_tokens: bool = False
) -> str:
    """`tokenizer.decode` for one sequence, narrowed to `str`."""
    text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    if not isinstance(text, str):
        raise TypeError(f"Tokenizer decode returned {type(text).__name__}, expected str.")
    return text


def stop_token_ids(config: GenerationConfig) -> set[int]:
    """Token ids that end a generated sequence: EOS (one or several) and PAD."""
    eos = config.eos_token_id
    ids = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
    if config.pad_token_id is not None:
        ids.add(config.pad_token_id)
    if not ids:
        raise ValueError(
            "Model generation_config defines neither eos_token_id nor pad_token_id; "
            "cannot tell where a sampled sequence ends."
        )
    return ids


def raw_token_logprobs(
    logits: tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    stop_token_ids: set[int],
) -> list[list[tuple[int, float]]]:
    """Per-sample (token_id, logprob) pairs under the model's *raw* distribution.

    Args:
        logits: `generate(..., output_logits=True).logits` — one `[n, vocab]`
            tensor per generated step, taken *before* temperature / top-k /
            top-p processing. Using the processed `scores` instead would make
            the logprobs depend on the sampling temperature, and predictive
            entropy would measure the sampler rather than the model.
        generated_ids: `[n, new_tokens]` ids generated after the prompt.
        stop_token_ids: EOS / PAD ids. Each sample keeps its first stop token
            and drops everything after it. The stop token stays because the
            probability of ending there is part of the sequence's likelihood
            (and vLLM's chat logprobs report it the same way). What follows
            it is batch padding: sequences that finish early are padded to
            the longest one, and counting that padding would distort the
            length normalization in predictive entropy.

    Returns:
        One list per sample of `(token_id, logprob)`, ending at the first
        stop token if the sample produced one.
    """
    if len(logits) != generated_ids.shape[1]:
        raise ValueError(
            f"Got {len(logits)} logit steps for {generated_ids.shape[1]} generated tokens; "
            "they must match one-to-one."
        )
    log_probs = torch.log_softmax(torch.stack(logits, dim=1).float(), dim=-1)  # [n, new, vocab]
    chosen = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)  # [n, new]

    per_sample: list[list[tuple[int, float]]] = []
    for ids_row, lps_row in zip(generated_ids.tolist(), chosen.tolist(), strict=True):
        pairs: list[tuple[int, float]] = []
        for tok_id, lp in zip(ids_row, lps_row, strict=True):
            pairs.append((tok_id, lp))
            if tok_id in stop_token_ids:
                break
        per_sample.append(pairs)
    return per_sample


def build_default_provider(model_id: str | None = None) -> SLMProvider:
    """Construct the SLMProvider selected by config.slm_provider.

    When `model_id` is None the active model from config.vllm_model_id is
    used (single-model commands like `python -m demo.cli chat`). When provided, the same
    config.slm_provider kind is built but bound to the requested model id —
    this is how `eval.run_benchmark --sweep` builds each model in
    config.target_model_ids.

    For VLLMOpenAIProvider, the per-model base_url override (if any) is
    pulled from config.target_model_base_urls; missing -> falls back to
    config.vllm_base_url.
    """
    settings = get_settings()
    kind = settings.slm_provider
    if kind == ProviderKind.VLLM:
        base_url_override: str | None = None
        if model_id is not None:
            base_url_override = settings.target_model_base_urls.get(model_id)
        return VLLMOpenAIProvider(model_id=model_id, base_url=base_url_override)
    if kind == ProviderKind.HF_TRANSFORMERS:
        return HFTransformersProvider(model_id=model_id)
    raise ValueError(
        f"Unknown SLM_PROVIDER={kind!r}. Expected one of: {[str(k) for k in ProviderKind]}."
    )


def sweep_model_ids() -> list[str]:
    """The ids in config.target_model_ids, after checking each is in the registry.

    Returns ids rather than providers so the sweep can build one provider at
    a time (`build_default_provider(model_id=...)`) and let each model's
    weights go before the next one loads.
    """
    settings = get_settings()
    known = set(list_model_ids())
    unknown = [mid for mid in settings.target_model_ids if mid not in known]
    if unknown:
        raise ValueError(
            f"TARGET_MODEL_IDS contains ids not in the model registry: {unknown!r}. "
            f"Known: {sorted(known)!r}. Add an entry in demo/model_registry.py first."
        )
    return list(settings.target_model_ids)
