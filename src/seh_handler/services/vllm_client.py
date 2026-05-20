"""vLLM OpenAI-compatible client. Issues a single multi-sample request with logprobs."""

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from seh_handler.config import get_settings
from seh_handler.constants import LOG_EXTRA_KEY
from seh_handler.custom_types.domain_types import Sample, SampleSet, TokenLogprob

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(base_url=settings.vllm_base_url, api_key=settings.vllm_api_key)
        self._model = settings.vllm_model_id
        self._n = settings.sample_n
        self._temperature = settings.sample_temperature
        self._top_logprobs = settings.top_logprobs
        self._max_tokens = settings.max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def sample(self, prompt: str) -> SampleSet:
        """Issue a single chat.completions request with n=N, logprobs=True.

        Returns a SampleSet of N samples. Raises with descriptive context on failure.
        """
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                n=self._n,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                logprobs=True,
                top_logprobs=self._top_logprobs,
            )
        except Exception as e:
            raise RuntimeError(
                f"vLLM sample() failed for prompt={prompt[:80]!r} "
                f"model={self._model!r} n={self._n}: {e!s}"
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
                    "model": self._model,
                    "prompt_preview": prompt[:80],
                }
            },
        )
        return SampleSet(prompt=prompt, samples=samples)

    def greedy(self, prompt: str) -> str:
        """Greedy single-sample completion (used for the default returned answer)."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(
                f"vLLM greedy() failed for prompt={prompt[:80]!r}: {e!s}"
            ) from e
