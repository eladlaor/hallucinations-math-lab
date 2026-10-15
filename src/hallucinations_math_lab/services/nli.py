"""DeBERTa-MNLI wrapper for bidirectional entailment judgments used by semantic clustering."""

import logging
from functools import lru_cache
from typing import Protocol

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import DEBERTA_MNLI_LABEL_ORDER, LOG_EXTRA_KEY, NLILabel

logger = logging.getLogger(__name__)


class EntailmentScorer(Protocol):
    """What Semantic Entropy needs from an NLI backend: one equivalence judgment.

    `NLIScorer` (DeBERTa-MNLI) is the reference implementation. Swap in any
    object with this method to experiment with other equivalence criteria,
    e.g. an LLM-based judge or a cheap string-normalization baseline.
    """

    def bidirectional_entails(self, a: str, b: str) -> bool: ...


class NLIScorer:
    """Single-process NLI scorer. Lazily loads the model on first use."""

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or get_settings().nli_model_id
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    def _ensure_loaded(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load tokenizer and model on first call; return both, never None."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            logger.info(
                "Loading NLI model.",
                extra={LOG_EXTRA_KEY: {"model_id": self._model_id, "device": self._device}},
            )
            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_id)
            model.to(self._device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load NLI model {self._model_id!r}: {e!s}") from e
        self._tokenizer, self._model = tokenizer, model
        return tokenizer, model

    @torch.no_grad()
    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        """Return the predicted NLI label (entailment / neutral / contradiction)."""
        tokenizer, model = self._ensure_loaded()
        try:
            inputs = tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            logits = model(**inputs).logits
            idx = int(torch.argmax(logits, dim=-1).item())
            return DEBERTA_MNLI_LABEL_ORDER[idx]
        except Exception as e:
            raise RuntimeError(
                f"NLI classification failed for premise={premise[:60]!r}, "
                f"hypothesis={hypothesis[:60]!r}: {e!s}"
            ) from e

    def bidirectional_entails(self, a: str, b: str) -> bool:
        """True iff a entails b AND b entails a (the SE paper's clustering criterion)."""
        return (
            self.classify(a, b) == NLILabel.ENTAILMENT
            and self.classify(b, a) == NLILabel.ENTAILMENT
        )


@lru_cache(maxsize=1)
def get_nli_scorer() -> NLIScorer:
    return NLIScorer()
