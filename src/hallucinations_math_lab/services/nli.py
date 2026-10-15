"""DeBERTa-MNLI wrapper for bidirectional entailment judgments used by semantic clustering."""

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import DEBERTA_MNLI_LABEL_ORDER, LOG_EXTRA_KEY, NLILabel

logger = logging.getLogger(__name__)


class NLIScorer:
    """Single-process NLI scorer. Lazily loads the model on first use."""

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or get_settings().nli_model_id
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSequenceClassification | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                logger.info(
                    "Loading NLI model.",
                    extra={LOG_EXTRA_KEY: {"model_id": self._model_id, "device": self._device}},
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(self._model_id)
                self._model.to(self._device)  # type: ignore[union-attr]
                self._model.eval()  # type: ignore[union-attr]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load NLI model {self._model_id!r}: {e!s}"
                ) from e

    @torch.no_grad()
    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        """Return the predicted NLI label (entailment / neutral / contradiction)."""
        self._ensure_loaded()
        try:
            inputs = self._tokenizer(  # type: ignore[misc]
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            logits = self._model(**inputs).logits  # type: ignore[misc]
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
