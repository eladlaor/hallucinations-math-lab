"""DeBERTa-MNLI wrapper for bidirectional entailment judgments used by semantic clustering."""

import logging
from collections.abc import Mapping
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
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, NLILabel

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
    def label_probs(self, premise: str, hypothesis: str) -> dict[NLILabel, float]:
        """Softmax over the three NLI labels, named by the model's own `id2label`."""
        tokenizer, model = self._ensure_loaded()
        label_by_index = nli_label_by_index(model.config.id2label, self._model_id)
        try:
            inputs = tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).tolist()
        except Exception as e:
            raise RuntimeError(
                f"NLI classification failed for premise={premise[:60]!r}, "
                f"hypothesis={hypothesis[:60]!r}: {e!s}"
            ) from e
        return {label_by_index[idx]: float(p) for idx, p in enumerate(probs)}

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        """Return the most likely NLI label (entailment / neutral / contradiction)."""
        probs = self.label_probs(premise, hypothesis)
        return max(probs, key=lambda label: probs[label])

    def bidirectional_entails(self, a: str, b: str) -> bool:
        """True iff a entails b AND b entails a (the SE paper's clustering criterion)."""
        return (
            self.classify(a, b) == NLILabel.ENTAILMENT
            and self.classify(b, a) == NLILabel.ENTAILMENT
        )


def nli_label_by_index(
    id2label: Mapping[int, str] | Mapping[str, str] | None, model_id: str
) -> dict[int, NLILabel]:
    """Map a checkpoint's output indices to `NLILabel`, from its `config.id2label`.

    NLI checkpoints disagree on label order (MNLI models commonly put
    contradiction first, others entailment first), so the order is read from
    the model, never assumed. Fails fast unless the model has exactly the
    three NLI labels.
    """
    if not id2label:
        raise ValueError(f"NLI model {model_id!r} has no config.id2label; cannot name its outputs.")
    known = {str(label): label for label in NLILabel}
    mapping: dict[int, NLILabel] = {}
    for idx, name in id2label.items():
        label = known.get(str(name).strip().lower())
        if label is None:
            raise ValueError(
                f"NLI model {model_id!r} has output label {name!r} at index {idx}, which is "
                f"not one of {sorted(known)}. Use an MNLI-style checkpoint with named labels."
            )
        mapping[int(idx)] = label
    missing = set(NLILabel) - set(mapping.values())
    if missing or len(mapping) != len(NLILabel):
        raise ValueError(
            f"NLI model {model_id!r} labels {dict(id2label)!r} must be exactly "
            f"{sorted(known)}; missing {sorted(str(m) for m in missing)}."
        )
    return mapping


@lru_cache(maxsize=1)
def get_nli_scorer() -> NLIScorer:
    return NLIScorer()
