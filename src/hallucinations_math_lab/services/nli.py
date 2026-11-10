"""DeBERTa-MNLI wrapper for bidirectional entailment judgments used by semantic clustering."""

import logging
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Protocol, runtime_checkable

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import HUB_COMMIT_SHA_PATTERN, LOG_EXTRA_KEY, NLILabel

logger = logging.getLogger(__name__)


class EntailmentScorer(Protocol):
    """What Semantic Entropy needs from an NLI backend: one equivalence judgment.

    `NLIScorer` (DeBERTa-MNLI) is the reference implementation. Swap in any
    object with this method to experiment with other equivalence criteria,
    e.g. an LLM-based judge or a cheap string-normalization baseline.
    """

    def bidirectional_entails(self, a: str, b: str) -> bool: ...


@runtime_checkable
class DirectionalEntailmentScorer(EntailmentScorer, Protocol):
    """An `EntailmentScorer` that also labels one direction at a time.

    Only the opt-in SE trace needs it, to show each direction's verdict. Its
    labels must agree with `bidirectional_entails`: that is True iff
    `classify(a, b)` and `classify(b, a)` are both entailment.
    """

    def classify(self, premise: str, hypothesis: str) -> NLILabel: ...


class NLIScorer:
    """Single-process NLI scorer. Lazily loads the model on first use.

    With neither argument, loads the configured `nli_model_id` at its pinned
    `nli_revision`. Another model is always pinned too: `model_id` and
    `revision` are given together or not at all."""

    def __init__(self, model_id: str | None = None, revision: str | None = None) -> None:
        if (model_id is None) != (revision is None):
            raise ValueError(
                f"NLIScorer needs model_id and revision together, got model_id={model_id!r}, "
                f"revision={revision!r}; an unpinned NLI model would change the SE labels "
                "(see NLI_MODEL_ID and NLI_REVISION)."
            )
        if revision is not None and not re.fullmatch(HUB_COMMIT_SHA_PATTERN, revision):
            raise ValueError(
                f"NLIScorer got revision {revision!r} for {model_id!r}; it needs a 40-hex Hub "
                "commit sha, since a branch or tag can move and change the SE labels."
            )
        if model_id is None or revision is None:
            settings = get_settings()
            model_id, revision = settings.nli_model_id, settings.nli_revision
        self._model_id: str = model_id
        self._revision = revision
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
                extra={
                    LOG_EXTRA_KEY: {
                        "model_id": self._model_id,
                        "revision": self._revision,
                        "device": self._device,
                    }
                },
            )
            tokenizer = AutoTokenizer.from_pretrained(self._model_id, revision=self._revision)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_id, revision=self._revision
            )
            model.to(self._device)
            model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load NLI model {self._model_id!r} at revision {self._revision!r} "
                f"(set NLI_REVISION together with NLI_MODEL_ID): {e!s}"
            ) from e
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
