"""App-wide immutable constants. Do not put configurable values here — those go in config.py."""

from enum import StrEnum
from typing import Final


class NLILabel(StrEnum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class AnswerSource(StrEnum):
    SMALL_MODEL_GREEDY = "small_model_greedy"
    JUDGE_MODEL = "judge_model"


class DetectorSignal(StrEnum):
    PREDICTIVE_ENTROPY = "predictive_entropy"
    SEMANTIC_ENTROPY = "semantic_entropy"


# DeBERTa-MNLI label index → string (HF model card order: contradiction, neutral, entailment)
DEBERTA_MNLI_LABEL_ORDER: Final[tuple[NLILabel, NLILabel, NLILabel]] = (
    NLILabel.CONTRADICTION,
    NLILabel.NEUTRAL,
    NLILabel.ENTAILMENT,
)

# Reproducibility
RANDOM_SEED: Final[int] = 42

# Logging key (used in `extra={"key": ...}`)
LOG_EXTRA_KEY: Final[str] = "ctx"
