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


class DetectorName(StrEnum):
    """Identifies a hallucination detector. All concrete `Detector` implementations
    expose one of these as `Detector.name`. Order here reflects increasing cost /
    increasing signal strength on the uncertainty axis; LLM-judge detectors are a
    different category and are not ordered against the uncertainty estimators."""

    PREDICTIVE_ENTROPY = "predictive_entropy"
    SEMANTIC_ENTROPY_PROBE = "semantic_entropy_probe"
    SEMANTIC_ENTROPY = "semantic_entropy"
    LLAMA_JUDGE_QA = "llama_judge_qa"
    LLAMA_JUDGE_QA_SAMPLES = "llama_judge_qa_samples"


class DetectorCategory(StrEnum):
    """Distinguishes what kind of signal a detector produces.

    UNCERTAINTY: derived from the SLM's own output distribution (token logprobs,
    sample-meaning dispersion, hidden-state probes). Cheap, model-internal.

    EXTERNAL_JUDGMENT: an external classifier (typically another LLM) scores the
    SLM's answer. Stronger but more expensive and not strictly an uncertainty
    measure. AUROC comparisons between categories are detector-comparisons, not
    uncertainty-method comparisons.
    """

    UNCERTAINTY = "uncertainty"
    EXTERNAL_JUDGMENT = "external_judgment"


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
