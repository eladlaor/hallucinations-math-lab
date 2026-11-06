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
    DISCRETE_SEMANTIC_ENTROPY = "discrete_semantic_entropy"
    LLAMA_JUDGE_QA = "llama_judge_qa"
    LLAMA_JUDGE_QA_SAMPLES = "llama_judge_qa_samples"


# Detectors whose score is an entropy in nats, so SE_THRESHOLD is on their
# scale. SEP scores p(high SE) and has its own threshold (SEP_THRESHOLD).
NATS_SCORED_DETECTORS: Final[frozenset[DetectorName]] = frozenset(
    {
        DetectorName.SEMANTIC_ENTROPY,
        DetectorName.DISCRETE_SEMANTIC_ENTROPY,
        DetectorName.PREDICTIVE_ENTROPY,
    }
)


class SemanticEntropyEstimator(StrEnum):
    """How Semantic Entropy turns clusters into a probability distribution.

    LIKELIHOOD_WEIGHTED: Farquhar et al.'s main estimator. p(c) is the sum of
    its samples' sequence likelihoods (length-normalized: exp of the mean
    token logprob), normalized over the N samples. Needs token logprobs.
    DISCRETE: p(c) = |c| / N. The paper's "discrete semantic entropy", for
    black-box models with no logprobs.
    """

    LIKELIHOOD_WEIGHTED = "likelihood_weighted"
    DISCRETE = "discrete"


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


# SE NLI input: the question is prepended to each answer before DeBERTa judges it,
# as in Farquhar et al.'s reference code (condition_on_question=True), so short
# answers like "Paris" vs "The capital is Paris" are compared in context.
SE_NLI_INPUT_TEMPLATE: Final[str] = "{question} {answer}"

# What the SLM is sent for one question: the configured brief-answer
# instruction, then the question (Farquhar et al.'s reference prompt shape).
# `SampleSet.prompt` keeps the bare question; only the model sees this.
ANSWER_MESSAGE_TEMPLATE: Final[str] = "{instruction}\nQuestion: {question}"

# SEP probes: one per model, at SEP_PROBE_DIR/<model slug>.pt unless SEP_PROBE_PATH says otherwise.
SEP_PROBE_DIR: Final[str] = "data/probes"
SEP_PROBE_SUFFIX: Final[str] = ".pt"
# Model id -> filesystem-safe slug: lowercase, runs of other characters become "-".
MODEL_SLUG_PATTERN: Final[str] = r"[^a-z0-9]+"
MODEL_SLUG_SEPARATOR: Final[str] = "-"

# SEP classifier: the paper notebook's bare LogisticRegression() (Kossen et al.
# 2024, train-latent-probe.ipynb). l1_ratio=0 is pure L2, scikit-learn >= 1.8's
# spelling of penalty="l2".
SEP_LOGREG_C: Final[float] = 1.0
SEP_LOGREG_L1_RATIO: Final[float] = 0.0
SEP_LOGREG_SOLVER: Final[str] = "lbfgs"
SEP_LOGREG_MAX_ITER: Final[int] = 100
# Pre-registered fallback when lbfgs does not converge at SEP_LOGREG_MAX_ITER (deviation D10).
SEP_LOGREG_FALLBACK_MAX_ITER: Final[int] = 1000


class SEPStandardization(StrEnum):
    """Feature scaling inside a SEP probe.

    NONE: the paper's setting; the probe reads raw hidden states.
    FOLDED_INTO_WEIGHTS: a standard scaler fitted on the training rows, folded
    into the linear weights so the probe is still one linear layer (deviation D12).
    """

    NONE = "none"
    FOLDED_INTO_WEIGHTS = "folded_into_weights"


# SEP layer selection (Kossen et al. 2024, App. B.3): a window of this many
# consecutive layers, concatenated, chosen by nested cross-validation with this
# many folds in both the outer and the inner loop.
SEP_LAYER_WINDOW: Final[int] = 5
SEP_CV_FOLDS: Final[int] = 5


class SEPLayerSelection(StrEnum):
    """How a SEP probe's layers were chosen.

    NESTED_CV: the window was picked by an inner CV on training rows only.
    FIXED: the layers were given (e.g. `--layers final`); no inner selection ran.
    """

    NESTED_CV = "nested_cv"
    FIXED = "fixed"


class SEPThresholdSource(StrEnum):
    """Where a SEP probe's recommended threshold came from.

    OOF_YOUDEN: Youden's J on the out-of-fold scores of the outer CV.
    """

    OOF_YOUDEN = "oof_youden"


# Reproducibility
RANDOM_SEED: Final[int] = 42

# Logging key (used in `extra={"key": ...}`)
LOG_EXTRA_KEY: Final[str] = "ctx"


class JudgeThinkingLevel(StrEnum):
    """Gemini 3.x reasoning depth for the judge (Google Gen AI `ThinkingLevel`).

    Gemini 3.5 models on Vertex ignore temperature / top-p / top-k; thinking
    level and a structured schema are the documented controls instead.
    """

    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class VLLMAuthMode(StrEnum):
    """How the vLLM client authenticates (see `Settings.vllm_auth`)."""

    API_KEY = "api_key"
    GOOGLE = "google"


# OAuth scope for calling Vertex AI with Application Default Credentials.
GOOGLE_CLOUD_PLATFORM_SCOPE: Final[str] = "https://www.googleapis.com/auth/cloud-platform"
