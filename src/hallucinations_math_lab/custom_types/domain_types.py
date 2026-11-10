"""Domain types: SampleSet, Cluster, the SE trace, DetectionResult."""

from pydantic import BaseModel, ConfigDict, Field

from hallucinations_math_lab.constants import (
    AnswerSource,
    DetectorName,
    NLILabel,
    SemanticEntropyEstimator,
)


def is_none(value: object) -> bool:
    """`exclude_if` predicate: leave an opt-in field out of the serialized
    model when it is absent, so untraced payloads keep their old shape."""
    return value is None


class TokenLogprob(BaseModel):
    model_config = ConfigDict(frozen=True)
    token: str
    logprob: float


class Sample(BaseModel):
    """One of N completions for a given prompt."""

    text: str
    token_logprobs: list[TokenLogprob] = Field(default_factory=list)

    @property
    def mean_token_logprob(self) -> float:
        if not self.token_logprobs:
            raise ValueError("Cannot compute mean token logprob: no token logprobs available.")
        return sum(t.logprob for t in self.token_logprobs) / len(self.token_logprobs)


class SampleSet(BaseModel):
    """The N samples returned for a single prompt."""

    prompt: str
    samples: list[Sample]


class Cluster(BaseModel):
    """A semantic cluster — group of sample indices judged to share meaning."""

    member_indices: list[int]
    representative_text: str

    @property
    def size(self) -> int:
        return len(self.member_indices)


class ClusteringComparison(BaseModel):
    """One NLI comparison the greedy clustering ran: an answer against a cluster's
    representative (its first member), both prefixed with the question.

    `forward` judges representative -> answer, `backward` answer -> representative.
    `backward` is None when it never ran: the clustering stops at the first
    direction that is not entailment. `joined` is True iff both are entailment.
    """

    answer_index: int
    cluster_index: int
    representative_index: int
    forward: NLILabel
    backward: NLILabel | None
    joined: bool


class ClusterPlacement(BaseModel):
    """Where one answer ended up: the cluster it joined, or the one it opened."""

    answer_index: int
    cluster_index: int
    opened_new_cluster: bool


class ClusteringTrace(BaseModel):
    """The clustering run, step by step: every comparison in the order it ran,
    then one placement per answer, in answer order."""

    comparisons: list[ClusteringComparison]
    placements: list[ClusterPlacement]


class SemanticEntropyTrace(BaseModel):
    """Everything needed to replay one SE computation.

    `cluster_probabilities` is the p(c) SE was computed from, aligned with the
    clusters. `answer_likelihoods` is each answer's length-normalized sequence
    likelihood, exp(mean token logprob), when the estimator weighs by it;
    None for the discrete estimator, whose p(c) is |c| / N.
    """

    clustering: ClusteringTrace
    estimator: SemanticEntropyEstimator
    answer_likelihoods: list[float] | None
    cluster_probabilities: list[float]


class DetectionResult(BaseModel):
    """Output of the detector orchestrator for a single prompt.

    `detector_scores` is keyed by `DetectorName` and carries one float per
    detector that ran. SE-specific clustering is in `clusters` (None if SE was
    not enabled this run). The escalation decision is recorded as the
    `escalation_detector` name, the comparand value, and the threshold used —
    so a reader can reconstruct *why* the system escalated without
    re-reading config. For multi-detector policies the human-readable reason
    lives in `escalation_policy_desc`, while `escalation_detector` /
    `escalation_threshold` remain populated for the single-detector path."""

    prompt: str
    greedy_answer: str
    samples: list[str]
    # Per-sample token logprobs, aligned with `samples`; lets a saved run be
    # re-scored offline by the sample-based detectors (eval/rescore_benchmark.py).
    sample_token_logprobs: list[list[TokenLogprob]] = Field(default_factory=list)
    detector_scores: dict[DetectorName, float]
    clusters: list[Cluster] | None = None
    escalation_detector: DetectorName
    escalation_threshold: float
    escalated: bool
    final_answer: str
    final_answer_source: AnswerSource
    judge_reasoning: str | None = None
    escalation_policy_desc: str | None = None
    # Opt-in (`trace_semantic_entropy`); left out of the serialized result when
    # absent, so untraced results serialize exactly as before the trace existed.
    se_trace: SemanticEntropyTrace | None = Field(default=None, exclude_if=is_none)
