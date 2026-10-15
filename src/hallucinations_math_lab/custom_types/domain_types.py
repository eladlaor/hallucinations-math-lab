"""Domain types: SampleSet, Cluster, DetectionResult."""

from pydantic import BaseModel, ConfigDict, Field

from hallucinations_math_lab.constants import AnswerSource, DetectorName


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
    detector_scores: dict[DetectorName, float]
    clusters: list[Cluster] | None = None
    escalation_detector: DetectorName
    escalation_threshold: float
    escalated: bool
    final_answer: str
    final_answer_source: AnswerSource
    judge_reasoning: str | None = None
    escalation_policy_desc: str | None = None
