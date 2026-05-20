"""Domain types: SampleSet, Cluster, DetectionResult."""

from pydantic import BaseModel, Field, ConfigDict

from seh_handler.constants import AnswerSource


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
    """Output of the detector for a single prompt."""
    prompt: str
    greedy_answer: str
    samples: list[str]
    predictive_entropy: float
    semantic_entropy: float
    clusters: list[Cluster]
    escalated: bool
    final_answer: str
    final_answer_source: AnswerSource
    judge_reasoning: str | None = None
