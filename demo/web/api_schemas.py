"""Request and response models of the web playground API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from demo.web.constants import AskMode, ResultsEntryKind
from hallucinations_math_lab import Cluster, Sample, TokenLogprob
from hallucinations_math_lab.constants import SemanticEntropyEstimator


class StatusResponse(BaseModel):
    """What the Ask tab needs to know before it lets the user press a button."""

    provider: str
    model_id: str
    judge_configured: bool
    judge_unavailable_reason: str | None
    se_threshold: float
    nli_model_id: str


class Preset(BaseModel):
    """One scenario of examples/score_precollected_samples.py."""

    name: str
    question: str
    samples: list[Sample]


class ScoreAnswer(BaseModel):
    """One answer to score. Token logprobs are optional: typed answers have none."""

    text: str = Field(min_length=1)
    token_logprobs: list[TokenLogprob] | None = None

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An answer cannot be blank.")
        return value


class ScoreRequest(BaseModel):
    question: str = Field(min_length=1)
    answers: list[ScoreAnswer] = Field(min_length=1)


class ScoreResponse(BaseModel):
    """SE and its meaning clusters always; PE only when every answer carries logprobs.

    `se_estimator` names how SE weighed the clusters: by the answers'
    likelihoods when every answer carries logprobs (the paper's estimator),
    otherwise by counting answers per cluster (discrete SE).
    """

    semantic_entropy: float
    se_estimator: SemanticEntropyEstimator
    predictive_entropy: float | None
    clusters: list[Cluster]
    n_answers: int
    se_threshold: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: AskMode


class ResultsEntry(BaseModel):
    path: str
    kind: ResultsEntryKind
    n_files: int


class ResultsListing(BaseModel):
    results_dir: str
    entries: list[ResultsEntry]


class ResultsReport(BaseModel):
    """`auroc_report` output for a run, `comparison_report` output for a sweep."""

    path: str
    kind: ResultsEntryKind
    report: dict[str, Any]
