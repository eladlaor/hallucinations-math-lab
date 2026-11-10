"""Request and response models of the web playground API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from demo.web.constants import AskMode, ResultsEntryKind
from hallucinations_math_lab import Cluster, Sample, SemanticEntropyTrace, TokenLogprob
from hallucinations_math_lab.constants import SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import is_none


class StatusResponse(BaseModel):
    """What the Ask tab needs to know before it lets the user press a button.

    `se_threshold` is in nats (SE, discrete SE, PE); `sep_threshold` is on
    SEP's p(high SE) scale. `escalation_threshold` is the one `/ask` applies
    to the configured escalation detector. `se_trace_unavailable_reason` is
    None when `/score` can record `se_trace`, else why the NLI scorer cannot.
    `ask_se_trace_unavailable_reason` is the same for `/ask`, which also needs
    an SE detector in ENABLED_DETECTORS.
    """

    provider: str
    model_id: str
    judge_configured: bool
    judge_unavailable_reason: str | None
    se_threshold: float
    sep_threshold: float
    escalation_threshold: float
    nli_model_id: str
    se_trace_unavailable_reason: str | None
    ask_se_trace_unavailable_reason: str | None


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
    """`se_trace` asks for the SE clustering step by step (`ScoreResponse.se_trace`)."""

    question: str = Field(min_length=1)
    answers: list[ScoreAnswer] = Field(min_length=1)
    se_trace: bool = False


class ScoreResponse(BaseModel):
    """SE and its meaning clusters always; PE only when every answer carries logprobs.

    `se_estimator` names how SE weighed the clusters: by the answers'
    likelihoods when every answer carries logprobs (the paper's estimator),
    otherwise by counting answers per cluster (discrete SE).

    `se_trace` is present only when the request asked for it: the clustering
    comparison by comparison, and the p(c) SE was computed from.
    """

    semantic_entropy: float
    se_estimator: SemanticEntropyEstimator
    predictive_entropy: float | None
    clusters: list[Cluster]
    n_answers: int
    se_threshold: float
    se_trace: SemanticEntropyTrace | None = Field(default=None, exclude_if=is_none)


class AskRequest(BaseModel):
    """`se_trace` asks for the SE clustering step by step (`DetectionResult.se_trace`);
    it needs SE among the enabled detectors."""

    question: str = Field(min_length=1)
    mode: AskMode
    se_trace: bool = False


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
