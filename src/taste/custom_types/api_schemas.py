"""Request/response schemas exposed by the (optional) chat HTTP API."""

from pydantic import BaseModel

from taste.constants import AnswerSource, DetectorName


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str
    source: AnswerSource
    detector_scores: dict[DetectorName, float]
    escalation_detector: DetectorName
    escalated: bool
    cluster_count: int
