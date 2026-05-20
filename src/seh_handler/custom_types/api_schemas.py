"""Request/response schemas exposed by the (optional) chat HTTP API."""

from pydantic import BaseModel

from seh_handler.constants import AnswerSource


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str
    source: AnswerSource
    semantic_entropy: float
    predictive_entropy: float
    escalated: bool
    cluster_count: int
