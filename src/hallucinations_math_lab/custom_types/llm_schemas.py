"""Schemas for structured LLM output (the model fills these, the SDK validates)."""

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """What the escalation judge must return for one question."""

    answer: str = Field(
        description="A concise, accurate answer. If the answer is unknown, say so explicitly."
    )
    reasoning: str = Field(description="A brief justification of the answer.")
