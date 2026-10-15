"""Schemas for structured LLM output (the model fills these, the SDK validates)."""

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """What the escalation judge must return for one question.

    Field order is generation order: Gemini emits structured output in the
    schema's property order. `reasoning` comes first so the answer is
    conditioned on it, rather than the reasoning justifying an answer the
    model has already committed to.
    """

    reasoning: str = Field(description="Brief reasoning toward the answer.")
    answer: str = Field(
        description="A concise, accurate answer. If the answer is unknown, say so explicitly."
    )
