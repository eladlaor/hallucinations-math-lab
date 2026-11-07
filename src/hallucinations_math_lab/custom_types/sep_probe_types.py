"""Config blocks that tie SEP hidden states to the exact backbone, prompt and
labels that produced them (SEP_CLASSIFIER.md, "Artifact format").

A collected dataset's `collection_config.json` is `SEPCollectionConfig`. The
blocks are shared with the trained probe's config, so a probe fitted on a
dataset carries the same backbone and prompt it was collected with. Every
block forbids unknown fields: a misspelled or foreign field fails at load
instead of being ignored.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hallucinations_math_lab.constants import HUB_COMMIT_SHA_PATTERN, SEP_DATASET_SCHEMA_VERSION

_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


class _StrictBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SEPBackbone(_StrictBlock):
    """The model whose hidden states were read, pinned to one Hub commit."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=HUB_COMMIT_SHA_PATTERN)
    num_hidden_layers: int = Field(ge=1)  # L; tensors hold layers 0..L
    hidden_size: int = Field(ge=1)
    dtype: str = Field(min_length=1)


class SEPPromptSpec(_StrictBlock):
    """How each question was formatted: a changed template or instruction
    moves every hidden state."""

    chat_template_sha256: str = Field(pattern=_SHA256_HEX_PATTERN)
    add_generation_prompt: Literal[True] = True
    answer_instruction: str = Field(min_length=1)


class SEPCollectionLabels(_StrictBlock):
    """How the SE labels were produced.

    `top_p` and `top_k` are the values sampling actually used. The sampler
    sends only the temperature, so `generate` takes both from the
    checkpoint's generation_config (deviation D8)."""

    sample_n: int = Field(ge=2)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    max_tokens: int = Field(ge=1)
    nli_model_id: str = Field(min_length=1)
    nli_revision: str = Field(pattern=HUB_COMMIT_SHA_PATTERN)


class SEPCollectionConfig(_StrictBlock):
    """`collection_config.json`: the calibration set, and the backbone, prompt
    and labels blocks of a collected SEP dataset. It holds no γ: the label
    split is a fit-time choice.

    `calibration_set` is the question file's name and `calibration_set_sha256`
    the SHA-256 of its bytes, so the fit, which reads only the dataset
    directory, can name the exact questions it trained on.
    `n_questions_requested` is the number of questions in that set. A run that
    stopped early writes fewer rows, and the count tells that partial dataset
    apart from a complete one."""

    schema_version: int = SEP_DATASET_SCHEMA_VERSION
    calibration_set: str = Field(min_length=1)
    calibration_set_sha256: str = Field(pattern=_SHA256_HEX_PATTERN)
    n_questions_requested: int = Field(ge=1)
    backbone: SEPBackbone
    prompt: SEPPromptSpec
    labels: SEPCollectionLabels

    @field_validator("schema_version")
    @classmethod
    def _known_schema_version(cls, value: int) -> int:
        if value != SEP_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unknown SEP dataset schema_version {value}; this code reads "
                f"version {SEP_DATASET_SCHEMA_VERSION}."
            )
        return value
