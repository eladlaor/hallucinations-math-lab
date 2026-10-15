"""Pydantic Settings — all configurable values come from env (.env file)."""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hallucinations_math_lab.constants import DetectorName


_DEFAULT_TARGET_MODEL_IDS: list[str] = [
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SLM provider selection. "hf_transformers" loads the model locally via
    # HuggingFace transformers (default for dev — runs against the local
    # default model from the registry). "vllm" delegates to a running vLLM
    # OpenAI-compat server (used in deployed and remote dev modes).
    slm_provider: str = Field(default="hf_transformers")

    # vLLM (OpenAI-compatible endpoint). Used when slm_provider="vllm".
    # Acts as the *fallback* base URL: any model in target_model_ids without
    # a per-model override in target_model_base_urls uses this URL.
    vllm_base_url: str = Field(default="http://localhost:8000/v1")
    vllm_api_key: str = Field(default="local-dummy")
    # The "active" model for single-model commands (e.g., `python -m demo.cli chat`).
    # Must appear in the model registry. Distinct from target_model_ids,
    # which is the evaluation sweep.
    vllm_model_id: str = Field(default="google/gemma-4-E2B-it")
    # Optional sibling vLLM instance launched with --task embed. When set,
    # VLLMOpenAIProvider hits /v1/embeddings on this URL to obtain a pooled
    # hidden state for SEP. Unset = SEP disabled at the vLLM provider.
    vllm_embed_base_url: str = Field(default="")
    vllm_embed_model_id: str = Field(default="")

    # Evaluation sweep — the set of SLM target models the project compares
    # against each other (SE / predictive-entropy / judge pipeline run once
    # per id). Must contain at least 2 ids, and every id must exist in
    # services.model_registry. Default = all four Gemma 4 SKUs.
    target_model_ids: list[str] = Field(default_factory=lambda: list(_DEFAULT_TARGET_MODEL_IDS))
    # Optional per-model base_url override. Keys must be a subset of
    # target_model_ids. Missing keys fall back to vllm_base_url. Lets the
    # operator run the same harness against:
    #   - 1 shared endpoint that's reconfigured between runs (sequential
    #     sweep, cheap — leave this empty), or
    #   - N parallel endpoints, one per model (expensive, true side-by-side
    #     — populate one entry per id).
    target_model_base_urls: dict[str, str] = Field(default_factory=dict)

    # Google Cloud
    gcp_project_id: str = Field(default="")
    gcp_region: str = Field(default="us-central1")

    # Judge models
    judge_model_id: str = Field(default="gemini-3.5-flash-lite")
    # 0.0 keeps the judge's verdict as reproducible as the API allows.
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # NLI for SE clustering
    nli_model_id: str = Field(default="microsoft/deberta-large-mnli")

    # Detection
    se_threshold: float = Field(default=1.0, ge=0.0)
    sample_n: int = Field(default=10, ge=2, le=50)
    sample_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_logprobs: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=64, ge=1, le=512)

    # Detector dispatch
    enabled_detectors: list[DetectorName] = Field(
        default_factory=lambda: [
            DetectorName.PREDICTIVE_ENTROPY,
            DetectorName.SEMANTIC_ENTROPY,
        ]
    )
    escalation_detector: DetectorName = Field(default=DetectorName.SEMANTIC_ENTROPY)

    # SEP probe weights. Default is the per-model slug under data/probes/.
    # Resolved at agent construction time from vllm_model_id.
    sep_probe_path: str = Field(default="")

    @field_validator("target_model_ids")
    @classmethod
    def _validate_target_model_ids(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError(
                f"target_model_ids must contain at least 2 ids for a meaningful "
                f"sweep; got {len(value)}: {value!r}."
            )
        if len(set(value)) != len(value):
            raise ValueError(f"target_model_ids must be unique; got duplicates in {value!r}.")
        # Registry membership is checked by demo.slm_provider.build_sweep_providers:
        # the library must not import the demo package, which is not shipped in
        # the wheel.
        return value

    @model_validator(mode="after")
    def _validate_target_model_base_urls(self) -> "Settings":
        targets = set(self.target_model_ids)
        stray = [mid for mid in self.target_model_base_urls if mid not in targets]
        if stray:
            raise ValueError(
                f"target_model_base_urls has keys not present in "
                f"target_model_ids: {stray!r}. Either add the id to the "
                f"sweep or remove the override."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
