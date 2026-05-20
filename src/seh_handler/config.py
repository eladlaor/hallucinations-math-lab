"""Pydantic Settings — all configurable values come from env (.env file)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM (OpenAI-compatible endpoint)
    vllm_base_url: str = Field(default="http://localhost:8000/v1")
    vllm_api_key: str = Field(default="local-dummy")
    vllm_model_id: str = Field(default="hugging-quants/gemma-2-9b-it-AWQ-INT4")

    # Google Cloud
    gcp_project_id: str = Field(default="")
    gcp_region: str = Field(default="us-central1")

    # Judge models
    judge_model_id: str = Field(default="gemini-2.5-flash")
    eval_judge_model_id: str = Field(default="gemini-2.5-pro")

    # NLI for SE clustering
    nli_model_id: str = Field(default="microsoft/deberta-large-mnli")

    # Detection
    se_threshold: float = Field(default=1.0, ge=0.0)
    sample_n: int = Field(default=10, ge=2, le=50)
    sample_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_logprobs: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=64, ge=1, le=512)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
