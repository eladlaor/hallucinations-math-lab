"""Pydantic Settings — all configurable values come from env (.env file)."""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hallucinations_math_lab.constants import DetectorName, JudgeThinkingLevel, VLLMAuthMode

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
    # How requests to vLLM authenticate. `api_key` sends VLLM_API_KEY as a
    # static bearer token (a local or self-hosted server). `google` sends a
    # short-lived Google access token from Application Default Credentials,
    # refreshed before it expires (a Vertex AI endpoint).
    vllm_auth: VLLMAuthMode = Field(default=VLLMAuthMode.API_KEY)
    vllm_api_key: str = Field(default="local-dummy")
    # The "active" model for single-model commands (e.g., `python -m demo.cli chat`).
    # Must appear in the model registry. Distinct from target_model_ids,
    # which is the evaluation sweep.
    vllm_model_id: str = Field(default="google/gemma-4-E2B-it")
    # Optional second vLLM server running the `embed` runner (see
    # serving/Dockerfile). When set, VLLMOpenAIProvider hits /v1/embeddings
    # on this URL to obtain a pooled hidden state for SEP. Unset = SEP
    # disabled at the vLLM provider. Authenticates the same way as above.
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
    # Gemini 3.5 on Vertex ignores temperature, so the judge sends none; its
    # reasoning depth is set here instead. 3.5 Flash-Lite defaults to MINIMAL,
    # which Google warns can cut multi-step reasoning short.
    judge_thinking_level: JudgeThinkingLevel = Field(default=JudgeThinkingLevel.LOW)

    # NLI for SE clustering
    nli_model_id: str = Field(default="microsoft/deberta-large-mnli")

    # Detection
    # Escalate when the escalation detector's score exceeds this, in nats.
    # 1.0 is a placeholder, not a measured value: calibrate it on a dev
    # slice disjoint from the test slice with eval/calibrate_threshold.py.
    se_threshold: float = Field(default=1.0, ge=0.0)
    sample_n: int = Field(default=10, ge=2, le=50)
    sample_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_logprobs: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=64, ge=1, le=512)
    # Prepended to every question the SLM answers (see
    # constants.ANSWER_MESSAGE_TEMPLATE). Chat-tuned models otherwise answer
    # TriviaQA in long sentences that max_tokens truncates and that make
    # correctness matching and NLI clustering noisy. The default is the
    # short-answer instruction of Farquhar et al.'s reference code. Recorded
    # in every benchmark row, since changing it changes every score.
    answer_instruction: str = Field(
        default="Answer the following question as briefly as possible.", min_length=1
    )

    # Detector dispatch
    enabled_detectors: list[DetectorName] = Field(
        default_factory=lambda: [
            DetectorName.PREDICTIVE_ENTROPY,
            DetectorName.SEMANTIC_ENTROPY,
        ]
    )
    escalation_detector: DetectorName = Field(default=DetectorName.SEMANTIC_ENTROPY)

    # SEP probe weights. Default is the per-model slug under data/probes/.
    # Resolved per evaluated model when the evaluator is built. Must stay unset
    # for a multi-model sweep (one file cannot serve several models).
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
        # Registry membership is checked by demo.slm_provider.sweep_model_ids:
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
