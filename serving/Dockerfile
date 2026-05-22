# vLLM serving image. The model is selected via $MODEL_ID at run time and the
# vLLM launch flags are resolved from the project's model registry, so a single
# image can serve any registered SLM. CUDA 12.4, vLLM pinned >=0.5.4 for the
# Gemma-2 attention soft-cap fix.

FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    VLLM_NO_USAGE_STATS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        "vllm>=0.5.4,<0.7" autoawq pydantic pydantic-settings

# Copy the model registry (lives in demo/, not the library) plus the shim.
# Build context is the repo root so demo/* resolves; the registry stays the
# single source of truth and the image picks up new model entries on rebuild.
COPY demo/__init__.py /opt/taste/demo/__init__.py
COPY demo/model_registry.py /opt/taste/demo/model_registry.py
COPY serving/model_flags.py /opt/taste/model_flags.py

# Vertex AI Prediction custom container contract: serve on $AIP_HTTP_PORT and
# respond to GET $AIP_HEALTH_ROUTE. vLLM's OpenAI server exposes /health.
# MODEL_ID defaults to the registry's is_deployed_default (Gemma 4 E4B). To
# serve a different registered SLM, override MODEL_ID at deploy time. The
# Gemma 4 E2B/E4B SKUs run unquantized in bf16 on a single L4 (24 GB) and
# keep the project's ~$0.70/hr cost story honest.
# VLLM_EMBED_PORT, when set, launches a sibling vLLM process with --task embed
# on that port; the SEP detector consumes it via /v1/embeddings.
ENV AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/v1/chat/completions \
    MODEL_ID=google/gemma-4-E4B-it \
    VLLM_EMBED_PORT="" \
    PYTHONPATH=/opt/taste

EXPOSE 8080

# Resolve registry-driven flags once, then exec vLLM. When VLLM_EMBED_PORT is
# set, also start a sibling embed-task vLLM in the background and wait on both.
CMD ["sh", "-c", "\
        set -e; \
        VLLM_FLAGS=\"$(python3 /opt/taste/model_flags.py \"$MODEL_ID\" vllm)\"; \
        if [ -n \"$VLLM_EMBED_PORT\" ]; then \
            echo \"Starting embed-task vLLM on port $VLLM_EMBED_PORT\"; \
            python3 -m vllm.entrypoints.openai.api_server \
                $VLLM_FLAGS \
                --task embed \
                --host 0.0.0.0 \
                --port \"$VLLM_EMBED_PORT\" \
                --gpu-memory-utilization 0.30 & \
        fi; \
        exec python3 -m vllm.entrypoints.openai.api_server \
            $VLLM_FLAGS \
            --host 0.0.0.0 \
            --port \"${AIP_HTTP_PORT}\" \
            --gpu-memory-utilization 0.60"]
