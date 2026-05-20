# vLLM serving image for Gemma 2 9B AWQ-INT4. Targets Vertex AI Prediction (custom container).
# CUDA 12.4, vLLM pinned >=0.5.4 for the Gemma-2 attention soft-cap fix.

FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    VLLM_NO_USAGE_STATS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir "vllm>=0.5.4,<0.7" autoawq

# Vertex AI Prediction custom container contract: serve on $AIP_HTTP_PORT (default 8080)
# and respond to GET $AIP_HEALTH_ROUTE. vLLM's OpenAI server exposes /health.
ENV AIP_HTTP_PORT=8080 \
    AIP_HEALTH_ROUTE=/health \
    AIP_PREDICT_ROUTE=/v1/chat/completions \
    MODEL_ID=hugging-quants/gemma-2-9b-it-AWQ-INT4

EXPOSE 8080

CMD ["sh", "-c", "python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_ID \
        --quantization awq \
        --dtype float16 \
        --host 0.0.0.0 \
        --port ${AIP_HTTP_PORT} \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90"]
