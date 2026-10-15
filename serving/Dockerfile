# vLLM serving image for the Gemma 4 registry entries.
#
# One image, one model, one runner per container:
#   MODEL_ID     which registered model to serve (demo/model_registry.py)
#   VLLM_RUNNER  `generate` (chat completions with raw logprobs, for PE / SE)
#                or `embed` (pooled last-token hidden state, for SEP)
# serving/launch.py turns those into a `vllm serve` argv from the registry.
#
# Base: the official vLLM image. v0.19.0 was the first release with Gemma 4
# support; v0.31.0 carries the later Gemma 4 fixes. The cu129 build is used
# because the default tag targets CUDA 13, whose minimum NVIDIA driver is
# newer than what Vertex documents for its GPU nodes.
FROM vllm/vllm-openai:v0.31.0-cu129

# Vertex mounts the host NVIDIA libraries under /usr/local/nvidia
# (per Google's vLLM-on-Vertex guide).
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    VLLM_NO_USAGE_STATS=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib64" \
    PYTHONPATH=/opt/app \
    AIP_HTTP_PORT=8080 \
    MODEL_ID=google/gemma-4-E4B-it \
    VLLM_RUNNER=generate

# Only the registry and the launcher are needed; the library is not.
COPY demo/__init__.py /opt/app/demo/__init__.py
COPY demo/model_registry.py /opt/app/demo/model_registry.py
COPY serving/launch.py /opt/app/serving/launch.py

EXPOSE 8080

# vLLM's OpenAI server answers GET /health, which Vertex uses as the health
# route. Requests reach the server unchanged through Vertex's invoke route
# (see knowledge/usage_guides/DEVOPS_GUIDE.md).
ENTRYPOINT ["python3", "/opt/app/serving/launch.py"]
