#!/usr/bin/env bash
# Build the vLLM image, push to Artifact Registry, deploy to a Vertex AI
# Endpoint. The model id selects every model-specific parameter (vLLM flags,
# GPU SKU, accelerator count, machine type) from the project's model registry,
# so a single deploy script handles every registered SLM.
#
# Override `MODEL_ID` to deploy a non-default SLM:
#     MODEL_ID=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 ./serving/deploy_vertex.sh
#
# Set `WITH_EMBED_ENDPOINT=1` to launch the sibling embed-task vLLM inside the
# container for SEP support. This roughly doubles VRAM usage.

set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:=us-central1}"
: "${REPO:=seh-handler}"
: "${MODEL_ID:=}"
: "${WITH_EMBED_ENDPOINT:=}"
: "${VLLM_EMBED_PORT:=8081}"
: "${ENDPOINT_DISPLAY_NAME:=seh-detector-endpoint}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Resolve the deployed-default model id from the registry when MODEL_ID is unset.
if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID="$(PYTHONPATH="${REPO_ROOT}/src" "${PYTHON_BIN}" -c \
        'from taste.services.model_registry import get_deployed_default; print(get_deployed_default().model_id)')"
fi

# Resolve machine/accelerator from the registry for this model.
mapfile -t REGISTRY_VALS < <(PYTHONPATH="${REPO_ROOT}/src" \
    "${PYTHON_BIN}" "${REPO_ROOT}/serving/model_flags.py" "${MODEL_ID}" vertex)
MACHINE_TYPE="${REGISTRY_VALS[0]}"
ACCELERATOR_TYPE="${REGISTRY_VALS[1]}"
ACCELERATOR_COUNT="${REGISTRY_VALS[2]}"

# Image tag derives from a slugified model id so different SLMs don't collide.
SAFE_MODEL_TAG="$(echo "${MODEL_ID}" | tr '/:.' '-' | tr '[:upper:]' '[:lower:]')"
: "${IMAGE:=vllm-${SAFE_MODEL_TAG}}"
: "${TAG:=$(date +%Y%m%d-%H%M%S)}"
: "${MODEL_DISPLAY_NAME:=slm-${SAFE_MODEL_TAG}}"

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo ">> Model: ${MODEL_ID}"
echo ">> Machine: ${MACHINE_TYPE}  Accelerator: ${ACCELERATOR_TYPE} x${ACCELERATOR_COUNT}"
echo ">> Image:   ${IMAGE_URI}"

echo ">> Ensuring Artifact Registry repo exists"
gcloud artifacts repositories describe "${REPO}" --location="${GCP_REGION}" \
    >/dev/null 2>&1 || \
    gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker --location="${GCP_REGION}"

echo ">> Building image (context: repo root, Dockerfile: serving/Dockerfile)"
gcloud builds submit "${REPO_ROOT}" \
    --config="${REPO_ROOT}/serving/cloudbuild.yaml" \
    --substitutions="_IMAGE_URI=${IMAGE_URI}"

CONTAINER_ENV="MODEL_ID=${MODEL_ID}"
if [[ -n "${WITH_EMBED_ENDPOINT}" ]]; then
    CONTAINER_ENV="${CONTAINER_ENV},VLLM_EMBED_PORT=${VLLM_EMBED_PORT}"
fi

echo ">> Uploading model to Vertex AI Model Registry"
VERTEX_MODEL_ID=$(gcloud ai models upload \
    --region="${GCP_REGION}" \
    --display-name="${MODEL_DISPLAY_NAME}-${TAG}" \
    --container-image-uri="${IMAGE_URI}" \
    --container-ports=8080 \
    --container-health-route=/health \
    --container-predict-route=/v1/chat/completions \
    --container-env-vars="${CONTAINER_ENV}" \
    --format="value(model)")
echo "Vertex Model: ${VERTEX_MODEL_ID}"

echo ">> Creating or reusing endpoint"
ENDPOINT_ID=$(gcloud ai endpoints list --region="${GCP_REGION}" \
    --filter="display_name=${ENDPOINT_DISPLAY_NAME}" \
    --format="value(name)" | head -n1 || true)
if [[ -z "${ENDPOINT_ID}" ]]; then
    ENDPOINT_ID=$(gcloud ai endpoints create \
        --region="${GCP_REGION}" \
        --display-name="${ENDPOINT_DISPLAY_NAME}" \
        --format="value(name)")
fi
echo "Endpoint: ${ENDPOINT_ID}"

echo ">> Deploying model to endpoint (this can take 10-20 minutes)"
gcloud ai endpoints deploy-model "${ENDPOINT_ID}" \
    --region="${GCP_REGION}" \
    --model="${VERTEX_MODEL_ID}" \
    --display-name="${MODEL_DISPLAY_NAME}-${TAG}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split=0=100

echo "Done. Endpoint: ${ENDPOINT_ID}"
