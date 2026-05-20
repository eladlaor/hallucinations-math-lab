#!/usr/bin/env bash
# Build the vLLM image, push to Artifact Registry, deploy to a Vertex AI Endpoint
# on a single L4 GPU.
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCP_REGION:=us-central1}"
: "${REPO:=seh-handler}"
: "${IMAGE:=vllm-gemma2-9b-awq}"
: "${TAG:=$(date +%Y%m%d-%H%M%S)}"
: "${MODEL_DISPLAY_NAME:=gemma-2-9b-awq-vllm}"
: "${ENDPOINT_DISPLAY_NAME:=seh-detector-endpoint}"
: "${MACHINE_TYPE:=g2-standard-12}"
: "${ACCELERATOR_TYPE:=NVIDIA_L4}"
: "${ACCELERATOR_COUNT:=1}"

IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo ">> Ensuring Artifact Registry repo exists"
gcloud artifacts repositories describe "${REPO}" --location="${GCP_REGION}" \
    >/dev/null 2>&1 || \
    gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker --location="${GCP_REGION}"

echo ">> Building image: ${IMAGE_URI}"
gcloud builds submit --tag "${IMAGE_URI}" serving/

echo ">> Uploading model to Vertex AI Model Registry"
MODEL_ID=$(gcloud ai models upload \
    --region="${GCP_REGION}" \
    --display-name="${MODEL_DISPLAY_NAME}-${TAG}" \
    --container-image-uri="${IMAGE_URI}" \
    --container-ports=8080 \
    --container-health-route=/health \
    --container-predict-route=/v1/chat/completions \
    --format="value(model)")
echo "Model: ${MODEL_ID}"

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
    --model="${MODEL_ID}" \
    --display-name="${MODEL_DISPLAY_NAME}-${TAG}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split=0=100

echo "Done. Endpoint: ${ENDPOINT_ID}"
