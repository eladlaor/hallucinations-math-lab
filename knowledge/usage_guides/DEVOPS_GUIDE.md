# DevOps Guide

## Table of Contents

- [Summary](#summary)
- [Do I need any of this?](#do-i-need-any-of-this)
- [Known gaps before you deploy](#known-gaps-before-you-deploy)
- [GCP project prerequisites](#gcp-project-prerequisites)
- [Terraform bootstrap](#terraform-bootstrap)
- [Migrating state to GCS](#migrating-state-to-gcs)
- [Building and pushing the vLLM image](#building-and-pushing-the-vllm-image)
- [Syncing the model registry into Terraform](#syncing-the-model-registry-into-terraform)
- [Bringing the Vertex endpoint up](#bringing-the-vertex-endpoint-up)
- [Running vLLM on your own GPU box instead](#running-vllm-on-your-own-gpu-box-instead)
- [Tearing the endpoint down](#tearing-the-endpoint-down)
- [Continuous integration](#continuous-integration)
- [Cost notes](#cost-notes)

## Summary

All cloud resources live in Terraform under `terraform/`. The always-on part
(API enablements, state/artifact/results buckets, Artifact Registry repo,
service account, IAM) costs about nothing. The Vertex AI endpoint is the only
paid resource and is gated behind `deploy_vertex_endpoint` (default `false`):
turn it on for a run, then turn it off. CI is a GitHub Actions workflow that
needs no cloud access.

## Do I need any of this?

Usually not. The library, the examples, the tests and the benchmark with the
local `hf_transformers` provider run without any cloud. You need this guide
only to serve a model with vLLM on a Vertex GPU endpoint, or to use the
Gemini judge (which needs a GCP project and credentials, but none of the
Terraform below).

## Known gaps before you deploy

- **The serving image pins `vllm>=0.5.4,<0.7`**, which predates the Gemma 4
  models in the registry. Raise the pin in `serving/Dockerfile` to a vLLM
  release that supports Gemma 4 before building.
- **Calling a Vertex endpoint from the client is not wired up or tested.**
  `VLLMOpenAIProvider` sends a static `VLLM_API_KEY` to `VLLM_BASE_URL`,
  while Vertex endpoints expect a short-lived Google access token and their
  own URL scheme. A vLLM server you can reach directly (your own GPU box, or
  a port-forward) is the supported path today.

## GCP project prerequisites

One-time manual steps outside Terraform:

1. Create a GCP project and link a billing account.
2. Configure Application Default Credentials:
   ```bash
   gcloud auth application-default login
   gcloud config set project <your-project-id>
   ```
   Terraform's Google provider picks these up; no key file is needed.

## Terraform bootstrap

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # gitignored; set project_id
terraform init
terraform plan
terraform apply
```

The first apply creates, with the default `name_prefix = "taste"`:

- the required API enablements;
- the `taste-vertex-runtime` service account and its IAM roles;
- the Artifact Registry Docker repo `taste-containers`;
- three buckets: `<your-project-id>-tfstate`, `<your-project-id>-artifacts`,
  `<your-project-id>-eval-results`.

Override `name_prefix` in `terraform.tfvars` if you want different resource
names. No Vertex endpoint is created yet.

## Migrating state to GCS

Once the first apply has created the state bucket:

1. In `terraform/backend.tf`, uncomment the `backend "gcs"` block and set
   `bucket = "<your-project-id>-tfstate"`.
2. Run `terraform init -migrate-state`. Terraform copies local state to
   `gs://<your-project-id>-tfstate/terraform/state/` and uses it from then on.
3. Delete the local `terraform.tfstate` if you like (it is gitignored).

## Building and pushing the vLLM image

The build runs on Cloud Build, so your machine needs no GPU drivers or
Docker. The build context is the repo root because the image copies
`demo/model_registry.py`.

```bash
REPO_URL=$(cd terraform && terraform output -raw artifact_registry_repository)
TAG=$(date +%Y%m%d-%H%M%S)
IMAGE_URI="${REPO_URL}/vllm-gemma:${TAG}"

gcloud builds submit . \
  --config=serving/cloudbuild.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI}"
```

Keep `IMAGE_URI` — it becomes `vllm_image_uri` below.

## Syncing the model registry into Terraform

Terraform reads the machine type, accelerator type and count from the Python
model registry so the two never drift. Regenerate after changing the deployed
model:

```bash
uv run python serving/model_flags.py google/gemma-4-E4B-it terraform \
  > terraform/model_registry.auto.tfvars.json
```

Terraform loads `*.auto.tfvars.json` automatically.

## Bringing the Vertex endpoint up

```bash
cd terraform
terraform apply \
  -var="deploy_vertex_endpoint=true" \
  -var="vllm_image_uri=${IMAGE_URI}"
```

Terraform creates the endpoint and uploads the model; a `null_resource` then
runs `gcloud ai endpoints deploy-model` with the registry's machine type,
accelerator and the runtime service account. This starts the GPU node and
the hourly meter. Expect 10–20 minutes. `terraform output vertex_endpoint_id`
prints the endpoint.

## Running vLLM on your own GPU box instead

On any machine with an NVIDIA GPU and the NVIDIA container runtime:

```bash
docker build -t vllm-gemma -f serving/Dockerfile .
docker run --gpus all -p 8080:8080 -e MODEL_ID=google/gemma-4-E4B-it vllm-gemma
```

Then in `.env`: `SLM_PROVIDER=vllm`, `VLLM_BASE_URL=http://<host>:8080/v1`,
`VLLM_MODEL_ID=google/gemma-4-E4B-it`. For SEP, start the container with
`VLLM_EMBED_PORT` set and point `VLLM_EMBED_BASE_URL` at that port. See
[`serving/README.md`](../../serving/README.md) for a curl smoke test.

## Tearing the endpoint down

```bash
cd terraform
terraform apply -var="deploy_vertex_endpoint=false"
```

The `null_resource` destroy step undeploys every model on the endpoint
before Terraform removes the endpoint and model. No manual `gcloud` step is
needed.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request:

- **lint** — `ruff check .`
- **test** — `uv sync --locked --all-extras`, then `pytest`, on Python 3.12
  and 3.13. GPU-only tests skip themselves.

It needs no secrets. `--locked` fails the job if `uv.lock` is out of date
with `pyproject.toml` — run `uv lock` and commit the result. Details of the
suites are in the [QA Guide](QA_GUIDE.md).

## Cost notes

- Always-on resources: about $0.
- Endpoint on one L4: about $0.70/hour (~$17/day). The A100 entries in the
  registry cost several times more.
- Forgetting the teardown is the most expensive mistake available in this
  repo. Run `terraform apply -var="deploy_vertex_endpoint=false"` as soon as
  the run is done.
