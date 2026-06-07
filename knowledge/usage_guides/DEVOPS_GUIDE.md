# DevOps Guide

## Table of Contents

- [Summary](#summary)
- [GCP project prerequisites](#gcp-project-prerequisites)
- [Terraform bootstrap](#terraform-bootstrap)
- [Migrating state to GCS](#migrating-state-to-gcs)
- [Building and pushing the vLLM image](#building-and-pushing-the-vllm-image)
- [Syncing the model registry into Terraform](#syncing-the-model-registry-into-terraform)
- [Bringing the Vertex endpoint up for a benchmark](#bringing-the-vertex-endpoint-up-for-a-benchmark)
- [Running the benchmark](#running-the-benchmark)
- [Tearing the endpoint down](#tearing-the-endpoint-down)
- [Cost notes](#cost-notes)

## Summary

All cloud resources are managed by Terraform under `terraform/`. The state-bucket,
Artifact Registry repo, GCS buckets, service account, and IAM bindings are
always-on (cost ≈ $0). The Vertex AI endpoint is the only cost-bearing resource
and is gated behind `deploy_vertex_endpoint` (default `false`). Flip the flag
on for benchmark runs, then back off.

## GCP project prerequisites

One-time, manual steps that live outside Terraform:

1. A GCP project exists (this repo uses `<your-project-id>`).
2. A billing account is linked to that project.
3. Local Application Default Credentials are configured:

   ```bash
   gcloud auth application-default login
   gcloud config set project <your-project-id>
   ```

   Terraform's Google provider picks ADC up automatically; no service-account
   JSON key file is required.

## Terraform bootstrap

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars  # gitignored; edit if needed
terraform init
terraform plan
terraform apply
```

The first apply creates:

- All required API enablements.
- The `taste-vertex-runtime` service account and its IAM roles.
- The Artifact Registry Docker repo `taste-containers`.
- Three GCS buckets: `<your-project-id>-tfstate`, `<your-project-id>-artifacts`,
  `<your-project-id>-eval-results`.

No Vertex endpoint is created at this stage (the variable defaults to false).

## Migrating state to GCS

After the first apply succeeds the state bucket exists, so state can move off
disk:

1. Edit `terraform/backend.tf` and uncomment the `backend "gcs"` block.
2. Run:

   ```bash
   terraform init -migrate-state
   ```

   Terraform copies the local `terraform.tfstate` into
   `gs://<your-project-id>-tfstate/terraform/state/` and from then on reads/writes
   remote state.
3. Commit the updated `backend.tf`.
4. The local `terraform.tfstate` file may be deleted (it is gitignored either
   way).

## Building and pushing the vLLM image

The image build runs on Cloud Build (the local machine does not need GPU
drivers, docker, or large bandwidth). The build context is the repo root so
the image can `COPY demo/model_registry.py`.

```bash
REPO_URL=$(cd terraform && terraform output -raw artifact_registry_repository)
TAG=$(date +%Y%m%d-%H%M%S)
IMAGE_URI="${REPO_URL}/vllm-gemma:${TAG}"

gcloud builds submit . \
  --config=serving/cloudbuild.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI}"
```

Record `IMAGE_URI` — it becomes the value of `vllm_image_uri` below.

## Syncing the model registry into Terraform

The Terraform stack reads machine-type, accelerator-type and accelerator-count
from the Python model registry (`demo/model_registry.py`) so the two never
drift. Regenerate the auto-loaded tfvars file whenever the deployed-default
SLM changes:

```bash
uv run python serving/model_flags.py google/gemma-4-E4B-it terraform \
  > terraform/model_registry.auto.tfvars.json
```

The file is `*.auto.tfvars.json`, so Terraform picks it up on the next plan
or apply with no extra flags.

## Bringing the Vertex endpoint up for a benchmark

```bash
cd terraform
terraform apply \
  -var="deploy_vertex_endpoint=true" \
  -var="vllm_image_uri=${IMAGE_URI}"
```

Terraform creates the endpoint and uploads the model, then a `null_resource`
runs `gcloud ai endpoints deploy-model` to attach the model with the
configured machine type, accelerator, and the runtime service account. This
is the step that spins up the L4 node and starts the hourly meter. Expect
10–20 minutes for the deploy to converge.

The `null_resource` is also the resource that performs `undeploy-model` on
destroy, so a single Terraform apply with the flag flipped back to false
tears everything down cleanly.

## Running the benchmark

```bash
uv run python eval/datasets/load_triviaqa.py --n 1000
uv run python eval/run_benchmark.py --in eval/datasets/triviaqa_subset.jsonl \
                                    --out eval/results/triviaqa.jsonl
uv run python eval/compute_auroc.py --in eval/results/triviaqa.jsonl
uv run python eval/vertex_genai_eval.py --in eval/results/triviaqa.jsonl
```

## Tearing the endpoint down

```bash
cd terraform
terraform apply -var="deploy_vertex_endpoint=false"
```

The `null_resource.deploy_model` destroy provisioner runs `undeploy-model`
for every deployed model on the endpoint before Terraform removes the
endpoint and model resources themselves. No manual `gcloud` step is needed.

## Cost notes

- Always-on resources (APIs, IAM, buckets, AR repo): ≈ $0.
- Vertex endpoint with deployed model on L4: ≈ $0.70/hr (~$17/day).
- Skipping teardown is the single most expensive mistake possible in this
  repo. Always run `terraform apply -var="deploy_vertex_endpoint=false"` (or
  the equivalent `gcloud undeploy-model`) when the benchmark finishes.
