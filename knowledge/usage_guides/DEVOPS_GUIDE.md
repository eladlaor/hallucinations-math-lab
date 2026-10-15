# DevOps Guide

## Table of Contents

- [Summary](#summary)
- [Do I need any of this?](#do-i-need-any-of-this)
- [Caveats before you deploy](#caveats-before-you-deploy)
- [GCP project prerequisites](#gcp-project-prerequisites)
- [Terraform bootstrap](#terraform-bootstrap)
- [Migrating state to GCS](#migrating-state-to-gcs)
- [Building and pushing the vLLM image](#building-and-pushing-the-vllm-image)
- [Syncing the model registry into Terraform](#syncing-the-model-registry-into-terraform)
- [Bringing the Vertex endpoints up](#bringing-the-vertex-endpoints-up)
- [Pointing the client at Vertex](#pointing-the-client-at-vertex)
- [Running vLLM on your own GPU box instead](#running-vllm-on-your-own-gpu-box-instead)
- [Tearing the endpoints down](#tearing-the-endpoints-down)
- [Continuous integration](#continuous-integration)
- [Cost notes](#cost-notes)

## Summary

All cloud resources live in Terraform under `terraform/`. The always-on part
(API enablements, state/artifact/results buckets, Artifact Registry repo,
service account, IAM) costs about nothing. The Vertex AI endpoints are the
only paid resources and are gated behind `deploy_vertex_endpoint` (default
`false`): turn them on for a run, then turn them off.

One serving image (`serving/Dockerfile`, on `vllm/vllm-openai:v0.31.0-cu129`)
serves one registry model with one vLLM *runner* per container: `generate`
(chat completions with raw logprobs, for PE and SE) or `embed` (pooled hidden
states, for SEP). Terraform puts each runner on its own **dedicated** Vertex
endpoint and prints an OpenAI-compatible base URL for each; the client calls
them with `VLLM_AUTH=google`. CI is a GitHub Actions workflow that needs no
cloud access.

## Do I need any of this?

Usually not. The library, the examples, the tests and the benchmark with the
local `hf_transformers` provider run without any cloud. You need this guide
only to serve a model with vLLM on a Vertex GPU endpoint or your own GPU box,
or to use the Gemini judge (which needs a GCP project and credentials, but
none of the Terraform below).

## Caveats before you deploy

- **Not yet exercised on live GCP.** The Terraform validates
  (`terraform validate` in CI) and every Python piece is unit-tested, but no
  endpoint has been brought up from this revision. Do one `curl` against the
  endpoint (see [Pointing the client at Vertex](#pointing-the-client-at-vertex))
  before a long benchmark.
- **The invoke route is a Vertex Public Preview feature.** It is what lets
  Vertex forward `/v1/chat/completions` and `/v1/embeddings` unchanged to vLLM.
  Neither `gcloud` nor the Terraform provider can set it, so
  `serving/vertex_deploy.py` uploads and deploys through the Vertex Python SDK.
- **A dedicated endpoint's DNS name can take a while to resolve** after
  creation — reports range from minutes to about half an hour. A
  name-resolution error right after `terraform apply` usually means "wait".
- **Terraform tracks the deploy hooks, not the deployed model.** If you
  undeploy from the Cloud Console, `terraform plan` shows no drift. See the
  comment block in `terraform/vertex.tf`.

## GCP project prerequisites

One-time manual steps outside Terraform:

1. Create a GCP project and link a billing account.
2. Configure Application Default Credentials:
   ```bash
   gcloud auth application-default login
   gcloud config set project <your-project-id>
   ```
   Terraform's Google provider, `serving/vertex_deploy.py` and the client's
   `VLLM_AUTH=google` mode all use these; no key file is needed.
3. Make sure the machine that runs `terraform apply` has
   [`uv`](https://docs.astral.sh/uv/) — the deploy step runs
   `uv run --extra deploy python serving/vertex_deploy.py`.

## Terraform bootstrap

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # gitignored; set project_id
terraform init
terraform plan
terraform apply
```

The provider versions are pinned by the committed `.terraform.lock.hcl`
(Google provider `~> 8.0`). The first apply creates, with the default
`name_prefix = "taste"`:

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
`demo/model_registry.py` and `serving/launch.py`.

```bash
REPO_URL=$(cd terraform && terraform output -raw artifact_registry_repository)
TAG=$(date +%Y%m%d-%H%M%S)
IMAGE_URI="${REPO_URL}/vllm-gemma:${TAG}"

gcloud builds submit . \
  --config=serving/cloudbuild.yaml \
  --substitutions="_IMAGE_URI=${IMAGE_URI}"
```

Keep `IMAGE_URI` — it becomes `vllm_image_uri` below. The same image serves
both runners; the runner is chosen at deploy time.

## Syncing the model registry into Terraform

Terraform reads the machine type, accelerator type and count from the Python
model registry so the two never drift. Regenerate after changing the deployed
model:

```bash
uv run python serving/model_flags.py google/gemma-4-E4B-it terraform \
  > terraform/model_registry.auto.tfvars.json
```

Terraform loads `*.auto.tfvars.json` automatically.

## Bringing the Vertex endpoints up

```bash
cd terraform
terraform apply \
  -var="deploy_vertex_endpoint=true" \
  -var="vllm_image_uri=${IMAGE_URI}"
# Add -var="with_embed_endpoint=true" to also serve SEP's hidden states.
```

For each runner (`generate`, plus `embed` when `with_embed_endpoint = true`)
Terraform:

1. creates a dedicated endpoint whose id is `vertex_endpoint_id` (default
   `1000000001`; the embed endpoint uses that value + 1 — Vertex requires
   numeric ids, unique per project and region);
2. runs `serving/vertex_deploy.py deploy`, which uploads the image as a model
   with invoke route prefix `/*` and health route `/health`, sets `MODEL_ID`
   and `VLLM_RUNNER` on the container, and deploys it with the registry's
   machine type and accelerator under the runtime service account.

This starts the GPU nodes and the hourly meter. Expect 10–20 minutes.

## Pointing the client at Vertex

```bash
cd terraform
terraform output vertex_endpoint_ids
terraform output -raw vllm_base_url         # .../endpoints/<id>/invoke/v1
terraform output -raw vllm_embed_base_url   # empty unless with_embed_endpoint
```

Put them in `.env`:

```bash
SLM_PROVIDER=vllm
VLLM_AUTH=google
VLLM_MODEL_ID=google/gemma-4-E4B-it
VLLM_BASE_URL=<vllm_base_url output>
VLLM_EMBED_BASE_URL=<vllm_embed_base_url output, or blank>
```

`VLLM_AUTH=google` makes the client send a Google access token from
Application Default Credentials and refresh it before it expires (tokens
last about an hour). The identity behind your credentials needs
`roles/aiplatform.user` on the project.

Smoke test before a long run:

```bash
curl "$(cd terraform && terraform output -raw vllm_base_url)/chat/completions" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-E4B-it",
       "messages": [{"role": "user", "content": "Who wrote 1984?"}],
       "n": 2, "logprobs": true}'
```

## Running vLLM on your own GPU box instead

On any machine with an NVIDIA GPU and the NVIDIA container runtime:

```bash
docker build -t vllm-gemma -f serving/Dockerfile .
docker run --gpus all -p 8080:8080 -e MODEL_ID=google/gemma-4-E4B-it vllm-gemma
```

Then in `.env`: `SLM_PROVIDER=vllm`, `VLLM_AUTH=api_key`,
`VLLM_BASE_URL=http://<host>:8080/v1`, `VLLM_MODEL_ID=google/gemma-4-E4B-it`.

For SEP, run a second container with the embed runner and point
`VLLM_EMBED_BASE_URL` at it:

```bash
docker run --gpus device=1 -p 8081:8080 \
  -e MODEL_ID=google/gemma-4-E4B-it -e VLLM_RUNNER=embed vllm-gemma
# .env: VLLM_EMBED_BASE_URL=http://<host>:8081/v1
```

Each container loads its own copy of the weights and by default claims 90%
of its GPU's memory (`VLLM_GPU_MEMORY_UTILIZATION`), so give each its own
GPU. With one GPU, get SEP's hidden states from the local `hf_transformers`
provider instead.

Without Docker, print the equivalent `vllm serve` command from the registry
and run it in an environment that has vLLM 0.31:

```bash
uv run python serving/model_flags.py google/gemma-4-E4B-it vllm generate
uv run python serving/model_flags.py google/gemma-4-E4B-it vllm embed
```

See [`serving/README.md`](../../serving/README.md) for a curl smoke test.

## Tearing the endpoints down

```bash
cd terraform
terraform apply -var="deploy_vertex_endpoint=false"
```

The `null_resource` destroy step runs `serving/vertex_deploy.py undeploy`,
which undeploys every model on each endpoint and deletes the uploaded models
before Terraform removes the endpoints. No manual `gcloud` step is needed.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull
request, with four jobs:

| Job | What it runs |
|---|---|
| **lint** | `ruff check .` and `ruff format --check .`, ruff pinned to 0.17.0 |
| **terraform** | `terraform fmt -check -recursive`, `init -backend=false`, `validate` (Terraform 1.16.6) |
| **typecheck** | `uv sync --locked --all-extras`, then `mypy` (config in `pyproject.toml`) |
| **test** | `uv sync --locked --all-extras`, then `pytest`, on Python 3.12 and 3.13 |

It needs no secrets or cloud credentials. `--locked` fails the job if
`uv.lock` is out of date with `pyproject.toml` — run `uv lock` and commit the
result. Ruff is pinned because its default rule set changes between releases.
Details of the suites are in the [QA Guide](QA_GUIDE.md).

## Cost notes

- Always-on resources: about $0.
- Generate endpoint on one L4: about $0.70/hour (~$17/day). The A100 entries
  in the registry cost several times more.
- `with_embed_endpoint = true` deploys a second endpoint with its own GPU and
  doubles the cost. Leave it off unless you are running SEP against vLLM; the
  local `hf_transformers` provider gives SEP its hidden states for free.
- Forgetting the teardown is the most expensive mistake available in this
  repo. Run `terraform apply -var="deploy_vertex_endpoint=false"` as soon as
  the run is done.
