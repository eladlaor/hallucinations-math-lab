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
- [Publishing a SEP probe to the Hub](#publishing-a-sep-probe-to-the-hub)
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
(chat completions with raw logprobs, for PE and SE) or `embed` (the
last-layer hidden state, for last-layer SEP probes only). Terraform puts
each runner on its own **dedicated** Vertex endpoint and prints an
OpenAI-compatible base URL for each; the client calls them with
`VLLM_AUTH=google`. CI is a GitHub Actions workflow that needs no cloud
access.

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
`name_prefix = "hml"`:

- the required API enablements;
- the `hml-vertex-runtime` service account and its IAM roles;
- the Artifact Registry Docker repo `hml-containers`;
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

**The embed runner serves last-layer probes only.** It pools the last
token's vector after the final layer, so it can serve a probe trained with
`eval.train_sep_probe --layers final` and nothing else. A probe that reads
another layer window is refused when the evaluator is built, with
`UnsupportedHiddenStateRequest`; serve such a probe through
`SLM_PROVIDER=hf_transformers`. Whether the embed runner's vectors match the
in-process transformers ones within tolerance has not been measured yet.

**Both runners load a pinned checkpoint.** The registry's `revision` (a full
commit sha) goes to `vllm serve` as `--revision` and `--tokenizer-revision`,
so a server always loads the exact weights and chat template a probe was
trained against. vLLM does not report which revision it loaded, so the
launch flag is what the client trusts; to move to a newer checkpoint, update
the registry entry (and retrain its probe), then rebuild the image.

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

## Publishing a SEP probe to the Hub

The release sequence for a trained SEP probe and its calibration dataset. No probe has been
published yet. The owner made both release decisions on 2026-11-08: the Hub namespace is
`eladlaor`, and the probe is released under Apache 2.0, the base model's license
([`PUBLISHED_SEP_PROBE.md`](../plans/PUBLISHED_SEP_PROBE.md#risks-and-decisions)).

Prerequisites: a GPU host that fits the backbone, `SLM_PROVIDER=hf_transformers` and
`VLLM_MODEL_ID=<model>` in `.env`, `uv sync --all-extras`, and `hf auth login` with write
access to the target namespace (the publisher reads that stored token; it takes none on the
command line). Commit and push before step 2: a probe fitted from uncommitted code is refused,
and the model card links to the fitting commit.

On a GPU pod whose NVIDIA driver is older than the locked torch build needs (the locked
PyPI build targets CUDA 13; RunPod hosts currently run older drivers), install the CUDA 12.6
build of the same torch version after syncing, and from then on run every command with
`uv run --no-sync`: a plain `uv run` re-syncs the environment back to the locked build.

```bash
uv sync --frozen --all-extras
uv pip install --reinstall "torch==<locked version>" \
  --index-url https://download.pytorch.org/whl/cu126
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The swap is recorded, not hidden: the collector writes `torch.__version__` (e.g.
`2.14.1+cu126`) and `torch.version.cuda` (`None` for a CPU-only build) into the dataset's
collection config, the trainer copies both into the probe's config, and both Hub cards
print the build.

```bash
# 1. Collect the calibration rows: SE labels and every layer's hidden states (GPU).
uv run python -m eval.collect_sep_dataset
# 2. Fit the probe on CPU; writes data/probes/<slug>/.
uv run python -m eval.train_sep_probe --dataset data/sep_datasets/<slug>
# 3. Benchmark the disjoint 200-question slice with that probe (GPU).
ENABLED_DETECTORS='["predictive_entropy","semantic_entropy_probe","semantic_entropy","discrete_semantic_entropy"]' \
  uv run python -m eval.run_benchmark --out eval/results/sep_heldout.jsonl --no-judge
# 4. Score it to JSON, with every scored question's id.
uv run python -m eval.compute_auroc --in eval/results/sep_heldout.jsonl \
  --json-out eval/results/sep_heldout_auroc.json
# 5. Create both Hub repos private first: the publisher creates a missing repo with the
#    account's default visibility. They go public with the GitHub repo (TODO.md, B1).
hf repos create eladlaor/sep-gemma-4-E2B-it --private
hf repos create eladlaor/sep-gemma-4-E2B-it-calibration --repo-type dataset --private
# 6. Check everything and print both cards; then run again without --dry-run.
uv run python -m eval.publish_sep_probe --probe-dir data/probes/<slug> \
  --dataset data/sep_datasets/<slug> --heldout eval/results/sep_heldout_auroc.json \
  --repo eladlaor/sep-gemma-4-E2B-it \
  --dataset-repo eladlaor/sep-gemma-4-E2B-it-calibration --tag v1 --dry-run
```

Both cards state the license. The probe is released under its backbone's license, which
the Hub lists as Apache 2.0 for the Gemma 4 family (`license_link`
https://ai.google.dev/gemma/docs/gemma_4_license). The dataset card releases the derived
columns under Apache 2.0 and credits TriviaQA for the questions, whose Hub license is
listed as unknown. A backbone without an entry in `BACKBONE_LICENSES`
(`eval/sep_model_card.py`) renders no card, so add its Hub license there before
publishing a probe for it.

The publisher refuses, before writing or uploading anything:

- a `--repo` or `--dataset-repo` that is not a valid `<owner>/<name>` Hub repo id, or a
  `--tag` that is not a plain tag name (letters, digits, `.`, `_`, `-`; e.g. `v1`);
- a probe directory with a missing file, fitted from uncommitted code (`repo_dirty`), or
  whose `metrics.json` reports a TBG window other than the layers the probe reads;
- a held-out report whose `sep_probe` hashes (`weights_sha256`, `config_sha256`) differ from
  the probe directory's files, whose model is not the probe's backbone, or that has no SEP
  AUROC;
- a held-out report without one `question_ids` entry per scored row (write it with
  `--json-out`), or with any id the probe was trained on;
- a held-out run that recorded the probe under an absolute path, which would be published:
  leave `SEP_PROBE_PATH` unset (or relative) in step 3;
- a dataset directory with a missing file, a training log other than the probe's, or a
  backbone, prompt, calibration set or question ids that differ from the probe's.

A real run uploads the probe directory (with the held-out report merged into
`metrics.json` and the rendered `README.md`) to the model repo, and the dataset directory
with its own rendered dataset card (`README.md`) to the dataset repo, one commit each. It
then tags both commits. A `--tag` that either repo already has is
refused before anything is uploaded; publish under a new tag. If tagging the model repo
fails, the publisher deletes the dataset repo's new tag, so the same `--tag` can be retried.
If that deletion fails too, the error names the tag left behind: delete it on the Hub before
retrying.

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
