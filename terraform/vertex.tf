# Vertex AI endpoints + model upload + deployed-model wiring.
#
# Gated behind var.deploy_vertex_endpoint because the deployed model is the
# only resource in the stack with non-trivial hourly cost (~$0.70/hr per L4).
# Flip the variable to true immediately before a benchmark run, then back to
# false (and re-apply) to tear the endpoints and their GPU nodes down.
#
# One dedicated endpoint per vLLM runner:
#   generate  chat completions with raw logprobs (PE / SE). Always deployed.
#   embed     pooled hidden states (SEP). Only when var.with_embed_endpoint.
# vLLM serves one runner per process, and an endpoint splits traffic between
# its deployed models by percentage rather than by path, so each runner gets
# its own endpoint.
#
# Each model is uploaded with an invoke route prefix of "/*": Vertex then
# forwards https://<dedicated DNS>/v1/<endpoint>/invoke/<path> to the
# container's /<path> unchanged, so the stock OpenAI client reaches vLLM's
# /v1/chat/completions and /v1/embeddings directly (see the vllm_*_base_url
# outputs). Invoke-route models can only be deployed to dedicated endpoints.
#
# Provider-gap note. The Google provider manages the endpoint natively, but
# has no resource for uploading a custom-container model or deploying it,
# and neither the provider nor gcloud can set the invoke route prefix. Upload
# + deploy therefore run through serving/vertex_deploy.py (Vertex Python SDK)
# from a null_resource, with a matching destroy provisioner that undeploys
# and deletes the uploaded model. `terraform apply -var deploy_vertex_endpoint=false`
# tears the whole stack down.
#
# Tradeoff worth being explicit about: Terraform here manages only the
# lifecycle hooks, not the state of the deployed model. If someone undeploys
# it from the Cloud Console, `terraform plan` reports no drift. Triggers
# cover config changes in this stack, not out-of-band mutations.
#
# The image at var.vllm_image_uri must already exist in Artifact Registry
# (serving/cloudbuild.yaml), and the deploy host needs `uv` plus Application
# Default Credentials. See knowledge/usage_guides/DEVOPS_GUIDE.md.

locals {
  vllm_runners = var.deploy_vertex_endpoint ? (
    var.with_embed_endpoint ? toset(["generate", "embed"]) : toset(["generate"])
  ) : toset([])
  # Vertex endpoint ids must be numeric; see var.vertex_endpoint_id.
  endpoint_ids = {
    generate = tostring(var.vertex_endpoint_id)
    embed    = tostring(var.vertex_endpoint_id + 1)
  }
}

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_vertex_ai_endpoint" "vllm" {
  for_each = local.vllm_runners

  name                       = local.endpoint_ids[each.key]
  display_name               = "${var.name_prefix} vLLM ${each.key} endpoint"
  location                   = var.region
  project                    = var.project_id
  dedicated_endpoint_enabled = true

  depends_on = [
    google_project_service.enabled,
    google_artifact_registry_repository.containers,
  ]
}

# Re-runs when the image, model, runner or any machine parameter changes, so
# a new image rolls out without manual steps.
resource "null_resource" "deploy_model" {
  for_each = local.vllm_runners

  triggers = {
    endpoint_id       = google_vertex_ai_endpoint.vllm[each.key].name
    runner            = each.key
    image_uri         = var.vllm_image_uri
    model_id          = var.model_id
    machine_type      = var.endpoint_machine_type
    accelerator_type  = var.endpoint_accelerator_type
    accelerator_count = var.endpoint_accelerator_count
    service_account   = google_service_account.vertex_runtime.email
    region            = var.region
    project_id        = var.project_id
    display_name      = "${var.name_prefix}-${each.key}-model"
  }

  lifecycle {
    precondition {
      condition     = var.vllm_image_uri != ""
      error_message = "vllm_image_uri must be set when deploy_vertex_endpoint = true."
    }
  }

  provisioner "local-exec" {
    when        = create
    working_dir = "${path.module}/.."
    command     = <<-EOT
      set -euo pipefail
      uv run --extra deploy python serving/vertex_deploy.py deploy \
        --project "${self.triggers.project_id}" \
        --region "${self.triggers.region}" \
        --endpoint-id "${self.triggers.endpoint_id}" \
        --image-uri "${self.triggers.image_uri}" \
        --model-id "${self.triggers.model_id}" \
        --runner "${self.triggers.runner}" \
        --machine-type "${self.triggers.machine_type}" \
        --accelerator-type "${self.triggers.accelerator_type}" \
        --accelerator-count "${self.triggers.accelerator_count}" \
        --service-account "${self.triggers.service_account}" \
        --display-name "${self.triggers.display_name}"
    EOT
  }

  provisioner "local-exec" {
    when        = destroy
    working_dir = "${path.module}/.."
    command     = <<-EOT
      set -euo pipefail
      uv run --extra deploy python serving/vertex_deploy.py undeploy \
        --project "${self.triggers.project_id}" \
        --region "${self.triggers.region}" \
        --endpoint-id "${self.triggers.endpoint_id}"
    EOT
  }

  depends_on = [
    google_vertex_ai_endpoint.vllm,
    google_artifact_registry_repository.containers,
    google_service_account.vertex_runtime,
    google_project_iam_member.vertex_runtime,
  ]
}
