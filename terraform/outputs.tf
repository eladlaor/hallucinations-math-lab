output "artifact_registry_repository" {
  description = "Full path of the Docker repo for pushing the vLLM image."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "tfstate_bucket" {
  description = "GCS bucket holding Terraform state once the backend is migrated."
  value       = google_storage_bucket.tfstate.name
}

output "artifacts_bucket" {
  description = "GCS bucket for build / model artifacts."
  value       = google_storage_bucket.artifacts.name
}

output "eval_results_bucket" {
  description = "GCS bucket for evaluation outputs."
  value       = google_storage_bucket.eval_results.name
}

output "vertex_runtime_service_account" {
  description = "Service account email used by the Vertex endpoint runtime."
  value       = google_service_account.vertex_runtime.email
}

output "vertex_endpoint_ids" {
  description = "Vertex AI endpoint ids by vLLM runner (empty unless deploy_vertex_endpoint = true)."
  value       = { for runner, endpoint in google_vertex_ai_endpoint.vllm : runner => endpoint.name }
}

# OpenAI-compatible base URLs through each endpoint's invoke route, in the
# form the Vertex SDK's Endpoint.invoke builds:
#   https://<dedicated DNS>/v1/projects/<number>/locations/<region>/endpoints/<id>/invoke/<path>
# The dedicated DNS name can take several minutes to resolve after creation.
output "vllm_base_url" {
  description = "Set as VLLM_BASE_URL (with VLLM_AUTH=google). Empty unless deployed."
  value       = try(local.invoke_base_urls["generate"], "")
}

output "vllm_embed_base_url" {
  description = "Set as VLLM_EMBED_BASE_URL. Empty unless with_embed_endpoint = true."
  value       = try(local.invoke_base_urls["embed"], "")
}

locals {
  invoke_base_urls = {
    for runner, endpoint in google_vertex_ai_endpoint.vllm : runner => format(
      "https://%s/v1/projects/%s/locations/%s/endpoints/%s/invoke/v1",
      trimprefix(endpoint.dedicated_endpoint_dns, "https://"),
      data.google_project.this.number,
      var.region,
      endpoint.name,
    )
  }
}
