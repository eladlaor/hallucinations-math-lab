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

output "vertex_endpoint_id" {
  description = "Vertex AI endpoint resource ID (empty unless deploy_vertex_endpoint = true)."
  value       = try(google_vertex_ai_endpoint.gemma[0].id, "")
}
