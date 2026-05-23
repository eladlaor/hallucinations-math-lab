# Vertex AI endpoint and deployed model.
#
# Gated behind var.deploy_vertex_endpoint because this is the only resource in
# the stack with non-trivial hourly cost (~$0.70/hr on L4). Flip the variable
# to true immediately before a benchmark run, then back to false (and re-apply)
# to tear the endpoint down.
#
# The container image at var.vllm_image_uri must already exist in Artifact
# Registry before flipping the flag.

resource "google_vertex_ai_endpoint" "gemma" {
  count = var.deploy_vertex_endpoint ? 1 : 0

  name         = "${var.name_prefix}-gemma-endpoint"
  display_name = "${var.name_prefix} Gemma SE detector endpoint"
  location     = var.region
  project      = var.project_id

  depends_on = [
    google_project_service.enabled,
    google_artifact_registry_repository.containers,
  ]
}

resource "google_vertex_ai_model" "gemma" {
  count    = var.deploy_vertex_endpoint ? 1 : 0
  provider = google-beta

  name         = "${var.name_prefix}-gemma-model"
  display_name = "${var.name_prefix} Gemma vLLM model"
  region       = var.region
  project      = var.project_id

  container_spec {
    image_uri = var.vllm_image_uri
    ports {
      container_port = 8080
    }
    health_route  = "/health"
    predict_route = "/v1/completions"
  }

  lifecycle {
    precondition {
      condition     = var.vllm_image_uri != ""
      error_message = "vllm_image_uri must be set when deploy_vertex_endpoint = true."
    }
  }

  depends_on = [google_artifact_registry_repository.containers]
}
