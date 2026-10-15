resource "google_artifact_registry_repository" "containers" {
  project       = var.project_id
  location      = var.region
  repository_id = "${var.name_prefix}-containers"
  description   = "Docker images for hallucinations-math-lab (vLLM serving container)."
  format        = "DOCKER"

  depends_on = [google_project_service.enabled]
}
