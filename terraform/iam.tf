resource "google_service_account" "vertex_runtime" {
  account_id   = "${var.name_prefix}-vertex-runtime"
  display_name = "Service account used by the Vertex AI endpoint runtime"
  project      = var.project_id

  depends_on = [google_project_service.enabled]
}

locals {
  vertex_runtime_roles = [
    "roles/aiplatform.user",
    "roles/artifactregistry.reader",
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ]
}

resource "google_project_iam_member" "vertex_runtime" {
  for_each = toset(local.vertex_runtime_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_runtime.email}"
}
