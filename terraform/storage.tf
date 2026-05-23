resource "google_storage_bucket" "tfstate" {
  name                        = "${var.project_id}-tfstate"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.enabled]
}

resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-artifacts"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  depends_on = [google_project_service.enabled]
}

resource "google_storage_bucket" "eval_results" {
  name                        = "${var.project_id}-eval-results"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  depends_on = [google_project_service.enabled]
}
