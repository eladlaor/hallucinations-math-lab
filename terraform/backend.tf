# State backend.
#
# Bootstrap phase (current): local state. The GCS state bucket does not exist
# yet, so we cannot use it as a backend. The first `terraform apply` will
# create the bucket (see storage.tf -> google_storage_bucket.tfstate).
#
# Migration phase (after first apply):
#   1. Uncomment the `backend "gcs"` block below.
#   2. Run: terraform init -migrate-state
#      Terraform will copy the local terraform.tfstate into the GCS bucket
#      and from then on read/write state remotely.
#   3. Commit backend.tf with the block uncommented.
#   4. The local terraform.tfstate can then be deleted (still gitignored).

terraform {
  # backend "gcs" {
  #   bucket = "<your-project-id>-tfstate"
  #   prefix = "terraform/state"
  # }
}
