# Vertex AI endpoint + model upload + deployed-model wiring.
#
# Gated behind var.deploy_vertex_endpoint because the deployed model is the
# only resource in the stack with non-trivial hourly cost (~$0.70/hr on L4).
# Flip the variable to true immediately before a benchmark run, then back to
# false (and re-apply) to tear the endpoint and its GPU node down.
#
# Provider-gap note. As of provider 5.x the Terraform Google provider does
# not expose a resource for uploading a custom-container model to the Vertex
# Model Registry, nor for deploying a custom model to an endpoint. The only
# adjacent resource (google_vertex_ai_endpoint_with_model_garden_deployment)
# is restricted to Model Garden publisher models and rejects custom
# containers. Endpoint creation IS supported by a native resource, so it is
# managed declaratively here; model upload + deploy-model are bundled into a
# single null_resource that shells out to gcloud, with a matching destroy
# provisioner that undeploys and deletes the uploaded model. A single
# `terraform apply -var deploy_vertex_endpoint=false` therefore tears the
# whole stack down cleanly.
#
# The container image at var.vllm_image_uri must already exist in Artifact
# Registry before flipping the flag. Build and push it via Cloud Build using
# serving/cloudbuild.yaml; see knowledge/usage_guides/DEVOPS_GUIDE.md.

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

# Uploads the custom-container model to the Vertex Model Registry and deploys
# it to the endpoint. Triggers re-run when the image URI, model id, or any
# machine/accelerator parameter changes, so a new image rolls out without
# manual intervention.
#
# Tradeoff worth being explicit about. A `null_resource + local-exec` is the
# most elegant *available* path given the provider gap above, but it is a
# compromise on Terraform's core guarantee. Terraform here manages only the
# lifecycle *hooks* (run gcloud on create, run gcloud on destroy); it does
# NOT model the state of the underlying Vertex resources. If someone deletes
# the deployed model from the Cloud Console, `terraform plan` will report no
# drift and `terraform apply` will not repair it. Triggers protect against
# the most common drift sources (config changes in this stack) but cannot
# observe out-of-band mutations. When the Google provider ships native
# resources for `models upload` and `endpoints deploy-model`, this block
# should be replaced — at that point we get real state reconciliation back.
resource "null_resource" "deploy_model" {
  count = var.deploy_vertex_endpoint ? 1 : 0

  triggers = {
    endpoint_id        = google_vertex_ai_endpoint.gemma[0].name
    image_uri          = var.vllm_image_uri
    model_id           = var.model_id
    machine_type       = var.endpoint_machine_type
    accelerator_type   = var.endpoint_accelerator_type
    accelerator_count  = var.endpoint_accelerator_count
    with_embed         = var.with_embed_endpoint ? "1" : ""
    embed_port         = var.embed_port
    service_account    = google_service_account.vertex_runtime.email
    region             = var.region
    project_id         = var.project_id
    name_prefix        = var.name_prefix
    model_display_name = "${var.name_prefix}-gemma-model"
  }

  lifecycle {
    precondition {
      condition     = var.vllm_image_uri != ""
      error_message = "vllm_image_uri must be set when deploy_vertex_endpoint = true."
    }
  }

  provisioner "local-exec" {
    when = create
    environment = {
      ENDPOINT_NAME      = self.triggers.endpoint_id
      REGION             = self.triggers.region
      PROJECT_ID         = self.triggers.project_id
      IMAGE_URI          = self.triggers.image_uri
      MODEL_ID           = self.triggers.model_id
      MACHINE_TYPE       = self.triggers.machine_type
      ACCELERATOR_TYPE   = self.triggers.accelerator_type
      ACCELERATOR_COUNT  = tostring(self.triggers.accelerator_count)
      SERVICE_ACCOUNT    = self.triggers.service_account
      MODEL_DISPLAY_NAME = self.triggers.model_display_name
      WITH_EMBED         = self.triggers.with_embed
      EMBED_PORT         = tostring(self.triggers.embed_port)
    }
    command = <<-EOT
      set -euo pipefail

      CONTAINER_ENV="MODEL_ID=$MODEL_ID"
      if [ -n "$WITH_EMBED" ]; then
        CONTAINER_ENV="$CONTAINER_ENV,VLLM_EMBED_PORT=$EMBED_PORT"
      fi

      VERTEX_MODEL_ID=$(gcloud ai models upload \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --display-name="$MODEL_DISPLAY_NAME" \
        --container-image-uri="$IMAGE_URI" \
        --container-ports=8080 \
        --container-health-route=/health \
        --container-predict-route=/v1/chat/completions \
        --container-env-vars="$CONTAINER_ENV" \
        --format="value(model)")

      echo "Uploaded model: $VERTEX_MODEL_ID"

      gcloud ai endpoints deploy-model "$ENDPOINT_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --model="$VERTEX_MODEL_ID" \
        --display-name="$MODEL_DISPLAY_NAME" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT" \
        --service-account="$SERVICE_ACCOUNT" \
        --min-replica-count=1 \
        --max-replica-count=1 \
        --traffic-split=0=100
    EOT
  }

  provisioner "local-exec" {
    when = destroy
    environment = {
      ENDPOINT_NAME = self.triggers.endpoint_id
      REGION        = self.triggers.region
      PROJECT_ID    = self.triggers.project_id
    }
    command = <<-EOT
      set -euo pipefail

      DEPLOYED_MODEL_IDS=$(gcloud ai endpoints describe "$ENDPOINT_NAME" \
        --project="$PROJECT_ID" --region="$REGION" \
        --format="value(deployedModels[].id)" 2>/dev/null | tr ';' '\n' || true)

      MODELS_TO_DELETE=""
      for id in $DEPLOYED_MODEL_IDS; do
        [ -z "$id" ] && continue
        REFERENCED_MODEL=$(gcloud ai endpoints describe "$ENDPOINT_NAME" \
          --project="$PROJECT_ID" --region="$REGION" \
          --format="value(deployedModels.filter(\"id:$id\").extract(model))" 2>/dev/null || true)
        MODELS_TO_DELETE="$MODELS_TO_DELETE $REFERENCED_MODEL"
        gcloud ai endpoints undeploy-model "$ENDPOINT_NAME" \
          --project="$PROJECT_ID" --region="$REGION" \
          --deployed-model-id="$id" --quiet
      done

      for model in $MODELS_TO_DELETE; do
        [ -z "$model" ] && continue
        gcloud ai models delete "$model" \
          --project="$PROJECT_ID" --region="$REGION" --quiet || true
      done
    EOT
  }

  depends_on = [
    google_vertex_ai_endpoint.gemma,
    google_artifact_registry_repository.containers,
    google_service_account.vertex_runtime,
    google_project_iam_member.vertex_runtime,
  ]
}
