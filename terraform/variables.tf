variable "project_id" {
  description = "GCP project ID that owns all taste resources."
  type        = string
}

variable "region" {
  description = "Primary GCP region for Vertex, Artifact Registry, and GCS buckets."
  type        = string
  default     = "us-central1"
}

variable "name_prefix" {
  description = "Prefix applied to resource names to keep them grouped and easy to clean up."
  type        = string
  default     = "taste"
}

variable "deploy_vertex_endpoint" {
  description = "When true, provisions the Vertex AI endpoint and deploys the Gemma vLLM model. This is the only resource in the stack that incurs hourly cost (~$0.70/hr on L4). Keep false except during active benchmark runs."
  type        = bool
  default     = false
}

variable "vllm_image_uri" {
  description = "Fully qualified Artifact Registry image URI for the vLLM serving container (only consumed when deploy_vertex_endpoint = true)."
  type        = string
  default     = ""
}

variable "endpoint_machine_type" {
  description = "Machine type for the Vertex endpoint deployed model."
  type        = string
  default     = "g2-standard-12"
}

variable "endpoint_accelerator_type" {
  description = "Accelerator type for the Vertex endpoint deployed model."
  type        = string
  default     = "NVIDIA_L4"
}

variable "endpoint_accelerator_count" {
  description = "Number of accelerators attached to the Vertex endpoint deployed model."
  type        = number
  default     = 1
}
