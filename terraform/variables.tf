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

variable "model_id" {
  description = "Hugging Face model id passed to the serving container as MODEL_ID. Must match an entry in demo/model_registry.py. Normally written into terraform/model_registry.auto.tfvars.json by `serving/model_flags.py <id> terraform`."
  type        = string
  default     = "google/gemma-4-E4B-it"
}

variable "with_embed_endpoint" {
  description = "When true, the serving container also launches a sibling vLLM in --task embed mode on var.embed_port for SEP support. Roughly doubles VRAM usage."
  type        = bool
  default     = false
}

variable "embed_port" {
  description = "Port for the sibling embed-task vLLM inside the serving container."
  type        = number
  default     = 8081
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
