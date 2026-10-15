"""Upload the vLLM serving image to Vertex AI and deploy it to an endpoint.

Called by Terraform (terraform/vertex.tf) because neither the Terraform
provider nor gcloud can set the model's invoke route prefix. With the invoke
route, Vertex forwards `.../invoke/<path>` to the container's `/<path>`
unchanged, so the stock OpenAI client reaches vLLM's `/v1/chat/completions`
and `/v1/embeddings` directly. Invoke-route models must be deployed to a
dedicated endpoint, which Terraform creates.

Usage:
    python serving/vertex_deploy.py deploy --project P --region R --endpoint-id E \
        --image-uri IMG --model-id google/gemma-4-E4B-it --runner generate \
        --machine-type g2-standard-12 --accelerator-type NVIDIA_L4 \
        --accelerator-count 1 --service-account SA --display-name NAME
    python serving/vertex_deploy.py undeploy --project P --region R --endpoint-id E
"""

import argparse
import logging

from google.cloud import aiplatform

logger = logging.getLogger(__name__)

# Same key as hallucinations_math_lab.constants.LOG_EXTRA_KEY.
LOG_EXTRA_KEY = "ctx"

CONTAINER_PORT = 8080
HEALTH_ROUTE = "/health"
# The only value Vertex accepts: forward every path under /invoke/.
INVOKE_ROUTE_PREFIX = "/*"
# vLLM uses shared memory for its worker processes; tensor parallelism needs
# more than the container default.
SHARED_MEMORY_SIZE_MB = 16384
ENV_MODEL_ID = "MODEL_ID"
ENV_RUNNER = "VLLM_RUNNER"


def deploy(args: argparse.Namespace) -> None:
    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(args.endpoint_id)
    try:
        model = aiplatform.Model.upload(
            display_name=args.display_name,
            serving_container_image_uri=args.image_uri,
            serving_container_invoke_route_prefix=INVOKE_ROUTE_PREFIX,
            serving_container_health_route=HEALTH_ROUTE,
            serving_container_ports=[CONTAINER_PORT],
            serving_container_environment_variables={
                ENV_MODEL_ID: args.model_id,
                ENV_RUNNER: args.runner,
            },
            serving_container_shared_memory_size_mb=SHARED_MEMORY_SIZE_MB,
        )
        logger.info(
            "Uploaded model.",
            extra={LOG_EXTRA_KEY: {"model": model.resource_name, "runner": args.runner}},
        )
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=args.display_name,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            service_account=args.service_account,
            min_replica_count=1,
            max_replica_count=1,
            traffic_percentage=100,
        )
    except Exception as e:
        raise RuntimeError(
            f"Vertex deploy failed for model_id={args.model_id!r} runner={args.runner!r} "
            f"endpoint={args.endpoint_id!r}: {e!s}"
        ) from e
    logger.info(
        "Deployed model.",
        extra={
            LOG_EXTRA_KEY: {
                "endpoint": endpoint.resource_name,
                "dedicated_endpoint_dns": endpoint.dedicated_endpoint_dns,
            }
        },
    )


def undeploy(args: argparse.Namespace) -> None:
    """Undeploy everything from the endpoint and delete the uploaded models."""
    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(args.endpoint_id)
    try:
        model_names = [deployed.model for deployed in endpoint.list_models()]
        endpoint.undeploy_all()
        for name in model_names:
            aiplatform.Model(name).delete()
    except Exception as e:
        raise RuntimeError(
            f"Vertex undeploy failed for endpoint={args.endpoint_id!r}: {e!s}"
        ) from e
    logger.info(
        "Undeployed and deleted models.",
        extra={LOG_EXTRA_KEY: {"endpoint": args.endpoint_id, "models": model_names}},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("deploy", "undeploy"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--project", required=True)
        cmd.add_argument("--region", required=True)
        cmd.add_argument("--endpoint-id", required=True)
    dep = sub.choices["deploy"]
    dep.add_argument("--image-uri", required=True)
    dep.add_argument("--model-id", required=True)
    dep.add_argument("--runner", required=True, choices=["generate", "embed"])
    dep.add_argument("--machine-type", required=True)
    dep.add_argument("--accelerator-type", required=True)
    dep.add_argument("--accelerator-count", required=True, type=int)
    dep.add_argument("--service-account", required=True)
    dep.add_argument("--display-name", required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    if args.command == "deploy":
        deploy(args)
    else:
        undeploy(args)


if __name__ == "__main__":
    main()
