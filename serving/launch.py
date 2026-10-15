"""Container entrypoint: start `vllm serve` for one registered model and runner.

The argument list is built in Python from the model registry and handed to
`os.execvp`, so no flag (the embed runner's JSON pooler config included)
passes through shell word-splitting or quoting.

Environment:
    MODEL_ID                  Registered model id (required).
    VLLM_RUNNER               `generate` (default) or `embed`.
    AIP_HTTP_PORT             Port to bind; Vertex sets it, default 8080.
    VLLM_GPU_MEMORY_UTILIZATION  Fraction of GPU memory vLLM may use.
"""

import logging
import os
import sys
from pathlib import Path

# Inside the image PYTHONPATH already covers the app dir; these inserts make
# the script runnable from a checkout too.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from demo.model_registry import ServingRunner, get_model_entry  # noqa: E402

logger = logging.getLogger(__name__)

ENV_MODEL_ID = "MODEL_ID"
ENV_RUNNER = "VLLM_RUNNER"
ENV_PORT = "AIP_HTTP_PORT"
ENV_GPU_MEMORY_UTILIZATION = "VLLM_GPU_MEMORY_UTILIZATION"

# Same key as hallucinations_math_lab.constants.LOG_EXTRA_KEY; the serving
# image ships only the model registry, not the library.
LOG_EXTRA_KEY = "ctx"

DEFAULT_PORT = "8080"
# Leaves headroom for CUDA graphs and activations on a 24 GB L4.
DEFAULT_GPU_MEMORY_UTILIZATION = "0.90"
BIND_HOST = "0.0.0.0"
VLLM_EXECUTABLE = "vllm"


def build_argv(env: dict[str, str]) -> list[str]:
    """The full `vllm serve` command line for the given environment."""
    model_id = env.get(ENV_MODEL_ID, "")
    if not model_id:
        raise ValueError(f"{ENV_MODEL_ID} is not set; cannot choose a model to serve.")
    runner_value = env.get(ENV_RUNNER, str(ServingRunner.GENERATE))
    try:
        runner = ServingRunner(runner_value)
    except ValueError as e:
        raise ValueError(
            f"{ENV_RUNNER}={runner_value!r} is not one of {[str(r) for r in ServingRunner]}."
        ) from e
    entry = get_model_entry(model_id)
    return [
        VLLM_EXECUTABLE,
        "serve",
        entry.model_id,
        *entry.vllm_flags(runner),
        "--host",
        BIND_HOST,
        "--port",
        env.get(ENV_PORT, DEFAULT_PORT),
        "--gpu-memory-utilization",
        env.get(ENV_GPU_MEMORY_UTILIZATION, DEFAULT_GPU_MEMORY_UTILIZATION),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    argv = build_argv(dict(os.environ))
    logger.info("Launching vLLM.", extra={LOG_EXTRA_KEY: {"argv": argv}})
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
