"""serving/launch.py tests: the container's `vllm serve` argv.

Contract pinned here:
  - MODEL_ID is required and must be registered; the runner defaults to
    generate and rejects unknown values.
  - The argv is `vllm serve <model> <registry flags> --host --port --gpu-...`,
    with Vertex's AIP_HTTP_PORT honored.
"""

import pytest

from demo.model_registry import ServingRunner, get_model_entry
from serving.launch import build_argv

MODEL_ID = "google/gemma-4-E4B-it"


def test_generate_is_the_default_runner():
    argv = build_argv({"MODEL_ID": MODEL_ID})
    entry = get_model_entry(MODEL_ID)
    assert argv[:3] == ["vllm", "serve", MODEL_ID]
    assert argv[3 : 3 + len(entry.vllm_flags(ServingRunner.GENERATE))] == entry.vllm_flags(
        ServingRunner.GENERATE
    )


def test_embed_runner_and_vertex_port_are_honored():
    argv = build_argv({"MODEL_ID": MODEL_ID, "VLLM_RUNNER": "embed", "AIP_HTTP_PORT": "9000"})
    assert "--runner" in argv
    assert argv[argv.index("--port") + 1] == "9000"
    assert argv[argv.index("--host") + 1] == "0.0.0.0"


def test_missing_model_id_fails_fast():
    with pytest.raises(ValueError, match="MODEL_ID is not set"):
        build_argv({})


def test_unknown_runner_fails_fast():
    with pytest.raises(ValueError, match="VLLM_RUNNER='classify'"):
        build_argv({"MODEL_ID": MODEL_ID, "VLLM_RUNNER": "classify"})


def test_unregistered_model_fails_fast():
    with pytest.raises(KeyError, match="not in the model registry"):
        build_argv({"MODEL_ID": "acme/imaginary-model"})
