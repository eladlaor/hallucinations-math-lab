"""Shell shim: print registry-derived serving parameters for a given model id.

The registry stays the single source of truth; this shim exposes it to two
non-Python consumers:

  - `serving/Dockerfile` CMD reads `vllm` to get the launch flags.
  - `terraform/` reads `terraform` to get a JSON blob with the Vertex
    machine/accelerator hints, written to
    `terraform/model_registry.auto.tfvars.json` so `terraform apply` picks it
    up automatically.

Usage:
    python3 serving/model_flags.py <model_id> vllm
    python3 serving/model_flags.py <model_id> terraform

`vllm` prints the launch flags as a single line, suitable for shell expansion:
    --model X --dtype Y --max-model-len Z [--quantization Q]

`terraform` prints a single JSON object on stdout with keys
`endpoint_machine_type`, `endpoint_accelerator_type`,
`endpoint_accelerator_count`, `model_id`. Suitable for
`> terraform/model_registry.auto.tfvars.json`.
"""

import json
import sys
from pathlib import Path

# Make the project layout importable when run from a checkout (deploy host).
# `demo.model_registry` lives at <repo_root>/demo/, and the library lives at
# <repo_root>/src/hallucinations_math_lab/. Inside the serving container PYTHONPATH already
# covers both, so these inserts are no-ops there.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from demo.model_registry import get_model_entry  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <model_id> {{vllm|terraform}}", file=sys.stderr)
        return 2

    model_id, mode = argv[1], argv[2]
    entry = get_model_entry(model_id)

    if mode == "vllm":
        print(" ".join(entry.vllm_flags()))
        return 0
    if mode == "terraform":
        print(json.dumps({
            "model_id": entry.model_id,
            "endpoint_machine_type": entry.machine_type,
            "endpoint_accelerator_type": entry.accelerator_type,
            "endpoint_accelerator_count": entry.accelerator_count,
        }, indent=2))
        return 0

    print(f"Unknown mode {mode!r}; expected 'vllm' or 'terraform'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
