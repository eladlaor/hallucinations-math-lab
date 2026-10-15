"""Print registry-derived serving parameters for a model id.

The model registry (demo/model_registry.py) is the single source of truth;
this CLI exposes it outside Python:

  - `vllm <runner>` prints a ready-to-paste `vllm serve` command for running
    the model on your own GPU box (runner: `generate` or `embed`). The
    container does not use this; `serving/launch.py` builds the same argv.
  - `terraform` prints the Vertex machine / accelerator hints as JSON, for
    `> terraform/model_registry.auto.tfvars.json`.

Usage:
    python3 serving/model_flags.py <model_id> vllm {generate|embed}
    python3 serving/model_flags.py <model_id> terraform
"""

import json
import shlex
import sys
from pathlib import Path

# Make the project layout importable when run from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from demo.model_registry import ServingRunner, get_model_entry

MODE_VLLM = "vllm"
MODE_TERRAFORM = "terraform"
USAGE = (
    "Usage: model_flags.py <model_id> vllm {generate|embed}\n"
    "       model_flags.py <model_id> terraform"
)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(USAGE, file=sys.stderr)
        return 2
    model_id, mode, rest = argv[1], argv[2], argv[3:]
    entry = get_model_entry(model_id)

    if mode == MODE_VLLM and len(rest) == 1:
        runner = ServingRunner(rest[0])
        print(shlex.join(["vllm", "serve", entry.model_id, *entry.vllm_flags(runner)]))
        return 0
    if mode == MODE_TERRAFORM and not rest:
        print(
            json.dumps(
                {
                    "model_id": entry.model_id,
                    "endpoint_machine_type": entry.machine_type,
                    "endpoint_accelerator_type": entry.accelerator_type,
                    "endpoint_accelerator_count": entry.accelerator_count,
                },
                indent=2,
            )
        )
        return 0

    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
