"""Shell shim: print vLLM launch flags for a given model id, read from the
registry. Used by `serving/Dockerfile` CMD and `serving/deploy_vertex.sh` so
the registry stays the single source of truth.

Usage:
    python3 serving/model_flags.py <model_id> vllm
    python3 serving/model_flags.py <model_id> vertex

`vllm` prints the launch flags as a single line, suitable for shell expansion:
    --model X --dtype Y --max-model-len Z [--quantization Q]

`vertex` prints machine-type, accelerator-type, accelerator-count on three
lines, in that order — easy for `deploy_vertex.sh` to read via `mapfile`.
"""

import sys
from pathlib import Path

# Make the project layout importable when run from a checkout (deploy host).
# `demo.model_registry` lives at <repo_root>/demo/, and the library lives at
# <repo_root>/src/taste/. Inside the serving container PYTHONPATH already
# covers both, so these inserts are no-ops there.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_REPO_ROOT, _REPO_ROOT / "src"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from demo.model_registry import get_model_entry  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <model_id> {{vllm|vertex}}", file=sys.stderr)
        return 2

    model_id, mode = argv[1], argv[2]
    entry = get_model_entry(model_id)

    if mode == "vllm":
        print(" ".join(entry.vllm_flags()))
        return 0
    if mode == "vertex":
        print(entry.machine_type)
        print(entry.accelerator_type)
        print(entry.accelerator_count)
        return 0

    print(f"Unknown mode {mode!r}; expected 'vllm' or 'vertex'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
