"""Train a Semantic Entropy Probe.

For each prompt in a calibration set:
  1. Sample N completions from the configured SLM.
  2. Compute Semantic Entropy (SE) over those samples (the supervision label).
  3. Collect the SLM's pooled hidden state for the same prompt.
After iterating the set, fit a linear regression of SE on hidden states and
save the resulting `nn.Linear` weights as a torch state-dict, keyed by SLM
model id under `data/probes/<slug>.pt`.

Why local training, not a released probe checkpoint:
  - The probe is specific to the SLM that produced the training hidden
    states. A checkpoint released against Llama-3.1-70B is not transferable
    to Gemma-4-E4B, and vice versa.
  - Training keeps the project self-contained and reproducible per backbone.

Usage:
    uv run python -m eval.train_sep_probe \\
        --calibration-set data/calibration/trivia_qa_500.jsonl \\
        --output-path data/probes/auto

`auto` writes to `data/probes/<slugified model id>.pt`. Pass an explicit
filename to override.

The calibration JSONL is expected to have one prompt per line under the
`prompt` key. Any additional fields are ignored. Expected size: a few hundred
prompts is enough for the linear probe; the SE pipeline is the expensive
part (N samples per prompt + NLI clustering).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropy
from hallucinations_math_lab.services.semantic_entropy_probe import SEPProbe
from demo.slm_provider import build_default_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("train_sep_probe")


def _slugify(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def _iter_prompts(path: Path):
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Calibration set {path} line {i + 1} is not valid JSON: {e}"
                ) from e
            if "prompt" not in obj:
                raise RuntimeError(
                    f"Calibration set {path} line {i + 1} is missing the required 'prompt' field."
                )
            yield obj["prompt"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-set", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        default="auto",
        help="Path to write probe state-dict; 'auto' = data/probes/<slug>.pt",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    settings = get_settings()
    model_id = settings.vllm_model_id
    slug = _slugify(model_id)
    if args.output_path == "auto":
        output_path = Path("data") / "probes" / f"{slug}.pt"
    else:
        output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    provider = build_default_provider()

    hidden_states: list[np.ndarray] = []
    se_labels: list[float] = []

    n_prompts = 0
    for prompt in _iter_prompts(args.calibration_set):
        n_prompts += 1
        try:
            sample_set = provider.sample(prompt)
            se_result = compute_semantic_entropy(sample_set)
            pooled = provider.pooled_hidden_state(prompt)
        except Exception as e:
            raise RuntimeError(
                f"Failed to collect training row for prompt={prompt[:80]!r}: {e!s}"
            ) from e

        if pooled is None:
            raise RuntimeError(
                f"SLM provider returned no pooled hidden state for prompt={prompt[:80]!r}. "
                "Training a SEP probe requires hidden-state access; for vLLM, set "
                "VLLM_EMBED_BASE_URL to a sibling --task embed instance."
            )
        hidden_states.append(pooled.detach().to(dtype=torch.float32).numpy())
        se_labels.append(se_result.semantic_entropy)
        logger.info(
            "Collected training row.",
            extra={
                LOG_EXTRA_KEY: {
                    "index": n_prompts,
                    "se_nats": round(se_result.semantic_entropy, 4),
                    "prompt_preview": prompt[:80],
                }
            },
        )

    if n_prompts == 0:
        raise RuntimeError(f"Calibration set {args.calibration_set} produced zero prompts.")

    X = np.stack(hidden_states)
    y = np.asarray(se_labels, dtype=np.float64)
    logger.info(
        "Fitting linear probe.",
        extra={
            LOG_EXTRA_KEY: {
                "n_examples": X.shape[0],
                "hidden_dim": X.shape[1],
                "ridge_alpha": args.ridge_alpha,
            }
        },
    )
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(X, y)

    hidden_dim = X.shape[1]
    probe = SEPProbe(hidden_dim)
    with torch.no_grad():
        probe.linear.weight.copy_(
            torch.tensor(ridge.coef_, dtype=torch.float32).unsqueeze(0)
        )
        probe.linear.bias.copy_(torch.tensor([float(ridge.intercept_)], dtype=torch.float32))

    torch.save(probe.state_dict(), output_path)
    logger.info(
        "Saved SEP probe.",
        extra={
            LOG_EXTRA_KEY: {
                "output_path": str(output_path),
                "model_id": model_id,
                "n_examples": X.shape[0],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
