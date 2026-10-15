"""Train a Semantic Entropy Probe.

For each prompt in a calibration set:
  1. Sample N completions from the configured SLM.
  2. Compute Semantic Entropy (SE) over those samples (the supervision label).
  3. Collect the SLM's pooled hidden state for the same prompt.
After iterating the set, fit a linear regression of SE on hidden states and
save the resulting `nn.Linear` weights as a torch state-dict, keyed by SLM
model id under `data/probes/<slug>.pt`.

Before the final fit, a held-out split reports how well the probe tracks SE
on prompts it never saw (Pearson r between predicted and true SE). The saved
probe is then refit on every row. An in-sample r would be meaningless here:
with hidden_dim in the thousands and a few hundred rows, a linear model can
fit the training labels almost perfectly.

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

from demo.slm_provider import build_default_provider
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, RANDOM_SEED
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropy
from hallucinations_math_lab.services.semantic_entropy_probe import SEPProbe

logger = logging.getLogger("train_sep_probe")

DEFAULT_HOLDOUT_FRACTION = 0.2
# Pearson r needs at least 3 points to be more than a line through 2 points.
MIN_HOLDOUT_ROWS = 3


def holdout_pearson_r(
    X: np.ndarray, y: np.ndarray, ridge_alpha: float, holdout_fraction: float, seed: int
) -> float:
    """Fit on a random (1 - holdout_fraction) of rows; return Pearson r between
    the probe's predictions and true SE on the held-out rows."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}.")
    n_holdout = round(len(y) * holdout_fraction)
    n_train = len(y) - n_holdout
    if n_holdout < MIN_HOLDOUT_ROWS or n_train < MIN_HOLDOUT_ROWS:
        raise ValueError(
            f"{len(y)} rows with holdout_fraction={holdout_fraction} leaves {n_train} train / "
            f"{n_holdout} held-out rows; need at least {MIN_HOLDOUT_ROWS} of each."
        )
    order = np.random.default_rng(seed).permutation(len(y))
    holdout, train = order[:n_holdout], order[n_holdout:]
    ridge = Ridge(alpha=ridge_alpha).fit(X[train], y[train])
    predicted = ridge.predict(X[holdout])
    if np.std(y[holdout]) == 0 or np.std(predicted) == 0:
        raise ValueError(
            "Held-out SE labels or predictions are constant, so Pearson r is undefined. "
            "Use a calibration set with a spread of easy and hard questions."
        )
    return float(np.corrcoef(predicted, y[holdout])[0, 1])


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
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Share of rows held out to report SEP-vs-SE Pearson r before the final fit.",
    )
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
                "Training a SEP probe requires hidden-state access: use SLM_PROVIDER="
                "hf_transformers, or for vLLM set VLLM_EMBED_BASE_URL to an embedding "
                "server (see the DevOps Guide)."
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
    r = holdout_pearson_r(X, y, args.ridge_alpha, args.holdout_fraction, seed=RANDOM_SEED)
    logger.info(
        "Held-out SEP-vs-SE agreement.",
        extra={
            LOG_EXTRA_KEY: {
                "pearson_r": round(r, 4),
                "holdout_fraction": args.holdout_fraction,
                "n_examples": X.shape[0],
            }
        },
    )
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
        probe.linear.weight.copy_(torch.tensor(ridge.coef_, dtype=torch.float32).unsqueeze(0))
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
