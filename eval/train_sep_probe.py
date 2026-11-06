"""Train a Semantic Entropy Probe.

For each prompt in a calibration set:
  1. Sample N completions from the configured SLM.
  2. Compute Semantic Entropy (SE) over those samples (the supervision label).
  3. Collect the SLM's pooled hidden state for the same prompt.
After iterating the set, fit a linear regression of SE on hidden states and
save the resulting `nn.Linear` weights as a torch state-dict, keyed by SLM
model id under `data/probes/<slug>.pt`.

Before the final fit, K-fold cross-validation reports how well the probe
tracks SE on prompts it never saw: Pearson r between predicted and true SE on
each held-out fold, as mean +/- std across folds (one random split of a few
hundred rows is too noisy to trust). The saved probe is then refit on every
row. An in-sample r would be meaningless here: with hidden_dim in the
thousands and a few hundred rows, a linear model can fit the training labels
almost perfectly.

Next to the probe it writes `<probe stem>.train_ids.json` with the training
question ids; run_benchmark.py refuses to evaluate the probe on any of them.

Why local training, not a released probe checkpoint:
  - The probe is specific to the SLM that produced the training hidden
    states. A checkpoint released against Llama-3.1-70B is not transferable
    to Gemma-4-E4B, and vice versa.
  - Training keeps the project self-contained and reproducible per backbone.

Usage:
    uv run python -m eval.train_sep_probe

Without `--calibration-set`, the default 500-question TriviaQA calibration
slice is used, downloaded on first use (eval/datasets/default_question_sets.py).

`auto` writes to `data/probes/<slugified model id>.pt`. Pass an explicit
filename to override.

The calibration JSONL is load_triviaqa.py output: one row per line with `id`
and `question` (other fields are ignored). Build it from a slice that does not
overlap the benchmark. Expected size: a few hundred
prompts is enough for the linear probe; the SE pipeline is the expensive
part (N samples per prompt + NLI clustering).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from demo.slm_provider import build_default_provider
from eval.datasets.default_question_sets import (
    CALIBRATION_SLICE,
    QuestionLoader,
    QuestionSlice,
    ensure_question_set,
)
from eval.sep_probe_provenance import write_training_ids
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, RANDOM_SEED
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropy
from hallucinations_math_lab.services.semantic_entropy_probe import SEPProbe, default_probe_path

logger = logging.getLogger("train_sep_probe")

DEFAULT_CV_FOLDS = 5
# Pearson r needs at least 3 points to be more than a line through 2 points.
MIN_FOLD_ROWS = 3
ID_FIELD = "id"
QUESTION_FIELD = "question"


def cross_validated_pearson_r(
    X: np.ndarray, y: np.ndarray, ridge_alpha: float, n_folds: int, seed: int
) -> tuple[float, float]:
    """Mean and std across K folds of Pearson r between the probe's predictions
    and true SE on each held-out fold."""
    if n_folds < 2:
        raise ValueError(f"Cross-validation needs at least 2 folds, got {n_folds}.")
    if len(y) // n_folds < MIN_FOLD_ROWS:
        raise ValueError(
            f"{len(y)} rows in {n_folds} folds leaves fewer than {MIN_FOLD_ROWS} held-out rows "
            "per fold; use more calibration rows or fewer folds."
        )
    rs: list[float] = []
    for train, held_out in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        predicted = Ridge(alpha=ridge_alpha).fit(X[train], y[train]).predict(X[held_out])
        if np.std(y[held_out]) == 0 or np.std(predicted) == 0:
            raise ValueError(
                "A held-out fold has constant SE labels or predictions, so Pearson r is "
                "undefined. Use a calibration set with a spread of easy and hard questions."
            )
        rs.append(float(np.corrcoef(predicted, y[held_out])[0, 1]))
    return float(np.mean(rs)), float(np.std(rs))


def _iter_questions(path: Path):
    """(id, question) per calibration row; fails fast on a malformed row."""
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
            missing = [k for k in (ID_FIELD, QUESTION_FIELD) if k not in obj]
            if missing:
                raise RuntimeError(
                    f"Calibration set {path} line {i + 1} lacks {missing}. Build it with "
                    "eval.datasets.load_triviaqa (see the README)."
                )
            yield str(obj[ID_FIELD]), obj[QUESTION_FIELD]


def resolve_calibration_path(
    calibration_set: Path | None,
    loader: QuestionLoader | None = None,
    slice_: QuestionSlice = CALIBRATION_SLICE,
) -> Path:
    """The given calibration file, or the default calibration slice (built on first use)."""
    if calibration_set is not None:
        return calibration_set
    return ensure_question_set(slice_, loader=loader)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-set",
        type=Path,
        default=None,
        help=f"Question JSONL; default: {CALIBRATION_SLICE.path}, downloaded on first use.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="auto",
        help="Path to write probe state-dict; 'auto' = data/probes/<slug>.pt",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Folds for the cross-validated SEP-vs-SE Pearson r reported before the final fit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration_set = resolve_calibration_path(args.calibration_set)

    settings = get_settings()
    model_id = settings.vllm_model_id
    if args.output_path == "auto":
        output_path = default_probe_path(model_id)
    else:
        output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    provider = build_default_provider()

    hidden_states: list[np.ndarray] = []
    se_labels: list[float] = []
    question_ids: list[str] = []

    n_prompts = 0
    for question_id, prompt in _iter_questions(calibration_set):
        n_prompts += 1
        question_ids.append(question_id)
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
    r_mean, r_std = cross_validated_pearson_r(
        X, y, args.ridge_alpha, args.cv_folds, seed=RANDOM_SEED
    )
    logger.info(
        "Cross-validated SEP-vs-SE agreement.",
        extra={
            LOG_EXTRA_KEY: {
                "pearson_r_mean": round(r_mean, 4),
                "pearson_r_std": round(r_std, 4),
                "cv_folds": args.cv_folds,
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
    sidecar = write_training_ids(output_path, question_ids, model_id, args.calibration_set)
    logger.info(
        "Saved SEP probe.",
        extra={
            LOG_EXTRA_KEY: {
                "output_path": str(output_path),
                "training_ids_path": str(sidecar),
                "model_id": model_id,
                "n_examples": X.shape[0],
            }
        },
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
