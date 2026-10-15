"""Semantic Entropy Probe (SEP) detector.

Kossen et al.'s follow-up to Farquhar — predicts Semantic Entropy from a
single forward pass via a linear probe over the SLM's pooled hidden state.
The probe trades a small accuracy loss for a large cost saving: one forward
pass instead of N=10 generations plus O(N) NLI calls.

Design choices specific to this implementation:

  - The probe is a single `nn.Linear(hidden_dim, 1)` regressor trained to
    predict SE (in nats) as a continuous target. Score units match SE's
    units, so the same threshold scale carries over and SEP can be slotted
    into the escalation comparator as a drop-in for SE.

  - Weights are loaded lazily on first `score(...)` call. Construction takes
    only the probe file path and a hidden-dim expectation; if the file is
    missing the detector raises immediately at construction time so the
    misconfiguration surfaces before the first request, not on the first
    request.

  - Per-model probes. The probe weights are specific to the SLM that
    produced the training hidden states; the registry tracks per-model
    probe paths under `data/probes/<slug>.pt`.

Why a regressor over a classifier:
  - SE is itself a continuous score that the orchestrator's threshold
    comparator already understands.
  - A regressor's output sits on the same scale as the SE label, so
    SEP-vs-SE correlation (the validation target in CLAUDE.md's
    acceptance criteria) reads naturally as Pearson r.
"""

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

from hallucinations_math_lab.constants import LOG_EXTRA_KEY, DetectorCategory, DetectorName
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorPrerequisiteError,
    DetectorScore,
)

logger = logging.getLogger(__name__)


class SEPProbe(nn.Module):
    """Linear regressor predicting SE (nats) from a pooled hidden state."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [hidden_dim] or [batch, hidden_dim]
        return self.linear(x).squeeze(-1)


class SemanticEntropyProbeDetector(Detector):
    """SEP detector implementing the `Detector` protocol.

    `probe_path` points at a torch state-dict produced by
    `eval/train_sep_probe.py`. The file must include `linear.weight` and
    `linear.bias` keys matching a `SEPProbe(hidden_dim)` module."""

    name = DetectorName.SEMANTIC_ENTROPY_PROBE
    requires_samples = False
    requires_hidden_state = True
    metadata = DetectorMetadata(
        signal="linear probe over the SLM's pooled last-layer hidden state at the final prompt token, predicting SE in nats",
        assumption="hallucination uncertainty is linearly decodable from a single forward pass over the prompt",
        failure_modes=(
            "probe is SLM-specific — weights trained on model A do not transfer to model B",
            "underperforms full SE on adversarial / out-of-distribution prompts where the linear approximation breaks",
            "requires hidden-state access; useless against any provider that does not expose pooled embeddings",
        ),
        category=DetectorCategory.UNCERTAINTY,
    )

    def __init__(self, probe_path: str | Path) -> None:
        probe_path = Path(probe_path)
        if not probe_path.is_file():
            raise FileNotFoundError(
                f"SEP probe file not found at {probe_path!s}. Train it first with "
                "`uv run python eval/train_sep_probe.py`."
            )
        self._probe_path = probe_path
        state = torch.load(probe_path, map_location="cpu", weights_only=True)
        weight = state["linear.weight"]
        hidden_dim = weight.shape[1]
        self._probe = SEPProbe(hidden_dim)
        self._probe.load_state_dict(state)
        self._probe.eval()
        self._hidden_dim = hidden_dim
        logger.info(
            "Loaded SEP probe.",
            extra={
                LOG_EXTRA_KEY: {
                    "probe_path": str(probe_path),
                    "hidden_dim": hidden_dim,
                }
            },
        )

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.pooled_hidden_state is None:
            raise DetectorPrerequisiteError(
                "SemanticEntropyProbeDetector requires a pooled hidden state; "
                "DetectionContext.pooled_hidden_state is None."
            )
        h = ctx.pooled_hidden_state.to(dtype=torch.float32)
        if h.shape[-1] != self._hidden_dim:
            raise RuntimeError(
                f"Pooled hidden state has dim {h.shape[-1]} but probe expects {self._hidden_dim}. "
                "Probe was likely trained against a different SLM."
            )
        # SEP is the cheapest detector by design — one linear forward pass.
        # cost_usd is 0 (local CPU); latency_ms captures the wall-clock cost
        # which is what makes SEP attractive on the Pareto frontier.
        start = time.perf_counter()
        with torch.no_grad():
            predicted = self._probe(h).item()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectorScore(
            name=self.name,
            score=float(predicted),
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
