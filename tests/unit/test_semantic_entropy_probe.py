"""Semantic Entropy Probe detector tests.

Pin the probe's contract end-to-end:
  - A probe state-dict saved to disk loads back into the detector.
  - The detector declares the correct prerequisites
    (`requires_samples=False`, `requires_hidden_state=True`).
  - `score(...)` runs the linear forward and returns a numeric DetectorScore.
  - A mismatched hidden-dim between the probe and the supplied tensor
    raises with an actionable message.
  - A missing probe file raises at construction time, not request time.
"""

import torch

import pytest

from taste.constants import DetectorName
from taste.services.detector_protocol import (
    DetectionContext,
    DetectorPrerequisiteError,
)
from taste.services.semantic_entropy_probe import (
    SEPProbe,
    SemanticEntropyProbeDetector,
)


@pytest.fixture
def trained_probe_file(tmp_path):
    """Write a trivial SEPProbe state-dict to disk and return the path.

    Weights are set deterministically so the detector's output is predictable:
    `score = sum(h)` (weight=ones, bias=0)."""
    hidden_dim = 8
    probe = SEPProbe(hidden_dim)
    with torch.no_grad():
        probe.linear.weight.copy_(torch.ones(1, hidden_dim))
        probe.linear.bias.zero_()
    path = tmp_path / "fake-probe.pt"
    torch.save(probe.state_dict(), path)
    return path


def test_missing_probe_file_raises_at_construction(tmp_path):
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError, match="SEP probe file not found"):
        SemanticEntropyProbeDetector(probe_path=missing)


def test_detector_declares_correct_prerequisites(trained_probe_file):
    detector = SemanticEntropyProbeDetector(probe_path=trained_probe_file)
    assert detector.name == DetectorName.SEMANTIC_ENTROPY_PROBE
    assert detector.requires_samples is False
    assert detector.requires_hidden_state is True


def test_score_runs_linear_forward(trained_probe_file):
    detector = SemanticEntropyProbeDetector(probe_path=trained_probe_file)
    hidden = torch.arange(8, dtype=torch.float32)  # sum = 28
    ctx = DetectionContext(prompt="q", samples=None, pooled_hidden_state=hidden)
    result = detector.score(ctx)
    assert result.name == DetectorName.SEMANTIC_ENTROPY_PROBE
    assert result.score == pytest.approx(28.0, abs=1e-5)


def test_score_raises_without_hidden_state(trained_probe_file):
    detector = SemanticEntropyProbeDetector(probe_path=trained_probe_file)
    ctx = DetectionContext(prompt="q", samples=None, pooled_hidden_state=None)
    with pytest.raises(DetectorPrerequisiteError, match="pooled hidden state"):
        detector.score(ctx)


def test_score_raises_on_hidden_dim_mismatch(trained_probe_file):
    detector = SemanticEntropyProbeDetector(probe_path=trained_probe_file)
    wrong_dim = torch.zeros(16)
    ctx = DetectionContext(prompt="q", samples=None, pooled_hidden_state=wrong_dim)
    with pytest.raises(RuntimeError, match="probe expects"):
        detector.score(ctx)
