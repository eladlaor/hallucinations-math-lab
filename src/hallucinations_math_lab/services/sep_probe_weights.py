"""The SEP probe's weight file, `probe.safetensors`: one linear layer, no code.

    weight  float32 [F]   F = len(layers) * hidden_size, layers concatenated in increasing order
    bias    float32 [1]
    header  {"format": "sep-logistic", "schema_version": "1"}

p(high SE) = sigmoid(x · weight + bias). The file is read with safetensors
alone: unlike a pickled torch state-dict, it cannot run code when loaded, so
a probe downloaded from the Hub is safe to open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from hallucinations_math_lab.constants import (
    SEP_PROBE_BIAS_TENSOR,
    SEP_PROBE_FORMAT,
    SEP_PROBE_FORMAT_KEY,
    SEP_PROBE_SCHEMA_VERSION,
    SEP_PROBE_SCHEMA_VERSION_KEY,
    SEP_PROBE_WEIGHT_TENSOR,
)

PROBE_DTYPE: Final = np.float32
_TENSOR_FRAMEWORK: Final[str] = "numpy"
_HEADER: Final[dict[str, str]] = {
    SEP_PROBE_FORMAT_KEY: SEP_PROBE_FORMAT,
    SEP_PROBE_SCHEMA_VERSION_KEY: str(SEP_PROBE_SCHEMA_VERSION),
}
_BIAS_SHAPE: Final[tuple[int]] = (1,)


@dataclass(frozen=True, eq=False)
class ProbeWeights:
    """A SEP probe's linear layer, float32: `weight` [F] and `bias` [1]."""

    weight: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.weight.dtype != PROBE_DTYPE or self.bias.dtype != PROBE_DTYPE:
            raise ValueError(
                f"Probe weight and bias must be float32; got {self.weight.dtype} and "
                f"{self.bias.dtype}."
            )
        if self.weight.ndim != 1 or self.weight.size == 0 or self.bias.shape != _BIAS_SHAPE:
            raise ValueError(
                f"Probe weight must be [F] with F >= 1 and bias [1]; got {self.weight.shape} "
                f"and {self.bias.shape}."
            )
        if not (np.all(np.isfinite(self.weight)) and np.all(np.isfinite(self.bias))):
            raise ValueError("Probe weight and bias must be finite.")


def write_probe_weights(path: Path, weights: ProbeWeights) -> None:
    """Write `weights` to `path` as a safetensors file with the format header."""
    try:
        save_file(
            {SEP_PROBE_WEIGHT_TENSOR: weights.weight, SEP_PROBE_BIAS_TENSOR: weights.bias},
            path,
            metadata=dict(_HEADER),
        )
    except (OSError, SafetensorError) as e:
        raise RuntimeError(f"Failed to write SEP probe weights to {path}: {e!s}") from e


def read_probe_weights(path: Path) -> ProbeWeights:
    """Read a probe written by `write_probe_weights`; raises ValueError on anything
    else, including a file with other tensors or another format header."""
    try:
        with safe_open(path, framework=_TENSOR_FRAMEWORK) as f:
            header = f.metadata() or {}
            tensors = {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118 (not a dict)
    except (OSError, SafetensorError) as e:
        raise ValueError(f"Cannot read SEP probe weights {path}: {e!s}") from e
    if header != _HEADER:
        raise ValueError(
            f"SEP probe weights {path} have header {header}; expected {_HEADER}. It is not a "
            "probe this code wrote, or it was written by another schema version."
        )
    expected_names = {SEP_PROBE_WEIGHT_TENSOR, SEP_PROBE_BIAS_TENSOR}
    if set(tensors) != expected_names:
        raise ValueError(
            f"SEP probe weights {path} hold tensors {sorted(tensors)}; expected "
            f"{sorted(expected_names)}."
        )
    try:
        return ProbeWeights(
            weight=tensors[SEP_PROBE_WEIGHT_TENSOR], bias=tensors[SEP_PROBE_BIAS_TENSOR]
        )
    except ValueError as e:
        raise ValueError(f"SEP probe weights {path} are malformed: {e!s}") from e
