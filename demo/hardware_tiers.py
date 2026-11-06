"""Which Gemma 4 model a machine can run locally, and why.

Holds each registry model's hardware figures and a pure `recommend` function:
facts about the machine in, a tier and model out. Gathering the facts is
`check_hardware.py`'s job. The README's hardware table is the prose version of
`HARDWARE_BY_MODEL`; keep the two in step.

Memory ≈ 2 bytes per parameter (bf16, unquantized) plus room for generation.
The Mac minimums are higher than the NVIDIA ones because macOS lets the GPU
use only part of the shared memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final


class Accelerator(StrEnum):
    NVIDIA = "nvidia"
    APPLE = "apple"
    CPU = "cpu"
    NONE = "none"


SUPPORTED_OS: Final[frozenset[str]] = frozenset({"Linux", "Darwin"})
# Free disk to leave beyond a model's download (cache metadata, other files).
DISK_HEADROOM_GB: Final[float] = 5.0
MEGABITS_PER_GB: Final[float] = 8000.0
SECONDS_PER_MINUTE: Final[float] = 60.0
TIER_NONE: Final[int] = 0


@dataclass(frozen=True)
class ModelHardware:
    model_id: str
    tier: int
    download_gb: float  # bf16 safetensors on the Hugging Face Hub
    min_nvidia_gb: float  # total GPU memory across GPUs
    min_mac_gb: float  # total Apple Silicon memory
    min_cpu_memory_gb: float | None  # None: not practical without a GPU
    mac_estimate_untested: bool = False


# Ordered smallest to largest; `recommend` tries them largest first.
HARDWARE: Final[tuple[ModelHardware, ...]] = (
    ModelHardware("google/gemma-4-E2B-it", 1, 10.2, 16.0, 24.0, 24.0),
    ModelHardware("google/gemma-4-E4B-it", 2, 16.0, 24.0, 32.0, None),
    ModelHardware("google/gemma-4-26B-A4B-it", 3, 51.6, 80.0, 96.0, None, True),
    ModelHardware("google/gemma-4-31B-it", 3, 62.5, 160.0, 128.0, None, True),
)
HARDWARE_BY_MODEL: Final[dict[str, ModelHardware]] = {h.model_id: h for h in HARDWARE}


@dataclass(frozen=True)
class MachineFacts:
    os: str  # platform.system(): "Linux", "Darwin", ...
    arch: str  # platform.machine(): "x86_64", "arm64", ...
    memory_gb: float  # total system memory
    nvidia_gb: float  # total NVIDIA GPU memory; 0 when there is none
    cuda: bool  # PyTorch can use an NVIDIA GPU
    mps: bool  # PyTorch can use the Apple Silicon GPU
    free_disk_gb: float  # where Hugging Face caches models
    download_mbps: float | None  # None when not measured


@dataclass(frozen=True)
class Recommendation:
    tier: int
    model_id: str | None
    accelerator: Accelerator
    download_gb: float | None
    download_minutes: float | None
    notes: list[str] = field(default_factory=list)


def download_minutes(size_gb: float, mbps: float) -> float:
    """Minutes to download `size_gb` at `mbps` megabits per second."""
    if mbps <= 0:
        raise ValueError(f"Download speed must be positive, got {mbps} Mbit/s.")
    return size_gb * MEGABITS_PER_GB / mbps / SECONDS_PER_MINUTE


def _gpu_accelerator(facts: MachineFacts, hw: ModelHardware) -> Accelerator | None:
    if facts.cuda and facts.nvidia_gb >= hw.min_nvidia_gb:
        return Accelerator.NVIDIA
    if facts.mps and facts.memory_gb >= hw.min_mac_gb:
        return Accelerator.APPLE
    return None


def _fits_disk(facts: MachineFacts, hw: ModelHardware) -> bool:
    return facts.free_disk_gb >= hw.download_gb + DISK_HEADROOM_GB


def _chosen(
    facts: MachineFacts, hw: ModelHardware, accelerator: Accelerator, notes: list[str]
) -> Recommendation:
    if accelerator is Accelerator.CPU:
        notes.append("No usable GPU: it runs on the CPU, but each question is slow.")
    if accelerator is Accelerator.APPLE and hw.mac_estimate_untested:
        notes.append(f"The Mac memory figure for {hw.model_id} is an estimate, untested.")
    minutes = (
        download_minutes(hw.download_gb, facts.download_mbps)
        if facts.download_mbps is not None
        else None
    )
    return Recommendation(hw.tier, hw.model_id, accelerator, hw.download_gb, minutes, notes)


def recommend(facts: MachineFacts) -> Recommendation:
    """The largest model this machine can run locally, or tier 0 if none fits."""
    if facts.os not in SUPPORTED_OS:
        raise ValueError(
            f"Unsupported OS {facts.os!r}: only {sorted(SUPPORTED_OS)} are supported for now."
        )
    notes: list[str] = []
    for hw in reversed(HARDWARE):
        accelerator = _gpu_accelerator(facts, hw)
        if accelerator is None:
            continue
        if _fits_disk(facts, hw):
            return _chosen(facts, hw, accelerator, notes)
        notes.append(
            f"{hw.model_id} fits in memory but not on disk: needs {hw.download_gb} GB "
            f"+ {DISK_HEADROOM_GB} GB free, have {facts.free_disk_gb:.1f} GB."
        )
    for hw in reversed(HARDWARE):
        if hw.min_cpu_memory_gb is None or facts.memory_gb < hw.min_cpu_memory_gb:
            continue
        if _fits_disk(facts, hw):
            return _chosen(facts, hw, Accelerator.CPU, notes)
        notes.append(f"{hw.model_id} could run on the CPU but there is not enough free disk.")
    notes.append(
        "No model fits locally: use the library and scoring scripts, or point "
        "SLM_PROVIDER=vllm at a remote server (see the DevOps Guide)."
    )
    return Recommendation(TIER_NONE, None, Accelerator.NONE, None, None, notes)
