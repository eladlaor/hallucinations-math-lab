"""Check which hardware tier this machine fits, and which model to run.

    uv run python -m demo.check_hardware            # human-readable
    uv run python -m demo.check_hardware --json     # for agents and scripts
    uv run python -m demo.check_hardware --no-network

Collects the facts (OS, memory, NVIDIA GPU memory, what PyTorch can use, free
disk where models are cached, and a short download-speed test), then asks
`hardware_tiers.recommend` for the tier. Linux and macOS only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Final

from demo.hardware_tiers import MachineFacts, Recommendation, recommend
from hallucinations_math_lab.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)

BYTES_PER_GB: Final[int] = 2**30
KIB_PER_GB: Final[int] = 2**20
MIB_PER_GB: Final[int] = 2**10
BITS_PER_BYTE: Final[int] = 8
BITS_PER_MEGABIT: Final[int] = 10**6
OS_MACOS: Final[str] = "Darwin"
OS_LINUX: Final[str] = "Linux"
MEMINFO_PATH: Final[Path] = Path("/proc/meminfo")
MEMINFO_TOTAL_KEY: Final[str] = "MemTotal:"
NVIDIA_SMI: Final[str] = "nvidia-smi"
NVIDIA_SMI_ARGS: Final[tuple[str, ...]] = (
    "--query-gpu=memory.total",
    "--format=csv,noheader,nounits",
)
HF_HOME_ENV: Final[str] = "HF_HOME"
DEFAULT_HF_HOME: Final[Path] = Path.home() / ".cache" / "huggingface"
# ~7 MB file on the Hugging Face Hub, where the models come from.
SPEED_TEST_URL: Final[str] = (
    "https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/parquet/"
    "rc.nocontext/validation/0.parquet"
)
SPEED_TEST_TIMEOUT_S: Final[float] = 60.0
SUBPROCESS_TIMEOUT_S: Final[float] = 30.0
ENV_MODEL_KEY: Final[str] = "VLLM_MODEL_ID"


def parse_nvidia_smi_gb(output: str) -> float:
    """Total memory in GB from `nvidia-smi --query-gpu=memory.total` (MiB per line)."""
    try:
        return sum(float(line) for line in output.split()) / MIB_PER_GB
    except ValueError as e:
        raise ValueError(f"Unexpected nvidia-smi memory output {output!r}: {e}") from e


def parse_meminfo_gb(text: str) -> float:
    """Total memory in GB from the text of /proc/meminfo."""
    for line in text.splitlines():
        if line.startswith(MEMINFO_TOTAL_KEY):
            return float(line.split()[1]) / KIB_PER_GB
    raise ValueError(f"No {MEMINFO_TOTAL_KEY} line in /proc/meminfo.")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.error("Hardware probe command failed", extra={LOG_EXTRA_KEY: {"cmd": cmd}})
        raise RuntimeError(f"Command {' '.join(cmd)!r} failed: {e}") from e


def memory_gb(os_name: str) -> float:
    if os_name == OS_MACOS:
        return int(_run(["sysctl", "-n", "hw.memsize"]).strip()) / BYTES_PER_GB
    try:
        return parse_meminfo_gb(MEMINFO_PATH.read_text())
    except OSError as e:
        raise RuntimeError(f"Cannot read {MEMINFO_PATH}: {e}") from e


def nvidia_gb() -> float:
    """Total NVIDIA GPU memory; 0 when nvidia-smi is not installed."""
    if shutil.which(NVIDIA_SMI) is None:
        return 0.0
    return parse_nvidia_smi_gb(_run([NVIDIA_SMI, *NVIDIA_SMI_ARGS]))


def torch_accelerators() -> tuple[bool, bool]:
    """(cuda, mps): which GPUs PyTorch can actually use."""
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is not installed; run `uv sync --all-extras` first.") from e
    return torch.cuda.is_available(), torch.backends.mps.is_available()


def free_disk_gb() -> float:
    """Free disk where Hugging Face caches models (nearest existing folder)."""
    path = Path(os.environ.get(HF_HOME_ENV, DEFAULT_HF_HOME))
    while not path.exists():
        path = path.parent
    try:
        return shutil.disk_usage(path).free / BYTES_PER_GB
    except OSError as e:
        raise RuntimeError(f"Cannot read free disk space at {path}: {e}") from e


def download_mbps() -> float:
    """Rough download speed from the Hugging Face Hub, in Mbit/s."""
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(SPEED_TEST_URL, timeout=SPEED_TEST_TIMEOUT_S) as resp:
            n_bytes = len(resp.read())
    except OSError as e:
        raise RuntimeError(
            f"Download speed test against {SPEED_TEST_URL} failed: {e}. "
            "Check the network, or rerun with --no-network."
        ) from e
    elapsed = time.perf_counter() - start
    return n_bytes * BITS_PER_BYTE / BITS_PER_MEGABIT / elapsed


def collect_facts(measure_speed: bool) -> MachineFacts:
    os_name = platform.system()
    cuda, mps = torch_accelerators()
    return MachineFacts(
        os=os_name,
        arch=platform.machine(),
        memory_gb=memory_gb(os_name),
        nvidia_gb=nvidia_gb(),
        cuda=cuda,
        mps=mps,
        free_disk_gb=free_disk_gb(),
        download_mbps=download_mbps() if measure_speed else None,
    )


def format_report(facts: MachineFacts, rec: Recommendation) -> str:
    speed = f"{facts.download_mbps:.0f} Mbit/s" if facts.download_mbps else "not measured"
    lines = [
        f"OS            {facts.os} {facts.arch}",
        f"Memory        {facts.memory_gb:.0f} GB",
        f"NVIDIA GPU    {f'{facts.nvidia_gb:.0f} GB' if facts.nvidia_gb else 'none'}",
        f"PyTorch GPU   cuda={facts.cuda}  mps={facts.mps}",
        f"Free disk     {facts.free_disk_gb:.0f} GB (model cache)",
        f"Download      {speed}",
        "",
        f"Tier {rec.tier}: "
        + (f"{rec.model_id} on {rec.accelerator}" if rec.model_id else "no local model"),
    ]
    if rec.download_gb is not None:
        eta = f", about {rec.download_minutes:.0f} min" if rec.download_minutes else ""
        lines.append(f"Download      {rec.download_gb} GB once{eta}")
    lines += [f"Note          {n}" for n in rec.notes]
    if rec.model_id:
        lines += ["", "Set in .env:", f"{ENV_MODEL_KEY}={rec.model_id}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Find this machine's hardware tier.")
    p.add_argument("--json", action="store_true", help="Print facts and tier as JSON.")
    p.add_argument("--no-network", action="store_true", help="Skip the download-speed test.")
    args = p.parse_args()
    facts = collect_facts(measure_speed=not args.no_network)
    rec = recommend(facts)
    if args.json:
        print(json.dumps({"facts": asdict(facts), "recommendation": asdict(rec)}, indent=2))
    else:
        print(format_report(facts, rec))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
