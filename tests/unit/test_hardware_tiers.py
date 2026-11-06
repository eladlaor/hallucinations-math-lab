"""Hardware tier recommendation: which Gemma 4 model a machine can run locally.

`recommend` is pure — facts in, recommendation out — so every branch is tested
without touching the machine. The probe helpers are tested on canned text.
"""

from dataclasses import replace
from typing import Any

import pytest

from demo.check_hardware import parse_meminfo_gb, parse_nvidia_smi_gb
from demo.hardware_tiers import (
    HARDWARE_BY_MODEL,
    Accelerator,
    MachineFacts,
    download_minutes,
    recommend,
)
from demo.model_registry import list_model_ids

PLENTY_OF_DISK_GB = 500.0


BASE_FACTS = MachineFacts(
    os="Linux",
    arch="x86_64",
    memory_gb=64.0,
    nvidia_gb=0.0,
    cuda=False,
    mps=False,
    free_disk_gb=PLENTY_OF_DISK_GB,
    download_mbps=None,
)


def _facts(**overrides: Any) -> MachineFacts:
    return replace(BASE_FACTS, **overrides)


def test_every_registry_model_has_hardware_figures_and_no_extras() -> None:
    assert set(HARDWARE_BY_MODEL) == set(list_model_ids())


@pytest.mark.parametrize(
    ("nvidia_gb", "model", "tier"),
    [
        (16.0, "google/gemma-4-E2B-it", 1),
        (24.0, "google/gemma-4-E4B-it", 2),
        (80.0, "google/gemma-4-26B-A4B-it", 3),
        (160.0, "google/gemma-4-31B-it", 3),
    ],
)
def test_nvidia_picks_the_largest_model_that_fits(nvidia_gb: float, model: str, tier: int) -> None:
    r = recommend(_facts(nvidia_gb=nvidia_gb, cuda=True))
    assert (r.model_id, r.tier, r.accelerator) == (model, tier, Accelerator.NVIDIA)


def test_small_nvidia_gpu_falls_back_to_cpu_when_memory_allows() -> None:
    r = recommend(_facts(nvidia_gb=8.0, cuda=True, memory_gb=32.0))
    assert (r.model_id, r.accelerator) == ("google/gemma-4-E2B-it", Accelerator.CPU)


def test_apple_silicon_36gb_runs_e4b() -> None:
    r = recommend(_facts(os="Darwin", arch="arm64", memory_gb=36.0, mps=True))
    assert (r.model_id, r.tier, r.accelerator) == ("google/gemma-4-E4B-it", 2, Accelerator.APPLE)


def test_apple_silicon_16gb_is_tier_0() -> None:
    r = recommend(_facts(os="Darwin", arch="arm64", memory_gb=16.0, mps=True))
    assert (r.model_id, r.tier, r.accelerator) == (None, 0, Accelerator.NONE)


def test_large_mac_estimates_are_flagged_untested() -> None:
    r = recommend(_facts(os="Darwin", arch="arm64", memory_gb=128.0, mps=True))
    assert r.model_id == "google/gemma-4-31B-it"
    assert any("untested" in n for n in r.notes)


def test_cpu_only_with_enough_memory_runs_e2b_slowly() -> None:
    r = recommend(_facts(memory_gb=32.0))
    assert (r.model_id, r.tier, r.accelerator) == ("google/gemma-4-E2B-it", 1, Accelerator.CPU)
    assert any("slow" in n for n in r.notes)


def test_cpu_only_with_little_memory_is_tier_0() -> None:
    r = recommend(_facts(memory_gb=16.0))
    assert (r.model_id, r.tier) == (None, 0)


def test_short_disk_drops_to_a_model_that_fits() -> None:
    # 36 GB Mac fits E4B (16.0 GB download) but only has room for E2B (10.2 GB).
    r = recommend(_facts(os="Darwin", arch="arm64", memory_gb=36.0, mps=True, free_disk_gb=20.0))
    assert r.model_id == "google/gemma-4-E2B-it"
    assert any("disk" in n for n in r.notes)


def test_no_disk_for_any_model_is_tier_0() -> None:
    r = recommend(_facts(nvidia_gb=80.0, cuda=True, free_disk_gb=5.0))
    assert (r.model_id, r.tier) == (None, 0)


def test_download_time_reported_when_speed_known() -> None:
    r = recommend(_facts(nvidia_gb=16.0, cuda=True, download_mbps=100.0))
    assert r.download_minutes == pytest.approx(10.2 * 8000 / 100 / 60)


def test_download_minutes_formula() -> None:
    assert download_minutes(10.0, 100.0) == pytest.approx(10.0 * 8000 / 100 / 60)


def test_unsupported_os_fails_fast() -> None:
    with pytest.raises(ValueError, match="Windows"):
        recommend(_facts(os="Windows"))


def test_parse_nvidia_smi_sums_gpus_in_gb() -> None:
    assert parse_nvidia_smi_gb("81920\n81920\n") == pytest.approx(160.0)


def test_parse_nvidia_smi_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="nvidia-smi"):
        parse_nvidia_smi_gb("N/A\n")


def test_parse_meminfo_reads_mem_total() -> None:
    text = "MemTotal:       33554432 kB\nMemFree:         1000 kB\n"
    assert parse_meminfo_gb(text) == pytest.approx(32.0)
