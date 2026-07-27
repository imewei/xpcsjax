"""Regression tests for the CPU HPC configuration helpers.

The thread-reservation arithmetic and the ``lscpu`` NUMA fallback are exercised
without letting ``configure_cpu_hpc`` mutate ``os.environ`` (OMP/BLAS thread
counts leak into every sibling test in the same xdist worker otherwise).
"""

from __future__ import annotations

from typing import Any

import pytest

from xpcsjax.device import benchmark_device_performance
from xpcsjax.device import cpu as device_cpu


@pytest.fixture
def isolated_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter the env-mutating / JAX-touching stages of ``configure_cpu_hpc``."""
    monkeypatch.setattr(device_cpu, "_set_cpu_environment_variables", lambda *a, **k: {})
    monkeypatch.setattr(device_cpu, "_configure_jax_cpu", lambda *a, **k: {})


@pytest.mark.parametrize(
    ("physical_cores", "expected_threads"),
    [
        (8, 8),  # below both tiers: no reservation
        (16, 14),  # exact 16-core boundary: 2 reserved, not 0
        (20, 18),
        (31, 29),
        (32, 28),  # exact 32-core boundary: 4 reserved, not 0
        (36, 32),
        (128, 124),
    ],
)
def test_thread_reservation_at_tier_boundaries(
    physical_cores: int,
    expected_threads: int,
    isolated_configure: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cores are reserved for the OS even at the exact tier thresholds."""
    monkeypatch.setattr(
        device_cpu,
        "detect_cpu_info",
        lambda: {
            "physical_cores": physical_cores,
            "logical_cores": physical_cores * 2,
            "numa_nodes": 1,
            "optimization_flags": [],
        },
    )

    config = device_cpu.configure_cpu_hpc()

    assert config["threads_configured"] == expected_threads


def _fake_lscpu(monkeypatch: pytest.MonkeyPatch, numa_value: str) -> None:
    """Make ``detect_cpu_info``'s lscpu probe return a given NUMA node value."""

    class _Result:
        returncode = 0
        stdout = f"Architecture: x86_64\nNUMA node(s): {numa_value}\n"

    monkeypatch.setattr(device_cpu.shutil, "which", lambda _name: "/usr/bin/lscpu")
    monkeypatch.setattr(device_cpu.subprocess, "run", lambda *a, **k: _Result())


def test_malformed_lscpu_numa_value_does_not_skip_optimization_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer NUMA count falls back to 1 without losing the flags block."""
    _fake_lscpu(monkeypatch, "2")
    good: dict[str, Any] = device_cpu.detect_cpu_info()

    _fake_lscpu(monkeypatch, "two")
    malformed: dict[str, Any] = device_cpu.detect_cpu_info()

    assert good["numa_nodes"] == 2
    assert malformed["numa_nodes"] == 1
    assert malformed["optimization_flags"] == good["optimization_flags"]


def test_benchmark_rejects_test_size_larger_than_available_memory() -> None:
    """An oversized benchmark raises ValueError instead of dying on allocation."""
    with pytest.raises(ValueError, match="test_size"):
        device_cpu.benchmark_cpu_performance(test_size=10**6)


def test_benchmark_device_performance_reports_oversized_test_size() -> None:
    """The public wrapper returns the documented error dict, not a crash."""
    results = benchmark_device_performance(test_size=10**6)

    assert "error" in results
    assert "test_size" in results["error"]
