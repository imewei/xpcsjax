"""Regression tests for the CPU HPC configuration helpers.

The thread-reservation arithmetic and the ``lscpu`` NUMA fallback are exercised
without letting ``configure_cpu_hpc`` mutate ``os.environ`` (OMP/BLAS thread
counts leak into every sibling test in the same xdist worker otherwise).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import mock_open

import pytest

import xpcsjax.device as device_pkg
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


def _fake_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin detect_cpu_info's /proc/cpuinfo brand+flags read to a fixed Intel/AVX2
    CPU, so ``optimization_flags`` is deterministic regardless of the runner's
    real hardware (an unrecognized brand/flag set would otherwise degrade
    ``optimization_flags`` to ``[]``, letting a broken flags-preservation path
    pass as ``[] == []``).
    """
    fake_cpuinfo = "model name\t: Intel(R) Xeon(R) CPU\nflags\t\t: fpu avx avx2\n"
    monkeypatch.setattr(device_cpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", mock_open(read_data=fake_cpuinfo))


def test_malformed_lscpu_numa_value_does_not_skip_optimization_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer NUMA count falls back to 1 without losing the flags block."""
    _fake_cpuinfo(monkeypatch)
    _fake_lscpu(monkeypatch, "2")
    good: dict[str, Any] = device_cpu.detect_cpu_info()

    _fake_lscpu(monkeypatch, "two")
    malformed: dict[str, Any] = device_cpu.detect_cpu_info()

    assert good["numa_nodes"] == 2
    assert malformed["numa_nodes"] == 1
    assert good["optimization_flags"] == ["intel_mkl", "avx2"]
    assert malformed["optimization_flags"] == good["optimization_flags"]


def test_benchmark_device_performance_reports_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MemoryError from the CPU benchmark surfaces as the documented error
    dict rather than crashing the caller.

    ``benchmark_cpu_performance``'s own ``test_size``-vs-available-memory guard
    (a ValueError) now fires before a real allocation failure would, so this
    exercises the sibling MemoryError branch directly via a mock rather than
    relying on an actual out-of-memory condition.
    """

    def _raise_memory_error(*_a: object, **_k: object) -> None:
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(device_pkg, "benchmark_cpu_performance", _raise_memory_error)

    results = device_pkg.benchmark_device_performance(test_size=100)

    assert "error" in results
    assert "simulated allocation failure" in results["error"]


def test_benchmark_rejects_test_size_larger_than_available_memory() -> None:
    """An oversized benchmark raises ValueError instead of dying on allocation."""
    with pytest.raises(ValueError, match="test_size"):
        device_cpu.benchmark_cpu_performance(test_size=10**6)


def test_configure_cpu_optimal_downgrades_on_jax_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded JAX CPU config error must not be reported as full success.

    configure_cpu_hpc merges _configure_jax_cpu's result (which may carry an
    "error" key) into its return dict. _configure_cpu_optimal previously
    reported configuration_successful=True/performance_ready=True
    unconditionally, hiding that failure.
    """
    monkeypatch.setattr(device_pkg, "HAS_CPU_MODULE", True)
    monkeypatch.setattr(
        device_pkg,
        "configure_cpu_hpc",
        lambda **_kw: {"threads_configured": 4, "error": "simulated JAX config failure"},
    )

    result = device_pkg._configure_cpu_optimal({}, cpu_threads=None)

    assert result["configuration_successful"] is True
    assert result["performance_ready"] is False
    assert "simulated JAX config failure" in result["warnings"][0]


def test_benchmark_device_performance_reports_oversized_test_size() -> None:
    """The public wrapper returns the documented error dict, not a crash."""
    results = benchmark_device_performance(test_size=10**6)

    assert "error" in results
    assert "test_size" in results["error"]
