"""Regression tests for the CPU HPC configuration helpers.

The thread-reservation arithmetic and the ``lscpu`` NUMA fallback are exercised
without letting ``configure_cpu_hpc`` mutate ``os.environ`` (OMP/BLAS thread
counts leak into every sibling test in the same xdist worker otherwise).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import mock_open

import pytest

from xpcsjax.device import cpu as device_cpu


@pytest.fixture
def isolated_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter the env-mutating / JAX-touching stages of ``configure_cpu_hpc``.

    Also pins the fit-concurrency divisor to 1. ``configure_cpu_hpc`` splits the
    auto-detected thread count across concurrent fits, and these targets run
    under ``-n auto`` (pytest-xdist sets ``PYTEST_XDIST_WORKER_COUNT``), which
    would otherwise make the expected counts a function of the runner's core
    count.
    """
    monkeypatch.setattr(device_cpu, "_set_cpu_environment_variables", lambda *a, **k: {})
    monkeypatch.setattr(device_cpu, "_configure_jax_cpu", lambda *a, **k: {})
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)


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


@pytest.mark.parametrize(
    ("concurrency", "expected_threads"),
    [
        ("1", 124),  # lone fit keeps the full post-reservation count
        ("8", 15),  # 128 -> 124 after OS reservation -> //8
        ("512", 1),  # never drops below one thread
    ],
)
def test_thread_count_is_divided_across_concurrent_fits(
    concurrency: str,
    expected_threads: int,
    isolated_configure: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N concurrent fit processes must not each pin the whole node's cores."""
    monkeypatch.setattr(
        device_cpu,
        "detect_cpu_info",
        lambda: {
            "physical_cores": 128,
            "logical_cores": 256,
            "numa_nodes": 1,
            "optimization_flags": [],
        },
    )
    monkeypatch.setenv("XPCSJAX_FIT_CONCURRENCY", concurrency)

    assert device_cpu.configure_cpu_hpc()["threads_configured"] == expected_threads


def test_explicit_thread_count_is_not_divided(
    isolated_configure: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``num_threads=`` is caller intent and must survive verbatim."""
    monkeypatch.setattr(
        device_cpu,
        "detect_cpu_info",
        lambda: {"physical_cores": 128, "logical_cores": 256, "numa_nodes": 1},
    )
    monkeypatch.setenv("XPCSJAX_FIT_CONCURRENCY", "8")

    assert device_cpu.configure_cpu_hpc(num_threads=64)["threads_configured"] == 64


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


def _cpu_info_no_extras() -> dict[str, Any]:
    """A CPU description with no AVX-512/Intel extras, so _configure_jax_cpu's
    xla_flags reduces to exactly the one deterministic flag under test.
    """
    return {"supports_avx512": False, "cpu_brand": "Generic"}


def test_configure_jax_cpu_skips_warning_when_flag_already_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live backend with the exact flag already in XLA_FLAGS is a harmless
    no-op: no warning, and xla_flags_applied reports True.

    This is the case xpcsjax/__init__.py's pre-import _DEFAULT_XLA_FLAGS entry
    creates on every real run -- device.cpu's own copy of the same flag
    should recognize it's redundant instead of always warning.
    """
    monkeypatch.setattr(device_cpu, "_jax_backend_initialized", lambda: True)
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")

    caplog_records: list[str] = []
    monkeypatch.setattr(
        device_cpu.logger,
        "warning",
        lambda *a, **k: caplog_records.append(str(a)),
    )

    result = device_cpu._configure_jax_cpu(4, _cpu_info_no_extras())

    assert result["xla_flags_applied"] is True
    assert caplog_records == []


def test_configure_jax_cpu_warns_when_flag_value_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag NAME present with a DIFFERENT value must still warn.

    Regression: a substring check on the flag name alone (`"...eigen" not in
    existing_flags`) would treat "--xla_cpu_multi_thread_eigen=false" as
    "already applied" even though the requested value ("=true") never took
    effect -- silently misreporting a real misconfiguration as success.
    """
    monkeypatch.setattr(device_cpu, "_jax_backend_initialized", lambda: True)
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

    result = device_cpu._configure_jax_cpu(4, _cpu_info_no_extras())

    assert result["xla_flags_applied"] is False


def test_configure_jax_cpu_warns_when_flag_entirely_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live backend with no matching flag at all must still warn (the
    pre-PR behavior, preserved for the genuinely-missing case)."""
    monkeypatch.setattr(device_cpu, "_jax_backend_initialized", lambda: True)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    result = device_cpu._configure_jax_cpu(4, _cpu_info_no_extras())

    assert result["xla_flags_applied"] is False
