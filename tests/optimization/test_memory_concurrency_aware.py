"""Concurrency-aware memory-budget routing (OOM-overcommit prevention).

Background
----------
Running the suite under pytest-xdist ``-n auto`` (one worker per CPU) used to
kernel-OOM-kill: each worker's ``get_adaptive_memory_threshold`` independently
budgeted ``total * 0.75`` as if it owned the whole box, so N workers overcommit
RAM ~N-fold. The same defect lives in the production multistart / parallel
accumulator ``ProcessPoolExecutor`` pools.

The fix makes the budget *concurrency-aware*: ``effective = available * fraction
/ max(1, worker_count)`` where ``worker_count`` is detected from an explicit
argument, the ``XPCSJAX_FIT_CONCURRENCY`` env-var (set by the production pools),
or ``PYTEST_XDIST_WORKER_COUNT`` (set by xdist), defaulting to 1 (single fit →
full budget, no behavior change).

These tests pin the new contract and act as the regression guard the Makefile's
``PARALLEL_DESELECT`` denylist could never provide.
"""

from __future__ import annotations

import os

import pytest

from xpcsjax.optimization.nlsq import heterodyne_memory as het_mem
from xpcsjax.optimization.nlsq import memory as mem

_GB = 1024**3


# ---------------------------------------------------------------------------
# Concurrency detection precedence
# ---------------------------------------------------------------------------


def test_detect_fit_concurrency_default_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    assert mem._detect_fit_concurrency() == 1


def test_detect_fit_concurrency_explicit_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPCSJAX_FIT_CONCURRENCY", "8")
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "16")
    assert mem._detect_fit_concurrency(3) == 3


def test_detect_fit_concurrency_fit_env_beats_xdist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPCSJAX_FIT_CONCURRENCY", "8")
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "16")
    assert mem._detect_fit_concurrency() == 8


def test_detect_fit_concurrency_reads_xdist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "10")
    assert mem._detect_fit_concurrency() == 10


def test_detect_fit_concurrency_floors_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "0")
    assert mem._detect_fit_concurrency() == 1
    assert mem._detect_fit_concurrency(-5) == 1


def test_detect_fit_concurrency_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XPCSJAX_FIT_CONCURRENCY", "not-an-int")
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    assert mem._detect_fit_concurrency() == 1


# ---------------------------------------------------------------------------
# Threshold scales down with concurrency (homodyne)
# ---------------------------------------------------------------------------


def test_threshold_shrinks_with_explicit_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 32.0 * _GB)
    single, _ = mem.get_adaptive_memory_threshold(0.5, concurrency=1)
    octo, _ = mem.get_adaptive_memory_threshold(0.5, concurrency=8)
    assert single == pytest.approx(16.0)
    assert octo == pytest.approx(2.0)


def test_threshold_shrinks_under_pytest_xdist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item-3 regression guard: PYTEST_XDIST_WORKER_COUNT shrinks the budget.

    This is the exact assertion whose absence let the denylist drift go unnoticed.
    """
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 60.0 * _GB)
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    single, _ = mem.get_adaptive_memory_threshold(0.5, concurrency=1)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "12")
    parallel, info = mem.get_adaptive_memory_threshold(0.5)
    assert info["concurrency"] == 12
    assert parallel == pytest.approx(single / 12)


def test_threshold_prefers_available_over_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 10.0 * _GB)
    monkeypatch.setattr(mem, "detect_total_system_memory", lambda: 64.0 * _GB)
    threshold, info = mem.get_adaptive_memory_threshold(0.5, concurrency=1)
    assert info["memory_basis"] == "available"
    assert info["available_memory_gb"] == pytest.approx(10.0)
    assert info["total_memory_gb"] == pytest.approx(64.0)
    # threshold tracks AVAILABLE (10*0.5), not TOTAL (64*0.5).
    assert threshold == pytest.approx(5.0)


def test_threshold_falls_back_to_total_when_available_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: None)
    monkeypatch.setattr(mem, "detect_total_system_memory", lambda: 40.0 * _GB)
    threshold, info = mem.get_adaptive_memory_threshold(0.5, concurrency=1)
    assert info["memory_basis"] == "total"
    assert threshold == pytest.approx(20.0)


def test_threshold_concurrency_recorded_in_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 32.0 * _GB)
    _, info = mem.get_adaptive_memory_threshold(0.5, concurrency=4)
    assert info["concurrency"] == 4


# ---------------------------------------------------------------------------
# Strategy escalation under concurrency (the actual OOM-prevention behaviour)
# ---------------------------------------------------------------------------


def test_select_strategy_escalates_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit that is STANDARD when alone escalates to OUT_OF_CORE under load."""
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 16.0 * _GB)
    # threshold(conc=1, frac=0.75) = 12 GB. Size the peak at ~6 GB.
    n_points = int(6 * _GB / (14 * 8 * 6.5))
    d1 = mem.select_nlsq_strategy(n_points, 14, memory_fraction=0.75, concurrency=1)
    assert d1.strategy is mem.NLSQStrategy.STANDARD  # 6 GB < 12 GB
    # threshold(conc=4) = 3 GB; peak 6 GB > 3 GB, index tiny -> OUT_OF_CORE.
    d4 = mem.select_nlsq_strategy(n_points, 14, memory_fraction=0.75, concurrency=4)
    assert d4.strategy is mem.NLSQStrategy.OUT_OF_CORE


# ---------------------------------------------------------------------------
# Heterodyne flavour mirrors the contract
# ---------------------------------------------------------------------------


def test_heterodyne_threshold_shrinks_with_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(het_mem, "detect_available_system_memory", lambda: 32.0)
    single = het_mem._get_memory_threshold(0.5, concurrency=1)
    quad = het_mem._get_memory_threshold(0.5, concurrency=4)
    assert single == pytest.approx(16.0)
    assert quad == pytest.approx(4.0)


def test_heterodyne_select_strategy_accepts_concurrency() -> None:
    d = het_mem.select_nlsq_strategy(1000, 10, concurrency=4)
    assert d.strategy is het_mem.NLSQStrategy.STANDARD  # tiny fit, always STANDARD


def test_heterodyne_select_strategy_escalates_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(het_mem, "detect_available_system_memory", lambda: 16.0)
    n_points = int(6 * _GB / (14 * 8 * 6.5))
    d1 = het_mem.select_nlsq_strategy(n_points, 14, memory_fraction=0.75, concurrency=1)
    assert d1.strategy is het_mem.NLSQStrategy.STANDARD
    d4 = het_mem.select_nlsq_strategy(n_points, 14, memory_fraction=0.75, concurrency=4)
    assert d4.strategy is het_mem.NLSQStrategy.LARGE


# ---------------------------------------------------------------------------
# Production multistart / accumulator pool twin: the env-var seam
# ---------------------------------------------------------------------------


def test_set_fit_concurrency_env_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPCSJAX_FIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
    try:
        mem.set_fit_concurrency_env(7)
        assert os.environ["XPCSJAX_FIT_CONCURRENCY"] == "7"
        assert mem._detect_fit_concurrency() == 7
    finally:
        os.environ.pop("XPCSJAX_FIT_CONCURRENCY", None)


# ---------------------------------------------------------------------------
# Item 2: XLA host-device count drops under xdist (80-device explosion fix)
# ---------------------------------------------------------------------------


def test_xla_host_device_count_full_when_single_worker() -> None:
    import xpcsjax

    assert xpcsjax._xla_host_device_count(1) == 4


def test_xla_host_device_count_drops_under_parallel() -> None:
    import xpcsjax

    assert xpcsjax._xla_host_device_count(20) == 1
    assert xpcsjax._xla_host_device_count(2) == 1
