"""Red tests for the Codex round-1 `optrest` shard (assorted xpcsjax/optimization modules).

Each test asserts the claimed symptom through the real code path and is
expected to FAIL until the corresponding defect is fixed.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def test_optrest_b1(monkeypatch):
    """A compute_accumulators timeout must not then join the still-running worker."""
    from xpcsjax.optimization.nlsq import parallel_accumulator as pa

    hang_s = 1.5
    release = threading.Event()
    monkeypatch.setattr(pa, "_batch_timeout", lambda _n: 0.05)
    monkeypatch.setattr(pa, "_ooc_compute_chunk", lambda _args: release.wait(hang_s))

    # Real executor + real pool methods; only the spawn/JAX worker init is skipped.
    pool = pa.OOCComputePool.__new__(pa.OOCComputePool)
    pool._executor = ThreadPoolExecutor(max_workers=1)
    pool._n_chunks = 1
    pool._shutdown = False

    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            with pool:
                pool.compute_accumulators(np.ones(1))
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < hang_s / 2, (
        f"pool exit blocked {elapsed:.2f}s on the worker that already timed out"
    )


def test_optrest_b2():
    """A retry that converges must yield a successful public result."""
    from xpcsjax.optimization.nlsq.fallback_chain import OptimizationStrategy
    from xpcsjax.optimization.nlsq.recovery import execute_with_recovery
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    x = np.arange(3.0)
    solver = Mock(
        side_effect=[
            RuntimeError("max iterations"),
            (np.array([2.0]), np.array([[4.0]])),
        ]
    )
    p, cov, _, actions, status = execute_with_recovery(
        lambda x, a: a * x,
        x,
        2 * x,
        np.array([1.0]),
        None,
        OptimizationStrategy.STANDARD,
        logging.getLogger(__name__),
        "linear",
        "jac",
        None,
        solver,
        None,
    )
    result = OptimizationResult(
        p, np.sqrt(np.diag(cov)), cov, 0.0, 0.0, status, 1, 0.0, {}, recovery_actions=actions
    )
    assert solver.call_count == 2 and actions
    assert result.success, f"status={status!r}, message={result.message!r}"


def test_optrest_b5(monkeypatch):
    """With no explicit fraction, strategy routing must honour NLSQ_MEMORY_FRACTION."""
    from xpcsjax.optimization.nlsq import memory as mem

    monkeypatch.setenv("NLSQ_MEMORY_FRACTION", "0.25")
    monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 16 * 1024**3)
    monkeypatch.setattr(mem, "detect_total_system_memory", lambda: 16 * 1024**3)
    decision = mem.select_nlsq_strategy(8_000_000, 14, concurrency=1)
    assert decision.threshold_gb == pytest.approx(4.0)
    assert decision.strategy == mem.NLSQStrategy.OUT_OF_CORE


def test_optrest_b6():
    """Two distinct angle bins 0.05 deg apart must each get their own contrast."""
    from xpcsjax.optimization.nlsq.parameter_utils import compute_consistent_per_angle_init

    t1, t2 = (a.ravel() for a in np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij"))
    g1_sq = np.exp(-np.abs(t1 - t2))
    data = SimpleNamespace(
        phi_flat=np.repeat([0.0, 0.05], t1.size),
        t1_flat=np.tile(t1, 2),
        t2_flat=np.tile(t2, 2),
        g2_flat=np.concatenate([1 + 0.2 * g1_sq, 1 + 0.6 * g1_sq]),
        q=1.0,
        L=1.0,
        dt=1.0,
    )
    contrasts, _ = compute_consistent_per_angle_init(
        data, np.array([1.0, 0.0, 0.0]), ["D0", "alpha", "D_offset"]
    )
    np.testing.assert_allclose(contrasts, [0.2, 0.6], rtol=1e-6)
