"""Scientific tests for xpcsjax.optimization.nlsq.parallel_accumulator.

Three layers, each with its own correctness contract:

* Reduction core (``accumulate_chunks_*``): parallel == sequential **bit-exact**
  (matrix addition is associative/commutative). Verified with integer-valued
  matrices so float summation order cannot perturb the result.
* JIT kernels (``create_ooc_kernels``): validated without ground-truth physics
  via two exact invariants — an all-diagonal chunk gives chi2 == 0 (the
  ``t1==t2`` mask), and the accumulator chi2 equals the standalone chi2 kernel.
* Shared-memory + worker layer: exercised in-process (calling the worker
  initializer directly) plus one real ``OOCComputePool`` spawn. All guarded by
  a shared-memory availability skip.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import jax.numpy as jnp
import numpy as np
import pytest

from xpcsjax.optimization.nlsq import parallel_accumulator as pa


def _shm_available() -> bool:
    try:
        s = shared_memory.SharedMemory(create=True, size=64)
        s.close()
        s.unlink()
        return True
    except Exception:  # noqa: BLE001 - any failure means SHM is unusable here
        return False


shm_required = pytest.mark.skipif(
    not _shm_available(), reason="POSIX shared memory unavailable in this environment"
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, False), (9, False), (10, True), (50, True)],
)
def test_should_use_parallel_accumulation(n: int, expected: bool) -> None:
    assert pa.should_use_parallel_accumulation(n) is expected


@pytest.mark.parametrize(("n", "expected"), [(9, False), (10, True)])
def test_should_use_parallel_compute(n: int, expected: bool) -> None:
    assert pa.should_use_parallel_compute(n) is expected


# ---------------------------------------------------------------------------
# Sequential reduction
# ---------------------------------------------------------------------------


def _int_chunks(n: int, n_params: int = 2):
    """Integer-valued (JtJ, Jtr, chi2) chunks — exact under float summation."""
    chunks = []
    for i in range(1, n + 1):
        chunks.append(
            (
                np.full((n_params, n_params), float(i)),
                np.full(n_params, float(i)),
                float(i),
            )
        )
    return chunks


def test_accumulate_sequential_sums_correctly() -> None:
    chunks = _int_chunks(4)  # values 1..4
    jtj, jtr, chi2, count = pa.accumulate_chunks_sequential(chunks)
    assert count == 4
    np.testing.assert_array_equal(jtj, np.full((2, 2), 10.0))  # 1+2+3+4
    np.testing.assert_array_equal(jtr, np.full(2, 10.0))
    assert chi2 == pytest.approx(10.0)


def test_accumulate_sequential_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty chunks list"):
        pa.accumulate_chunks_sequential([])


# ---------------------------------------------------------------------------
# Parallel reduction — identical to sequential
# ---------------------------------------------------------------------------


def test_parallel_below_threshold_falls_back() -> None:
    chunks = _int_chunks(5)  # < _MIN_CHUNKS_FOR_PARALLEL
    out = pa.accumulate_chunks_parallel(chunks, n_workers=4)
    expected = pa.accumulate_chunks_sequential(chunks)
    np.testing.assert_array_equal(out[0], expected[0])
    assert out[3] == expected[3] == 5


def test_parallel_zero_workers_falls_back() -> None:
    chunks = _int_chunks(12)
    out = pa.accumulate_chunks_parallel(chunks, n_workers=0)
    expected = pa.accumulate_chunks_sequential(chunks)
    np.testing.assert_array_equal(out[0], expected[0])


def test_parallel_matches_sequential_bit_exact() -> None:
    # >= threshold so the real process-pool path runs; integer values make the
    # commutative/associative reduction bit-exact regardless of worker order.
    chunks = _int_chunks(16)
    par = pa.accumulate_chunks_parallel(chunks, n_workers=2)
    seq = pa.accumulate_chunks_sequential(chunks)
    np.testing.assert_array_equal(par[0], seq[0])
    np.testing.assert_array_equal(par[1], seq[1])
    assert par[2] == pytest.approx(seq[2])
    assert par[3] == seq[3] == 16


def test_parallel_fallback_on_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the pool path to raise -> the except clause falls back to sequential.
    import concurrent.futures

    class _BoomExecutor:
        def __init__(self, *a: object, **k: object) -> None:
            raise OSError("simulated pool creation failure")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BoomExecutor)
    chunks = _int_chunks(12)
    out = pa.accumulate_chunks_parallel(chunks, n_workers=2)
    expected = pa.accumulate_chunks_sequential(chunks)
    np.testing.assert_array_equal(out[0], expected[0])


# ---------------------------------------------------------------------------
# JIT kernel factory
# ---------------------------------------------------------------------------

_Q, _L, _DT = 0.01, 1.0, 1.0
_T_GRID = [1.0, 2.0, 3.0]


def _static_physics_config(per_angle: bool):
    phi = [0.0, 45.0] if per_angle else [0.0]
    return {
        "per_angle_scaling": per_angle,
        "n_phi": len(phi),
        "phi_unique": jnp.asarray(phi),
        "t1_unique": jnp.asarray(_T_GRID),
        "t2_unique": jnp.asarray(_T_GRID),
        "n_t1": len(_T_GRID),
        "n_t2": len(_T_GRID),
        "q": _Q,
        "L": _L,
        "dt": _DT,
    }


def _kernels(cfg):
    return pa.create_ooc_kernels(
        per_angle_scaling=cfg["per_angle_scaling"],
        n_phi=cfg["n_phi"],
        phi_unique=cfg["phi_unique"],
        t1_unique_global=cfg["t1_unique"],
        t2_unique_global=cfg["t2_unique"],
        n_t1=cfg["n_t1"],
        n_t2=cfg["n_t2"],
        q_val=cfg["q"],
        L_val=cfg["L"],
        dt_val=cfg["dt"],
    )


@pytest.mark.parametrize("per_angle", [False, True])
def test_kernel_diagonal_chunk_has_zero_chi2(per_angle: bool) -> None:
    cfg = _static_physics_config(per_angle)
    acc, chi2_fn = _kernels(cfg)
    if per_angle:
        p = jnp.asarray([0.3, 0.3, 1.0, 1.0, 1.0e-3, 1.0, 0.0])
    else:
        p = jnp.asarray([0.3, 1.0, 1.0e-3, 1.0, 0.0])

    # All points on the diagonal (t1 == t2) -> residuals masked to zero.
    phi_c = jnp.asarray([0.0, 0.0, 0.0])
    t1_c = jnp.asarray([1.0, 2.0, 3.0])
    t2_c = jnp.asarray([1.0, 2.0, 3.0])
    g2_c = jnp.asarray([5.0, 5.0, 5.0])  # arbitrary; masked away
    sigma = jnp.ones(3)

    jtj, jtr, chi2 = acc(p, phi_c, t1_c, t2_c, g2_c, sigma)
    assert float(chi2) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.asarray(jtj), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jtr), 0.0, atol=1e-12)
    assert float(chi2_fn(p, phi_c, t1_c, t2_c, g2_c, sigma)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("per_angle", [False, True])
def test_kernel_accumulator_chi2_matches_chi2_kernel(per_angle: bool) -> None:
    cfg = _static_physics_config(per_angle)
    acc, chi2_fn = _kernels(cfg)
    if per_angle:
        p = jnp.asarray([0.3, 0.25, 1.0, 1.05, 1.0e-3, 1.0, 0.0])
        phi_c = jnp.asarray([0.0, 45.0, 0.0])
    else:
        p = jnp.asarray([0.3, 1.0, 1.0e-3, 1.0, 0.0])
        phi_c = jnp.asarray([0.0, 0.0, 0.0])

    # Off-diagonal points produce non-trivial residuals.
    t1_c = jnp.asarray([1.0, 2.0, 3.0])
    t2_c = jnp.asarray([2.0, 3.0, 1.0])
    g2_c = jnp.asarray([1.2, 1.1, 1.05])
    sigma = jnp.ones(3)

    jtj, jtr, chi2_acc = acc(p, phi_c, t1_c, t2_c, g2_c, sigma)
    chi2_only = chi2_fn(p, phi_c, t1_c, t2_c, g2_c, sigma)

    n_params = p.shape[0]
    assert np.asarray(jtj).shape == (n_params, n_params)
    assert np.asarray(jtr).shape == (n_params,)
    # J^T J is symmetric by construction.
    np.testing.assert_allclose(np.asarray(jtj), np.asarray(jtj).T, atol=1e-10)
    # The two kernels must agree on chi2 (same residual definition).
    assert float(chi2_acc) == pytest.approx(float(chi2_only), rel=1e-9)
    assert np.isfinite(float(chi2_acc))


# ---------------------------------------------------------------------------
# OOCSharedArrays
# ---------------------------------------------------------------------------


@shm_required
def test_shared_arrays_roundtrip_and_cleanup() -> None:
    phi = np.array([0.0, 0.0, 0.0])
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([2.0, 3.0, 1.0])
    g2 = np.array([1.1, 1.2, 1.3])
    sigma = np.ones(3)
    boundaries = [(0, 3)]

    with pa.OOCSharedArrays(phi, t1, t2, g2, sigma, boundaries) as shared:
        refs = shared.get_refs()
        assert set(refs) == {"phi", "t1", "t2", "g2", "sigma"}
        # Attach to the 'g2' block and confirm the data round-tripped.
        ref = refs["g2"]
        blk = shared_memory.SharedMemory(name=ref["shm_name"], create=False)
        try:
            view = np.ndarray(ref["shape"], dtype=ref["dtype"], buffer=blk.buf)
            np.testing.assert_array_equal(view, g2)
        finally:
            blk.close()
    # After the context exits, the blocks are unlinked.
    assert shared._shm_blocks == []


@shm_required
def test_shared_arrays_without_sigma() -> None:
    phi = np.array([0.0])
    t1 = np.array([1.0])
    t2 = np.array([1.0])
    g2 = np.array([1.0])
    with pa.OOCSharedArrays(phi, t1, t2, g2, None, [(0, 1)]) as shared:
        assert "sigma" not in shared.get_refs()


# ---------------------------------------------------------------------------
# Worker functions (in-process: call the initializer directly, no spawn)
# ---------------------------------------------------------------------------


@shm_required
def test_worker_functions_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pre-register the env vars the worker initializer mutates; monkeypatch then
    # restores their original values on teardown regardless of the worker's writes.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(var, "1")

    cfg = _static_physics_config(per_angle=False)
    phi = np.array([0.0, 0.0, 0.0])
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([2.0, 3.0, 1.0])
    g2 = np.array([1.2, 1.1, 1.05])
    sigma = np.ones(3)
    boundaries = [(0, 3)]
    params = np.array([0.3, 1.0, 1.0e-3, 1.0, 0.0])

    with pa.OOCSharedArrays(phi, t1, t2, g2, sigma, boundaries) as shared:
        physics_config = {
            "phi_unique": np.asarray(cfg["phi_unique"]),
            "t1_unique": np.asarray(cfg["t1_unique"]),
            "t2_unique": np.asarray(cfg["t2_unique"]),
            "per_angle_scaling": False,
            "n_phi": 1,
            "n_t1": 3,
            "n_t2": 3,
            "q": _Q,
            "L": _L,
            "dt": _DT,
        }
        try:
            pa._ooc_worker_init(shared.get_refs(), physics_config, boundaries, 1)
            jtj, jtr, chi2 = pa._ooc_compute_chunk((params, 0))
            assert jtj.shape == (5, 5)
            assert jtr.shape == (5,)
            assert np.isfinite(chi2)
            # chi2-only worker agrees with the accumulator worker.
            chi2_only = pa._ooc_compute_chi2_chunk((params, 0))
            assert chi2_only == pytest.approx(chi2, rel=1e-9)
        finally:
            pa._ooc_worker_cleanup()


# ---------------------------------------------------------------------------
# OOCComputePool (one real spawn)
# ---------------------------------------------------------------------------


@shm_required
def test_compute_pool_matches_direct_kernel() -> None:
    cfg = _static_physics_config(per_angle=False)
    phi = np.array([0.0, 0.0, 0.0])
    t1 = np.array([1.0, 2.0, 3.0])
    t2 = np.array([2.0, 3.0, 1.0])
    g2 = np.array([1.2, 1.1, 1.05])
    sigma = np.ones(3)
    boundaries = [(0, 3)]
    params = np.array([0.3, 1.0, 1.0e-3, 1.0, 0.0])

    physics_config = {
        "phi_unique": np.asarray(cfg["phi_unique"]),
        "t1_unique": np.asarray(cfg["t1_unique"]),
        "t2_unique": np.asarray(cfg["t2_unique"]),
        "per_angle_scaling": False,
        "n_phi": 1,
        "n_t1": 3,
        "n_t2": 3,
        "q": _Q,
        "L": _L,
        "dt": _DT,
    }

    # Direct (in-process) reference chi2 for the same chunk.
    _, chi2_fn = _kernels(cfg)
    ref_chi2 = float(
        chi2_fn(
            jnp.asarray(params),
            jnp.asarray(phi),
            jnp.asarray(t1),
            jnp.asarray(t2),
            jnp.asarray(g2),
            jnp.asarray(sigma),
        )
    )

    with pa.OOCSharedArrays(phi, t1, t2, g2, sigma, boundaries) as shared:
        with pa.OOCComputePool(
            n_workers=1,
            shared_arrays=shared,
            physics_config=physics_config,
            chunk_boundaries=boundaries,
            threads_per_worker=1,
        ) as pool:
            results = pool.compute_accumulators(params)
            assert len(results) == 1
            jtj, jtr, chi2 = results[0]
            assert jtj.shape == (5, 5)
            assert float(chi2) == pytest.approx(ref_chi2, rel=1e-6)

            total_chi2 = pool.compute_chi2(params)
            assert total_chi2 == pytest.approx(ref_chi2, rel=1e-6)
            # shutdown is idempotent
            pool.shutdown()
            pool.shutdown()
