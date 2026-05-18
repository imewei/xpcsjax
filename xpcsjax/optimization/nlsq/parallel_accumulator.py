"""Parallel chunk accumulation for NLSQ streaming optimizer.

Dispatches chunk computations to a process pool and reduces
J^T J, J^T r, chi2 accumulators. Falls back to sequential
when n_chunks < 10 or pool creation fails.

Also provides ``create_ooc_kernels`` factory and ``OOCComputePool``
for parallelizing the per-chunk JIT compute across persistent workers.

Matrix addition is associative and commutative, so parallel
accumulation produces identical results to sequential.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.shared_memory
import os
import pickle  # noqa: S403 — used for ProcessPoolExecutor error handling only
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_CHUNKS_FOR_PARALLEL = 10
_MIN_CHUNKS_FOR_PARALLEL_COMPUTE = 10


def should_use_parallel_accumulation(n_chunks: int) -> bool:
    """Determine if parallel accumulation is worthwhile.

    Parameters
    ----------
    n_chunks : int
        Number of chunks to accumulate.

    Returns
    -------
    bool
        True if n_chunks >= threshold for parallel accumulation.
    """
    return n_chunks >= _MIN_CHUNKS_FOR_PARALLEL


def accumulate_chunks_sequential(
    chunks: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate chunk results sequentially.

    Parameters
    ----------
    chunks : list of (JtJ, Jtr, chi2) tuples
        Each tuple contains:
        - JtJ: (n_params, n_params) symmetric matrix
        - Jtr: (n_params,) vector
        - chi2: scalar cost contribution

    Returns
    -------
    total_JtJ : np.ndarray
        Sum of all JtJ matrices.
    total_Jtr : np.ndarray
        Sum of all Jtr vectors.
    total_chi2 : float
        Sum of all chi2 values.
    count : int
        Number of chunks accumulated.
    """
    total_JtJ: np.ndarray | None = None
    total_Jtr: np.ndarray | None = None
    total_chi2 = 0.0
    count = 0

    for JtJ, Jtr, chi2 in chunks:
        if total_JtJ is None:
            total_JtJ = np.zeros_like(JtJ)
            total_Jtr = np.zeros_like(Jtr)
        total_JtJ += JtJ
        total_Jtr += Jtr
        total_chi2 += chi2
        count += 1

    if total_JtJ is None or total_Jtr is None:
        raise ValueError("Cannot accumulate empty chunks list")
    return total_JtJ, total_Jtr, total_chi2, count


def accumulate_chunks_parallel(
    chunks: list[tuple[np.ndarray, np.ndarray, float]],
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate chunk results in parallel via process pool.

    Partitions chunks across workers, each computes partial sums,
    then reduces. Falls back to sequential on failure.

    Parameters
    ----------
    chunks : list of (JtJ, Jtr, chi2) tuples
        Each tuple contains:
        - JtJ: (n_params, n_params) symmetric matrix
        - Jtr: (n_params,) vector
        - chi2: scalar cost contribution
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    total_JtJ : np.ndarray
        Sum of all JtJ matrices.
    total_Jtr : np.ndarray
        Sum of all Jtr vectors.
    total_chi2 : float
        Sum of all chi2 values.
    count : int
        Number of chunks accumulated.
    """
    from concurrent.futures import (
        ProcessPoolExecutor,
        as_completed,
    )
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    if n_workers < 1:
        return accumulate_chunks_sequential(chunks)

    if len(chunks) < _MIN_CHUNKS_FOR_PARALLEL:
        return accumulate_chunks_sequential(chunks)

    # Partition chunks across workers
    partitions: list[list[tuple[np.ndarray, np.ndarray, float]]] = [
        [] for _ in range(n_workers)
    ]
    for i, chunk in enumerate(chunks):
        partitions[i % n_workers].append(chunk)

    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(accumulate_chunks_sequential, partition)
                for partition in partitions
                if partition
            ]

            total_JtJ: np.ndarray | None = None
            total_Jtr: np.ndarray | None = None
            total_chi2 = 0.0
            total_count = 0

            for future in as_completed(futures):
                JtJ, Jtr, chi2, count = future.result(timeout=300)
                if total_JtJ is None:
                    total_JtJ = np.zeros_like(JtJ)
                    total_Jtr = np.zeros_like(Jtr)
                total_JtJ += JtJ
                total_Jtr += Jtr
                total_chi2 += chi2
                total_count += count

        if total_JtJ is None or total_Jtr is None:
            raise ValueError("No partitions produced results")
        return total_JtJ, total_Jtr, total_chi2, total_count

    except (OSError, RuntimeError, pickle.PicklingError, FuturesTimeoutError) as e:
        logger.warning(
            "Parallel chunk accumulation failed (%s), falling back to sequential", e
        )
        return accumulate_chunks_sequential(chunks)


# ============================================================================
# OOC Parallel Compute: Factory, shared memory, and worker pool
# ============================================================================


def should_use_parallel_compute(n_chunks: int) -> bool:
    """Determine if parallel chunk COMPUTE is worthwhile.

    Parameters
    ----------
    n_chunks : int
        Number of chunks in the out-of-core iteration.

    Returns
    -------
    bool
        True if n_chunks >= threshold for parallel compute.
    """
    return n_chunks >= _MIN_CHUNKS_FOR_PARALLEL_COMPUTE


def create_ooc_kernels(
    per_angle_scaling: bool,
    n_phi: int,
    phi_unique: Any,
    t1_unique_global: Any,
    t2_unique_global: Any,
    n_t1: int,
    n_t2: int,
    q_val: float,
    L_val: float,
    dt_val: float,
) -> tuple[Callable, Callable]:
    """Create JIT-compiled OOC chunk kernels from physics constants.

    This is the single source of truth for the chunk accumulators and chi2
    kernels used by both sequential and parallel OOC paths.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle contrast/offset arrays are used.
    n_phi : int
        Number of unique phi angles.
    phi_unique : jnp.ndarray
        Sorted unique phi values.
    t1_unique_global, t2_unique_global : jnp.ndarray
        Sorted unique time arrays.
    n_t1, n_t2 : int
        Sizes of t1 and t2 unique arrays.
    q_val, L_val, dt_val : float
        Physics constants (wavevector, thickness, time step).

    Returns
    -------
    compute_chunk_accumulators : callable
        JIT kernel: ``(p, phi_c, t1_c, t2_c, g2_c, sigma) -> (JtJ, Jtr, chi2)``
    compute_chunk_chi2 : callable
        JIT kernel: ``(p, phi_c, t1_c, t2_c, g2_c, sigma) -> chi2``
    """
    import jax
    import jax.numpy as jnp

    from xpcsjax.core.physics_nlsq import compute_g2_scaled

    @jax.jit
    def compute_chunk_accumulators(
        p: Any, phi_c: Any, t1_c: Any, t2_c: Any, g2_c: Any, sigma: Any
    ) -> Any:
        """Compute J^T J, J^T r, and chi2 for a chunk."""

        def r_fn(curr_p: Any) -> Any:
            if per_angle_scaling:
                contrast_arr = curr_p[:n_phi]
                offset_arr = curr_p[n_phi : 2 * n_phi]
                physical_params = curr_p[2 * n_phi :]
            else:
                contrast_scalar = curr_p[0]
                offset_scalar = curr_p[1]
                physical_params = curr_p[2:]

            if per_angle_scaling:
                compute_g2_vmap = jax.vmap(
                    lambda phi_val, c_val, o_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t1_unique_global,
                            t2=t2_unique_global,
                            phi=phi_val,
                            q=q_val,
                            L=L_val,
                            contrast=c_val,
                            offset=o_val,
                            dt=dt_val,
                        )
                    ),
                    in_axes=(0, 0, 0),
                )
                g2_theory_grid = compute_g2_vmap(phi_unique, contrast_arr, offset_arr)
            else:
                compute_g2_vmap = jax.vmap(  # type: ignore[assignment]
                    lambda phi_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t1_unique_global,
                            t2=t2_unique_global,
                            phi=phi_val,
                            q=q_val,
                            L=L_val,
                            contrast=contrast_scalar,
                            offset=offset_scalar,
                            dt=dt_val,
                        )
                    ),
                    in_axes=0,
                )
                g2_theory_grid = compute_g2_vmap(phi_unique)  # type: ignore[call-arg]

            g2_theory_flat = g2_theory_grid.flatten()
            # Cast to int64 BEFORE multiplication to prevent int32 overflow.
            # jnp.searchsorted returns int32; for large OOC datasets
            # (n_phi=100, n_t1=5000, n_t2=5000) the product 99 * 25_000_000 = 2.475B
            # exceeds int32 max (2.147B), silently wrapping to a negative index.
            phi_indices = jnp.searchsorted(phi_unique, phi_c).astype(jnp.int64)
            t1_indices = jnp.searchsorted(t1_unique_global, t1_c).astype(jnp.int64)
            t2_indices = jnp.searchsorted(t2_unique_global, t2_c).astype(jnp.int64)
            flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices
            g2_theory_chunk = g2_theory_flat[flat_indices]

            w = 1.0 / sigma
            res = (g2_c - g2_theory_chunk) * w
            return jnp.where(jnp.abs(t1_c - t2_c) > 1e-15, res, 0.0)

        J = jax.jacfwd(r_fn)(p)
        r = r_fn(p)
        return J.T @ J, J.T @ r, jnp.sum(r**2)

    @jax.jit
    def compute_chunk_chi2(
        p: Any, phi_c: Any, t1_c: Any, t2_c: Any, g2_c: Any, sigma: Any
    ) -> Any:
        """Compute chi2 for a chunk (no Jacobian)."""
        if per_angle_scaling:
            contrast_arr = p[:n_phi]
            offset_arr = p[n_phi : 2 * n_phi]
            physical_params = p[2 * n_phi :]
        else:
            contrast_scalar = p[0]
            offset_scalar = p[1]
            physical_params = p[2:]

        if per_angle_scaling:
            compute_g2_vmap = jax.vmap(
                lambda phi_val, c_val, o_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=t1_unique_global,
                        t2=t2_unique_global,
                        phi=phi_val,
                        q=q_val,
                        L=L_val,
                        contrast=c_val,
                        offset=o_val,
                        dt=dt_val,
                    )
                ),
                in_axes=(0, 0, 0),
            )
            g2_theory_grid = compute_g2_vmap(phi_unique, contrast_arr, offset_arr)
        else:
            compute_g2_vmap = jax.vmap(  # type: ignore[assignment]
                lambda phi_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=t1_unique_global,
                        t2=t2_unique_global,
                        phi=phi_val,
                        q=q_val,
                        L=L_val,
                        contrast=contrast_scalar,
                        offset=offset_scalar,
                        dt=dt_val,
                    )
                ),
                in_axes=0,
            )
            g2_theory_grid = compute_g2_vmap(phi_unique)  # type: ignore[call-arg]

        g2_theory_flat = g2_theory_grid.flatten()
        # Cast to int64 BEFORE multiplication to prevent int32 overflow (same fix
        # as compute_chunk_accumulators above).
        phi_indices = jnp.searchsorted(phi_unique, phi_c).astype(jnp.int64)
        t1_indices = jnp.searchsorted(t1_unique_global, t1_c).astype(jnp.int64)
        t2_indices = jnp.searchsorted(t2_unique_global, t2_c).astype(jnp.int64)
        flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices
        g2_theory_chunk = g2_theory_flat[flat_indices]

        w = 1.0 / sigma
        res = (g2_c - g2_theory_chunk) * w
        res = jnp.where(jnp.abs(t1_c - t2_c) > 1e-15, res, 0.0)
        return jnp.sum(res**2)

    return compute_chunk_accumulators, compute_chunk_chi2


# ---- OOC Worker globals (set by _ooc_worker_init) ----
_w_phi: np.ndarray | None = None
_w_t1: np.ndarray | None = None
_w_t2: np.ndarray | None = None
_w_g2: np.ndarray | None = None
_w_sigma: np.ndarray | None = None
_w_chunk_boundaries: list[tuple[int, int]] = []
_w_compute_accumulators: Callable | None = None
_w_compute_chi2: Callable | None = None
_w_shm_handles: list[multiprocessing.shared_memory.SharedMemory] = []


def _ooc_worker_cleanup() -> None:
    """Close shared memory handles on worker exit."""
    for shm in _w_shm_handles:
        try:
            shm.close()
        except (OSError, ValueError):
            pass


def _ooc_worker_init(
    shm_refs: dict[str, dict[str, Any]],
    physics_config: dict[str, Any],
    chunk_boundaries: list[tuple[int, int]],
    threads_per_worker: int,
) -> None:
    """Initialize a persistent OOC compute worker.

    Sets up JAX/OMP, attaches to shared memory (zero-copy views),
    and creates JIT kernels from physics constants.
    """
    import atexit

    global _w_phi, _w_t1, _w_t2, _w_g2, _w_sigma  # noqa: PLW0603
    global _w_chunk_boundaries, _w_compute_accumulators  # noqa: PLW0603
    global _w_compute_chi2, _w_shm_handles  # noqa: PLW0603

    # Thread pinning
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)

    # JAX float64 before import (CLAUDE.md rule #8)
    os.environ["JAX_ENABLE_X64"] = "true"

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    # Persistent compilation cache (CLAUDE.md rule #9)
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(os.path.expanduser("~/.cache/homodyne/jax_cache"))),
    )
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Attach to shared memory arrays (zero-copy)
    _w_shm_handles = []
    arrays: dict[str, np.ndarray] = {}
    for name, ref in shm_refs.items():
        shm = multiprocessing.shared_memory.SharedMemory(
            name=ref["shm_name"], create=False
        )
        _w_shm_handles.append(shm)
        arrays[name] = np.ndarray(ref["shape"], dtype=ref["dtype"], buffer=shm.buf)

    # Register cleanup to close shared memory handles on worker exit
    atexit.register(_ooc_worker_cleanup)

    _w_phi = arrays["phi"]
    _w_t1 = arrays["t1"]
    _w_t2 = arrays["t2"]
    _w_g2 = arrays["g2"]
    _w_sigma = arrays.get("sigma")
    _w_chunk_boundaries = chunk_boundaries

    # Create JIT kernels from physics config
    phi_unique = jnp.asarray(physics_config["phi_unique"])
    t1_unique = jnp.asarray(physics_config["t1_unique"])
    t2_unique = jnp.asarray(physics_config["t2_unique"])

    _w_compute_accumulators, _w_compute_chi2 = create_ooc_kernels(
        per_angle_scaling=physics_config["per_angle_scaling"],
        n_phi=physics_config["n_phi"],
        phi_unique=phi_unique,
        t1_unique_global=t1_unique,
        t2_unique_global=t2_unique,
        n_t1=physics_config["n_t1"],
        n_t2=physics_config["n_t2"],
        q_val=physics_config["q"],
        L_val=physics_config["L"],
        dt_val=physics_config["dt"],
    )


def _ooc_compute_chunk(
    args: tuple[np.ndarray, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute JtJ, Jtr, chi2 for a single chunk using worker globals.

    Parameters
    ----------
    args : (params_np, chunk_id)
        params_np: current parameter values as numpy array.
        chunk_id: index into _w_chunk_boundaries.

    Returns
    -------
    JtJ, Jtr, chi2 as numpy arrays and float.
    """
    import jax.numpy as jnp

    params_np, chunk_id = args
    start, end = _w_chunk_boundaries[chunk_id]

    phi_c = _w_phi[start:end]  # type: ignore[index]
    t1_c = _w_t1[start:end]  # type: ignore[index]
    t2_c = _w_t2[start:end]  # type: ignore[index]
    g2_c = _w_g2[start:end]  # type: ignore[index]
    sigma_c = _w_sigma[start:end] if _w_sigma is not None else 1.0

    p = jnp.asarray(params_np)
    JtJ, Jtr, chi2 = _w_compute_accumulators(  # type: ignore[misc]
        p,
        jnp.asarray(phi_c),
        jnp.asarray(t1_c),
        jnp.asarray(t2_c),
        jnp.asarray(g2_c),
        jnp.asarray(sigma_c) if isinstance(sigma_c, np.ndarray) else sigma_c,
    )
    return np.asarray(JtJ), np.asarray(Jtr), float(chi2)


def _ooc_compute_chi2_chunk(
    args: tuple[np.ndarray, int],
) -> float:
    """Compute chi2 for a single chunk using worker globals (no Jacobian).

    Parameters
    ----------
    args : (params_np, chunk_id)
        params_np: current parameter values as numpy array.
        chunk_id: index into _w_chunk_boundaries.

    Returns
    -------
    chi2 as float.
    """
    import jax.numpy as jnp

    params_np, chunk_id = args
    start, end = _w_chunk_boundaries[chunk_id]

    phi_c = _w_phi[start:end]  # type: ignore[index]
    t1_c = _w_t1[start:end]  # type: ignore[index]
    t2_c = _w_t2[start:end]  # type: ignore[index]
    g2_c = _w_g2[start:end]  # type: ignore[index]
    sigma_c = _w_sigma[start:end] if _w_sigma is not None else 1.0

    p = jnp.asarray(params_np)
    chi2 = _w_compute_chi2(  # type: ignore[misc]
        p,
        jnp.asarray(phi_c),
        jnp.asarray(t1_c),
        jnp.asarray(t2_c),
        jnp.asarray(g2_c),
        jnp.asarray(sigma_c) if isinstance(sigma_c, np.ndarray) else sigma_c,
    )
    return float(chi2)


class OOCSharedArrays:
    """Shared memory manager for OOC flat data arrays.

    Parameters
    ----------
    phi_flat, t1_flat, t2_flat, g2_flat : np.ndarray
        Flat data arrays for the OOC iteration.
    sigma_flat : np.ndarray or None
        Per-point uncertainty weights.
    chunk_boundaries : list of (start, end) tuples
        Index boundaries for each chunk (using sorted/stratified indices).
    """

    def __init__(
        self,
        phi_flat: np.ndarray,
        t1_flat: np.ndarray,
        t2_flat: np.ndarray,
        g2_flat: np.ndarray,
        sigma_flat: np.ndarray | None,
        chunk_boundaries: list[tuple[int, int]],
    ) -> None:
        self._shm_blocks: list[multiprocessing.shared_memory.SharedMemory] = []
        self._refs: dict[str, dict[str, Any]] = {}
        self._chunk_boundaries = chunk_boundaries

        for name, arr in [
            ("phi", phi_flat),
            ("t1", t1_flat),
            ("t2", t2_flat),
            ("g2", g2_flat),
        ]:
            self._create_shm(name, arr)

        if sigma_flat is not None:
            self._create_shm("sigma", sigma_flat)

    def _create_shm(self, name: str, arr: np.ndarray) -> None:
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=arr.nbytes)
        buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        buf[:] = arr
        self._shm_blocks.append(shm)
        self._refs[name] = {
            "shm_name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }

    def get_refs(self) -> dict[str, dict[str, Any]]:
        """Get picklable shared memory references."""
        return self._refs

    def cleanup(self) -> None:
        """Close and unlink all shared memory blocks."""
        for shm in self._shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except (OSError, ValueError):
                pass
        self._shm_blocks.clear()

    def __enter__(self) -> OOCSharedArrays:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


class OOCComputePool:
    """Persistent process pool for parallel OOC chunk computation.

    Workers share flat data arrays via shared memory and cache JIT kernels.
    The pool persists across L-M iterations (JIT compile once, reuse).

    Parameters
    ----------
    n_workers : int
        Number of parallel compute workers.
    shared_arrays : OOCSharedArrays
        Shared memory manager with data arrays.
    physics_config : dict
        Physics constants for JIT kernel creation.
    chunk_boundaries : list of (start, end)
        Index boundaries for each chunk.
    threads_per_worker : int
        OMP/MKL threads per worker process.
    """

    def __init__(
        self,
        n_workers: int,
        shared_arrays: OOCSharedArrays,
        physics_config: dict[str, Any],
        chunk_boundaries: list[tuple[int, int]],
        threads_per_worker: int = 1,
    ) -> None:
        from concurrent.futures import ProcessPoolExecutor

        self._n_workers = n_workers
        self._n_chunks = len(chunk_boundaries)
        self._shutdown = False

        ctx = multiprocessing.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_ooc_worker_init,
            initargs=(
                shared_arrays.get_refs(),
                physics_config,
                chunk_boundaries,
                threads_per_worker,
            ),
        )
        logger.info(
            "OOCComputePool started: %d workers, %d chunks",
            n_workers,
            self._n_chunks,
        )

    def compute_accumulators(
        self, params: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Dispatch all chunks to workers and collect (JtJ, Jtr, chi2) tuples.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values.

        Returns
        -------
        list of (JtJ, Jtr, chi2) tuples, one per chunk.
        """
        from concurrent.futures import as_completed

        futures = [
            self._executor.submit(_ooc_compute_chunk, (params, chunk_id))
            for chunk_id in range(self._n_chunks)
        ]

        results: list[tuple[np.ndarray, np.ndarray, float]] = []
        for future in as_completed(futures):
            results.append(future.result(timeout=300))
        return results

    def compute_chi2(self, params: np.ndarray, stride: int = 1) -> float:
        """Dispatch chi2-only computation across workers (no Jacobian).

        Parameters
        ----------
        params : np.ndarray
            Current parameter values.
        stride : int
            Chunk stride for subsampling (1 = all chunks).

        Returns
        -------
        Estimated total chi2 (scaled if stride > 1).
        """
        from concurrent.futures import as_completed

        chunk_ids = list(range(0, self._n_chunks, stride))
        futures = [
            self._executor.submit(_ooc_compute_chi2_chunk, (params, cid))
            for cid in chunk_ids
        ]

        total_chi2 = 0.0
        for future in as_completed(futures):
            total_chi2 += future.result(timeout=300)

        if stride > 1 and len(chunk_ids) > 0:
            total_chi2 *= self._n_chunks / len(chunk_ids)
        return total_chi2

    def shutdown(self) -> None:
        """Shut down the pool. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("OOCComputePool shut down")

    def __enter__(self) -> OOCComputePool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
