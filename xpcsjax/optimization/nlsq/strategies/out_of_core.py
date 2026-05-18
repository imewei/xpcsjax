"""Out-of-Core Global Accumulation strategy for NLSQ optimization.

Extracted from wrapper.py to reduce file size and improve maintainability.

This module provides:
- Out-of-core J^T J / J^T r accumulation for massive datasets
- Levenberg-Marquardt iteration with chunk-wise gradient accumulation
- Parallel chunk computation with shared memory pools
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from xpcsjax.optimization.nlsq.strategies.chunking import (
    calculate_adaptive_chunk_size,
    get_stratified_chunk_iterator,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def _effective_param_count_for_ooc(
    per_angle_scaling: bool,
    n_params: int,
    n_phi: int,
    n_physical: int,
    anti_degeneracy_config: dict | None = None,
) -> int:
    """Return the parameter count used for out-of-core covariance scaling."""
    if not per_angle_scaling:
        return n_params

    ad_config = anti_degeneracy_config or {}
    per_angle_mode = ad_config.get("per_angle_mode", "auto")
    threshold = int(ad_config.get("constant_scaling_threshold", 3))

    if per_angle_mode == "constant":
        return n_physical
    if per_angle_mode == "auto" and n_phi >= threshold and n_params == n_physical + 2:
        return 2 * n_phi + n_physical

    return n_params


def fit_with_out_of_core_accumulation(
    stratified_data: Any,
    data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    config: Any,
    fast_chi2_mode: bool = False,
    anti_degeneracy_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using Out-of-Core Global Accumulation for massive datasets.

    This strategy virtually chunks the dataset using Index-Based Stratification,
    accumulates the full Hessian and Gradient (J^T J, J^T r) by iterating
    over chunks, and takes a global Levenberg-Marquardt step.

    Guarantees identical convergence to standard NLSQ but with minimal memory.

    Note (v2.14.1+):
        This method now uses FULL homodyne physics via compute_g2_scaled(),
        identical to stratified least-squares. Anti-Degeneracy Defense System
        support is planned for a future release.

    Args:
        stratified_data: Stratified data object (unused, kept for API compat)
        data: Original XPCS data object with .phi, .t1, .t2, .g2, .q, .L
        per_angle_scaling: Whether per-angle scaling is enabled
        physical_param_names: Names of physical parameters
        initial_params: Initial parameter guess
        bounds: Parameter bounds (lower, upper) or None
        log: Logger instance
        config: Configuration object or dict
        fast_chi2_mode: If True, subsample chunks for chi2 evaluation
        anti_degeneracy_config: Anti-degeneracy configuration (reserved)

    Returns:
        (popt, pcov, info) tuple
    """
    import jax.numpy as jnp

    _start_time = time.perf_counter()  # noqa: F841
    log.info(
        "Initializing Out-of-Core Global Stratified Optimization (Full Physics)..."
    )

    # 1. Setup Chunking
    # Use StratifiedIndices if available (Zero-Copy)
    _use_index_based = False  # noqa: F841
    # We operate on the ORIGINAL flattened data to avoid pre-materializing
    # a giant stratified copy (which causes OOM).
    # We assume `data` object has .phi, .t1, .t2, .g2
    # We need to flatten them carefully (using ravel/reshape to avoid copies if possible)

    # Helper to flatten dimensions
    def _get_flat_arrays(
        d: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        # Same logic as _prepare_data but trying to be lazy/view-based
        phi_arr = np.asarray(d.phi)
        t1_arr = np.asarray(d.t1)
        t2_arr = np.asarray(d.t2)
        g2_arr = np.asarray(d.g2)
        sigma_arr = getattr(d, "sigma", None)

        # Extract 1D from meshgrids if needed (borrowed from _prepare_data)
        if t1_arr.ndim == 2 and t1_arr.size > 0:
            t1_arr = t1_arr[:, 0]
        if t2_arr.ndim == 2 and t2_arr.size > 0:
            t2_arr = t2_arr[0, :]

        phi_grid, t1_grid, t2_grid = np.meshgrid(phi_arr, t1_arr, t2_arr, indexing="ij")

        # Flatten sigma if available
        if sigma_arr is not None:
            sigma_arr = np.asarray(sigma_arr)
            sigma_flat = sigma_arr.ravel()
        else:
            sigma_flat = None

        # These flattens create copies usually, but for 25M points (200MB) it's acceptable ONCE
        # The OOM comes from creating SECOND and THIRD copies during stratification.
        return (
            phi_grid.ravel(),
            t1_grid.ravel(),
            t2_grid.ravel(),
            g2_arr.ravel(),
            sigma_flat,
        )

    phi_flat, t1_flat, t2_flat, g2_flat, sigma_flat = _get_flat_arrays(data)

    # Calculate optimal chunk size
    n_points = len(phi_flat)
    n_params = len(initial_params)
    n_angles = len(np.unique(phi_flat))

    chunk_size = calculate_adaptive_chunk_size(
        total_points=n_points,
        n_params=n_params,
        n_angles=n_angles,
        safety_factor=5.0,
    )

    # Get iterator that yields INDICES for stratified chunks
    # This allows us to pull stratified data from the flat arrays on demand
    iterator = get_stratified_chunk_iterator(phi_flat, chunk_size)
    log.info(
        f"Out-of-Core Strategy: {len(iterator)} chunks of size ~{chunk_size}\n"
        f"  Pipeline: Chunk(Indices) -> Load -> JIT(Acc) -> Global Step"
    )

    # Pre-compute unique phi for JAX mapping
    phi_unique = jnp.sort(jnp.unique(phi_flat))

    # 2. Setup Optimization State
    params_curr = jnp.array(initial_params)

    cfg_dict = (
        config.config
        if hasattr(config, "config")
        else (config if isinstance(config, dict) else {})
    )

    # Extract physics constants from data (v2.14.1+: Full homodyne physics)
    q_val = float(data.q)
    L_val = float(data.L)
    dt_raw = getattr(data, "dt", cfg_dict.get("dt", None))
    if dt_raw is None:
        log.warning(
            "_fit_with_stratified_least_squares (OOC): dt not found in data or config; "
            "using dt=0.001 s as fallback."
        )
        dt_val = 0.001
    else:
        dt_val = float(dt_raw)

    # Extract global unique time arrays for meshgrid construction.
    # IMPORTANT: t1 and t2 must remain separate -- merging them into a single
    # union array creates a padded square grid (n_t x n_t) which is wrong
    # for non-symmetric XPCS data where n_t1 != n_t2.  All flat-index
    # arithmetic downstream uses (n_t1, n_t2) as the grid shape.
    t1_unique_global = jnp.sort(jnp.unique(jnp.asarray(t1_flat)))
    t2_unique_global = jnp.sort(jnp.unique(jnp.asarray(t2_flat)))
    n_phi = len(phi_unique)
    n_t1 = len(t1_unique_global)
    n_t2 = len(t2_unique_global)
    n_physical = len(physical_param_names)

    # Effective parameter count for DOF in s^2 computation.
    # auto_averaged uses a compressed vector (contrast_avg, offset_avg, physical)
    # but consumes expanded DOF; constant mode keeps scaling fixed and must not
    # be expanded or covariance is over-inflated.
    n_params_effective = _effective_param_count_for_ooc(
        per_angle_scaling,
        n_params,
        n_phi,
        n_physical,
        anti_degeneracy_config,
    )

    log.info(
        f"Full Physics Setup: n_phi={n_phi}, n_t1={n_t1}, n_t2={n_t2}, "
        f"q={q_val:.4e}, L={L_val:.4e}, dt={dt_val:.4e}"
    )
    max_iter = cfg_dict.get("optimization", {}).get("max_iterations", 50)

    # Convergence tolerances (v2.22.0: multi-criteria, matching standard NLSQ)
    xtol = 1e-6  # Relative parameter change (per-component max, not norm)
    ftol = 1e-6  # Relative cost function change
    lm_lambda = 0.01  # Initial damping
    rel_change = float("inf")  # Initialize to prevent NameError at loop exit
    cost_change = float("inf")  # Initialize for multi-criteria convergence

    # ====================================================================
    # JIT-compiled Chunk Kernels via factory (single source of truth)
    # ====================================================================
    from xpcsjax.optimization.nlsq.parallel_accumulator import (
        create_ooc_kernels,
    )

    compute_chunk_accumulators, compute_chunk_chi2 = create_ooc_kernels(
        per_angle_scaling=per_angle_scaling,
        n_phi=n_phi,
        phi_unique=phi_unique,
        t1_unique_global=t1_unique_global,
        t2_unique_global=t2_unique_global,
        n_t1=n_t1,
        n_t2=n_t2,
        q_val=q_val,
        L_val=L_val,
        dt_val=dt_val,
    )

    # Lazy import for parallel chunk accumulation
    from xpcsjax.optimization.nlsq.parallel_accumulator import (
        OOCComputePool,
        OOCSharedArrays,
        accumulate_chunks_parallel,
        accumulate_chunks_sequential,
        should_use_parallel_accumulation,
        should_use_parallel_compute,
    )

    # Create parallel compute pool if beneficial
    ooc_pool: OOCComputePool | None = None
    ooc_shared: OOCSharedArrays | None = None
    n_total_chunks = len(iterator)

    if should_use_parallel_compute(n_total_chunks):
        try:
            # Build chunk boundaries from the stratified iterator
            chunk_boundaries: list[tuple[int, int]] = []
            # Flatten all indices in iterator order into a single array
            all_indices = []
            offset = 0
            for indices_chunk in iterator:
                all_indices.append(indices_chunk)
                chunk_boundaries.append((offset, offset + len(indices_chunk)))
                offset += len(indices_chunk)
            all_indices_arr = np.concatenate(all_indices)

            # Reorder flat arrays to match iterator order (contiguous chunks)
            phi_ordered = np.asarray(phi_flat)[all_indices_arr]
            t1_ordered = np.asarray(t1_flat)[all_indices_arr]
            t2_ordered = np.asarray(t2_flat)[all_indices_arr]
            g2_ordered = np.asarray(g2_flat)[all_indices_arr]
            sigma_ordered = (
                np.asarray(sigma_flat)[all_indices_arr]
                if sigma_flat is not None
                else None
            )

            ooc_shared = OOCSharedArrays(
                phi_ordered,
                t1_ordered,
                t2_ordered,
                g2_ordered,
                sigma_ordered,
                chunk_boundaries,
            )

            physics_config = {
                "per_angle_scaling": per_angle_scaling,
                "n_phi": n_phi,
                "phi_unique": np.asarray(phi_unique),
                "t1_unique": np.asarray(t1_unique_global),
                "t2_unique": np.asarray(t2_unique_global),
                "n_t1": n_t1,
                "n_t2": n_t2,
                "q": q_val,
                "L": L_val,
                "dt": dt_val,
            }

            n_ooc_workers = max(1, min(4, os.cpu_count() or 1))
            ooc_pool = OOCComputePool(
                n_workers=n_ooc_workers,
                shared_arrays=ooc_shared,
                physics_config=physics_config,
                chunk_boundaries=chunk_boundaries,
                threads_per_worker=max(1, (os.cpu_count() or 4) // n_ooc_workers),
            )
            log.info(
                "Parallel OOC compute: %d chunks across %d workers",
                n_total_chunks,
                n_ooc_workers,
            )
        except (OSError, RuntimeError, MemoryError) as exc:
            log.warning(
                "Parallel OOC pool creation failed (%s), using sequential",
                exc,
            )
            if ooc_shared is not None:
                ooc_shared.cleanup()
                ooc_shared = None
            ooc_pool = None

    def evaluate_total_chi2(params_eval: Any) -> float:
        stride = 10 if fast_chi2_mode else 1

        # Use parallel pool for chi2 evaluation when available
        if ooc_pool is not None:
            return ooc_pool.compute_chi2(np.asarray(params_eval), stride=stride)

        # Sequential fallback
        total_c2 = 0.0
        eval_count = 0
        for i, ind_c in enumerate(iterator):
            if i % stride != 0:
                continue

            p_c = phi_flat[ind_c]
            t1_c = t1_flat[ind_c]
            t2_c = t2_flat[ind_c]
            g2_c = g2_flat[ind_c]
            sigma_c = sigma_flat[ind_c] if sigma_flat is not None else 1.0
            c2_chunk = compute_chunk_chi2(params_eval, p_c, t1_c, t2_c, g2_c, sigma_c)
            total_c2 += c2_chunk
            eval_count += 1

        total_chunks = len(iterator)
        if eval_count > 0:
            scale = total_chunks / eval_count
            return total_c2 * scale
        return 0.0

    # Use per-point sigma if available from data, otherwise unit weighting
    if sigma_flat is None:
        log.info("No per-point sigma available - using unit weighting for OOC")

    # Optimization Loop
    log.info(f"Starting Out-of-Core Loop (Max iter: {max_iter})...")

    # Track early convergence result for return after cleanup
    _early_result: tuple[np.ndarray, np.ndarray, dict] | None = None

    try:
        for i in range(max_iter):
            _iter_start = time.perf_counter()  # noqa: F841

            if ooc_pool is not None:
                # Parallel compute: dispatch all chunks to pool
                chunk_results = ooc_pool.compute_accumulators(np.asarray(params_curr))
                count = sum(
                    end - start
                    for start, end in chunk_boundaries  # noqa: F821
                )
            else:
                # Sequential compute: iterate chunks locally
                chunk_results_local: list[tuple[np.ndarray, np.ndarray, float]] = []
                count = 0
                for indices_chunk in iterator:
                    phi_c = phi_flat[indices_chunk]
                    t1_c = t1_flat[indices_chunk]
                    t2_c = t2_flat[indices_chunk]
                    g2_c = g2_flat[indices_chunk]
                    sigma_c = (
                        sigma_flat[indices_chunk] if sigma_flat is not None else 1.0
                    )
                    JtJ, Jtr, chi2 = compute_chunk_accumulators(
                        params_curr, phi_c, t1_c, t2_c, g2_c, sigma_c
                    )
                    chunk_results_local.append(
                        (np.asarray(JtJ), np.asarray(Jtr), float(chi2))
                    )
                    count += len(indices_chunk)
                chunk_results = chunk_results_local

            # Reduce chunk results (parallel reduction when beneficial)
            n_chunks = len(chunk_results)
            if n_chunks == 0:
                total_JtJ = jnp.zeros((n_params, n_params))
                total_Jtr = jnp.zeros(n_params)
                total_chi2 = 0.0
            elif should_use_parallel_accumulation(n_chunks):
                if i == 0:
                    log.info(
                        "Parallel chunk reduction: %d chunks",
                        n_chunks,
                    )
                total_JtJ_np, total_Jtr_np, total_chi2, _ = accumulate_chunks_parallel(
                    chunk_results,
                    n_workers=max(1, min(4, n_chunks // 4)),
                )
                total_JtJ = jnp.asarray(total_JtJ_np)
                total_Jtr = jnp.asarray(total_Jtr_np)
            else:
                if i == 0:
                    log.debug(
                        "Sequential chunk reduction: %d chunks",
                        n_chunks,
                    )
                total_JtJ_np, total_Jtr_np, total_chi2, _ = (
                    accumulate_chunks_sequential(chunk_results)
                )
                total_JtJ = jnp.asarray(total_JtJ_np)
                total_Jtr = jnp.asarray(total_Jtr_np)

            # Robust Levenberg-Marquardt Step Loop
            step_accepted = False

            # Check for invalid Jacobian/Residuals
            if jnp.any(jnp.isnan(total_Jtr)) or jnp.any(jnp.isinf(total_JtJ)):
                log.warning("Gradient/Hessian contains NaNs/Infs. Checking params.")
                if i == 0:
                    raise RuntimeError("Initial parameters produced invalid gradients.")
                break

            diag_idx = jnp.diag_indices_from(total_JtJ)

            for _lm_iter in range(10):  # Max dampings per iter
                solver_matrix = total_JtJ.at[diag_idx].add(
                    lm_lambda * jnp.diag(total_JtJ)
                )

                try:
                    # use lstsq for robustness against singular matrices
                    step, _, _, _ = jnp.linalg.lstsq(
                        solver_matrix, -total_Jtr, rcond=1e-5
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    step = jnp.full_like(total_Jtr, jnp.nan)  # Signal fail

                # Check step validity
                if jnp.any(jnp.isnan(step)):
                    log.warning(
                        f"Bad step (NaN). Increasing damping ({lm_lambda:.1e} -> {lm_lambda * 10:.1e})"
                    )
                    lm_lambda *= 10
                    continue

                # Proposed parameters
                params_new = params_curr + step
                # Clip
                if bounds is not None:
                    lower, upper = bounds
                    params_new = jnp.clip(
                        params_new, jnp.asarray(lower), jnp.asarray(upper)
                    )

                # Evaluate New Cost
                try:
                    chi2_new = evaluate_total_chi2(params_new)
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    log.warning(f"Eval failed: {e}")
                    chi2_new = jnp.inf

                # Acceptance check
                if chi2_new < total_chi2:
                    # Accept
                    ratio = (total_chi2 - chi2_new) / total_chi2
                    log.info(
                        f"Iter {i + 1}: chi2={float(chi2_new):.4e} (dec {ratio:.1%}), "
                        f"lambda={lm_lambda:.1e}"
                    )
                    params_curr = params_new
                    lm_lambda *= 0.1  # Decrease damping (trust more)
                    if lm_lambda < 1e-7:
                        lm_lambda = 1e-7
                    step_accepted = True

                    # Multi-criteria convergence (v2.22.0)
                    # 1. Per-component relative parameter change (scale-invariant)
                    param_scale = jnp.maximum(jnp.abs(params_curr), 1e-10)
                    rel_change = float(jnp.max(jnp.abs(step) / param_scale))
                    # 2. Relative cost function change
                    cost_change = float(ratio)

                    log.debug(
                        f"  Convergence: xtol={rel_change:.2e} "
                        f"(thresh={xtol:.0e}), "
                        f"ftol={cost_change:.2e} "
                        f"(thresh={ftol:.0e})"
                    )

                    if rel_change < xtol and cost_change < ftol:
                        log.info(
                            f"Out-of-Core converged: xtol={rel_change:.2e}<{xtol:.0e}, "
                            f"ftol={cost_change:.2e}<{ftol:.0e}"
                        )
                        s2 = float(chi2_new) / max(count - n_params_effective, 1)
                        try:
                            pcov = s2 * np.linalg.inv(np.array(total_JtJ))
                        except np.linalg.LinAlgError:
                            log.warning(
                                "Singular J^T J in OOC - using pseudo-inverse for covariance"
                            )
                            pcov = s2 * np.linalg.pinv(np.array(total_JtJ))
                        _early_result = (
                            np.array(params_curr),
                            pcov,
                            {
                                "chi_squared": float(chi2_new),
                                "iterations": i + 1,
                                "convergence_status": "converged",
                                "message": "Out-of-Core converged (xtol+ftol)",
                            },
                        )
                        break
                    break  # Break inner LM loop, proceed to next accumulation
                else:
                    # Reject
                    log.debug(
                        f"Reject step (chi2 {float(chi2_new):.4e} >= {float(total_chi2):.4e}). Damping up."
                    )
                    lm_lambda *= 10

            if _early_result is not None:
                break
            if not step_accepted:
                log.warning("Could not find better step. Stopping.")
                break
    finally:
        # Clean up parallel compute pool and shared memory
        if ooc_pool is not None:
            ooc_pool.shutdown()
        if ooc_shared is not None:
            ooc_shared.cleanup()

    if _early_result is not None:
        return _early_result

    # Determine final status (rel_change initialized to inf before loop)
    converged = rel_change < xtol and cost_change < ftol
    info = {
        "chi_squared": float(total_chi2),
        "iterations": i + 1,
        "convergence_status": "converged" if converged else "max_iter",
        "message": "Out-of-Core accumulation completed",
    }
    # pcov = s^2 * (J^T J)^{-1}  where s^2 = RSS / (n - p_effective)
    # Uses n_params_effective for correct DOF in auto_averaged mode.
    s2 = float(total_chi2) / max(count - n_params_effective, 1)
    try:
        pcov = s2 * np.linalg.inv(np.array(total_JtJ))
    except np.linalg.LinAlgError:
        log.warning("Singular J^T J in OOC - using pseudo-inverse for covariance")
        pcov = s2 * np.linalg.pinv(np.array(total_JtJ))
    return np.array(params_curr), pcov, info
