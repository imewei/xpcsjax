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

from xpcsjax.optimization.nlsq.parameter_utils import ResolvedPhysicalParameters
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

    # Only treat "constant" as frozen-scaling when the caller actually passed
    # a reduced (physics-only) parameter vector. The OOC L-M loop has no
    # per-angle-mode reduction mechanism of its own (unlike stratified_ls.py's
    # fixed-scaling path or hybrid_streaming's "constant" mode) -- it always
    # optimizes the full vector it was given -- so collapsing to n_physical
    # here when n_params is still the full length would understate DOF.
    if per_angle_mode == "constant" and n_params == n_physical:
        return n_physical
    # Explicit "averaged" is a first-class token equivalent to resolved-auto-
    # averaged (mirrors anti_degeneracy_controller / hybrid_streaming).
    if per_angle_mode in ("auto", "averaged") and n_phi >= threshold and n_params == n_physical + 2:
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
    resolved_physical: ResolvedPhysicalParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using Out-of-Core Global Accumulation for massive datasets.

    This strategy virtually chunks the dataset using Index-Based Stratification,
    accumulates the full Hessian and Gradient (J^T J, J^T r) by iterating
    over chunks, and takes a global Levenberg-Marquardt step.

    Guarantees identical convergence to standard NLSQ but with minimal memory.

    Parameters
    ----------
    stratified_data : Any
        Stratified data object (unused, kept for API compatibility).
    data : Any
        Original XPCS data object exposing ``.phi``, ``.t1``, ``.t2``, ``.g2``,
        ``.q``, ``.L``.
    per_angle_scaling : bool
        Whether per-angle scaling is enabled.
    physical_param_names : list[str]
        Names of the physical parameters.
    initial_params : np.ndarray
        Initial parameter guess.
    bounds : tuple of np.ndarray or None
        Parameter bounds ``(lower, upper)`` or ``None`` for unbounded.
    log : logging.Logger or logging.LoggerAdapter
        Logger instance.
    config : Any
        Configuration object or dict (read for ``dt`` and ``max_iterations``).
    fast_chi2_mode : bool, optional
        If ``True``, subsample chunks (stride 10) for chi-squared evaluation
        during the line search (default: ``False``).
    anti_degeneracy_config : dict, optional
        Anti-degeneracy configuration. Only its ``per_angle_mode`` /
        ``constant_scaling_threshold`` keys are consulted, to pick the effective
        parameter count used for covariance DOF; the anti-degeneracy *layers*
        are not run on this path.
    resolved_physical : ResolvedPhysicalParameters, optional
        Free/fixed split for the trailing physical-parameter slice. A fixed
        physical slot is pinned to its configured value and masked out of
        every Levenberg-Marquardt step and the covariance solve (Task 7c);
        ``None`` (or an all-free mask) is a complete no-op.

    Returns
    -------
    popt : np.ndarray
        Optimized parameters.
    pcov : np.ndarray
        Parameter covariance matrix ``s^2 (J^T J)^{-1}``, falling back to the
        pseudo-inverse when ``J^T J`` is singular.
    info : dict
        Optimization information (``chi_squared``, ``iterations``,
        ``convergence_status``, ``message``, ``fast_chi2_mode``).

    Notes
    -----
    This method uses the full homodyne physics via
    ``compute_g2_scaled()``, identical to stratified least-squares. The
    Anti-Degeneracy Defense System layers are not yet wired on this path.
    """
    import jax.numpy as jnp

    _start_time = time.perf_counter()  # noqa: F841
    log.info("Initializing Out-of-Core Global Stratified Optimization (Full Physics)...")

    chi2_stride = 10 if fast_chi2_mode else 1
    if fast_chi2_mode:
        log.warning(
            "fast_chi2_mode enabled: line-search chi2 computed from every %dth chunk "
            "(subsampled, rescaled); final chi2/pcov are NOT subsampled",
            chi2_stride,
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

    # Fixed physical parameters (Task 7c). `compute_chunk_accumulators`
    # computes its Jacobian via `jax.jacfwd(r_fn)(p)` INSIDE its own
    # `@jax.jit` boundary, w.r.t. its own bound argument `p` -- wrapping the
    # *call* with a restore closure (the pattern the plain NLSQ tail uses)
    # cannot change what that Jacobian is differentiated against, since no
    # external autodiff call chains through it. Instead: keep `params_curr`
    # full-length everywhere (so the physics kernels always see a correctly
    # laid-out vector), and mask the returned (J^T J, J^T r) down to the free
    # submatrix before every LM step / covariance solve. This is exact --
    # `jax.jacfwd` builds each Jacobian column independently, so
    # `(J^T J)[free, free] == J[:, free]^T @ J[:, free]` and dropping a
    # column is identical to never differentiating along it.
    free_idx: np.ndarray | None = None
    if resolved_physical is not None and not resolved_physical.free_mask.all():
        n_physical_names = len(resolved_physical.physical_names)
        # Pin every fixed physical slot to its configured value before the
        # optimizer (or the physics kernels) ever see it.
        initial_params = np.concatenate(
            [
                np.asarray(initial_params[:-n_physical_names], dtype=np.float64),
                resolved_physical.values_full,
            ]
        )
        full_free_mask = np.ones(n_params, dtype=bool)
        full_free_mask[-n_physical_names:] = resolved_physical.free_mask
        free_idx = np.where(full_free_mask)[0]

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
        config.config if hasattr(config, "config") else (config if isinstance(config, dict) else {})
    )

    # Extract physics constants from data (full homodyne physics)
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
    # averaged uses a compressed vector (contrast_avg, offset_avg, physical)
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
    max_iter = (cfg_dict.get("optimization") or {}).get("max_iterations", 50)

    # Convergence tolerances (multi-criteria, matching standard NLSQ). Read
    # from the same optimization.nlsq block the sibling stratified_ls.py path
    # uses, so a user-configured xtol/ftol isn't silently discarded whenever
    # a fit routes to OUT_OF_CORE instead of STANDARD/CHUNKED.
    nlsq_cfg = cfg_dict.get("optimization", {}).get("nlsq", {})
    # Relative parameter change (per-component max, not norm). `.get(key,
    # default)` only substitutes when the key is absent — an explicit
    # `null` in the YAML must be guarded separately or `float(None)` raises.
    xtol_raw = nlsq_cfg.get("xtol")
    xtol = 1e-8 if xtol_raw is None else float(xtol_raw)
    # Relative cost function change
    ftol_raw = nlsq_cfg.get("ftol", nlsq_cfg.get("tolerance"))
    ftol = 1e-8 if ftol_raw is None else float(ftol_raw)
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
                np.asarray(sigma_flat)[all_indices_arr] if sigma_flat is not None else None
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
        stride = chi2_stride

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

    def _accumulate_at(params: Any) -> tuple[Any, Any, float, int]:
        """Accumulate (J^T J, J^T r, chi2, count) at ``params``.

        Single source of truth for the per-chunk accumulate+reduce step, used
        both inside the L-M loop and to refresh the accumulators at whatever
        point is actually being returned (post-step params), so the reported
        covariance never mixes a pre-step Hessian with a post-step point.
        """
        if ooc_pool is not None:
            chunk_results = ooc_pool.compute_accumulators(np.asarray(params))
            acc_count = sum(end - start for start, end in chunk_boundaries)  # noqa: F821
        else:
            chunk_results_local: list[tuple[np.ndarray, np.ndarray, float]] = []
            acc_count = 0
            for indices_chunk in iterator:
                phi_c = phi_flat[indices_chunk]
                t1_c = t1_flat[indices_chunk]
                t2_c = t2_flat[indices_chunk]
                g2_c = g2_flat[indices_chunk]
                sigma_c = sigma_flat[indices_chunk] if sigma_flat is not None else 1.0
                JtJ_c, Jtr_c, chi2_c = compute_chunk_accumulators(
                    params, phi_c, t1_c, t2_c, g2_c, sigma_c
                )
                chunk_results_local.append((np.asarray(JtJ_c), np.asarray(Jtr_c), float(chi2_c)))
                acc_count += len(indices_chunk)
            chunk_results = chunk_results_local

        acc_n_chunks = len(chunk_results)
        if acc_n_chunks == 0:
            return jnp.zeros((n_params, n_params)), jnp.zeros(n_params), 0.0, acc_count
        if should_use_parallel_accumulation(acc_n_chunks):
            jtj_np, jtr_np, chi2_total, _ = accumulate_chunks_parallel(
                chunk_results, n_workers=max(1, min(4, acc_n_chunks // 4))
            )
        else:
            jtj_np, jtr_np, chi2_total, _ = accumulate_chunks_sequential(chunk_results)
        return jnp.asarray(jtj_np), jnp.asarray(jtr_np), chi2_total, acc_count

    def _active_jtj_jtr(full_JtJ: Any, full_Jtr: Any) -> tuple[Any, Any]:
        """Slice (J^T J, J^T r) down to the free-parameter submatrix.

        A no-op (returns the inputs unchanged) when no physical parameter is
        fixed. Exact per the Task 7c note above `free_idx`.
        """
        if free_idx is None:
            return full_JtJ, full_Jtr
        free_idx_jnp = jnp.asarray(free_idx)
        return full_JtJ[free_idx_jnp][:, free_idx_jnp], full_Jtr[free_idx_jnp]

    def _embed_step(active_step: Any) -> Any:
        """Embed a free-only LM step into a full-length step.

        Zeros elsewhere so a fixed physical slot never moves.
        """
        if free_idx is None:
            return active_step
        return jnp.zeros(n_params).at[jnp.asarray(free_idx)].set(active_step)

    def _pcov_from_active_jtj(active_JtJ: Any, chi2: float, count: int) -> np.ndarray:
        """Invert the free-submatrix Hessian into a full-length covariance.

        Zeros on every fixed row/column -- exactly 0 uncertainty, matching
        the plain NLSQ tail's contract.
        """
        s2 = float(chi2) / max(count - n_params_effective, 1)
        try:
            active_pcov = s2 * np.linalg.inv(np.array(active_JtJ))
        except np.linalg.LinAlgError:
            log.warning("Singular J^T J in OOC - using pseudo-inverse for covariance")
            active_pcov = s2 * np.linalg.pinv(np.array(active_JtJ))
        if free_idx is None:
            return active_pcov
        full_pcov = np.zeros((n_params, n_params))
        full_pcov[np.ix_(free_idx, free_idx)] = active_pcov
        return full_pcov

    # Optimization Loop
    log.info(f"Starting Out-of-Core Loop (Max iter: {max_iter})...")

    # Track early convergence result for return after cleanup
    _early_result: tuple[np.ndarray, np.ndarray, dict] | None = None
    # Refreshed (J^T J, J^T r, chi2, count) at the final params_curr, computed
    # inside the try (before the finally's ooc_pool.shutdown()) so the
    # post-loop covariance build never calls back into an already-shut-down
    # pool.
    _final_accum: tuple[Any, Any, float, int] | None = None

    # Seed loop-carried accumulators so the post-loop summary stays well-defined
    # even when max_iter <= 0 (loop body never runs).
    i = -1
    count = 0
    # total_chi2/total_JtJ need no pre-loop seed: the post-loop `_final_accum`
    # recompute (below) always runs before either name is read, whether or
    # not the loop body executes for max_iter <= 0.
    # Tracks WHY the loop exited early (as opposed to normal max_iter
    # exhaustion), so the post-loop status report doesn't mislabel a
    # numerical-instability abort or a stalled line search as "max_iter".
    break_reason: str | None = None

    try:
        for i in range(max_iter):
            _iter_start = time.perf_counter()  # noqa: F841

            total_JtJ, total_Jtr, total_chi2, count = _accumulate_at(params_curr)

            # Robust Levenberg-Marquardt Step Loop
            step_accepted = False

            # Reduce to the free-parameter submatrix (Task 7c; a no-op when
            # no physical parameter is fixed) BEFORE the finite check, so a
            # fixed slot's (irrelevant, never-stepped) Jacobian column can't
            # spuriously abort a fit that's actually fine on the free
            # directions.
            active_JtJ, active_Jtr = _active_jtj_jtr(total_JtJ, total_Jtr)

            # Check for invalid Jacobian/Residuals: reject any non-finite value
            # (NaN OR Inf) in EITHER the gradient vector or the Hessian. The
            # prior asymmetric check (NaN-only on Jtr, Inf-only on JtJ) let a
            # NaN Hessian / Inf gradient slip past the i==0 hard-stop and
            # silently poison the covariance solve below.
            if not (jnp.all(jnp.isfinite(active_Jtr)) and jnp.all(jnp.isfinite(active_JtJ))):
                log.warning("Gradient/Hessian contains NaNs/Infs. Checking params.")
                if i == 0:
                    raise RuntimeError("Initial parameters produced invalid gradients.")
                break_reason = "non_finite_gradient"
                break

            diag_idx = jnp.diag_indices_from(active_JtJ)

            # Line-search baseline measured with the SAME estimator as the trial
            # cost. In fast_chi2_mode evaluate_total_chi2 returns a strided+scaled
            # estimate, so comparing that estimate against the exact full
            # accumulator chi2 (total_chi2) would bias step acceptance. Use a
            # like-for-like estimate of the current point. With fast_chi2_mode off
            # this is exactly total_chi2 (no extra evaluation, behavior unchanged).
            chi2_ref = evaluate_total_chi2(params_curr) if fast_chi2_mode else total_chi2

            for _lm_iter in range(10):  # Max dampings per iter
                solver_matrix = active_JtJ.at[diag_idx].add(lm_lambda * jnp.diag(active_JtJ))

                try:
                    # use lstsq for robustness against singular matrices
                    active_step, _, _, _ = jnp.linalg.lstsq(solver_matrix, -active_Jtr, rcond=1e-5)
                    step = _embed_step(active_step)
                except (ValueError, RuntimeError, FloatingPointError):
                    step = jnp.full(n_params, jnp.nan)  # Signal fail

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
                    params_new = jnp.clip(params_new, jnp.asarray(lower), jnp.asarray(upper))

                # Evaluate New Cost
                try:
                    chi2_new = evaluate_total_chi2(params_new)
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    log.warning(f"Eval failed: {e}")
                    chi2_new = jnp.inf

                # Acceptance check (against the like-for-like baseline)
                if chi2_new < chi2_ref:
                    # Accept
                    ratio = (chi2_ref - chi2_new) / chi2_ref
                    log.info(
                        f"Iter {i + 1}: chi2={float(chi2_new):.4e} (dec {ratio:.1%}), "
                        f"lambda={lm_lambda:.1e}"
                    )
                    params_curr = params_new
                    lm_lambda *= 0.1  # Decrease damping (trust more)
                    if lm_lambda < 1e-7:
                        lm_lambda = 1e-7
                    step_accepted = True

                    # Multi-criteria convergence
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
                        # Refresh the accumulators at the ACCEPTED (post-step)
                        # params_curr before building pcov: total_JtJ above is
                        # still the pre-step Hessian from this iteration's
                        # top-of-loop accumulation, so pairing it with the
                        # post-step chi2_new/params_curr would report a
                        # covariance for a point it wasn't computed at.
                        conv_JtJ, _conv_Jtr, conv_chi2, conv_count = _accumulate_at(params_curr)
                        conv_active_JtJ, _ = _active_jtj_jtr(conv_JtJ, _conv_Jtr)
                        pcov = _pcov_from_active_jtj(conv_active_JtJ, conv_chi2, conv_count)
                        _early_result = (
                            np.array(params_curr),
                            pcov,
                            {
                                "chi_squared": float(conv_chi2),
                                "iterations": i + 1,
                                "convergence_status": "converged",
                                "message": "Out-of-Core converged (xtol+ftol)",
                                "fast_chi2_mode": fast_chi2_mode,
                            },
                        )
                        break
                    break  # Break inner LM loop, proceed to next accumulation
                else:
                    # Reject
                    log.debug(
                        f"Reject step (chi2 {float(chi2_new):.4e} >= {float(chi2_ref):.4e}). Damping up."
                    )
                    lm_lambda *= 10

            if _early_result is not None:
                break
            if not step_accepted:
                log.warning("Could not find better step. Stopping.")
                break_reason = "line_search_stalled"
                break

        if _early_result is None:
            # Refresh the accumulators at whatever point params_curr actually
            # holds before building the final pcov: total_JtJ/total_chi2 above
            # are from the top of the LAST loop iteration, evaluated at the
            # pre-step params of that iteration -- not necessarily the
            # params_curr being returned below (e.g. a step was accepted this
            # iteration, or the loop exited via a mid-iteration break).
            # Recomputing here keeps pcov's Hessian/DOF paired with the actual
            # returned point instead of a stale pre-step one. MUST run before
            # the `finally` below shuts down ooc_pool -- _accumulate_at uses
            # it when set, so calling this after shutdown raises.
            _final_accum = _accumulate_at(params_curr)
    finally:
        # Clean up parallel compute pool and shared memory
        if ooc_pool is not None:
            ooc_pool.shutdown()
        if ooc_shared is not None:
            ooc_shared.cleanup()

    if _early_result is not None:
        return _early_result

    assert _final_accum is not None
    total_JtJ, _total_Jtr, total_chi2, count = _final_accum

    # Determine final status (rel_change initialized to inf before loop).
    # A break_reason set above means the loop exited via a numerical-
    # instability or stalled-line-search abort, NOT ordinary max_iter
    # exhaustion -- report that distinctly so a caller branching on
    # convergence_status doesn't mistake one for the other.
    converged = rel_change < xtol and cost_change < ftol
    if break_reason is not None:
        convergence_status = break_reason
    elif converged:
        convergence_status = "converged"
    else:
        convergence_status = "max_iter"
    info = {
        "chi_squared": float(total_chi2),
        "iterations": i + 1,
        "convergence_status": convergence_status,
        "message": "Out-of-Core accumulation completed",
        "fast_chi2_mode": fast_chi2_mode,
    }
    # pcov = s^2 * (J^T J)^{-1}  where s^2 = RSS / (n - p_effective)
    # Uses n_params_effective for correct DOF in averaged mode.
    final_active_JtJ, _ = _active_jtj_jtr(total_JtJ, _total_Jtr)
    pcov = _pcov_from_active_jtj(final_active_JtJ, total_chi2, count)
    return np.array(params_curr), pcov, info
