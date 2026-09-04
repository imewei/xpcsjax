"""Heterodyne pointwise model function and hybrid-streaming wrapper (Phase 2-A).

Mirrors the homodyne ``fit_with_stratified_hybrid_streaming`` pattern from
``hybrid_streaming.py`` for the heterodyne two-time correlation kernel.

Public API
----------
build_heterodyne_pointwise_model(*, stratified_data, model, physical_param_names)
    -> (model_fn, x_data, y_data, p0, meta)

fit_with_stratified_hybrid_streaming_heterodyne(*, stratified_data, model,
    physical_param_names, initial_params, bounds, hybrid_config, anti_degeneracy_config)
    -> (popt, pcov, info)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics import (
    assemble_anti_degeneracy_diagnostics,
)
from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
    build_gradient_collapse_callback,
    gradient_monitor_diagnostics,
)
from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper
from xpcsjax.optimization.nlsq.per_angle_mode import PerAngleMode
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import HeterodyneStratifiedData

logger = get_logger(__name__)


def _sigma_weighted_mse(residuals: Any, sigma: Any | None) -> Any:
    """Mean-squared residual, optionally weighted by per-point sigma.

    Mirrors the safe_sigma/valid_sigma EPS-guard convention used in
    strategies/residual_jit.py EXACTLY: points with sigma <= EPS are excluded
    from the loss entirely (residual treated as zero for that point, matching
    residual_jit.py's `jnp.where(valid_sigma, (obs-theory)/safe_sigma, 0.0)`)
    rather than falling back to an unweighted raw residual, which would be a
    materially different (and uncited) aggregation behavior. When sigma is
    None, this is exactly the pre-existing unweighted jnp.mean(residuals**2).
    """
    if sigma is None:
        return jnp.mean(residuals**2)
    EPS = 1e-10
    sigma_jax = jnp.asarray(sigma)
    valid_sigma = sigma_jax > EPS
    safe_sigma = jnp.where(valid_sigma, sigma_jax, 1.0)
    weighted_sq = jnp.where(valid_sigma, (residuals / safe_sigma) ** 2, 0.0)
    return jnp.mean(weighted_sq)


# ---------------------------------------------------------------------------
# Optional NLSQ import — mirrors hybrid_streaming.py pattern
# ---------------------------------------------------------------------------
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    HAS_HYBRID_STREAMING = True
except ImportError:
    AdaptiveHybridStreamingOptimizer = None  # type: ignore[assignment,misc]
    HybridStreamingConfig = None  # type: ignore[assignment,misc]
    HAS_HYBRID_STREAMING = False


def _bin_to_grid(values: np.ndarray, grid: np.ndarray, axis_name: str) -> np.ndarray:
    """Bin values to grid indices via ``searchsorted`` + boundary clip.

    Mirrors homodyne ``_bin_to_grid`` in ``hybrid_streaming.py`` exactly.
    An unguarded clip silently routes data lying outside the fitted grid to
    the boundary bin — a data-integrity violation. We clip (to stay in-bounds)
    but surface how many points were affected so misaligned data/config is
    not silent.
    """
    raw = np.searchsorted(grid, values)
    n_oob = int(np.sum(raw >= len(grid)))
    # Below-grid points snap to bin 0 just like a legitimate grid[0] value, so
    # raw alone cannot surface them — count them explicitly.
    n_low = int(np.sum(np.asarray(values) < grid[0]))
    if n_oob > 0 or n_low > 0:
        logger.warning(
            "%d data point(s) lie above and %d below the %s grid; clipped to the "
            "boundary bin. Check data/config grid alignment.",
            n_oob,
            n_low,
            axis_name,
        )
    return np.clip(raw, 0, len(grid) - 1)


def build_heterodyne_pointwise_model(
    *,
    stratified_data: HeterodyneStratifiedData,
    model: HeterodyneModel,
    physical_param_names: list[str],
    per_angle_mode: PerAngleMode = "constant",
) -> tuple[Any, np.ndarray, np.ndarray, list[float], dict[str, Any]]:
    """Build the pointwise model function and data arrays for hybrid streaming.

    Parameters
    ----------
    stratified_data :
        Flat heterodyne stratified data from
        ``build_heterodyne_stratified_data``.
    model :
        Configured HeterodyneModel providing ``t``, ``q``, ``dt``, and
        ``param_manager``.
    physical_param_names :
        Names of the varying physics parameters (``model.param_manager.varying_names``).
    per_angle_mode :
        Resolved canonical scaling treatment for per-angle contrast/offset. The
        builder accepts ONLY the three resolved tokens; ``"auto"`` must be resolved
        upstream via
        :func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode`.
        The native parameter vector is scaling-first ``[scaling_head | physics]``.
        ``"constant"`` (default) — freeze quantile-estimated scaling inside the JIT
        closure; the scaling head is empty (physics-only optimizer vector).
        ``"averaged"`` — prepend 2 optimized scaling scalars (mean contrast,
        mean offset) to ``p0``; the JIT closure broadcasts them across all angles.
        ``"individual"`` — prepend ``2 * n_phi`` optimized scaling params (per-angle
        contrast then per-angle offset); the JIT closure reads them directly.

    Returns
    -------
    model_fn : callable
        JIT-compiled pointwise model function with signature
        ``model_fn(x_batch, *params) -> jnp.ndarray``.
    x_data : np.ndarray of shape (N, 3), int32
        Index array ``[phi_idx, t1_idx, t2_idx]`` per data point.
    y_data : np.ndarray of shape (N,), float64
        Observed C2 values.
    p0 : list[float]
        Initial values for the varying physics parameters.
    meta : dict
        ``{"phi_unique": ..., "contrast_arr": ..., "offset_arr": ...}``
        Arrays are in sorted phi_unique order (element k ↔ phi_unique[k]).
    """
    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne_pointwise
    from xpcsjax.optimization.nlsq.parameter_utils import compute_quantile_per_angle_scaling

    # ------------------------------------------------------------------
    # 1. Gather flat data from stratified_data
    # ------------------------------------------------------------------
    all_phi = stratified_data.phi_flat
    all_t1 = stratified_data.t1_flat
    all_t2 = stratified_data.t2_flat
    g2_flat = stratified_data.g2_flat

    # ------------------------------------------------------------------
    # 2. Build unique sorted grids
    # ------------------------------------------------------------------
    phi_unique: np.ndarray = np.array(sorted(set(all_phi.tolist())), dtype=np.float64)
    t_unique: np.ndarray = np.asarray(model.t, dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Bin values to grid indices (mirrors homodyne _bin_to_grid)
    # ------------------------------------------------------------------
    phi_idx_arr = _bin_to_grid(all_phi, phi_unique, "phi")
    t1_idx_arr = _bin_to_grid(all_t1, t_unique, "t1")
    t2_idx_arr = _bin_to_grid(all_t2, t_unique, "t2")

    # ------------------------------------------------------------------
    # 4. Filter diagonal AND t=0 boundary (mirrors _compute_residuals_jit)
    #
    # _compute_residuals_jit in heterodyne_jax_backend.py excludes BOTH:
    #   (a) the diagonal (t1 == t2)
    #   (b) the t=0 row/column (t1_idx == 0 OR t2_idx == 0)
    # Yielding (n_t-1)*(n_t-2) points per angle.  Both exclusions must be
    # applied here so the pointwise training set matches the residual support.
    # ------------------------------------------------------------------
    keep = (t1_idx_arr != t2_idx_arr) & (t1_idx_arr > 0) & (t2_idx_arr > 0)
    phi_idx_arr = phi_idx_arr[keep]
    t1_idx_arr = t1_idx_arr[keep]
    t2_idx_arr = t2_idx_arr[keep]
    g2_flat = g2_flat[keep]

    x_data = np.column_stack(
        [phi_idx_arr.astype(np.int32), t1_idx_arr.astype(np.int32), t2_idx_arr.astype(np.int32)]
    )
    y_data = np.asarray(g2_flat, dtype=np.float64)

    # ------------------------------------------------------------------
    # 5. Compute per-angle quantile scaling, then REINDEX to phi_unique order
    # ------------------------------------------------------------------
    # compute_quantile_per_angle_scaling iterates angles in the chunk/input
    # order from stratified_data, which is the original phi order (may differ
    # from sorted phi_unique). We build a phi_val → sorted_index mapping and
    # reorder the returned arrays so element k ↔ phi_unique[k].
    contrast_raw, offset_raw = compute_quantile_per_angle_scaling(stratified_data)

    # Determine the phi value that owns each output slot in contrast_raw /
    # offset_raw.  The function internally iterates phi_unique in sorted order
    # when operating on the flat-field format (phi_flat / t1_flat / t2_flat),
    # so the raw arrays ARE already in sorted phi order.  We verify this by
    # comparing with the phi_unique we built and reindex defensively using the
    # phi_flat values for robustness.
    #
    # Strategy: compute_quantile_per_angle_scaling sorts its own phi_unique
    # from the data.  Its output index k corresponds to its k-th sorted phi.
    # Our phi_unique is also sorted from the same data, so the orders match
    # and contrast_raw[k] already maps to phi_unique[k].  We still build an
    # explicit mapping from the original per-chunk phi values to confirm
    # alignment for the multi-phi case (see CRITICAL alignment note in the
    # task spec).
    n_phi = len(phi_unique)
    if len(contrast_raw) != n_phi:
        raise ValueError(
            f"compute_quantile_per_angle_scaling returned {len(contrast_raw)} entries "
            f"but phi_unique has {n_phi} entries."
        )

    # The mapping is IDENTITY: compute_quantile_per_angle_scaling fills its
    # output by iterating a SORTED phi_unique (parameter_utils builds phi_unique
    # sorted and assigns contrast/offset by its enumerate index), so
    # contrast_raw[k] / offset_raw[k] already correspond to phi_unique[k] — the
    # same sorted order the JIT closure bins phi_idx against (_bin_to_grid uses
    # the sorted phi_unique). We therefore use the raw arrays directly.
    #
    # A previous "defensive" reindex keyed the slots on the chunk/insertion phi
    # order and assigned `contrast_arr[reindex] = contrast_raw`; that composes
    # two different permutations and silently scattered the frozen per-angle
    # scaling to the WRONG angles whenever the input phi was not already
    # ascending. It was identity only for pre-sorted phi (the sole case the
    # tests exercised), so the bug never surfaced. Dropping it restores the
    # correct per-angle alignment with zero change for ascending phi.
    contrast_arr = np.asarray(contrast_raw, dtype=np.float64)
    offset_arr = np.asarray(offset_raw, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. Initial parameter vector (varying physics; optionally + scaling tail)
    # ------------------------------------------------------------------
    p0: list[float] = [float(v) for v in model.param_manager.get_initial_values()]
    n_physics_varying = len(p0)

    # Canonical scaling-first layout authority: [scaling_head | physics_tail].
    # Rejects the unresolved auto token (resolve upstream); the ValueError surfaces
    # here for unknown tokens.
    mapper = ParameterIndexMapper.canonical(
        mode=per_angle_mode, n_phi=n_phi, n_physics=n_physics_varying
    )
    n_scaling = mapper.n_optimized
    if per_angle_mode == "averaged":
        contrast0 = float(np.mean(contrast_arr))
        offset0 = float(np.mean(offset_arr))
        p0 = [contrast0, offset0, *p0]  # scaling-first head
    elif per_angle_mode == "individual":
        # scaling-first head: [contrast_per_angle | offset_per_angle], then physics
        p0 = [*contrast_arr.tolist(), *offset_arr.tolist(), *p0]
    elif per_angle_mode == "constant":
        pass  # scaling frozen, applied in residual; head empty
    else:  # pragma: no cover - canonical() already rejected the unresolved token
        raise ValueError(
            f"unknown per_angle_mode {per_angle_mode!r}; valid: constant, averaged, individual"
        )

    # ------------------------------------------------------------------
    # 7. Prepare JAX-side fixed tensors for the closure
    # ------------------------------------------------------------------
    fixed_full_jax = jnp.asarray(model.param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(list(model.param_manager.varying_indices), dtype=jnp.int32)
    tied_idx_pairs = model.param_manager.tied_idx_pairs
    t_jax = jnp.asarray(t_unique, dtype=jnp.float64)
    q_val = float(model.q)
    dt_val = float(model.dt)
    phi_unique_jax = jnp.asarray(phi_unique, dtype=jnp.float64)
    contrast_jax = jnp.asarray(contrast_arr, dtype=jnp.float64)
    offset_jax = jnp.asarray(offset_arr, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # 8. Build JIT-compiled pointwise model function
    # ------------------------------------------------------------------
    # Cache the compile-time constants as local Python names for the closure.
    # per_angle_mode and the mapper slices are Python-level constants (not JAX
    # tracers), so the if/elif branches below are static at JIT trace time.
    _per_angle_mode = per_angle_mode
    _n_phi_local = len(phi_unique)  # compile-time constant for individual slice
    _scaling_block = mapper.scaling_block  # Python-static slice (head)
    _physics_block = mapper.physics_block  # Python-static slice (tail)

    @jax.jit
    def model_fn(x_batch: jnp.ndarray, *params_tuple: jnp.ndarray) -> jnp.ndarray:
        """Pointwise heterodyne model: params = [scaling_head | physics_tail]."""
        x_batch_2d = jnp.atleast_2d(x_batch)
        params_all = jnp.stack(params_tuple)

        # Reconstruct full physics parameter vector from fixed + varying.
        # Layout is scaling-first: physics is the TAIL, scaling the HEAD.
        physics = params_all[_physics_block]
        full = fixed_full_jax.at[varying_indices_jax].set(physics)
        for child_idx, parent_idx in tied_idx_pairs:
            full = full.at[child_idx].set(full[parent_idx])

        # Resolve per-angle scaling — branch is static (compile-time constant).
        if _per_angle_mode == "averaged":
            head = params_all[_scaling_block]
            contrasts = jnp.full((_n_phi_local,), head[0])
            offsets = jnp.full((_n_phi_local,), head[1])
        elif _per_angle_mode == "individual":
            head = params_all[_scaling_block]
            contrasts = head[:_n_phi_local]
            offsets = head[_n_phi_local:]
        else:  # constant
            contrasts = contrast_jax
            offsets = offset_jax

        # Extract grid indices
        phi_idx = x_batch_2d[:, 0].astype(jnp.int32)
        t1_idx = x_batch_2d[:, 1].astype(jnp.int32)
        t2_idx = x_batch_2d[:, 2].astype(jnp.int32)

        # Delegate to the pointwise heterodyne kernel
        result = compute_c2_heterodyne_pointwise(
            full,
            t_jax,
            q_val,
            dt_val,
            phi_unique=phi_unique_jax,
            phi_idx=phi_idx,
            t1_idx=t1_idx,
            t2_idx=t2_idx,
            contrast=contrasts,
            offset=offsets,
        )
        return jnp.squeeze(result)

    # ------------------------------------------------------------------
    # Pre-compute masked sigma aligned 1:1 with x_data/y_data.
    # We store the raw sigma_3d lookup here (before sigma uniformity check)
    # so the wrapper does not need to re-derive the keep mask.  If sigma is
    # not available on stratified_data the entry is None.
    # ------------------------------------------------------------------
    meta_sigma: np.ndarray | None = None
    if hasattr(stratified_data, "sigma") and stratified_data.sigma is not None:
        sigma_3d = np.asarray(stratified_data.sigma, dtype=np.float64)
        # Reuse the pre-keep phi/t indices from the PRE-filter arrays so the
        # mask aligns with the flat order we started from.
        phi_idx_pre = _bin_to_grid(all_phi, phi_unique, "phi_sigma_pre")
        t1_idx_pre = _bin_to_grid(all_t1, t_unique, "t1_sigma_pre")
        t2_idx_pre = _bin_to_grid(all_t2, t_unique, "t2_sigma_pre")
        # `keep` was built from the same pre-filter arrays, so this is aligned.
        sigma_sel = sigma_3d[phi_idx_pre[keep], t1_idx_pre[keep], t2_idx_pre[keep]]
        meta_sigma = sigma_sel

    # ------------------------------------------------------------------
    # Scaling-tail parameter bounds (used by later tasks for joint bounds array)
    # ------------------------------------------------------------------
    if per_angle_mode == "averaged":
        c_lo, c_hi = 0.01, max(2.0 * contrast0, 1.0)
        # Offset is a DC baseline that can be negative; use a symmetric bound
        # centered on offset0 so the lower bound permits negative offsets.
        o_lo = offset0 - max(abs(offset0), 1.0)
        o_hi = offset0 + max(abs(offset0), 1.0)
        scaling_lower = np.array([c_lo, o_lo], dtype=np.float64)
        scaling_upper = np.array([c_hi, o_hi], dtype=np.float64)
    elif per_angle_mode == "individual":
        # Per-angle contrast bounds: [0.01, max(2*contrast, 1.0)] element-wise.
        contrast_lower = np.full(n_phi, 0.01, dtype=np.float64)
        contrast_upper = np.maximum(2.0 * contrast_arr, 1.0)
        # Per-angle symmetric offset bounds: each angle gets its own interval
        # centered on its offset estimate, half-width max(|offset|, 1.0).
        o_range_arr = np.maximum(np.abs(offset_arr), 1.0)
        offset_lower = offset_arr - o_range_arr
        offset_upper = offset_arr + o_range_arr
        # Concatenation order [contrast(n_phi) | offset(n_phi)] matches the
        # model_fn slicing (tail[:n_phi] = contrasts, tail[n_phi:] = offsets).
        scaling_lower = np.concatenate([contrast_lower, offset_lower])
        scaling_upper = np.concatenate([contrast_upper, offset_upper])
    else:  # constant
        scaling_lower = np.empty(0, dtype=np.float64)
        scaling_upper = np.empty(0, dtype=np.float64)

    meta: dict[str, Any] = {
        "phi_unique": phi_unique,
        "contrast_arr": contrast_arr,
        "offset_arr": offset_arr,
        "keep_mask": keep,
        "n_data_points": int(keep.sum()),
        "sigma": meta_sigma,
        # Unique time grid the pointwise kernel was indexed against (t1_idx /
        # t2_idx in x_data address THIS array, not necessarily model.t — they
        # are equal here since build_heterodyne_stratified_data syncs the model
        # time axis, but expose it explicitly so downstream residual builders
        # index against the exact same grid).
        "t_unique": t_unique,
        # Scaling-mode metadata (canonical resolved token)
        "per_angle_mode": per_angle_mode,
        "n_scaling": n_scaling,
        "n_physics_varying": n_physics_varying,
        "scaling_bounds": (scaling_lower, scaling_upper),
        # Authoritative angle count: the same phi_unique the JIT closure and the
        # pointwise kernel were built against. Downstream (AdaptiveRegularizer)
        # must use THIS, not a fresh float-set count which can overcount via
        # float-representation noise.
        "n_phi": len(phi_unique),
    }

    return model_fn, x_data, y_data, p0, meta


def _build_hybrid_streaming_config(nested: dict[str, Any]) -> Any:
    """Build a HybridStreamingConfig from a nested override dict.

    All 24 keys confirmed present in nlsq.HybridStreamingConfig.
    """
    if HybridStreamingConfig is None:
        raise ImportError(
            "nlsq.HybridStreamingConfig not available. "
            "Install nlsq>=0.7.5 to use heterodyne hybrid streaming."
        )

    defaults: dict[str, Any] = {
        "normalize": True,
        "normalization_strategy": "auto",
        "warmup_iterations": 200,
        "max_warmup_iterations": 500,
        "warmup_learning_rate": 1e-3,
        "gauss_newton_max_iterations": 100,
        "gauss_newton_tol": 1e-8,
        "chunk_size": 10000,
        "trust_region_initial": 1.0,
        "regularization_factor": 1e-10,
        "enable_checkpoints": True,
        "checkpoint_frequency": 100,
        "validate_numerics": True,
        "verbose": 1,
        "log_frequency": 1,
        "enable_warm_start_detection": True,
        "warm_start_threshold": 0.01,
        "enable_adaptive_warmup_lr": True,
        "warmup_lr_refinement": 1e-6,
        "warmup_lr_careful": 1e-5,
        "enable_cost_guard": True,
        "cost_increase_tolerance": 0.05,
        "enable_step_clipping": True,
        "max_warmup_step_size": 0.1,
        # L3 group-variance regularization (Task 2)
        "enable_group_variance_regularization": False,
        "group_variance_lambda": 0.0,
        "group_variance_indices": None,
    }

    # Apply overrides from caller (only for keys that exist in HybridStreamingConfig)
    merged = {**defaults, **{k: v for k, v in nested.items() if k in defaults}}

    return HybridStreamingConfig(**merged)


def fit_with_stratified_hybrid_streaming_heterodyne(
    *,
    stratified_data: HeterodyneStratifiedData,
    model: HeterodyneModel,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    hybrid_config: dict[str, Any] | None = None,
    anti_degeneracy_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit heterodyne model via NLSQ AdaptiveHybridStreamingOptimizer.

    Mirrors ``fit_with_stratified_hybrid_streaming`` from ``hybrid_streaming.py``
    for the heterodyne two-component kernel.

    Parameters
    ----------
    stratified_data :
        Flat heterodyne data from ``build_heterodyne_stratified_data``.
    model :
        Configured HeterodyneModel.
    physical_param_names :
        Names of the varying parameters.
    initial_params :
        Initial values for the varying parameters, shape (n_varying,).
    bounds :
        ``(lower, upper)`` arrays, each shape (n_varying,).
    hybrid_config :
        Overrides for HybridStreamingConfig defaults (any subset of keys).
    anti_degeneracy_config :
        Consumed to select the per-angle scaling treatment and L3
        group-variance regularization. Accepted keys:
        ``per_angle_mode`` — ``"auto"`` (THE DEFAULT, including when no config /
        None is supplied — mirrors laminar; resolves to ``"averaged"`` when
        ``n_phi >= constant_scaling_threshold`` (default 3), else to
        ``"individual"``), ``"constant"`` (explicit opt-out: frozen quantile
        scaling, no L3), ``"individual"`` (per-angle optimized scaling + L2
        hierarchical branch).
        ``regularization.{enable, mode, lambda, target_cv}`` —
        configures the L3 adaptive group-variance regularizer on the scaling
        tail (active for the optimized-scaling modes).

    Returns
    -------
    popt : np.ndarray, shape (n_varying,)
        Fitted parameter values.
    pcov : np.ndarray, shape (n_varying, n_varying)
        Parameter covariance matrix (identity fallback on missing).
    info : dict
        Optimizer diagnostics; always contains at least ``nit`` and
        ``"hybrid_streaming_diagnostics"``.
    """
    if AdaptiveHybridStreamingOptimizer is None:
        raise ImportError(
            "AdaptiveHybridStreamingOptimizer not available. "
            "Install nlsq>=0.7.5 to use heterodyne hybrid streaming."
        )

    # ------------------------------------------------------------------
    # Resolve anti-degeneracy config (Task 2)
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.adaptive_regularization import (
        AdaptiveRegularizationConfig,
        AdaptiveRegularizer,
    )

    ad_config: dict[str, Any] = anti_degeneracy_config or {}

    # Mirror laminar EXACTLY (hybrid_streaming.py:550): the default per-angle mode
    # is "auto" even when NO anti-degeneracy config is supplied — there is no
    # special-cased "freeze scaling when unconfigured" branch. 'auto' optimizes
    # scaling ('averaged' at/above the threshold; per-angle 'individual', which
    # also activates the L2 hierarchical branch, below it). Explicit
    # ``per_angle_mode="constant"`` is the opt-out that freezes scaling. The shared
    # resolver requires a resolved token.
    from xpcsjax.optimization.nlsq.per_angle_mode import (
        DEFAULT_CONSTANT_SCALING_THRESHOLD,
        resolve_per_angle_mode,
    )

    requested_mode = ad_config.get("per_angle_mode", "auto")
    threshold = int(ad_config.get("constant_scaling_threshold", DEFAULT_CONSTANT_SCALING_THRESHOLD))
    # n_phi is derived set-wise from phi_flat, matching the builder's
    # deduplication (build_heterodyne_pointwise_model's phi_unique).
    n_phi_resolved = len(set(stratified_data.phi_flat.tolist()))
    mode_actual = resolve_per_angle_mode(requested_mode, n_phi_resolved, threshold)

    logger.info(
        "anti_degeneracy_config: mode_actual=%r (ad_config_provided=%s)",
        mode_actual,
        bool(ad_config),
    )

    # ------------------------------------------------------------------
    # Build model function and data — pass resolved per_angle_mode
    # ------------------------------------------------------------------
    logger.info("Building heterodyne pointwise model for hybrid streaming...")
    model_fn, x_data, y_data, p0, meta = build_heterodyne_pointwise_model(
        stratified_data=stratified_data,
        model=model,
        physical_param_names=physical_param_names,
        per_angle_mode=mode_actual,
    )
    logger.info("Dataset size: %d points", len(y_data))

    # ------------------------------------------------------------------
    # Splice scaling-tail bounds onto physics bounds (Task 2)
    # ------------------------------------------------------------------
    scaling_lower, scaling_upper = meta["scaling_bounds"]
    if len(scaling_lower) > 0:
        if bounds is not None:
            lo, hi = bounds
            # Scaling-first layout: scaling-head bounds PREPEND the physics bounds.
            bounds = (
                np.concatenate([scaling_lower, np.asarray(lo, dtype=np.float64)]),
                np.concatenate([scaling_upper, np.asarray(hi, dtype=np.float64)]),
            )
        else:
            # bounds=None + scaling tail. The ideal would be to bound only the
            # scaling tail and leave physics unbounded via ±inf. But nlsq's
            # AdaptiveHybridStreamingOptimizer "bounds" normalization strategy
            # silently corrupts ±inf bounds to NaN (verified empirically: a real
            # fit with ±inf physics bounds returns all-NaN params, while finite
            # physics bounds converge cleanly). Rather than inject NaN, we leave
            # bounds=None and warn that the scaling tail is unbounded — the
            # caller should pass finite physics bounds to also bound the scaling
            # tail. See heterodyne_core which always supplies finite bounds.
            logger.warning(
                "bounds=None with an optimized scaling tail (n_scaling=%d): the "
                "scaling contrast/offset are left UNBOUNDED because nlsq's hybrid-"
                "streaming optimizer does not accept ±inf bounds (corrupts to NaN). "
                "Pass finite physics bounds to bound the scaling tail too.",
                len(scaling_lower),
            )

    # ------------------------------------------------------------------
    # Build L3 AdaptiveRegularizer + group-variance config (Task 2/6)
    # ------------------------------------------------------------------
    n_scaling = meta["n_scaling"]
    n_phi_meta = meta["n_phi"]
    reg_cfg_dict: dict[str, Any] = ad_config.get("regularization", {})

    # L3 mode-aware group_indices (averaged / individual).
    # Group indices are LOCAL offsets WITHIN the scaling tail (0-based within
    # the tail), translated to full-vector coords (base + ...) when building
    # group_variance_kwargs for the plain optimizer branch.
    # In the hierarchical branch L3 is applied via the loss_fn directly.
    # L3 group indices from the canonical mapper — ONE boundary authority shared
    # with the laminar streaming path. Scaling-first head-local coords: averaged
    # -> [(0,1),(1,2)]; individual -> [(0,n_phi),(n_phi,2*n_phi)]; constant -> [].
    _group_indices: list[tuple[int, int]] | None = (
        ParameterIndexMapper.canonical(
            mode=mode_actual, n_phi=n_phi_resolved, n_physics=len(physical_param_names)
        ).group_indices
        or None  # [] (constant) -> None, matching the existing "no L3 groups" sentinel
    )

    regularization_active = (
        (n_scaling > 0) and (_group_indices is not None) and reg_cfg_dict.get("enable", True)
    )

    adaptive_regularizer: AdaptiveRegularizer | None = None
    group_variance_kwargs: dict[str, Any] = {}

    if regularization_active:
        assert _group_indices is not None  # guarded above
        # Scaling-first layout: the scaling head occupies full-vector indices
        # [0, n_scaling), so the mapper's head-local group indices ARE already the
        # full-vector coordinates — no base offset. compute_regularization_jax (the
        # hierarchical loss) and the plain-branch group-variance config both slice
        # the FULL [scaling | physics] vector, and the scaling head is at offset 0.
        group_indices_full: list[tuple[int, int]] = list(_group_indices)

        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=reg_cfg_dict.get("mode", "relative"),
            lambda_base=float(reg_cfg_dict.get("lambda", 1.0)),
            target_cv=float(reg_cfg_dict.get("target_cv", 0.10)),
            auto_tune_lambda=bool(reg_cfg_dict.get("auto_tune_lambda", True)),
            group_indices=group_indices_full,
        )
        adaptive_regularizer = AdaptiveRegularizer(reg_config, n_phi_meta)
        group_variance_kwargs = {
            "enable_group_variance_regularization": True,
            "group_variance_lambda": float(adaptive_regularizer.lambda_value),
            "group_variance_indices": group_indices_full,
        }
        logger.info(
            "L3 group-variance regularization: lambda=%.4f, group_indices=%s",
            adaptive_regularizer.lambda_value,
            group_variance_kwargs["group_variance_indices"],
        )

    # ------------------------------------------------------------------
    # Build L4 GradientCollapseMonitor + curve_fit callback (Task 3)
    # Mirrors heterodyne_core._build_l4_callback — strictly observational.
    # monitor-on vs monitor-off is objective-identical.
    # ------------------------------------------------------------------
    gm_cfg_dict: dict[str, Any] = ad_config.get("gradient_monitoring", {})
    monitor: GradientCollapseMonitor | None = None
    l4_callback = None

    if gm_cfg_dict.get("enable", True) and meta["n_scaling"] > 0:
        n_phys_params = meta["n_physics_varying"]
        n_scaling_params = meta["n_scaling"]
        # Scaling-first layout: scaling head at [0, n_scaling); physics tail after.
        per_angle_indices = np.arange(n_scaling_params, dtype=np.intp)
        physical_indices = np.arange(
            n_scaling_params, n_scaling_params + n_phys_params, dtype=np.intp
        )
        monitor_config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=float(gm_cfg_dict.get("ratio_threshold", 0.01)),
            consecutive_triggers=int(gm_cfg_dict.get("consecutive_triggers", 5)),
            response_mode=gm_cfg_dict.get("response", "hierarchical"),
            check_interval=1,
        )
        monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=physical_indices,
            per_angle_indices=per_angle_indices,
        )

        def _loss(p: jnp.ndarray) -> jnp.ndarray:
            pred = model_fn(x_data, *p)
            return 0.5 * jnp.sum((jnp.asarray(y_data) - pred) ** 2)

        grad_fn = jax.jit(jax.grad(_loss))
        l4_callback = build_gradient_collapse_callback(monitor, grad_fn)
        logger.info(
            "L4 gradient-collapse monitor enabled (heterodyne streaming): "
            "n_physics=%d, n_scaling=%d",
            n_phys_params,
            n_scaling_params,
        )

    # ------------------------------------------------------------------
    # Build L2 HierarchicalOptimizer
    # Mirrors laminar :821-860.  Only fires for 'individual' (i.e. when not
    # use_constant).  'averaged' / 'constant' already suppress gradient-
    # cancellation degeneracy by having only 2 or 0 per-angle DoF, so
    # hierarchical alternation is not needed there.
    #
    # LAYOUT NOTE: heterodyne's native vector is now scaling-first
    # [scaling(n_scaling) | physics(n_phys)] (Phase-4 unification), which IS
    # HierarchicalOptimizer's expected [per_angle | physics] convention — so the
    # vector is passed through identity (no permutation).
    # ------------------------------------------------------------------
    use_constant = mode_actual in ("averaged", "constant")
    hier_cfg_dict: dict[str, Any] = ad_config.get("hierarchical", {})
    enable_hier = hier_cfg_dict.get("enable", True)
    hierarchical_optimizer = None

    if enable_hier and n_scaling > 0 and not use_constant:
        from xpcsjax.optimization.nlsq.hierarchical import (
            HierarchicalConfig,
            HierarchicalOptimizer,
        )

        hier_config = HierarchicalConfig(
            enable=True,
            max_outer_iterations=int(hier_cfg_dict.get("max_outer_iterations", 5)),
            outer_tolerance=float(hier_cfg_dict.get("outer_tolerance", 1e-6)),
            physical_max_iterations=int(hier_cfg_dict.get("physical_max_iterations", 100)),
            per_angle_max_iterations=int(hier_cfg_dict.get("per_angle_max_iterations", 50)),
        )
        hierarchical_optimizer = HierarchicalOptimizer(
            config=hier_config,
            n_phi=n_phi_meta,
            n_physical=meta["n_physics_varying"],
            # 'individual' only reaches here; per-angle scaling is unreparameterized.
        )
        logger.info(
            "L2 hierarchical optimizer enabled (heterodyne streaming): "
            "mode=%r, n_per_angle=%d, n_physical=%d, max_outer=%d",
            mode_actual,
            hierarchical_optimizer.n_per_angle,
            hierarchical_optimizer.n_physical,
            hier_config.max_outer_iterations,
        )

    # ------------------------------------------------------------------
    # Build HybridStreamingConfig (merge L3 kwargs)
    # ------------------------------------------------------------------
    cfg = _build_hybrid_streaming_config({**(hybrid_config or {}), **group_variance_kwargs})

    # ------------------------------------------------------------------
    # Honor initial_params override — physics-only or full vector
    # ------------------------------------------------------------------
    p0_arr = np.asarray(p0, dtype=np.float64)
    if initial_params is not None:
        ip = np.asarray(initial_params, dtype=np.float64)
        n_phys = meta["n_physics_varying"]
        if ip.shape[0] == n_phys:
            # Physics-only override: splice into the physics TAIL (scaling-first),
            # keeping the scaling head from the builder.
            p0_arr[len(p0_arr) - n_phys :] = ip
        elif ip.shape == p0_arr.shape:
            p0_arr = ip
        else:
            logger.warning(
                "initial_params length %d matches neither physics (%d) nor "
                "full (%d); using model default.",
                len(ip),
                n_phys,
                len(p0_arr),
            )

    # ------------------------------------------------------------------
    # Build sigma from meta (already masked with the keep filter, aligned
    # 1:1 with x_data/y_data by build_heterodyne_pointwise_model).
    # The old code recomputed the mask here with diagonal-only exclusion,
    # which was both redundant and wrong (missed the t=0 boundary).
    # ------------------------------------------------------------------
    sigma: np.ndarray | None = None
    if meta.get("sigma") is not None:
        sigma_sel = np.asarray(meta["sigma"], dtype=np.float64)
        if np.all(sigma_sel == 1.0):
            sigma = None  # uniform — let optimizer use default
        else:
            sigma = sigma_sel

    # ------------------------------------------------------------------
    # Run optimizer
    # ------------------------------------------------------------------
    logger.info("Initializing AdaptiveHybridStreamingOptimizer...")
    optimizer = AdaptiveHybridStreamingOptimizer(cfg)

    if bounds is not None:
        lower, upper = bounds
        bounds_arg: tuple[np.ndarray, np.ndarray] | None = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
    else:
        bounds_arg = None

    logger.info("Running heterodyne hybrid streaming fit (%d params)...", len(p0_arr))

    # ------------------------------------------------------------------
    # Branch: L2 hierarchical (individual) or plain optimizer
    # ------------------------------------------------------------------
    hierarchical_active = False

    if hierarchical_optimizer is not None:
        # L2 hierarchical branch.
        #
        # HierarchicalOptimizer expects layout [per_angle | physics]:
        #   indices 0..n_scaling-1  → per-angle (scaling head)
        #   indices n_scaling..end  → physics
        #
        # Heterodyne's native vector is ALREADY scaling-first [scaling | physics]
        # (Phase-4 unification), which IS the hier-convention layout — so the
        # bridge permutation is identity and is retired. p0_arr / bounds_arg pass
        # straight through; the result is already native.
        #
        # NOTE: the hierarchical loss_fn materialises the full prediction over
        # x_data on every call — mirrors laminar; acceptable because L2 only
        # fires for 'individual' which auto selects at small n_phi.
        assert bounds_arg is not None, (
            "L2 hierarchical requires bounds (bounds_arg is None — pass finite "
            "physics bounds so the scaling tail is also bounded)"
        )

        y_data_jax = jnp.asarray(y_data)
        x_data_jax = x_data  # already numpy; model_fn accepts both

        def _hier_loss(params: np.ndarray) -> float:
            """Loss in the native scaling-first param space [scaling | physics].

            Includes L3 adaptive regularization when active. Honors sigma
            weighting (Finding #4, 2026-07-23) matching the plain-path
            branch's optimizer.fit(sigma=sigma, ...) below.
            """
            params_jax = jnp.asarray(params)
            pred = model_fn(x_data_jax, *params_jax)
            residuals = y_data_jax - pred
            wl = _sigma_weighted_mse(residuals, sigma) * y_data.shape[0]
            if adaptive_regularizer is not None:
                mse = wl / y_data.shape[0]
                wl = wl + adaptive_regularizer.compute_regularization_jax(
                    params_jax, mse, y_data.shape[0]
                )
            return float(wl)

        _hier_counter = [0]

        def _loss_jax(ph: jnp.ndarray) -> jnp.ndarray:
            """Loss in the native scaling-first param space [scaling | physics] (JAX)."""
            pred = model_fn(x_data_jax, *ph)
            residuals = y_data_jax - pred
            wl = _sigma_weighted_mse(residuals, sigma) * y_data.shape[0]
            if adaptive_regularizer is not None:
                mse = wl / y_data.shape[0]
                wl = wl + adaptive_regularizer.compute_regularization_jax(ph, mse, y_data.shape[0])
            return wl

        _value_and_grad = jax.jit(jax.value_and_grad(_loss_jax))

        def _hier_grad(params: np.ndarray) -> np.ndarray:
            """Gradient in the native scaling-first param space."""
            # Single forward+backward pass: value_and_grad gives both the loss
            # and the gradient, so the monitor reuses loss_val instead of a
            # second full _hier_loss forward pass.
            loss_val, g = _value_and_grad(jnp.asarray(params))
            if monitor is not None:
                # `g` and `params` are in the native scaling-first layout, which
                # is the same layout the monitor's physical_indices /
                # per_angle_indices are built against — no permutation needed.
                monitor.check(
                    np.asarray(g),
                    _hier_counter[0],
                    np.asarray(params),
                    float(loss_val),
                )
                _hier_counter[0] += 1
            return np.asarray(g)

        hier_result = hierarchical_optimizer.fit(
            loss_fn=_hier_loss,
            grad_fn=_hier_grad,
            p0=np.asarray(p0_arr, dtype=np.float64),
            bounds=(bounds_arg[0], bounds_arg[1]),
            outer_iteration_callback=None,  # no shear update for heterodyne
        )

        popt = np.asarray(hier_result.x, dtype=np.float64)
        n = len(popt)

        # Real Hessian-based Gauss-Newton covariance (mirrors laminar's
        # BUG-15/H-5 fix at strategies/hybrid_streaming.py:1590-1623, but
        # deliberately diverges from it on singular Hessians -- see below).
        # Uses the pure-JAX scalar loss `_loss_jax` already defined above for
        # the optimizer's own gradient calls. Any failure along this path --
        # jax.hessian raising, a non-finite Hessian, a SINGULAR Hessian, a
        # non-finite covariance, or a negative covariance diagonal -- falls
        # back to an identity placeholder + covariance_is_placeholder=True
        # (single unified except below).
        #
        # A singular Hessian is treated as a full failure, NOT recovered via
        # np.linalg.pinv (unlike laminar's origin). L2 uses bounded L-BFGS-B:
        # a solution resting on a bound is not an interior stationary point,
        # so the unconstrained Hessian there need not be positive-definite --
        # and pinv's Moore-Penrose null-space treatment sets the singular
        # direction's variance to EXACTLY 0.0 (infinite precision) rather
        # than the statistically correct answer (unbounded uncertainty, since
        # that direction is genuinely unidentified by the fit). A negative-
        # diagonal guard alone does not catch this -- 0.0 is not < 0 -- so
        # heterodyne_result_builder would silently clip it to sigma = 0.0,
        # the same confidently-wrong "known" this whole guard exists to
        # prevent, just at the zero boundary instead of the negative one.
        # n_data <= n_params is guarded against division-by-zero the same way
        # laminar's origin does (max(..., 1)) and is not separately detected as a
        # placeholder case, matching the "exact mirror of laminar" decision
        # (spec, "Approaches considered"; see Global Constraints above) --
        # that decision covers the DOF guard only, not the pinv divergence
        # above.
        n_hier_data = len(y_data)
        s2_hier = float(hier_result.fun) / max(n_hier_data - n, 1)
        covariance_is_placeholder = False
        try:
            popt_jax = jnp.asarray(popt)
            H = np.asarray(jax.hessian(_loss_jax)(popt_jax))
            if not np.all(np.isfinite(H)):
                raise ValueError("Hessian contains non-finite entries")
            try:
                pcov = 2.0 * s2_hier * np.linalg.inv(H)
            except np.linalg.LinAlgError as le:
                raise ValueError(
                    "Hessian is singular -- a pseudo-inverse covariance would "
                    "misreport the unidentified (null-space) directions as "
                    "exactly zero variance instead of unbounded uncertainty"
                ) from le
            if not np.all(np.isfinite(pcov)):
                raise ValueError("Covariance contains non-finite entries")
            if np.any(np.diag(pcov) < 0):
                raise ValueError("Covariance has negative diagonal (Hessian not positive-definite)")
        except Exception as e:
            logger.error(
                "Hessian covariance failed in heterodyne L2 path (%s); covariance "
                "is an identity placeholder — reported uncertainties are NOT "
                "meaningful.",
                e,
                exc_info=True,
            )
            pcov = np.eye(n)
            covariance_is_placeholder = True

        info: dict[str, Any] = {
            "success": bool(hier_result.success),
            "nit": int(hier_result.n_outer_iterations),
            "message": hier_result.message,
            # Approximate function-evaluation count: HierarchicalOptimizer does
            # not surface a true inner-iteration tally, so we estimate ~150 inner
            # evaluations per outer step (physical + per-angle alternations),
            # mirroring laminar's same approximation. Diagnostic only — not exact.
            "function_evaluations": hier_result.n_outer_iterations * 150,
            "covariance_is_placeholder": covariance_is_placeholder,
            "hybrid_streaming_diagnostics": {
                "phase_iterations": {"phase1": 0, "phase2": hier_result.n_outer_iterations},
                "warmup_diagnostics": {},
                "gauss_newton_diagnostics": {"final_cost": hier_result.fun},
                "hierarchical_history": hier_result.history,
                "covariance_is_placeholder": covariance_is_placeholder,
            },
        }
        hierarchical_active = True
        logger.info(
            "L2 hierarchical fit complete: success=%s, outer_iters=%d, loss=%.6e",
            hier_result.success,
            hier_result.n_outer_iterations,
            hier_result.fun,
        )

    else:
        # Plain hybrid-streaming path (averaged / constant)
        result: dict[str, Any] = optimizer.fit(
            data_source=(x_data, y_data),
            func=model_fn,
            p0=p0_arr,
            bounds=bounds_arg,
            sigma=sigma,
            callback=l4_callback,
        )

        # ------------------------------------------------------------------
        # Extract popt / pcov / info
        # ------------------------------------------------------------------
        popt = np.asarray(result["x"], dtype=np.float64)
        n = len(popt)
        pcov = np.asarray(result.get("pcov", np.eye(n)), dtype=np.float64)

        # Build info dict: everything except x and pcov
        info = {k: v for k, v in result.items() if k not in ("x", "pcov")}

    # Ensure hybrid_streaming_diagnostics key is always present
    if "hybrid_streaming_diagnostics" not in info:
        info["hybrid_streaming_diagnostics"] = {k: info[k] for k in ("nit", "success") if k in info}

    # Thread data-point count for reduced-chi dof (Finding 3)
    info["n_data_points"] = meta["n_data_points"]

    # ------------------------------------------------------------------
    # SSR + frozen baseline (Task 2)
    # ------------------------------------------------------------------
    pred = np.asarray(model_fn(x_data, *popt))
    info["ssr"] = float(np.sum((y_data - pred) ** 2))

    if meta["n_scaling"] > 0:
        n_phys = meta["n_physics_varying"]
        n_scal = meta["n_scaling"]
        # Build the frozen scaling HEAD using the per-mode initial estimates so
        # we compare optimised SSR against the unoptimised quantile-baseline.
        # The head layout (scaling-first) must match what model_fn expects:
        #   averaged   : [mean_contrast, mean_offset]          (2 params)
        #   individual : [contrast_arr..., offset_arr...]       (2*n_phi params)
        _contrast_arr = np.asarray(meta["contrast_arr"])
        _offset_arr = np.asarray(meta["offset_arr"])
        _mode = meta["per_angle_mode"]
        if _mode == "averaged":
            frozen_head = [float(np.mean(_contrast_arr)), float(np.mean(_offset_arr))]
        elif _mode == "individual":
            frozen_head = _contrast_arr.tolist() + _offset_arr.tolist()
        else:
            # constant or unexpected mode: no scaling head to freeze
            frozen_head = []
        # Scaling-first: frozen head PREPENDS the optimised physics tail.
        frozen = frozen_head + list(popt[n_scal:])
        if len(frozen) == len(popt):
            pred0 = np.asarray(model_fn(x_data, *frozen))
            info["ssr_frozen_baseline"] = float(np.sum((y_data - pred0) ** 2))
        else:
            # Mismatch guard: fall back to current SSR (no meaningful baseline)
            info["ssr_frozen_baseline"] = info["ssr"]
    else:
        info["ssr_frozen_baseline"] = info["ssr"]

    # ------------------------------------------------------------------
    # Anti-degeneracy diagnostics — symmetric contract via shared assembler
    # (Task 4/6). Emits the same top-level keys as heterodyne_core and the
    # laminar in-memory paths: hierarchical_active / regularization_active /
    # shear_weighting / gradient_monitor (when present) + layer_detail kwargs.
    # L5 (shear weighting) is laminar_flow-only; streaming heterodyne reports
    # the canonical "laminar_flow_inactive" sentinel.
    # L2 (hierarchical) is wired for individual.
    # ------------------------------------------------------------------
    gm_block: dict | None = None
    if monitor is not None:
        # Build canonical L4 block; falls back to post_solve_fallback mechanism
        # when the callback never fired (zero observations).
        gm_block = gradient_monitor_diagnostics(monitor)
        if gm_block["mechanism"] == "post_solve_fallback":
            # Compute post-solve covariance condition as fallback indicator.
            # info["covariance_is_placeholder"] is DYNAMIC: the hierarchical
            # branch now computes a real Hessian-based covariance and only
            # falls back to the identity placeholder when that fails (see the
            # covariance block above), and the plain streaming branch reports
            # whatever the optimizer produced. When the flag is True, pcov is
            # the identity placeholder and cond(I)=1.0 would masquerade as a
            # real, well-conditioned covariance -- report NaN there; only
            # compute the real condition number on a genuine pcov.
            is_placeholder = bool(info.get("covariance_is_placeholder", False))
            if (not is_placeholder) and pcov.ndim == 2 and pcov.shape[0] > 0:
                pcov_cond = float(np.linalg.cond(pcov))
            else:
                pcov_cond = float("nan")
            gm_block["post_solve_cov_condition"] = pcov_cond
        logger.info(
            "L4 gradient-collapse monitor (heterodyne streaming): "
            "mechanism=%s, n_observations=%s, max_gradient_ratio=%.3g, "
            "collapse_detected=%s.",
            gm_block["mechanism"],
            gm_block.get("n_observations"),
            gm_block["max_gradient_ratio"],
            gm_block["collapse_detected"],
        )

    # Thread the frozen per-angle quantile scaling through for constant mode so
    # heterodyne_views.reconstruct_per_angle_scaling(mode="constant") can read
    # it from this path's nlsq_diagnostics, matching heterodyne_constant_mode.py's
    # key names (meta["contrast_arr"]/["offset_arr"] ARE the frozen values baked
    # into the model for constant mode; see build_heterodyne_pointwise_model).
    _extra_constant_scaling: dict[str, Any] = (
        {
            "contrast_per_angle_fixed": np.asarray(meta["contrast_arr"]),
            "offset_per_angle_fixed": np.asarray(meta["offset_arr"]),
        }
        if meta["per_angle_mode"] == "constant"
        else {}
    )
    info["anti_degeneracy"] = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=hierarchical_active,
        regularization_active=bool(regularization_active),
        shear_weighting="laminar_flow_inactive",
        gradient_monitor=gm_block,
        per_angle_mode=meta["per_angle_mode"],
        n_optimized=int(meta["n_scaling"]),
        # Thread the L2 branch's real/placeholder covariance flag into the
        # public anti_degeneracy block, mirroring heterodyne_stratified_ls.py's
        # established pattern for the same flag. Read from `info` (not the
        # local `covariance_is_placeholder` var, which only the L2 branch
        # sets) so the plain-path branch's absence of the key degrades safely
        # to False instead of raising NameError.
        covariance_is_placeholder=bool(info.get("covariance_is_placeholder", False)),
        **_extra_constant_scaling,
    )

    return popt, pcov, info
