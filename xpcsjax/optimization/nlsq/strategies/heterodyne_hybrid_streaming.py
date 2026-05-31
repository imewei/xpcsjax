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

from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
    build_gradient_collapse_callback,
    gradient_monitor_diagnostics,
)
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import HeterodyneStratifiedData

logger = get_logger(__name__)

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
    """searchsorted + boundary clip, warning on out-of-grid points.

    Mirrors homodyne ``_bin_to_grid`` in ``hybrid_streaming.py`` exactly.
    An unguarded clip silently routes data lying outside the fitted grid to
    the boundary bin — a data-integrity violation. We clip (to stay in-bounds)
    but surface how many points were affected so misaligned data/config is
    not silent.
    """
    raw = np.searchsorted(grid, values)
    n_oob = int(np.sum(raw >= len(grid)))
    if n_oob > 0:
        logger.warning(
            "%d data point(s) lie beyond the %s grid; clipped to the boundary "
            "bin. Check data/config grid alignment.",
            n_oob,
            axis_name,
        )
    return np.clip(raw, 0, len(grid) - 1)


def build_heterodyne_pointwise_model(
    *,
    stratified_data: HeterodyneStratifiedData,
    model: HeterodyneModel,
    physical_param_names: list[str],
    per_angle_mode: str = "fixed_constant",
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
        Scaling treatment for per-angle contrast/offset.
        ``"fixed_constant"`` (default) — freeze quantile-estimated scaling inside
        the JIT closure (existing behaviour, backward-compatible).
        ``"auto_averaged"`` — append 2 optimized scaling scalars (mean contrast,
        mean offset) to ``p0``; the JIT closure reads them as uniform across all
        angles.  Later modes (``"individual"``, ``"fourier"``) are reserved for
        future tasks.

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

    # Build phi_value → sorted_index mapping for the REINDEX step
    # (handles any edge case where the raw function iterates in a different
    # order; in practice they match, but explicit reindex is the safe path).
    chunk_phi_order: list[float] = []
    if hasattr(stratified_data, "chunks") and stratified_data.chunks:
        for chunk in stratified_data.chunks:
            # Each chunk corresponds to one angle; take representative phi
            chunk_phi_order.append(float(chunk.phi[0]))
    else:
        # Flat format: phi_flat has one value per (t1,t2) pair per angle.
        # Recover the per-angle phi by reading the first element of each slab.
        seen: dict[float, int] = {}
        for phi_val in stratified_data.phi_flat.tolist():
            if phi_val not in seen:
                seen[phi_val] = len(seen)
        chunk_phi_order = list(seen.keys())

    if len(chunk_phi_order) == n_phi:
        # Build mapping: raw_slot → sorted_index
        phi_to_sorted = {float(p): int(i) for i, p in enumerate(phi_unique.tolist())}
        reindex = np.array([phi_to_sorted[float(p)] for p in chunk_phi_order], dtype=np.int64)
        contrast_arr = np.empty(n_phi, dtype=np.float64)
        offset_arr = np.empty(n_phi, dtype=np.float64)
        contrast_arr[reindex] = np.asarray(contrast_raw, dtype=np.float64)
        offset_arr[reindex] = np.asarray(offset_raw, dtype=np.float64)
    else:
        # Fallback: trust the raw order (matches phi_unique sorted order)
        contrast_arr = np.asarray(contrast_raw, dtype=np.float64)
        offset_arr = np.asarray(offset_raw, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. Initial parameter vector (varying physics; optionally + scaling tail)
    # ------------------------------------------------------------------
    p0: list[float] = [float(v) for v in model.param_manager.get_initial_values()]
    n_physics_varying = len(p0)

    if per_angle_mode == "auto_averaged":
        contrast0 = float(np.mean(contrast_arr))
        offset0 = float(np.mean(offset_arr))
        n_scaling = 2
        p0 = [*p0, contrast0, offset0]
    elif per_angle_mode == "fixed_constant":
        n_scaling = 0
    else:
        raise NotImplementedError(
            f"per_angle_mode={per_angle_mode!r} not yet wired in streaming "
            "(individual/fourier modes are not yet supported in streaming)."
        )

    # ------------------------------------------------------------------
    # 7. Prepare JAX-side fixed tensors for the closure
    # ------------------------------------------------------------------
    fixed_full_jax = jnp.asarray(model.param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(
        list(model.param_manager.varying_indices), dtype=jnp.int32
    )
    t_jax = jnp.asarray(t_unique, dtype=jnp.float64)
    q_val = float(model.q)
    dt_val = float(model.dt)
    phi_unique_jax = jnp.asarray(phi_unique, dtype=jnp.float64)
    contrast_jax = jnp.asarray(contrast_arr, dtype=jnp.float64)
    offset_jax = jnp.asarray(offset_arr, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # 8. Build JIT-compiled pointwise model function
    # ------------------------------------------------------------------
    @jax.jit
    def model_fn(x_batch: jnp.ndarray, *params_tuple: jnp.ndarray) -> jnp.ndarray:
        """Pointwise heterodyne model: params = [physics_varying | scaling_tail]."""
        x_batch_2d = jnp.atleast_2d(x_batch)
        params_all = jnp.stack(params_tuple)

        # Reconstruct full physics parameter vector from fixed + varying
        physics = params_all[:n_physics_varying]
        full = fixed_full_jax.at[varying_indices_jax].set(physics)

        # Resolve per-angle scaling: optimized tail or frozen quantile estimates
        if n_scaling == 2:
            contrasts = jnp.full((phi_unique_jax.shape[0],), params_all[n_physics_varying])
            offsets = jnp.full((phi_unique_jax.shape[0],), params_all[n_physics_varying + 1])
        else:
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
    if n_scaling == 2:
        c_lo, c_hi = 0.01, max(2.0 * contrast0, 1.0)
        # Offset is a DC baseline that can be negative; use a symmetric bound
        # centered on offset0 so the lower bound permits negative offsets.
        o_lo = offset0 - max(abs(offset0), 1.0)
        o_hi = offset0 + max(abs(offset0), 1.0)
        scaling_lower = np.array([c_lo, o_lo], dtype=np.float64)
        scaling_upper = np.array([c_hi, o_hi], dtype=np.float64)
    else:
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
        # Scaling-mode metadata (Task 1+)
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
            "Install nlsq>=0.6.10 to use heterodyne hybrid streaming."
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
        ``per_angle_mode`` — ``"auto"`` resolves to auto_averaged (2 optimized
        scaling scalars), ``"constant"`` resolves to fixed_constant (frozen
        quantile scaling); ``"fourier"``/``"individual"`` are not yet supported
        in streaming. ``regularization.{enable, mode, lambda, target_cv}`` —
        configures the L3 adaptive group-variance regularizer on the scaling
        tail. An empty/absent dict falls back to fixed_constant (no L3).

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
            "Install nlsq>=0.6.10 to use heterodyne hybrid streaming."
        )

    # ------------------------------------------------------------------
    # Resolve anti-degeneracy config (Task 2)
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.adaptive_regularization import (
        AdaptiveRegularizationConfig,
        AdaptiveRegularizer,
    )

    ad_config: dict[str, Any] = anti_degeneracy_config or {}

    if not ad_config:
        # Backward-compatible default: freeze scaling inside the JIT closure
        mode_actual = "fixed_constant"
    else:
        requested_mode = ad_config.get("per_angle_mode", "auto")
        if requested_mode == "auto":
            # auto always optimizes averaged scaling for now; Task 5 will route
            # sub-threshold n_phi to 'individual' once that mode is wired.
            mode_actual = "auto_averaged"
        elif requested_mode == "constant":
            mode_actual = "fixed_constant"
        else:
            mode_actual = requested_mode  # explicit 'fourier'/'individual' -> Task 5

    logger.info(
        "anti_degeneracy_config: mode_actual=%r (ad_config_provided=%s)",
        mode_actual, bool(ad_config),
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
            bounds = (
                np.concatenate([np.asarray(lo, dtype=np.float64), scaling_lower]),
                np.concatenate([np.asarray(hi, dtype=np.float64), scaling_upper]),
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
    # Build L3 AdaptiveRegularizer + group-variance config (Task 2)
    # ------------------------------------------------------------------
    n_scaling = meta["n_scaling"]
    reg_cfg_dict: dict[str, Any] = ad_config.get("regularization", {})
    # L3 only makes sense when there are actual scaling groups to regularize.
    # auto_averaged supplies the (contrast, offset) group pairs; other modes do
    # not (yet) — Task 5 will extend this to fourier/individual.
    regularization_active = (
        (n_scaling > 0)
        and (mode_actual == "auto_averaged")
        and reg_cfg_dict.get("enable", True)
    )

    adaptive_regularizer: AdaptiveRegularizer | None = None
    group_variance_kwargs: dict[str, Any] = {}

    if regularization_active:
        # Local offsets WITHIN the scaling tail; translated to full-vector coords (base+...) below.
        group_indices: list[tuple[int, int]] = [(0, 1), (1, 2)]

        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=reg_cfg_dict.get("mode", "relative"),
            lambda_base=float(reg_cfg_dict.get("lambda", 1.0)),
            target_cv=float(reg_cfg_dict.get("target_cv", 0.10)),
            group_indices=group_indices if group_indices else None,
        )
        adaptive_regularizer = AdaptiveRegularizer(reg_config, meta["n_phi"])
        base = meta["n_physics_varying"]
        group_variance_kwargs = {
            "enable_group_variance_regularization": True,
            "group_variance_lambda": float(adaptive_regularizer.lambda_value),
            "group_variance_indices": [(base + a, base + b) for (a, b) in (group_indices or [])],
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
        base_idx = meta["n_physics_varying"]
        n_scaling_params = meta["n_scaling"]
        physical_indices = np.arange(base_idx, dtype=np.intp)
        per_angle_indices = np.arange(base_idx, base_idx + n_scaling_params, dtype=np.intp)
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
            base_idx, n_scaling_params,
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
            # Physics-only override: splice in, keep scaling tail from builder
            p0_arr[:n_phys] = ip
        elif ip.shape == p0_arr.shape:
            p0_arr = ip
        else:
            logger.warning(
                "initial_params length %d matches neither physics (%d) nor "
                "full (%d); using model default.",
                len(ip), n_phys, len(p0_arr),
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
    info: dict[str, Any] = {k: v for k, v in result.items() if k not in ("x", "pcov")}

    # Ensure hybrid_streaming_diagnostics key is always present
    if "hybrid_streaming_diagnostics" not in info:
        info["hybrid_streaming_diagnostics"] = {
            k: info[k] for k in ("nit", "success") if k in info
        }

    # Thread data-point count for reduced-chi dof (Finding 3)
    info["n_data_points"] = meta["n_data_points"]

    # ------------------------------------------------------------------
    # SSR + frozen baseline (Task 2)
    # ------------------------------------------------------------------
    pred = np.asarray(model_fn(x_data, *popt))
    info["ssr"] = float(np.sum((y_data - pred) ** 2))

    if meta["n_scaling"] > 0:
        n_phys = meta["n_physics_varying"]
        frozen = list(popt[:n_phys]) + [
            float(np.mean(meta["contrast_arr"])),
            float(np.mean(meta["offset_arr"])),
        ]
        pred0 = np.asarray(model_fn(x_data, *frozen))
        info["ssr_frozen_baseline"] = float(np.sum((y_data - pred0) ** 2))
    else:
        info["ssr_frozen_baseline"] = info["ssr"]

    # ------------------------------------------------------------------
    # Seed anti_degeneracy diagnostics dict (Task 2; L4 block added Task 3)
    # ------------------------------------------------------------------
    info["anti_degeneracy"] = {"per_angle_mode": meta["per_angle_mode"]}

    if monitor is not None:
        # Build canonical L4 block; falls back to post_solve_fallback mechanism
        # when the callback never fired (zero observations).
        gm_block = gradient_monitor_diagnostics(monitor)
        if gm_block["mechanism"] == "post_solve_fallback":
            # Compute post-solve covariance condition as fallback indicator
            pcov_cond = float(np.linalg.cond(pcov)) if pcov.ndim == 2 and pcov.shape[0] > 0 else float("nan")
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
        info["anti_degeneracy"]["gradient_monitor"] = gm_block

    return popt, pcov, info
