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
    """Bin values to grid indices via ``searchsorted`` + boundary clip.

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
    fourier_order: int = 2,
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
        angles.
        ``"individual"`` — append ``2 * n_phi`` optimized scaling params (per-angle
        contrast then per-angle offset); the JIT closure reads them directly.
        ``"fourier"`` — append ``2 * (2*fourier_order + 1)`` Fourier coefficient
        params; the JIT closure expands them to per-angle via the
        :class:`~xpcsjax.optimization.nlsq.fourier_reparam.FourierReparameterizer`
        JIT-safe transform.
    fourier_order :
        Number of Fourier harmonics.  Only used when ``per_angle_mode="fourier"``.
        Default 2 gives ``2*(2*2+1)=10`` total scaling coefficients.

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

    # Fourier reparameterizer — built once here, closed over in model_fn below.
    fourier_reparam: Any = None  # set for "fourier" mode only

    # Effective fourier mode — authoritative only after the fourier branch runs.
    # For non-fourier modes it is the mode itself. For fourier it reports whether
    # FourierReparameterizer actually used the Fourier basis or silently fell back
    # to independent (per-angle) scaling because n_phi was too small.
    fourier_effective_mode: str = per_angle_mode

    if per_angle_mode == "auto_averaged":
        contrast0 = float(np.mean(contrast_arr))
        offset0 = float(np.mean(offset_arr))
        n_scaling = 2
        p0 = [*p0, contrast0, offset0]
    elif per_angle_mode == "individual":
        # Tail = [contrast_per_angle | offset_per_angle], length 2*n_phi.
        n_scaling = 2 * n_phi
        p0 = [*p0, *contrast_arr.tolist(), *offset_arr.tolist()]
    elif per_angle_mode == "fourier":
        # Import the real Fourier API (same import used by heterodyne_core).
        from xpcsjax.optimization.nlsq.fourier_reparam import (
            FourierReparamConfig,
            FourierReparameterizer,
        )

        fourier_config = FourierReparamConfig(
            mode="fourier",
            fourier_order=fourier_order,
        )
        fourier_reparam = FourierReparameterizer(phi_unique, fourier_config)
        # Inverse-transform initial per-angle estimates into Fourier coefficient space.
        init_coeffs = fourier_reparam.per_angle_to_fourier(contrast_arr, offset_arr)
        # NOTE: meta["n_scaling"] (set from fourier_reparam.n_coeffs) is the
        # AUTHORITATIVE scaling-tail length — downstream code must read it, never
        # recompute 2*(2K+1) from per_angle_mode+fourier_order. When n_phi is too
        # small for the requested fourier_order, FourierReparameterizer silently
        # falls back to independent mode (use_fourier=False), making n_coeffs =
        # 2*n_phi, not 2*(2K+1).
        n_scaling = fourier_reparam.n_coeffs  # = 2*(2*fourier_order+1) when use_fourier=True
        p0 = [*p0, *np.asarray(init_coeffs, dtype=np.float64).tolist()]
        # Expose the EFFECTIVE mode so consumers (Task 6) can detect the fallback.
        fourier_effective_mode = "fourier" if fourier_reparam.use_fourier else "individual"
        if not fourier_reparam.use_fourier:
            logger.warning(
                "per_angle_mode='fourier' requested with fourier_order=%d but "
                "n_phi=%d is too small (need n_phi >= 1+2*order); "
                "FourierReparameterizer fell back to independent per-angle scaling "
                "(n_scaling=%d).",
                fourier_order,
                n_phi,
                n_scaling,
            )
    elif per_angle_mode == "fixed_constant":
        n_scaling = 0
    else:
        raise NotImplementedError(
            f"per_angle_mode={per_angle_mode!r} not supported in build_heterodyne_pointwise_model."
        )

    # ------------------------------------------------------------------
    # 7. Prepare JAX-side fixed tensors for the closure
    # ------------------------------------------------------------------
    fixed_full_jax = jnp.asarray(model.param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(list(model.param_manager.varying_indices), dtype=jnp.int32)
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
    # per_angle_mode and fourier_reparam are Python-level constants (not JAX
    # tracers), so the if/elif branches below are static at JIT trace time.
    _per_angle_mode = per_angle_mode
    _fourier_reparam = fourier_reparam  # None unless per_angle_mode=="fourier"
    _n_phi_local = len(phi_unique)  # compile-time constant for individual slice

    @jax.jit
    def model_fn(x_batch: jnp.ndarray, *params_tuple: jnp.ndarray) -> jnp.ndarray:
        """Pointwise heterodyne model: params = [physics_varying | scaling_tail]."""
        x_batch_2d = jnp.atleast_2d(x_batch)
        params_all = jnp.stack(params_tuple)

        # Reconstruct full physics parameter vector from fixed + varying
        physics = params_all[:n_physics_varying]
        full = fixed_full_jax.at[varying_indices_jax].set(physics)

        # Resolve per-angle scaling — branch is static (compile-time constant).
        if _per_angle_mode == "auto_averaged":
            contrasts = jnp.full((_n_phi_local,), params_all[n_physics_varying])
            offsets = jnp.full((_n_phi_local,), params_all[n_physics_varying + 1])
        elif _per_angle_mode == "individual":
            tail = params_all[n_physics_varying:]
            contrasts = tail[:_n_phi_local]
            offsets = tail[_n_phi_local:]
        elif _per_angle_mode == "fourier":
            # _fourier_reparam.fourier_to_per_angle_jax is JIT-safe (uses jnp).
            tail = params_all[n_physics_varying:]
            contrasts, offsets = _fourier_reparam.fourier_to_per_angle_jax(tail)
        else:  # fixed_constant
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
    if per_angle_mode == "auto_averaged":
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
    elif per_angle_mode == "fourier":
        # Use FourierReparameterizer.get_bounds() which returns bounds for the
        # Fourier coefficient vector in the same layout as per_angle_to_fourier.
        # fourier_reparam was built in the fourier branch above (this branch only
        # runs when per_angle_mode == "fourier").
        scaling_lower, scaling_upper = fourier_reparam.get_bounds()
        scaling_lower = np.asarray(scaling_lower, dtype=np.float64)
        scaling_upper = np.asarray(scaling_upper, dtype=np.float64)
    else:  # fixed_constant
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
        # Effective fourier mode: "fourier" if the Fourier basis was actually
        # used, "individual" if FourierReparameterizer fell back (n_phi too small
        # for the requested order); for non-fourier modes it is the mode itself.
        # Consumers detect the silent fallback via meta.get("fourier_effective_mode").
        "fourier_effective_mode": fourier_effective_mode,
        "n_scaling": n_scaling,
        "n_physics_varying": n_physics_varying,
        "scaling_bounds": (scaling_lower, scaling_upper),
        # FourierReparameterizer object for "fourier" mode (consumed by the
        # HierarchicalOptimizer L2 wiring); None for every non-fourier mode so
        # `meta.get("fourier")` is always safe.
        "fourier": fourier_reparam,
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
        ``per_angle_mode`` — ``"auto"`` (THE DEFAULT, including when no config /
        None is supplied — mirrors laminar; resolves to ``"auto_averaged"`` when
        ``n_phi >= constant_scaling_threshold`` (default 3), else to
        ``"individual"``), ``"fixed_constant"`` (explicit opt-out: frozen quantile
        scaling, no L3), ``"individual"`` (per-angle optimized scaling + L2
        hierarchical branch), ``"fourier"`` (Fourier-coefficient scaling + L2
        hierarchical; falls back to independent when n_phi < 1+2K).
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

    # Mirror laminar EXACTLY (hybrid_streaming.py:550): the default per-angle mode
    # is "auto" even when NO anti-degeneracy config is supplied — there is no
    # special-cased "freeze scaling when unconfigured" branch. 'auto' optimizes
    # scaling (AVERAGED at/above the threshold; per-angle 'individual', which also
    # activates the L2 hierarchical branch, below it). Explicit
    # ``per_angle_mode="constant"`` is the opt-out that freezes scaling.
    requested_mode = ad_config.get("per_angle_mode", "auto")
    if requested_mode == "auto":
        # n_phi is derived set-wise from phi_flat, matching the builder's
        # deduplication (build_heterodyne_pointwise_model's phi_unique).
        threshold = int(ad_config.get("constant_scaling_threshold", 3))
        n_phi_resolved = len(set(stratified_data.phi_flat.tolist()))
        mode_actual = "auto_averaged" if n_phi_resolved >= threshold else "individual"
    elif requested_mode == "constant":
        mode_actual = "fixed_constant"
    else:
        mode_actual = requested_mode  # explicit 'individual' or 'fourier'

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
    # Build L3 AdaptiveRegularizer + group-variance config (Task 2/6)
    # ------------------------------------------------------------------
    n_scaling = meta["n_scaling"]
    n_phi_meta = meta["n_phi"]
    reg_cfg_dict: dict[str, Any] = ad_config.get("regularization", {})

    # L3 mode-aware group_indices (Task 6: extend to individual/fourier).
    # Group indices are LOCAL offsets WITHIN the scaling tail (0-based within
    # the tail), translated to full-vector coords (base + ...) when building
    # group_variance_kwargs for the plain optimizer branch.
    # In the hierarchical branch L3 is applied via the loss_fn directly.
    _group_indices: list[tuple[int, int]] | None = None
    if mode_actual == "auto_averaged":
        _group_indices = [(0, 1), (1, 2)]
    elif mode_actual == "individual":
        # individual: tail = [contrast_0..contrast_{n_phi-1} | offset_0..offset_{n_phi-1}]
        _group_indices = [(0, n_phi_meta), (n_phi_meta, 2 * n_phi_meta)]
    elif mode_actual == "fourier":
        # fourier: tail = [contrast_coeffs | offset_coeffs], each of length n_scaling//2
        # (n_scaling is always even: 2 * n_coeffs_per_param)
        c = n_scaling // 2
        _group_indices = [(0, c), (c, 2 * c)]
    # fixed_constant: _group_indices stays None (nothing to regularize)

    regularization_active = (
        (n_scaling > 0) and (_group_indices is not None) and reg_cfg_dict.get("enable", True)
    )

    adaptive_regularizer: AdaptiveRegularizer | None = None
    group_variance_kwargs: dict[str, Any] = {}

    if regularization_active:
        assert _group_indices is not None  # guarded above
        # Translate tail-LOCAL group indices to FULL-vector coordinates ONCE.
        # compute_regularization_jax (used by the hierarchical loss) and the
        # plain-branch group-variance config both slice the FULL
        # [physics(n_phys) | scaling] vector, so the regularizer must carry
        # full-vector indices (base + offset), not tail-local ones — otherwise
        # it would regularize the first n_phi PHYSICS params instead of the
        # scaling tail.
        base = meta["n_physics_varying"]
        group_indices_full: list[tuple[int, int]] = [
            (base + a, base + b) for (a, b) in _group_indices
        ]

        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=reg_cfg_dict.get("mode", "relative"),
            lambda_base=float(reg_cfg_dict.get("lambda", 1.0)),
            target_cv=float(reg_cfg_dict.get("target_cv", 0.10)),
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
            base_idx,
            n_scaling_params,
        )

    # ------------------------------------------------------------------
    # Build L2 HierarchicalOptimizer (Task 6)
    # Mirrors laminar :821-860.  Only fires for individual/fourier (i.e.
    # when not use_constant).  auto_averaged / fixed_constant already
    # suppress gradient-cancellation degeneracy by having only 2 or 0
    # per-angle DoF, so hierarchical alternation is not needed there.
    #
    # LAYOUT NOTE: heterodyne's native vector is [physics(n_phys) | scaling(n_scaling)].
    # HierarchicalOptimizer expects [per_angle(n_scaling) | physics(n_phys)] (per-angle
    # first, physics last — same as the laminar convention).  We permute the
    # vector before passing it to the optimizer and un-permute the result.
    # ------------------------------------------------------------------
    use_constant = mode_actual in ("auto_averaged", "fixed_constant")
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
            fourier_reparameterizer=meta.get("fourier"),  # None for individual
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
            # Physics-only override: splice in, keep scaling tail from builder
            p0_arr[:n_phys] = ip
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
    # Branch: L2 hierarchical (individual/fourier) or plain optimizer
    # ------------------------------------------------------------------
    hierarchical_active = False

    if hierarchical_optimizer is not None:
        # L2 hierarchical branch (Task 6).
        #
        # HierarchicalOptimizer expects layout [per_angle | physics]:
        #   indices 0..n_scaling-1  → per-angle (scaling tail)
        #   indices n_scaling..end  → physics
        #
        # Heterodyne's native layout is the reverse: [physics | scaling].
        # We permute to hier-convention, run the solver, then un-permute.
        #
        # NOTE: the hierarchical loss_fn materialises the full prediction over
        # x_data on every call — mirrors laminar; acceptable because L2 only
        # fires for individual/fourier which auto selects at small n_phi.
        assert bounds_arg is not None, (
            "L2 hierarchical requires bounds (bounds_arg is None — pass finite "
            "physics bounds so the scaling tail is also bounded)"
        )
        n_phys_h = meta["n_physics_varying"]
        n_scal_h = meta["n_scaling"]

        # Permutation: heterodyne [physics | scaling] -> hier [scaling | physics]
        perm = np.concatenate(
            [
                np.arange(n_phys_h, n_phys_h + n_scal_h, dtype=np.intp),  # scaling tail first
                np.arange(n_phys_h, dtype=np.intp),  # then physics
            ]
        )
        unperm = np.empty_like(perm)
        unperm[perm] = np.arange(len(perm), dtype=np.intp)

        p0_hier = p0_arr[perm]
        bounds_hier = (bounds_arg[0][perm], bounds_arg[1][perm])

        y_data_jax = jnp.asarray(y_data)
        x_data_jax = x_data  # already numpy; model_fn accepts both

        def _hier_loss(params_hier: np.ndarray) -> float:
            """Loss in hier-convention param space [scaling | physics].

            Permutes back to heterodyne convention [physics | scaling] before
            calling model_fn so the closure is consistent with x_data/y_data.
            Includes L3 adaptive regularization when active.
            """
            params_native = jnp.asarray(params_hier)[unperm]
            pred = model_fn(x_data_jax, *params_native)
            residuals = y_data_jax - pred
            wl = jnp.mean(residuals**2) * y_data.shape[0]
            if adaptive_regularizer is not None:
                mse = wl / y_data.shape[0]
                wl = wl + adaptive_regularizer.compute_regularization_jax(
                    params_native, mse, y_data.shape[0]
                )
            return float(wl)

        _hier_counter = [0]

        def _loss_jax(ph: jnp.ndarray) -> jnp.ndarray:
            """Loss in hier-convention param space [scaling | physics] (JAX)."""
            params_native = ph[unperm]
            pred = model_fn(x_data_jax, *params_native)
            residuals = y_data_jax - pred
            wl = jnp.mean(residuals**2) * y_data.shape[0]
            if adaptive_regularizer is not None:
                mse = wl / y_data.shape[0]
                wl = wl + adaptive_regularizer.compute_regularization_jax(
                    params_native, mse, y_data.shape[0]
                )
            return wl

        _value_and_grad = jax.jit(jax.value_and_grad(_loss_jax))

        def _hier_grad(params_hier: np.ndarray) -> np.ndarray:
            """Gradient in hier-convention param space."""
            # Single forward+backward pass: value_and_grad gives both the loss
            # and the gradient, so the monitor reuses loss_val instead of a
            # second full _hier_loss forward pass.
            loss_val, g = _value_and_grad(jnp.asarray(params_hier))
            if monitor is not None:
                # `g` and `params_hier` are in HIER layout [scaling | physics],
                # but the monitor's physical_indices/per_angle_indices are NATIVE
                # layout [physics | scaling]. Un-permute both before check() so the
                # group slices line up with the indices.
                g_native = np.asarray(g)[unperm]
                params_native_arr = np.asarray(params_hier)[unperm]
                monitor.check(
                    g_native,
                    _hier_counter[0],
                    params_native_arr,
                    float(loss_val),
                )
                _hier_counter[0] += 1
            return np.asarray(g)

        hier_result = hierarchical_optimizer.fit(
            loss_fn=_hier_loss,
            grad_fn=_hier_grad,
            p0=np.asarray(p0_hier, dtype=np.float64),
            bounds=bounds_hier,
            outer_iteration_callback=None,  # no shear update for heterodyne
        )

        # Un-permute result back to heterodyne convention [physics | scaling]
        x_hier_native = np.asarray(hier_result.x, dtype=np.float64)[unperm]
        n = len(x_hier_native)
        pcov = np.eye(n)  # Hessian covariance is optional; identity placeholder
        popt = x_hier_native
        info: dict[str, Any] = {
            "success": bool(hier_result.success),
            "nit": int(hier_result.n_outer_iterations),
            "message": hier_result.message,
            # Approximate function-evaluation count: HierarchicalOptimizer does
            # not surface a true inner-iteration tally, so we estimate ~150 inner
            # evaluations per outer step (physical + per-angle alternations),
            # mirroring laminar's same approximation. Diagnostic only — not exact.
            "function_evaluations": hier_result.n_outer_iterations * 150,
            "covariance_is_placeholder": True,
            "hybrid_streaming_diagnostics": {
                "phase_iterations": {"phase1": 0, "phase2": hier_result.n_outer_iterations},
                "warmup_diagnostics": {},
                "gauss_newton_diagnostics": {"final_cost": hier_result.fun},
                "hierarchical_history": hier_result.history,
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
        # Plain hybrid-streaming path (auto_averaged / fixed_constant)
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
        # Build the frozen scaling tail using the per-mode initial estimates so
        # we compare optimised SSR against the unoptimised quantile-baseline.
        # The tail layout must match what model_fn expects for this mode:
        #   auto_averaged : [mean_contrast, mean_offset]          (2 params)
        #   individual    : [contrast_arr..., offset_arr...]       (2*n_phi params)
        #   fourier       : Fourier coefficients from reparam      (n_scaling params)
        # For fourier, re-project the per-angle quantile estimates into Fourier
        # space via the same FourierReparameterizer that was used at build time.
        _contrast_arr = np.asarray(meta["contrast_arr"])
        _offset_arr = np.asarray(meta["offset_arr"])
        _mode = meta["per_angle_mode"]
        if _mode == "auto_averaged":
            frozen_tail = [float(np.mean(_contrast_arr)), float(np.mean(_offset_arr))]
        elif _mode == "individual":
            frozen_tail = _contrast_arr.tolist() + _offset_arr.tolist()
        elif _mode == "fourier" and meta.get("fourier") is not None:
            _fp = meta["fourier"]
            _coeffs = np.asarray(
                _fp.per_angle_to_fourier(_contrast_arr, _offset_arr), dtype=np.float64
            )
            frozen_tail = _coeffs.tolist()
        else:
            # fixed_constant or unexpected mode: no scaling tail to freeze
            frozen_tail = []
        frozen = list(popt[:n_phys]) + frozen_tail
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
    # L2 (hierarchical) is now wired for individual/fourier (Task 6).
    # ------------------------------------------------------------------
    gm_block: dict | None = None
    if monitor is not None:
        # Build canonical L4 block; falls back to post_solve_fallback mechanism
        # when the callback never fired (zero observations).
        gm_block = gradient_monitor_diagnostics(monitor)
        if gm_block["mechanism"] == "post_solve_fallback":
            # Compute post-solve covariance condition as fallback indicator.
            # On the hierarchical path pcov is an identity placeholder
            # (info["covariance_is_placeholder"] is True), so cond(I)=1.0 would
            # masquerade as a real, well-conditioned covariance. Report NaN there;
            # only compute the real condition number on a genuine pcov (plain
            # streaming branch).
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

    info["anti_degeneracy"] = assemble_anti_degeneracy_diagnostics(
        hierarchical_active=hierarchical_active,
        regularization_active=bool(regularization_active),
        shear_weighting="laminar_flow_inactive",
        gradient_monitor=gm_block,
        per_angle_mode=meta["per_angle_mode"],
    )

    return popt, pcov, info
