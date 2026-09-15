"""Anti-degeneracy layer (L1-L5) construction for laminar hybrid-streaming.

Extracted from ``fit_with_stratified_hybrid_streaming`` (optimization review
item D2): the narrow, banner-delimited slice that builds the per-angle mode
flags (L1), the hierarchical optimizer (L2), the adaptive regularizer (L3),
the gradient-collapse monitor (L4), and the shear-sensitivity weighter (L5).

Deliberately NOT extracted (stays in ``fit_with_stratified_hybrid_streaming``):
- The per-angle mode RESOLUTION call (``_resolve_streaming_per_angle_mode``) --
  ``tests/optimization/test_static_individual_invariant.py`` and
  ``test_explicit_averaged_mode_parity.py`` use
  ``inspect.getsource(fit_with_stratified_hybrid_streaming)`` to assert that
  exact call appears in THAT function's own source, so it cannot move.
- The group-variance regularization computation and the earlier,
  differently-numbered "4-layer" L-BFGS warm-up defense (warm-start
  detection / adaptive LR / cost guard / step clipping) -- a separate
  concern interleaved with this one in the original function.

Three dependencies are accepted as explicit parameters rather than imported
directly, because existing tests monkeypatch them on the ``hybrid_streaming``
module object (``monkeypatch.setattr(hs, "X", ...)``); passing them through
from the caller's own (patchable) module-level names preserves that contract
exactly the same way ``logger`` is already threaded through this file as a
parameter instead of a module-level import:
``HierarchicalOptimizer``, ``AdaptiveRegularizationConfig``,
``ShearSensitivityWeighting``, and the quantile-scaling estimator (passed as
``compute_quantile_per_angle_scaling``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.adaptive_regularization import AdaptiveRegularizer
from xpcsjax.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from xpcsjax.optimization.nlsq.hierarchical import HierarchicalConfig
from xpcsjax.optimization.nlsq.parameter_utils import ResolvedPhysicalParameters
from xpcsjax.optimization.nlsq.shear_weighting import ShearWeightingConfig


def _resolve_scaling_mode_for_indexing(mode: str, use_fixed_scaling: bool) -> str:
    """Resolve the canonical per-angle mode used for L3/L4 index construction.

    "constant" configured but quantile-based fixed-scaling estimation failed
    (use_fixed_scaling stayed False) builds the real 2-param
    [contrast_mean, offset_mean, *physical] vector -- the "averaged" layout,
    not the frozen 0-param "constant" layout. Every other combination maps
    to itself unchanged.
    """
    return "averaged" if (mode == "constant" and not use_fixed_scaling) else mode


@dataclass
class StreamingLayerSetup:
    """Outputs of :func:`configure_streaming_layers`.

    Mirrors the local variables ``fit_with_stratified_hybrid_streaming``
    previously built inline; the caller destructures this back into those
    same names so the ~1000 remaining lines of that function are untouched.
    """

    per_angle_mode_actual: str
    use_constant: bool
    use_averaged_scaling: bool
    use_fixed_scaling: bool
    fixed_contrast_per_angle: np.ndarray | None
    fixed_offset_per_angle: np.ndarray | None
    fixed_contrast_jax: jnp.ndarray | None
    fixed_offset_jax: jnp.ndarray | None
    averaged_contrast_init: float | None
    averaged_offset_init: float | None
    hierarchical_optimizer: Any | None
    adaptive_regularizer: AdaptiveRegularizer | None
    gradient_monitor: GradientCollapseMonitor | None
    shear_weighter: Any | None
    phi0_is_free_for_shear: bool
    enable_group_variance_regularization: bool
    group_variance_lambda: float
    anti_degeneracy_components: dict[str, Any]


def configure_streaming_layers(
    *,
    per_angle_mode_actual: str,
    constant_scaling_threshold: int,
    per_angle_scaling: bool,
    n_phi: int,
    n_physical: int,
    n_physical_free: int,
    free_physical_names: list[str],
    physical_param_names: list[str],
    is_laminar_flow: bool,
    phi_unique: np.ndarray,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    stratified_data: Any,
    resolved_physical: ResolvedPhysicalParameters | None,
    ad_config: dict,
    hierarchical_config: dict,
    regularization_config: dict,
    gradient_monitoring_config: dict,
    enable_group_variance_regularization: bool,
    group_variance_lambda: float,
    logger: Any,
    HierarchicalOptimizer: Any,
    AdaptiveRegularizationConfig: Any,
    ShearSensitivityWeighting: Any,
    compute_quantile_per_angle_scaling: Any,
) -> StreamingLayerSetup:
    """Build the L1-L5 anti-degeneracy components for the streaming solve.

    ``per_angle_mode_actual`` is already resolved by the caller (via
    ``_resolve_streaming_per_angle_mode`` -- kept inline there, see module
    docstring). ``enable_group_variance_regularization``/``group_variance_lambda``
    are passed through as both input (their config-derived defaults) and
    output (L3 overrides them when it enables adaptive regularization) since
    the caller's separate group-variance-regularization step (kept in place)
    needs whichever value won.
    """
    # T031: Determine mode flags
    # use_constant: True for both averaged and constant (constant-style mapping)
    # use_fixed_scaling: True only for constant (scaling NOT optimized)
    # use_averaged_scaling: True only for averaged (scaling optimized)
    use_constant = per_angle_mode_actual in ("averaged", "constant")
    use_averaged_scaling = per_angle_mode_actual == "averaged"
    # use_fixed_scaling will be set True after quantile estimation for constant mode

    # Per-angle reparameterization: averaged/constant/individual scaling layout.
    if per_angle_mode_actual == "constant" and per_angle_scaling:
        # constant mode: per-angle scaling is FIXED, not optimized
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Constant Scaling")
        logger.info(f"  Mode: {per_angle_mode_actual}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info("  Method: Quantile-based per-angle scaling (FIXED, not optimized)")
        logger.info("  Per-angle contrast/offset will be estimated from c2 data quantiles")
        logger.info("  These values are FIXED (not optimized) during fitting")
        logger.info(f"  Parameter reduction: {2 * n_phi} -> 0 (physical only)")
        logger.info("=" * 60)
    elif per_angle_mode_actual == "averaged" and per_angle_scaling:
        # averaged mode: averaged scaling is OPTIMIZED (9 params)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Averaged Scaling")
        logger.info(f"  Mode: {per_angle_mode_actual}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info("  Method: Quantile estimates -> averaged -> OPTIMIZED")
        logger.info("  Initial values: averaged from per-angle quantile estimates")
        logger.info(f"  Parameter reduction: {2 * n_phi} -> 2 (averaged contrast + offset)")
        logger.info("=" * 60)

    # Unified resolved-mode banner (laminar ↔ heterodyne parity). No controller
    # on this path, so values are computed inline; n_scaling is the OPTIMIZED
    # scaling count (constant -> 0).
    if per_angle_scaling:
        from xpcsjax.optimization.nlsq.anti_degeneracy_logging import (
            MODE_SHORT,
            log_effective_per_angle_mode,
        )

        if per_angle_mode_actual == "constant":
            _n_scaling = 0
        elif per_angle_mode_actual == "averaged":
            _n_scaling = 2
        else:  # individual
            _n_scaling = 2 * n_phi
        log_effective_per_angle_mode(
            logger,
            mode=MODE_SHORT.get(per_angle_mode_actual, per_angle_mode_actual),
            n_phi=n_phi,
            n_physics=n_physical,
            n_scaling=_n_scaling,
            threshold=constant_scaling_threshold,
        )

    # =====================================================================
    # CONSTANT/AUTO_AVERAGED MODES: Quantile-Based Scaling
    # =====================================================================
    # - constant: per-angle values are FIXED (not optimized), 7 params
    # - averaged: averaged values are OPTIMIZED as initial values, 9 params
    # =====================================================================
    use_fixed_scaling = False
    fixed_contrast_per_angle: np.ndarray | None = None
    fixed_offset_per_angle: np.ndarray | None = None
    fixed_contrast_jax: jnp.ndarray | None = None
    fixed_offset_jax: jnp.ndarray | None = None
    # For averaged mode: averaged values to use as initial optimization values
    averaged_contrast_init: float | None = None
    averaged_offset_init: float | None = None

    if use_constant and per_angle_scaling:
        logger.info("Computing quantile-based per-angle scaling estimates...")
        try:
            # Extract bounds for clipping
            contrast_bounds = (0.0, 1.0)  # Default
            offset_bounds = (0.5, 1.5)  # Default
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                if len(lower_bounds) >= n_phi and len(upper_bounds) >= n_phi:
                    contrast_bounds = (
                        float(lower_bounds[0]),
                        float(upper_bounds[0]),
                    )
                    offset_bounds = (
                        float(lower_bounds[n_phi]),
                        float(upper_bounds[n_phi]),
                    )

            # Compute quantile-based per-angle scaling
            fixed_contrast_per_angle, fixed_offset_per_angle = compute_quantile_per_angle_scaling(
                stratified_data=stratified_data,
                contrast_bounds=contrast_bounds,
                offset_bounds=offset_bounds,
                logger=logger,
            )

            if fixed_contrast_per_angle is not None and fixed_offset_per_angle is not None:
                if per_angle_mode_actual == "constant":
                    # constant: Use per-angle values DIRECTLY as FIXED
                    use_fixed_scaling = True
                    fixed_contrast_jax = jnp.asarray(fixed_contrast_per_angle)
                    fixed_offset_jax = jnp.asarray(fixed_offset_per_angle)

                    logger.info("Fixed per-angle scaling computed (FIXED, not optimized):")
                    logger.info(
                        f"  Contrast: mean={np.nanmean(fixed_contrast_per_angle):.4f}, "
                        f"range=[{np.nanmin(fixed_contrast_per_angle):.4f}, "
                        f"{np.nanmax(fixed_contrast_per_angle):.4f}]"
                    )
                    logger.info(
                        f"  Offset: mean={np.nanmean(fixed_offset_per_angle):.4f}, "
                        f"range=[{np.nanmin(fixed_offset_per_angle):.4f}, "
                        f"{np.nanmax(fixed_offset_per_angle):.4f}]"
                    )
                elif per_angle_mode_actual == "averaged":
                    # averaged: AVERAGE per-angle values → use as INITIAL for optimization
                    averaged_contrast_init = float(np.nanmean(fixed_contrast_per_angle))
                    averaged_offset_init = float(np.nanmean(fixed_offset_per_angle))

                    logger.info("Averaged scaling computed (initial values for optimization):")
                    logger.info(f"  Averaged contrast: {averaged_contrast_init:.4f}")
                    logger.info(f"  Averaged offset: {averaged_offset_init:.4f}")
                    logger.info("  These will be OPTIMIZED along with 7 physical params (9 total)")

                    # Do NOT set use_fixed_scaling = True for averaged
                    # The averaged values are just initial guesses for optimization
            else:  # pragma: no cover – defensive; function always returns arrays
                logger.warning(  # type: ignore[unreachable]
                    "Failed to compute quantile-based scaling, "
                    "falling back to standard constant mode (optimizing 2 params)"
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            logger.warning(
                f"Error computing quantile-based scaling: {e}, "
                f"falling back to standard constant mode"
            )
            use_fixed_scaling = False

    # Layer 2: Hierarchical Optimization Configuration
    # =====================================================================
    # CRITICAL FIX (Jan 2026): Auto-enable hierarchical when shear_weighting
    # is enabled. Shear weighting is ONLY applied inside hierarchical
    # optimizer's loss function. Without hierarchical, the gradient
    # cancellation for gamma_dot_t0 is NOT prevented!
    #
    # Root cause: The shear gradient ∂L/∂γ̇₀ ∝ Σ cos(φ₀-φ) cancels when
    # summing over angles spanning 360° (e.g., 23 angles → 94.6% cancellation).
    # Shear weighting emphasizes shear-sensitive angles to prevent this.
    # =====================================================================
    shear_weighting_config_early = ad_config.get("shear_weighting", {})
    shear_weighting_will_be_enabled = (
        shear_weighting_config_early.get("enable", True) and is_laminar_flow and n_phi > 3
    )

    enable_hierarchical = hierarchical_config.get("enable", True)

    # Override: shear weighting requires hierarchical optimization to function
    if shear_weighting_will_be_enabled and not enable_hierarchical:
        logger.warning("=" * 60)
        logger.warning("ANTI-DEGENERACY: Shear weighting enabled but hierarchical disabled!")
        logger.warning("  Auto-enabling hierarchical optimization to apply shear weights.")
        logger.warning("  Without this, gradient cancellation will collapse gamma_dot_t0.")
        logger.warning("=" * 60)
        enable_hierarchical = True

    hierarchical_optimizer = None
    # Skip hierarchical optimization in constant scaling mode:
    # - Constant mode already prevents per-angle absorption (2 DoF vs 46)
    # - HierarchicalOptimizer expects n_per_angle = 2*n_phi (contrast + offset)
    # - Using hierarchical with constant mode causes index mismatch error
    if enable_hierarchical and per_angle_scaling and not use_constant:
        # n_physical defined unconditionally above
        hier_config = HierarchicalConfig(
            enable=True,
            max_outer_iterations=hierarchical_config.get("max_outer_iterations", 5),
            outer_tolerance=float(hierarchical_config.get("outer_tolerance", 1e-6)),
            physical_max_iterations=hierarchical_config.get("physical_max_iterations", 100),
            per_angle_max_iterations=hierarchical_config.get("per_angle_max_iterations", 50),
        )
        hierarchical_optimizer = HierarchicalOptimizer(
            config=hier_config,
            n_phi=n_phi,
            # A fixed physical parameter is stripped out of the vector this
            # optimizer receives (`fit_initial_params`, wrapped via
            # `active_model_fn`/`loss_fn`), so its internal physical-index
            # split must be sized to the REDUCED count, not the full
            # `n_physical` -- otherwise `.physical_indices` runs past the
            # end of the actual (reduced-length) parameter array.
            n_physical=n_physical_free,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
        logger.info(f"  Enabled: {enable_hierarchical}")
        logger.info(f"  Max outer iterations: {hier_config.max_outer_iterations}")
        logger.info(f"  Outer tolerance: {hier_config.outer_tolerance}")
        if shear_weighting_will_be_enabled:
            logger.info("  Shear weighting: WILL BE APPLIED via hierarchical loss function")
        logger.info("=" * 60)
    elif use_constant and enable_hierarchical and per_angle_scaling:
        # Log that hierarchical is skipped due to constant scaling mode
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
        logger.info("  Skipped: constant scaling mode already prevents per-angle absorption")
        logger.info("  Reason: Only 2 per-angle DoF (vs 46), no need for hierarchical alternation")
        logger.info("=" * 60)

    # Layer 3: Adaptive Relative Regularization Configuration
    # Replaces/enhances the basic group variance regularization with CV-based approach
    regularization_mode = regularization_config.get("mode", "relative")
    regularization_lambda = float(regularization_config.get("lambda", 1.0))
    target_cv = float(regularization_config.get("target_cv", 0.10))
    target_contribution = float(regularization_config.get("target_contribution", 0.10))
    max_cv = float(regularization_config.get("max_cv", 0.20))
    auto_tune_lambda = bool(regularization_config.get("auto_tune_lambda", True))

    adaptive_regularizer = None
    if per_angle_scaling:
        # L3 group indices come from the canonical ParameterIndexMapper.
        # Bridge the controller's resolved flags to the canonical mode string.
        from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

        # Derive from per_angle_mode_actual directly (it is always one of
        # "constant"/"averaged"/"individual" per use_constant's definition
        # above), NOT from the use_fixed_scaling/use_averaged_scaling success
        # flags: a quantile-estimation failure in "constant" mode leaves
        # use_fixed_scaling=False while use_constant/per_angle_mode_actual
        # stay "constant" (see the `elif use_constant:` param-vector branch
        # below, added specifically for this fallback state) — the old
        # two-way ternary fell through to "individual" for that state,
        # mis-sizing L3's group indices against the real 2-element scaling
        # head.
        #
        # But literal "constant" is ALSO wrong for that fallback state:
        # ParameterIndexMapper.canonical(mode="constant", ...).n_optimized is
        # 0 (frozen, no scaling head at all), while the `elif use_constant:`
        # branch below builds a REAL 2-param [contrast_mean, offset_mean,
        # *physical] vector for this exact state (quantile estimation failed,
        # use_fixed_scaling stayed False). A true frozen "constant" fit
        # (use_fixed_scaling=True) has no scaling head to group; only the
        # fallback state has one, and it is shaped like "averaged" (2
        # params), not "individual". Resolve to "averaged" for indexing
        # purposes in that one case only — per_angle_mode_actual itself is
        # left untouched (still reported as "constant" in diagnostics). See
        # _resolve_scaling_mode_for_indexing.
        _canonical = _resolve_scaling_mode_for_indexing(per_angle_mode_actual, use_fixed_scaling)
        _mapper = ParameterIndexMapper.canonical(mode=_canonical, n_phi=n_phi, n_physics=n_physical)
        mode_group_indices = _mapper.group_indices or None  # [] (constant) -> None
        logger.debug(f"L3 group indices from mapper ({_canonical}): {_mapper.group_indices}")

        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=regularization_mode,
            lambda_base=regularization_lambda,
            target_cv=target_cv,
            target_contribution=target_contribution,
            max_cv=max_cv,
            auto_tune_lambda=auto_tune_lambda,
            group_indices=mode_group_indices,
        )
        adaptive_regularizer = AdaptiveRegularizer(reg_config, n_phi)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 3 - Adaptive Regularization")
        logger.info(f"  Mode: {regularization_mode}")
        logger.info(f"  Auto-tuned lambda: {adaptive_regularizer.lambda_value:.2f}")
        logger.info(f"  Target CV: {target_cv} ({target_cv * 100:.0f}% variation)")
        logger.info(f"  Max CV: {max_cv}")
        logger.info(f"  Group indices: {adaptive_regularizer.group_indices}")
        logger.info("=" * 60)

        # Update group variance settings to use adaptive regularizer's lambda
        # This ensures NLSQ's built-in regularization is consistent
        enable_group_variance_regularization = True
        group_variance_lambda = adaptive_regularizer.lambda_value

    # Layer 4: Gradient Collapse Monitor Configuration
    gradient_monitor_enabled = gradient_monitoring_config.get("enable", True)
    gradient_monitor = None
    if gradient_monitor_enabled and per_angle_scaling:
        # Phase 6: L4 per-angle count comes from the canonical mapper built in the L3
        # block above (same `if per_angle_scaling:` guard). Rebuild defensively in case
        # L3 was skipped (regularization disabled) — it is a cheap pure dataclass.
        from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

        # Same fix as _canonical above: derive from per_angle_mode_actual
        # directly, not from the use_fixed_scaling/use_averaged_scaling
        # success flags (which can disagree with per_angle_mode_actual on a
        # constant-mode quantile-estimation failure) -- AND resolve that one
        # fallback state ("constant" configured, quantile estimation failed)
        # to "averaged" for indexing, since its real vector is the 2-param
        # [contrast_mean, offset_mean, *physical] layout, not the frozen
        # 0-param "constant" layout (see the matching L3 comment above and
        # _resolve_scaling_mode_for_indexing).
        _canonical_l4 = _resolve_scaling_mode_for_indexing(per_angle_mode_actual, use_fixed_scaling)
        n_per_angle = ParameterIndexMapper.canonical(
            mode=_canonical_l4, n_phi=n_phi, n_physics=n_physical
        ).n_optimized  # 0 (constant) | 2 (averaged) | 2*n_phi (individual)
        # The monitor's `.check()` only ever sees the REDUCED gradient/params
        # array (it is called from `grad_fn`, which differentiates `loss_fn`
        # -> `active_model_fn`, wrapped to accept free-only physical params
        # when a fixed physical parameter is active) -- size its indices to
        # `n_physical_free`, not the full `n_physical`, or they run past the
        # end of that array.
        # Use numpy arrays for indices (JAX compatibility)
        per_angle_indices = np.arange(n_per_angle, dtype=np.intp)
        physical_indices = np.arange(n_per_angle, n_per_angle + n_physical_free, dtype=np.intp)

        # Compute gamma_dot_t0 index for watch_parameters. In laminar_flow,
        # physical params are [D0, alpha, D_offset, gamma_dot_t0, beta,
        # gamma_dot_t_offset, phi0] -- gamma_dot_t0 is nominally at
        # physical_indices[3]. When a physical parameter earlier in that list
        # is fixed (and therefore absent from `free_physical_names`),
        # gamma_dot_t0's position within the REDUCED physical block shifts;
        # look it up by name instead of assuming index 3. If gamma_dot_t0
        # ITSELF is the fixed parameter, there is nothing to watch (it is no
        # longer a free variable at all) -- `watch_parameters` stays empty.
        _watch_parameters: list[int] = []
        if "gamma_dot_t0" in free_physical_names:
            gamma_dot_t0_idx = n_per_angle + free_physical_names.index("gamma_dot_t0")
            _watch_parameters = [gamma_dot_t0_idx]

        monitor_config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=float(gradient_monitoring_config.get("ratio_threshold", 0.01)),
            consecutive_triggers=gradient_monitoring_config.get("consecutive_triggers", 5),
            response_mode=gradient_monitoring_config.get("response", "hierarchical"),
            # NEW (Dec 2025): Watch gamma_dot_t0 specifically for gradient collapse
            # This detects when shear parameter gradient vanishes during L-BFGS warmup
            watch_parameters=_watch_parameters,
            watch_threshold=float(gradient_monitoring_config.get("watch_threshold", 1e-8)),
        )
        gradient_monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=physical_indices,
            per_angle_indices=per_angle_indices,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 4 - Gradient Collapse Monitor")
        logger.info(f"  Enabled: {gradient_monitor_enabled}")
        logger.info(f"  Ratio threshold: {monitor_config.ratio_threshold}")
        logger.info(f"  Consecutive triggers: {monitor_config.consecutive_triggers}")
        logger.info(f"  Response mode: {monitor_config.response_mode}")
        logger.info("=" * 60)

    # Layer 5: Shear-Sensitivity Weighting
    # Prevents gradient cancellation for shear parameters by emphasizing
    # shear-sensitive angles (parallel/antiparallel to flow direction)
    shear_weighting_config = ad_config.get("shear_weighting", {})
    shear_weighting_enabled = shear_weighting_config.get("enable", True)
    shear_weighter: ShearSensitivityWeighting | None = None

    # `_phi0_is_free_for_shear` also gates `shear_weight_update_callback`,
    # defined much further below in this function -- both read/write the
    # same enclosing-function-scope name, standard Python closure capture.
    _phi0_is_free_for_shear = "phi0" in free_physical_names
    if is_laminar_flow and shear_weighting_enabled and n_phi > 3:
        # Get initial phi0 from config or use default
        initial_phi0 = shear_weighting_config.get("initial_phi0", None)
        if initial_phi0 is None:
            # Try to get from initial parameters. `initial_params` here is
            # still the FULL (pre-strip) vector -- the strip happens later,
            # after all Layer 1-5 anti-degeneracy object construction -- so
            # `initial_params[-1]` is phi0 (last physical param) UNLESS phi0
            # is fixed to a value different from its raw config initial
            # guess, in which case `initial_params[-1]` is the stale
            # unfixed guess, not the configured override (dev-suite:
            # three-brain deep-review finding). `resolved_physical.
            # values_full`, when available, already carries the correct
            # value at every position -- the fixed override at fixed slots,
            # the same raw initial value everywhere else -- so prefer it.
            if resolved_physical is not None and "phi0" in physical_param_names:
                initial_phi0 = float(
                    resolved_physical.values_full[physical_param_names.index("phi0")]
                )
            else:
                initial_phi0 = float(initial_params[-1]) if len(initial_params) > 0 else 0.0

        sw_config = ShearWeightingConfig(
            enable=True,
            min_weight=float(shear_weighting_config.get("min_weight", 0.3)),
            alpha=float(shear_weighting_config.get("alpha", 1.0)),
            update_frequency=int(shear_weighting_config.get("update_frequency", 1)),
            initial_phi0=initial_phi0,
            normalize=shear_weighting_config.get("normalize", True),
        )
        # `ShearSensitivityWeighting.update_phi0` is only ever called on the
        # REDUCED (free-only) parameter vector (via
        # shear_weight_update_callback, itself only invoked from the L2
        # hierarchical branch's per-angle-optimized params) -- size it to
        # n_physical_free and look phi0_index up by name (mirrors the L4
        # gamma_dot_t0_idx fix above), not the full n_physical / a hardcoded
        # dense-layout index 6. If phi0 itself is the fixed parameter, it is
        # absent from `free_physical_names`: phi0_index is left at its
        # default (unused, since `_phi0_is_free_for_shear` gates the update
        # callback below to a no-op in that case -- phi0 is pinned to
        # `initial_phi0`, exactly the configured fixed value, for the whole
        # solve, which is the correct behavior for a value that never
        # varies).
        shear_weighter = ShearSensitivityWeighting(
            phi_angles=phi_unique,
            n_physical=n_physical_free,
            phi0_index=(free_physical_names.index("phi0") if _phi0_is_free_for_shear else 0),
            config=sw_config,
        )
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY DEFENSE: Layer 5 - Shear-Sensitivity Weighting")
        logger.info(f"  Enabled: {shear_weighting_enabled}")
        logger.info(f"  n_phi: {n_phi}")
        logger.info(f"  min_weight: {sw_config.min_weight:.2f}")
        logger.info(f"  alpha: {sw_config.alpha:.1f}")
        logger.info(f"  initial_phi0: {initial_phi0:.1f} deg")
        logger.info("=" * 60)

    # Store anti-degeneracy components for diagnostics
    anti_degeneracy_components = {
        "per_angle_mode": per_angle_mode_actual,
        "use_constant": use_constant,  # T031: Track constant mode status
        "use_fixed_scaling": use_fixed_scaling,  # Track fixed scaling status
        "hierarchical_optimizer": hierarchical_optimizer,
        "adaptive_regularizer": adaptive_regularizer,
        "gradient_monitor": gradient_monitor,
        "shear_weighter": shear_weighter,
    }

    return StreamingLayerSetup(
        per_angle_mode_actual=per_angle_mode_actual,
        use_constant=use_constant,
        use_averaged_scaling=use_averaged_scaling,
        use_fixed_scaling=use_fixed_scaling,
        fixed_contrast_per_angle=fixed_contrast_per_angle,
        fixed_offset_per_angle=fixed_offset_per_angle,
        fixed_contrast_jax=fixed_contrast_jax,
        fixed_offset_jax=fixed_offset_jax,
        averaged_contrast_init=averaged_contrast_init,
        averaged_offset_init=averaged_offset_init,
        hierarchical_optimizer=hierarchical_optimizer,
        adaptive_regularizer=adaptive_regularizer,
        gradient_monitor=gradient_monitor,
        shear_weighter=shear_weighter,
        phi0_is_free_for_shear=_phi0_is_free_for_shear,
        enable_group_variance_regularization=enable_group_variance_regularization,
        group_variance_lambda=group_variance_lambda,
        anti_degeneracy_components=anti_degeneracy_components,
    )
