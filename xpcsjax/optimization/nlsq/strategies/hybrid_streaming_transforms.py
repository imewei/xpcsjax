"""Per-angle scaling parameter/covariance transforms for laminar hybrid-streaming.

``fit_with_stratified_hybrid_streaming`` reduces the optimizer-facing parameter
vector for the ``fixed``/``averaged``/``constant`` (quantile-fallback) per-angle
scaling modes before the solve (the FORWARD transform), then expands the
solution — parameters and covariance — back to the full per-angle layout
afterward (the INVERSE transform). Extracted verbatim from
``strategies/hybrid_streaming.py``; ``logger`` is threaded through
explicitly because the caller receives its logger as an
injected parameter (not the module-level ``get_logger(__name__)``), so these
functions must do the same rather than importing their own.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def forward_transform_per_angle_params(
    *,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    n_phi: int,
    use_fixed_scaling: bool,
    use_averaged_scaling: bool,
    use_constant: bool,
    averaged_contrast_init: float | None,
    averaged_offset_init: float | None,
    logger: Any,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Reduce the per-angle parameter vector/bounds ahead of the solve.

    Returns ``(fit_initial_params, fit_bounds)``. When none of the three mode
    flags is set (``individual`` mode), returns a copy of ``initial_params``
    and the unmodified ``bounds`` — matching the original inline fallthrough.
    """
    fit_initial_params = initial_params.copy()
    fit_bounds = bounds

    # T034-T038: Constant mode parameter transformation
    # When use_fixed_scaling=True, use physical params only (fixed contrast/offset from quantiles)
    # Fallback: Transform per-angle params (2*n_phi) to constant (2) by taking means
    if use_fixed_scaling:
        # FIXED SCALING MODE: Use quantile-derived fixed per-angle scaling
        # Parameters are physical-only, contrast/offset are NOT in the param vector
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Fixed Per-Angle Scaling")
        physical_params = initial_params[2 * n_phi :]

        # New parameter layout: [physical_params] only
        fit_initial_params = physical_params

        logger.info(f"  Original params: {len(initial_params)}")
        logger.info(f"  Fixed scaling params: {len(fit_initial_params)} (physical only)")
        logger.info(f"  Per-angle reduction: {2 * n_phi} -> 0 (using fixed arrays)")

        # Transform bounds to physical only
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            fit_bounds = (lower_bounds[2 * n_phi :], upper_bounds[2 * n_phi :])
            logger.info(f"  Bounds reduced to physical only: {len(fit_bounds[0])} params")
        logger.info("=" * 60)
    elif use_averaged_scaling:
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Auto Averaged Scaling Mode")
        # Transform per-angle params to single values (means) for optimization
        per_angle_params = initial_params[: 2 * n_phi]
        physical_params = initial_params[2 * n_phi :]

        # Split per-angle into contrast and offset groups
        contrast_per_angle = per_angle_params[:n_phi]
        offset_per_angle = per_angle_params[n_phi : 2 * n_phi]

        # Use quantile-based averaged values if computed, else take means
        if averaged_contrast_init is not None and averaged_offset_init is not None:
            contrast_mean = averaged_contrast_init
            offset_mean = averaged_offset_init
            logger.info("  Using quantile-based averaged initial values (OPTIMIZED)")
        else:
            contrast_mean = np.nanmean(contrast_per_angle)
            offset_mean = np.nanmean(offset_per_angle)
            logger.info("  Using parameter-based averaged initial values (OPTIMIZED)")

        # New parameter layout: [contrast_const, offset_const, physical_params]
        fit_initial_params = np.concatenate([[contrast_mean], [offset_mean], physical_params])

        logger.info(f"  Original params: {len(initial_params)}")
        logger.info(f"  Constant params: {len(fit_initial_params)}")
        logger.info(f"  Per-angle reduction: {2 * n_phi} -> 2")
        logger.info(f"  Contrast mean: {contrast_mean:.6f}")
        logger.info(f"  Offset mean: {offset_mean:.6f}")

        # T039: Transform bounds for constant mode
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            # For constant mode, use the bounds of the first per-angle param
            # (all per-angle bounds are typically the same)
            fit_lower = np.concatenate(
                [
                    [lower_bounds[0]],
                    [lower_bounds[n_phi]],
                    lower_bounds[2 * n_phi :],
                ]
            )
            fit_upper = np.concatenate(
                [
                    [upper_bounds[0]],
                    [upper_bounds[n_phi]],
                    upper_bounds[2 * n_phi :],
                ]
            )
            fit_bounds = (fit_lower, fit_upper)
        logger.info("=" * 60)
    elif use_constant:
        # Fallback: explicit "constant" mode where quantile-based fixed-scaling
        # estimation raised (use_fixed_scaling stayed False, see the except
        # block above). Without this branch fit_initial_params/fit_bounds stay
        # at the full per-angle length (2*n_phi + n_physical) while
        # model_fn_pointwise's `elif use_constant:` branch (and n_per_angle=2
        # above) expect only 2 + n_physical params — a silent parameter-vector
        # corruption (per-angle contrast/offset leaking into physics slots).
        # Mirror the averaged-mode transform above, but using the mean of the
        # raw per-angle initial values (quantile estimates are unavailable).
        logger.warning("=" * 60)
        logger.warning("ANTI-DEGENERACY EXECUTION: Constant Scaling (quantile fallback)")
        logger.warning("  Quantile-based fixed scaling failed; using mean of initial values")
        per_angle_params = initial_params[: 2 * n_phi]
        physical_params = initial_params[2 * n_phi :]
        contrast_per_angle = per_angle_params[:n_phi]
        offset_per_angle = per_angle_params[n_phi : 2 * n_phi]
        contrast_mean = np.nanmean(contrast_per_angle)
        offset_mean = np.nanmean(offset_per_angle)

        # New parameter layout: [contrast_const, offset_const, physical_params]
        fit_initial_params = np.concatenate([[contrast_mean], [offset_mean], physical_params])

        logger.warning(f"  Original params: {len(initial_params)}")
        logger.warning(f"  Constant params: {len(fit_initial_params)}")
        logger.warning(f"  Per-angle reduction: {2 * n_phi} -> 2")
        logger.warning(f"  Contrast mean: {contrast_mean:.6f}")
        logger.warning(f"  Offset mean: {offset_mean:.6f}")

        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            fit_lower = np.concatenate(
                [
                    [lower_bounds[0]],
                    [lower_bounds[n_phi]],
                    lower_bounds[2 * n_phi :],
                ]
            )
            fit_upper = np.concatenate(
                [
                    [upper_bounds[0]],
                    [upper_bounds[n_phi]],
                    upper_bounds[2 * n_phi :],
                ]
            )
            fit_bounds = (fit_lower, fit_upper)
        logger.warning("=" * 60)

    return fit_initial_params, fit_bounds


def _transform_covariance_via_jacobian(
    pcov_const: np.ndarray | None,
    *,
    n_constant_total: int,
    n_phi: int,
    n_physical: int,
    logger: Any,
    log_fn: Any,
    log_mismatch: bool,
) -> np.ndarray | None:
    """Expand a constant-space covariance to the full per-angle covariance.

    Maps ``[contrast_const, offset_const, physical]`` to the full per-angle
    ``[contrast(n_phi), offset(n_phi), physical]`` space via the broadcast
    Jacobian ``J_full`` (contrast/offset columns broadcast to every angle,
    physical block is identity): ``pcov_full = J_full @ pcov_const @ J_full.T``.

    Shared by the ``use_averaged_scaling`` and ``use_constant`` (quantile
    fallback) inverse-transform branches of
    :func:`inverse_transform_per_angle_params` — previously two near-identical
    ~30-line copies.

    ``log_fn`` is the success-path logger method (``logger.info`` for the real
    averaged-mode path, ``logger.warning`` for the constant quantile-fallback
    path, matching each caller's original level). ``log_mismatch`` preserves
    an existing asymmetry: the averaged-mode branch logs a detailed shape
    warning when ``pcov_const`` is missing/misshapen; the constant-fallback
    branch silently returns ``None`` there (not touched by this refactor).
    """
    if (
        pcov_const is None
        or pcov_const.shape[0] != n_constant_total
        or pcov_const.shape[1] != n_constant_total
    ):
        if log_mismatch:
            pcov_shape = pcov_const.shape if pcov_const is not None else None
            logger.warning(
                f"  Constant covariance unavailable or wrong shape (got {pcov_shape}, "
                f"expected ({n_constant_total}, {n_constant_total})). "
                "Using identity fallback."
            )
        return None

    n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
    n_total_restored = n_per_angle_total + n_physical

    # Build Jacobian for constant -> per-angle transformation
    J_full = np.zeros((n_total_restored, n_constant_total))
    # Contrast broadcast: d(contrast_per_angle[i])/d(contrast_const) = 1
    J_full[:n_phi, 0] = 1.0
    # Offset broadcast: d(offset_per_angle[i])/d(offset_const) = 1
    J_full[n_phi : 2 * n_phi, 1] = 1.0
    # Physical params: identity (pass-through)
    J_full[2 * n_phi :, 2:] = np.eye(n_physical)

    try:
        pcov_transformed = J_full @ pcov_const @ J_full.T
        log_fn("  Covariance transformed from constant to per-angle space")
        return pcov_transformed
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(f"  Covariance transformation failed: {e}. Using identity fallback.")
        return None


def inverse_transform_per_angle_params(
    *,
    popt: np.ndarray,
    result: dict[str, Any],
    n_phi: int,
    use_fixed_scaling: bool,
    use_averaged_scaling: bool,
    use_constant: bool,
    fixed_contrast_per_angle: np.ndarray | None,
    fixed_offset_per_angle: np.ndarray | None,
    logger: Any,
) -> np.ndarray:
    """Expand the optimizer's reduced ``popt`` back to the full per-angle layout.

    Mutates ``result["pcov_transformed"]`` in place exactly as the original
    inline code did; returns the (possibly expanded) ``popt``. When none of
    the three mode flags is set (``individual`` mode), ``popt`` is returned
    unchanged and ``result`` is not touched — matching the original inline
    fallthrough (no ``if``/``elif`` branch taken).
    """
    # Fixed scaling mode inverse transformation
    # Expand physical-only params back to per-angle format using fixed scaling arrays
    if use_fixed_scaling:
        assert fixed_contrast_per_angle is not None  # set when use_fixed_scaling is True
        assert fixed_offset_per_angle is not None  # set when use_fixed_scaling is True
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Inverse Fixed Scaling Transform")
        # Layout: [physical_params] - popt contains ONLY physical parameters
        physical_params_opt = popt

        # Use the pre-computed fixed per-angle scaling from quantiles
        contrast_per_angle_opt = fixed_contrast_per_angle
        offset_per_angle_opt = fixed_offset_per_angle

        # Reconstruct full parameter vector in original layout
        popt = np.concatenate([contrast_per_angle_opt, offset_per_angle_opt, physical_params_opt])

        logger.info(f"  Physical params: {len(physical_params_opt)}")
        logger.info(f"  Fixed per-angle scaling restored: {len(popt)} total params")
        logger.info(
            f"  Contrast (fixed): mean={np.nanmean(contrast_per_angle_opt):.4f}, "
            f"range=[{np.nanmin(contrast_per_angle_opt):.4f}, {np.nanmax(contrast_per_angle_opt):.4f}]"
        )
        logger.info(
            f"  Offset (fixed): mean={np.nanmean(offset_per_angle_opt):.4f}, "
            f"range=[{np.nanmin(offset_per_angle_opt):.4f}, {np.nanmax(offset_per_angle_opt):.4f}]"
        )

        # Transform covariance from physical-only space to full space
        # For fixed scaling mode, the Jacobian is simpler:
        # Per-angle params are fixed (variance = 0), physical params have identity
        # J[i, j] = 0 for per-angle params (i < 2*n_phi)
        # J[2*n_phi+i, i] = 1 for physical params (identity)
        pcov_physical = result.get("pcov", None)
        n_physical = len(physical_params_opt)

        if (
            pcov_physical is not None
            and pcov_physical.shape[0] == n_physical
            and pcov_physical.shape[1] == n_physical
        ):
            n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
            n_total_restored = n_per_angle_total + n_physical

            # Build full covariance matrix
            # Per-angle params have zero covariance (they're fixed)
            # Physical params have the original covariance
            try:
                pcov_full = np.zeros((n_total_restored, n_total_restored))
                # Physical params covariance block
                pcov_full[2 * n_phi :, 2 * n_phi :] = pcov_physical
                result["pcov_transformed"] = pcov_full
                logger.info("  Covariance expanded: per-angle=0 (fixed), physical=preserved")
            except (
                ValueError,
                RuntimeError,
                MemoryError,
                np.linalg.LinAlgError,
            ) as e:
                logger.warning(f"  Covariance expansion failed: {e}. Using identity fallback.")
                result["pcov_transformed"] = None
        else:
            pcov_shape = pcov_physical.shape if pcov_physical is not None else None
            logger.warning(
                f"  Physical covariance unavailable or wrong shape (got {pcov_shape}, "
                f"expected ({n_physical}, {n_physical})). "
                "Using identity fallback."
            )
            result["pcov_transformed"] = None

        logger.info("=" * 60)

    # T046-T049: Auto averaged mode inverse transformation
    # Expand averaged parameters back to per-angle format for backward compatibility
    elif use_averaged_scaling:
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY EXECUTION: Inverse Auto Averaged Transform")
        # Layout: [contrast_const, offset_const, physical_params]
        from xpcsjax.optimization.nlsq.data_prep import (
            expand_per_angle_parameters,
        )

        contrast_const = popt[0]
        offset_const = popt[1]
        n_physical_opt = len(popt) - 2
        expanded = expand_per_angle_parameters(
            popt,
            None,
            n_phi,
            n_physical_opt,
        )
        popt = expanded.params

        logger.info(f"  Constant params: 2 + {n_physical_opt} physical")
        logger.info(f"  Restored per-angle params: {len(popt)}")
        logger.info(f"  Contrast (uniform): {contrast_const:.6f}")
        logger.info(f"  Offset (uniform): {offset_const:.6f}")

        # Transform covariance from constant space to per-angle space
        # For constant mode, the Jacobian is simpler: broadcasting matrix
        # J[i, 0] = 1 for i in 0..n_phi-1 (contrast params)
        # J[n_phi+i, 1] = 1 for i in 0..n_phi-1 (offset params)
        # J[2*n_phi+i, 2+i] = 1 for physical params (identity)
        pcov_constant = result.get("pcov", None)
        n_constant_total = 2 + n_physical_opt

        result["pcov_transformed"] = _transform_covariance_via_jacobian(
            pcov_constant,
            n_constant_total=n_constant_total,
            n_phi=n_phi,
            n_physical=n_physical_opt,
            logger=logger,
            log_fn=logger.info,
            log_mismatch=True,
        )

        logger.info("=" * 60)
    elif use_constant:
        # Inverse of the forward-transform quantile-fallback branch above:
        # explicit "constant" mode whose quantile-based fixed-scaling estimation
        # failed used the same [contrast_const, offset_const, physical_params]
        # layout as use_averaged_scaling, so popt/pcov must be expanded back to
        # the per-angle layout the same way — otherwise popt stays at length
        # 2 + n_physical instead of the 2*n_phi + n_physical the caller
        # (residual_fn, diagnostics) expects.
        logger.warning("=" * 60)
        logger.warning("ANTI-DEGENERACY EXECUTION: Inverse Constant Transform (quantile fallback)")
        from xpcsjax.optimization.nlsq.data_prep import (
            expand_per_angle_parameters,
        )

        contrast_const = popt[0]
        offset_const = popt[1]
        n_physical_opt = len(popt) - 2
        expanded = expand_per_angle_parameters(
            popt,
            None,
            n_phi,
            n_physical_opt,
        )
        popt = expanded.params

        logger.warning(f"  Constant params: 2 + {n_physical_opt} physical")
        logger.warning(f"  Restored per-angle params: {len(popt)}")
        logger.warning(f"  Contrast (uniform): {contrast_const:.6f}")
        logger.warning(f"  Offset (uniform): {offset_const:.6f}")

        pcov_constant = result.get("pcov", None)
        n_constant_total = 2 + n_physical_opt

        # Preserves the original's silent-None-on-mismatch behavior (no
        # detailed warning here, unlike the averaged branch above) via
        # log_mismatch=False.
        result["pcov_transformed"] = _transform_covariance_via_jacobian(
            pcov_constant,
            n_constant_total=n_constant_total,
            n_phi=n_phi,
            n_physical=n_physical_opt,
            logger=logger,
            log_fn=logger.warning,
            log_mismatch=False,
        )

        logger.warning("=" * 60)

    return popt
