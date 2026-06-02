"""Stratified least-squares strategy for NLSQ optimization.

Extracted from wrapper.py to reduce file size and improve maintainability.

This module provides:
- Stratified chunk creation from angle-stratified data
- Stratified least-squares fitting with anti-degeneracy support
- Supports fixed_constant, auto_averaged, individual, and fourier modes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from nlsq import LeastSquares

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)
from xpcsjax.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """One stratified chunk of flat (phi, t1, t2, g2) arrays plus shared metadata."""

    phi: Any
    t1: Any
    t2: Any
    g2: Any
    q: Any
    L: Any
    dt: Any


@dataclass
class StratifiedChunkedData:
    """Container exposing chunked data via ``.chunks`` plus parent-level ``sigma``."""

    chunks: list[Any]
    sigma: Any


def create_stratified_chunks(
    stratified_data: Any,
    target_chunk_size: int = 100_000,
) -> Any:
    """Convert stratified flat arrays into chunks for StratifiedResidualFunction.

    Args:
        stratified_data: StratifiedData object with flat stratified arrays
        target_chunk_size: Target size for each chunk

    Returns:
        Object with .chunks attribute containing list of chunk objects
    """
    # Get flat stratified arrays
    phi_flat = stratified_data.phi_flat
    t1_flat = stratified_data.t1_flat
    t2_flat = stratified_data.t2_flat
    g2_flat = stratified_data.g2_flat

    # Get metadata (not chunked - shared across all chunks)
    sigma = stratified_data.sigma  # 3D array: (n_phi, n_t1, n_t2)
    q = stratified_data.q
    L = stratified_data.L
    dt = getattr(stratified_data, "dt", None)

    # CRITICAL FIX (Nov 10, 2025): Use original stratification boundaries
    # instead of naive sequential slicing to preserve angle completeness
    chunk_sizes_attr = getattr(stratified_data, "chunk_sizes", None)

    if chunk_sizes_attr is not None:
        # Use original chunk boundaries from stratification
        # This ensures each chunk contains all phi angles
        n_chunks = len(chunk_sizes_attr)
        chunks = []
        current_idx = 0

        for _, chunk_size in enumerate(chunk_sizes_attr):
            start_idx = current_idx
            end_idx = current_idx + chunk_size

            chunk = Chunk(
                phi=phi_flat[start_idx:end_idx],
                t1=t1_flat[start_idx:end_idx],
                t2=t2_flat[start_idx:end_idx],
                g2=g2_flat[start_idx:end_idx],
                q=q,
                L=L,
                dt=dt,
            )
            chunks.append(chunk)
            current_idx = end_idx
    else:
        # Fallback: Sequential chunking (for index-based stratification)
        # WARNING: This may still have angle incompleteness issues!
        n_total = len(g2_flat)
        n_chunks = max(1, (n_total + target_chunk_size - 1) // target_chunk_size)

        chunks = []
        for i in range(n_chunks):
            start_idx = i * target_chunk_size
            end_idx = min(start_idx + target_chunk_size, n_total)

            chunk = Chunk(
                phi=phi_flat[start_idx:end_idx],
                t1=t1_flat[start_idx:end_idx],
                t2=t2_flat[start_idx:end_idx],
                g2=g2_flat[start_idx:end_idx],
                q=q,
                L=L,
                dt=dt,
            )
            chunks.append(chunk)

    return StratifiedChunkedData(chunks, sigma)


def fit_with_stratified_least_squares(
    stratified_data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    target_chunk_size: int = 100_000,
    anti_degeneracy_config: dict | None = None,
    nlsq_config_dict: dict | None = None,
    analysis_mode: AnalysisMode | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using NLSQ's least_squares() with stratified residual function.

    This method solves the double-chunking problem by using NLSQ's least_squares()
    function directly with a StratifiedResidualFunction. This gives us full control
    over chunking, ensuring angle completeness in each chunk for proper per-angle
    parameter gradients.

    Args:
        stratified_data: StratifiedData object with flat stratified arrays
        per_angle_scaling: Whether per-angle parameters are enabled
        physical_param_names: List of physical parameter names
        initial_params: Initial parameter guess
        bounds: Parameter bounds (lower, upper) tuple
        log: Logger instance
        target_chunk_size: Target size for each chunk (default: 100k points)
        anti_degeneracy_config: Optional config dict for Anti-Degeneracy Defense System
        nlsq_config_dict: Optional NLSQ convergence config dict

    Returns:
        (popt, pcov, info) tuple
    """
    log.info("=" * 80)
    log.info("STRATIFIED LEAST-SQUARES OPTIMIZATION")
    log.info("Using NLSQ's least_squares() with angle-stratified chunks")
    log.info("=" * 80)

    # =====================================================================
    # Anti-Degeneracy Defense System (v2.14.0+)
    # =====================================================================
    is_laminar_flow = "gamma_dot_t0" in physical_param_names
    n_phi = len(np.unique(stratified_data.phi_flat))
    n_physical = len(physical_param_names)

    # Initialize anti-degeneracy controller
    ad_controller = None
    if anti_degeneracy_config is not None and per_angle_scaling and is_laminar_flow:
        phi_unique_rad = np.deg2rad(np.array(sorted(set(stratified_data.phi_flat))))
        ad_controller = AntiDegeneracyController.from_config(
            config_dict=anti_degeneracy_config,
            n_phi=n_phi,
            phi_angles=phi_unique_rad,
            n_physical=n_physical,
            per_angle_scaling=per_angle_scaling,
            is_laminar_flow=is_laminar_flow,
            analysis_mode=analysis_mode,
        )

        if ad_controller.is_enabled:
            log.info("=" * 60)
            log.info("ANTI-DEGENERACY DEFENSE: Enabled for Stratified LS")
            log.info(f"  per_angle_mode: {ad_controller.per_angle_mode_actual}")
            log.info(f"  use_constant: {ad_controller.use_constant}")
            log.info(f"  use_fixed_scaling: {ad_controller.use_fixed_scaling}")
            log.info(f"  use_averaged_scaling: {ad_controller.use_averaged_scaling}")
            log.info(f"  use_fourier: {ad_controller.use_fourier}")
            log.info(f"  use_shear_weighting: {ad_controller.use_shear_weighting}")
            log.info("=" * 60)

            # Transform initial parameters for Fourier mode only
            # CONSTANT MODE (v2.17.0): Parameter transformation is handled later
            # when computing fixed per-angle scaling from quantiles
            if ad_controller.use_fixed_scaling:
                log.info(
                    "Fixed constant mode: parameter transformation deferred to "
                    "quantile-based fixed scaling computation"
                )

            elif ad_controller.use_averaged_scaling:
                log.info(
                    "Auto averaged mode: parameter transformation deferred to "
                    "quantile-based averaged scaling computation"
                )

            elif ad_controller.use_fourier:
                log.info(
                    f"Transforming parameters: Fourier mode ({len(initial_params)} -> "
                    f"{ad_controller.n_per_angle_params + n_physical})"
                )
                initial_params, _ = ad_controller.transform_params_to_fourier(
                    initial_params
                )
                if bounds is not None:
                    # Transform bounds for Fourier mode
                    lower, upper = bounds
                    assert ad_controller.fourier is not None
                    n_coeffs = ad_controller.fourier.n_coeffs_per_param
                    # Use the mean of bounds for Fourier coefficients
                    lower_fourier = np.concatenate(
                        [
                            np.full(
                                n_coeffs, np.mean(lower[:n_phi])
                            ),  # contrast coeffs
                            np.full(
                                n_coeffs, np.mean(lower[n_phi : 2 * n_phi])
                            ),  # offset coeffs
                            lower[2 * n_phi :],  # physical lower
                        ]
                    )
                    upper_fourier = np.concatenate(
                        [
                            np.full(
                                n_coeffs, np.mean(upper[:n_phi])
                            ),  # contrast coeffs
                            np.full(
                                n_coeffs, np.mean(upper[n_phi : 2 * n_phi])
                            ),  # offset coeffs
                            upper[2 * n_phi :],  # physical upper
                        ]
                    )
                    bounds = (lower_fourier, upper_fourier)
                    log.debug(f"Transformed bounds to Fourier mode: {bounds[0].shape}")

    # Convert stratified flat arrays into chunks
    log.info(
        f"Creating chunks from stratified data (target size: {target_chunk_size:,})..."
    )
    chunked_data = create_stratified_chunks(stratified_data, target_chunk_size)
    log.info(f"Created {len(chunked_data.chunks)} chunks")

    # Start timing
    start_time = time.perf_counter()

    # Create JIT-compatible stratified residual function
    # CRITICAL UPDATE (v2.17.0): Constant mode now uses fixed per-angle scaling
    # from quantile estimation. The parameters contain ONLY physical parameters.
    # This replaces the old approach of using mean contrast/offset.
    effective_per_angle_scaling = per_angle_scaling
    fixed_contrast = None
    fixed_offset = None

    if ad_controller is not None and ad_controller.use_fixed_scaling:
        # FIXED_CONSTANT MODE (v2.18.0): Compute fixed per-angle scaling
        # from quantiles. Per-angle values are FIXED (not optimized).
        # Result: 7 physical params only.
        log.info("=" * 60)
        log.info(
            "FIXED_CONSTANT MODE: Computing fixed per-angle scaling from quantiles"
        )
        log.info("=" * 60)

        # Get contrast/offset bounds from initial bounds
        if bounds is not None:
            contrast_bounds = (
                float(np.min(bounds[0][:n_phi])),
                float(np.max(bounds[1][:n_phi])),
            )
            offset_bounds = (
                float(np.min(bounds[0][n_phi : 2 * n_phi])),
                float(np.max(bounds[1][n_phi : 2 * n_phi])),
            )
        else:
            contrast_bounds = (0.0, 1.0)
            offset_bounds = (0.5, 1.5)

        # Compute fixed per-angle scaling from quantiles
        ad_controller.compute_fixed_per_angle_scaling(
            stratified_data=stratified_data,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        # Get the fixed scaling values
        fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
        if fixed_scaling is not None:
            fixed_contrast, fixed_offset = fixed_scaling

            # Update initial_params to contain ONLY physical parameters
            # Original format was [contrast(n_phi), offset(n_phi), physical(n_physical)]
            # New format is [physical(n_physical)]
            initial_params = initial_params[2 * n_phi :]
            log.info(
                f"Reduced initial parameters to physical only: {len(initial_params)} params"
            )

            # Update bounds to contain ONLY physical parameter bounds
            if bounds is not None:
                lower, upper = bounds
                bounds = (lower[2 * n_phi :], upper[2 * n_phi :])
                log.info(f"Reduced bounds to physical only: {len(bounds[0])} params")

            # Mark that we're using fixed scaling (not per_angle_scaling from params)
            effective_per_angle_scaling = False
            log.info(
                f"Fixed per-angle scaling will be used:\n"
                f"  Contrast: mean={np.nanmean(fixed_contrast):.4f}, "
                f"range=[{np.nanmin(fixed_contrast):.4f}, {np.nanmax(fixed_contrast):.4f}]\n"
                f"  Offset: mean={np.nanmean(fixed_offset):.4f}, "
                f"range=[{np.nanmin(fixed_offset):.4f}, {np.nanmax(fixed_offset):.4f}]"
            )
        else:
            log.warning(
                "Failed to compute fixed per-angle scaling, "
                "falling back to standard mode"
            )
            effective_per_angle_scaling = False

    elif ad_controller is not None and ad_controller.use_averaged_scaling:
        # AUTO_AVERAGED MODE (v2.18.0): Estimate per-angle scaling from
        # quantiles, AVERAGE to single values, then OPTIMIZE them.
        # Result: 9 params (7 physical + 1 contrast_avg + 1 offset_avg).
        log.info("=" * 60)
        log.info("AUTO_AVERAGED MODE: Computing averaged scaling initial values")
        log.info("=" * 60)

        # Get contrast/offset bounds from initial bounds
        if bounds is not None:
            contrast_bounds = (
                float(np.min(bounds[0][:n_phi])),
                float(np.max(bounds[1][:n_phi])),
            )
            offset_bounds = (
                float(np.min(bounds[0][n_phi : 2 * n_phi])),
                float(np.max(bounds[1][n_phi : 2 * n_phi])),
            )
        else:
            contrast_bounds = (0.0, 1.0)
            offset_bounds = (0.5, 1.5)

        # Compute per-angle scaling from quantiles for initial estimates
        ad_controller.compute_fixed_per_angle_scaling(
            stratified_data=stratified_data,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        # Get per-angle estimates and AVERAGE them for optimization start
        fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
        if fixed_scaling is not None:
            contrast_per_angle, offset_per_angle = fixed_scaling
            avg_contrast = float(np.nanmean(contrast_per_angle))
            avg_offset = float(np.nanmean(offset_per_angle))

            # Build 9-param initial_params: [contrast_avg, offset_avg, physical(7)]
            physical_params_init = initial_params[2 * n_phi :]
            initial_params = np.concatenate(
                [[avg_contrast, avg_offset], physical_params_init]
            )
            log.info(
                f"Averaged initial parameters: {len(initial_params)} params "
                f"(contrast={avg_contrast:.4f}, offset={avg_offset:.4f})"
            )

            # Update bounds: [contrast_bounds, offset_bounds, physical_bounds]
            if bounds is not None:
                lower, upper = bounds
                bounds = (
                    np.concatenate(
                        [[contrast_bounds[0], offset_bounds[0]], lower[2 * n_phi :]]
                    ),
                    np.concatenate(
                        [[contrast_bounds[1], offset_bounds[1]], upper[2 * n_phi :]]
                    ),
                )
                log.info(f"Updated bounds for averaged mode: {len(bounds[0])} params")

            # Scalar contrast/offset will be OPTIMIZED (not fixed)
            # per_angle_scaling=False + no fixed arrays -> residual mode 3
            effective_per_angle_scaling = False
            # Do NOT set fixed_contrast/fixed_offset -- they are optimized
            log.info(
                f"Averaged scaling will be OPTIMIZED (not fixed):\n"
                f"  Initial contrast: {avg_contrast:.4f} "
                f"(from per-angle range [{np.nanmin(contrast_per_angle):.4f}, "
                f"{np.nanmax(contrast_per_angle):.4f}])\n"
                f"  Initial offset: {avg_offset:.4f} "
                f"(from per-angle range [{np.nanmin(offset_per_angle):.4f}, "
                f"{np.nanmax(offset_per_angle):.4f}])"
            )
        else:
            log.warning(
                "Failed to compute per-angle scaling estimates, "
                "falling back to mean of initial per-angle values"
            )
            # Fallback: average the initial per-angle values (NaN-safe)
            avg_contrast = float(np.nanmean(initial_params[:n_phi]))
            avg_offset = float(np.nanmean(initial_params[n_phi : 2 * n_phi]))
            physical_params_init = initial_params[2 * n_phi :]
            initial_params = np.concatenate(
                [[avg_contrast, avg_offset], physical_params_init]
            )
            if bounds is not None:
                lower, upper = bounds
                bounds = (
                    np.concatenate(
                        [[contrast_bounds[0], offset_bounds[0]], lower[2 * n_phi :]]
                    ),
                    np.concatenate(
                        [[contrast_bounds[1], offset_bounds[1]], upper[2 * n_phi :]]
                    ),
                )
            effective_per_angle_scaling = False

    log.info("Creating JIT-compatible stratified residual function...")
    residual_fn = StratifiedResidualFunctionJIT(
        stratified_data=chunked_data,  # Use chunked_data with .chunks attribute
        per_angle_scaling=effective_per_angle_scaling,
        physical_param_names=physical_param_names,
        logger=log,  # type: ignore[arg-type]
        fixed_contrast_per_angle=fixed_contrast,
        fixed_offset_per_angle=fixed_offset,
    )

    # Validate chunk structure
    log.info("Validating chunk structure...")
    residual_fn.validate_chunk_structure()

    # Log diagnostics
    residual_fn.log_diagnostics()

    # Gradient sanity check (CRITICAL)
    # Verify that gradients are non-zero before starting optimization
    # This catches parameter initialization issues early
    log.info("=" * 80)
    log.info("GRADIENT SANITY CHECK")
    log.info("=" * 80)

    try:
        # Compute residuals at initial parameters
        residuals_0 = residual_fn(initial_params)
        log.info(
            f"Initial residuals: shape={residuals_0.shape}, "
            f"min={float(np.min(residuals_0)):.6e}, "
            f"max={float(np.max(residuals_0)):.6e}, "
            f"mean={float(np.mean(residuals_0)):.6e}"
        )

        # Perturb the first physical parameter (D0) by 1%.
        # BUG-5: In auto_averaged mode, effective_per_angle_scaling=False but
        # params = [contrast_avg, offset_avg, D0, ...], so D0 is at index 2.
        # In individual mode, D0 is at index 2*n_phi. In fixed_constant, D0
        # is at index 0 (no scaling params in vector).
        if effective_per_angle_scaling:
            phys_idx = 2 * residual_fn.n_phi  # individual mode
        elif len(initial_params) > 2:
            phys_idx = 2  # auto_averaged: [contrast_avg, offset_avg, D0, ...]
        else:
            phys_idx = 0  # fixed_constant: [D0, alpha, ...]
        params_test = np.array(initial_params, copy=True)
        params_test[phys_idx] *= 1.01  # 1% perturbation
        residuals_1 = residual_fn(params_test)

        # Estimate gradient magnitude
        gradient_estimate = float(np.abs(np.sum(residuals_1 - residuals_0)))
        log.info(
            f"Gradient estimate (1% perturbation of param[{phys_idx}]): {gradient_estimate:.6e}"
        )

        # Check if gradient is suspiciously small
        if gradient_estimate < 1e-10:
            log.error("=" * 80)
            log.error("GRADIENT SANITY CHECK FAILED")
            log.error("=" * 80)
            log.error(f"Gradient estimate: {gradient_estimate:.6e} (expected > 1e-10)")
            log.error("This indicates:")
            log.error(
                "  - Parameter initialization issue (likely wrong parameter count)"
            )
            log.error("  - Residual function not sensitive to parameter changes")
            log.error("  - Optimization will fail with 0 iterations")
            log.error("")
            log.error("Diagnostic information:")
            log.error(f"  Initial parameters count: {len(initial_params)}")
            if effective_per_angle_scaling:
                expected_count = len(physical_param_names) + 2 * residual_fn.n_phi
                log.error(
                    f"  Expected for per-angle scaling: {len(physical_param_names)} physical + 2*{residual_fn.n_phi} scaling = {expected_count}"
                )
            else:
                expected_count = len(physical_param_names) + 2
                log.error(
                    f"  Expected for constant scaling: {len(physical_param_names)} physical + 2 scaling = {expected_count}"
                )
            log.error(
                f"  Residual function expects: per_angle_scaling={effective_per_angle_scaling}, n_phi={residual_fn.n_phi}"
            )
            log.error("=" * 80)
            raise ValueError(
                f"Gradient sanity check FAILED: gradient ~{gradient_estimate:.2e} "
                f"(expected > 1e-10). Optimization cannot proceed with zero gradients."
            )

        log.info(
            f"Gradient sanity check passed (gradient magnitude: {gradient_estimate:.6e})"
        )

    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
        if "Gradient sanity check FAILED" in str(e):
            raise  # Re-raise our custom error
        log.warning(f"Gradient sanity check encountered error: {e}")
        log.warning("Proceeding with optimization, but this may fail")

    log.info("=" * 80)

    # Prepare for optimization
    log.info("Starting NLSQ least_squares() optimization...")
    log.info(f"  Initial parameters: {len(initial_params)} parameters")
    log.info(f"  Bounds: {'provided' if bounds is not None else 'unbounded'}")
    log.info(f"  Residual chunks: {residual_fn.n_chunks}")
    log.info(f"  Real data points: {residual_fn.n_real_points:,}")

    # Call NLSQ's least_squares() - NO xdata/ydata needed!
    # Data is encapsulated in residual_fn
    optimization_start = time.perf_counter()

    # Extract convergence tolerances from NLSQConfig (BUG-14 fix)
    cfg = nlsq_config_dict or {}
    opt_ftol = float(cfg.get("ftol", cfg.get("tolerance", 1e-8)))
    opt_xtol = float(cfg.get("xtol", 1e-8))
    opt_gtol = float(cfg.get("gtol", 1e-8))
    opt_max_nfev = int(cfg.get("max_iterations", 1000))

    ls = LeastSquares(enable_stability=True, enable_diagnostics=True)

    from xpcsjax.optimization.nlsq.gradient_monitor import _get_debug_curvefit_callback

    _ls_kwargs: dict = dict(
        fun=residual_fn,
        x0=initial_params,
        jac=None,  # Use JAX autodiff for Jacobian
        bounds=bounds,  # type: ignore[arg-type]
        method="trf",  # Trust Region Reflective
        ftol=opt_ftol,
        xtol=opt_xtol,
        gtol=opt_gtol,
        max_nfev=opt_max_nfev,
        verbose=0,
    )
    _dbg_cb = _get_debug_curvefit_callback()
    if _dbg_cb is not None and "callback" not in _ls_kwargs:
        _ls_kwargs["callback"] = _dbg_cb

    result = ls.least_squares(**_ls_kwargs)

    optimization_time = time.perf_counter() - optimization_start
    log.info(f"Optimization completed in {optimization_time:.2f}s")

    # Extract results from NLSQ result object
    popt = np.asarray(result["x"])

    # CRITICAL: Enforce parameter bounds (post-optimization clipping)
    log.info(
        f"POST-OPTIMIZATION BOUNDS CHECK: bounds={'provided' if bounds is not None else 'None'}, popt shape={popt.shape}"
    )
    if bounds is not None:
        lower_bounds, upper_bounds = bounds
        bounds_violated = False

        log.info(
            f"Debug: lower_bounds type={type(lower_bounds)}, shape={getattr(lower_bounds, 'shape', 'N/A')}"
        )
        log.info(
            f"Debug: First 3 lower bounds: {lower_bounds[:3] if hasattr(lower_bounds, '__getitem__') else 'N/A'}"
        )
        log.info(
            f"Debug: Last 3 lower bounds: {lower_bounds[-3:] if hasattr(lower_bounds, '__getitem__') else 'N/A'}"
        )
        log.info(f"Debug: First 3 popt: {popt[:3]}")
        log.info(f"Debug: Last 3 popt: {popt[-3:]}")

        for i in range(len(popt)):
            original_value = popt[i]

            if original_value < lower_bounds[i] or original_value > upper_bounds[i]:
                popt[i] = np.clip(popt[i], lower_bounds[i], upper_bounds[i])
                bounds_violated = True

                # BUG-6: Use effective_per_angle_scaling (post anti-degeneracy)
                # not per_angle_scaling (original config), to match actual param layout.
                if effective_per_angle_scaling:
                    n_angles = residual_fn.n_phi
                    n_scaling = 2 * n_angles
                    if i < n_angles:
                        param_name = f"contrast_angle_{i}"
                    elif i < n_scaling:
                        param_name = f"offset_angle_{i - n_angles}"
                    else:
                        param_idx = i - n_scaling
                        param_name = (
                            physical_param_names[param_idx]
                            if param_idx < len(physical_param_names)
                            else f"param_{i}"
                        )
                else:
                    param_name = (
                        physical_param_names[i]
                        if i < len(physical_param_names)
                        else f"param_{i}"
                    )

                log.warning(
                    f"Parameter '{param_name}' violated bounds: "
                    f"{original_value:.6e} not in [{lower_bounds[i]:.6e}, {upper_bounds[i]:.6e}]"
                )
                log.warning(f"    Clipped to: {popt[i]:.6e} (bounds enforced)")

        if bounds_violated:
            log.warning("=" * 80)
            log.warning("BOUNDS VIOLATION DETECTED")
            log.warning("=" * 80)
            log.warning("One or more parameters violated physical bounds.")
            log.warning("Parameters have been clipped to valid ranges.")
            log.warning("This may indicate:")
            log.warning(
                "  - Poor initial conditions (check config initial_parameters.values)"
            )
            log.warning("  - Insufficient constraints (consider constrained optimizer)")
            log.warning("  - Optimizer exploring unphysical parameter space")
            log.warning("=" * 80)

    # Compute final residuals first (needed for both cost and covariance scaling)
    final_residuals = residual_fn(popt)
    final_cost = float(np.sum(final_residuals**2))
    n_data = len(final_residuals)
    n_params = len(popt)

    # Effective parameter count for DOF in s^2 computation.
    if (
        ad_controller is not None
        and ad_controller.is_enabled
        and ad_controller.use_averaged_scaling
    ):
        n_params_effective = 2 * n_phi + n_physical
    else:
        n_params_effective = n_params

    # Use actual data point count, not padded length from StratifiedResidualFunction
    n_data_real = (
        residual_fn.n_total_points if hasattr(residual_fn, "n_total_points") else n_data
    )

    # Compute covariance matrix from Jacobian
    if "pcov" in result and result["pcov"] is not None:
        pcov = np.asarray(result["pcov"])
        log.info("Using covariance matrix from NLSQ result")
    else:
        log.info("Computing covariance matrix from Jacobian...")

        s2 = final_cost / max(n_data_real - n_params_effective, 1)

        # Jacobian space consistency check
        if hasattr(residual_fn, "n_params") and residual_fn.n_params != len(popt):
            log.warning(
                f"Parameter space mismatch: residual_fn expects {residual_fn.n_params} "
                f"params but popt has {len(popt)}. Skipping Jacobian-based covariance."
            )
            pcov = np.eye(len(popt)) * s2
        else:
            jac_fn = jax.jacfwd(residual_fn)
            J = jac_fn(popt)
            J = np.asarray(J)

            try:
                JTJ = J.T @ J
                pcov = np.linalg.inv(JTJ) * s2
            except np.linalg.LinAlgError:
                log.warning("Singular Jacobian, using pseudo-inverse for covariance")
                pcov = np.linalg.pinv(JTJ) * s2

        log.info(
            f"Covariance scaling: s^2={s2:.6e} (n_data={n_data_real}, "
            f"n_params_effective={n_params_effective})"
        )

    # Extract convergence information
    success = result.get("success", False)
    message = result.get("message", "Optimization completed")
    nfev = result.get("nfev", 0)
    nit = result.get("nit", 0)

    # Determine if optimization actually improved
    initial_residuals = residual_fn(initial_params)
    initial_cost = float(np.sum(initial_residuals**2))
    cost_reduction = (
        (initial_cost - final_cost) / initial_cost if initial_cost > 0 else 0
    )
    params_changed = not np.allclose(popt, initial_params, rtol=1e-8)

    # Log results
    log.info("=" * 80)
    log.info("OPTIMIZATION RESULTS")
    log.info(f"  Status: {'SUCCESS' if success else 'FAILED'}")
    log.info(f"  Message: {message}")
    log.info(f"  Function evaluations: {nfev}")
    log.info(f"  Iterations: {nit}")
    log.info(f"  Initial cost: {initial_cost:.6e}")
    log.info(f"  Final cost: {final_cost:.6e}")
    log.info(f"  Cost reduction: {cost_reduction * 100:+.2f}%")
    log.info(f"  Parameters changed: {params_changed}")
    log.info(f"  Total time: {time.perf_counter() - start_time:.2f}s")
    log.info("=" * 80)

    # Check for optimization failure
    if not params_changed and cost_reduction < 0.001:
        log.warning(
            "Optimization may have failed: parameters unchanged and cost unchanged\n"
            f"  Cost reduction: {cost_reduction * 100:.2f}%\n"
            "This may indicate a problem with gradient computation"
        )
    elif cost_reduction < 0.01:
        log.debug(
            f"Cost reduction < 1% ({cost_reduction * 100:.2f}%): "
            "initial parameters may already be near-optimal"
        )

    # =====================================================================
    # Anti-Degeneracy: Inverse Transformation (v2.14.0+, v2.18.0 update)
    # =====================================================================
    anti_degeneracy_info: dict[str, Any] = {}
    if ad_controller is not None and ad_controller.is_enabled:
        if ad_controller.use_fixed_scaling:
            if ad_controller.has_fixed_per_angle_scaling():
                fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
                assert fixed_scaling is not None
                fixed_contrast, fixed_offset = fixed_scaling

                log.info(
                    f"Expanding parameters from fixed_constant mode:\n"
                    f"  Physical params: {len(popt)}\n"
                    f"  Fixed contrast: mean={np.nanmean(fixed_contrast):.4f}\n"
                    f"  Fixed offset: mean={np.nanmean(fixed_offset):.4f}\n"
                    f"  Expanded: {2 * n_phi + n_physical}"
                )

                popt_expanded = np.concatenate(
                    [
                        fixed_contrast,
                        fixed_offset,
                        popt,
                    ]
                )

                pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
                pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov

                popt = popt_expanded
                pcov = pcov_expanded
                log.info(
                    f"Expanded to {len(popt)} parameters with fixed per-angle scaling"
                )
                anti_degeneracy_info["mode"] = "fixed_constant_quantile"
                anti_degeneracy_info["original_n_params"] = n_physical
                anti_degeneracy_info["expanded_n_params"] = len(popt)
                anti_degeneracy_info["fixed_contrast_mean"] = float(
                    np.nanmean(fixed_contrast)
                )
                anti_degeneracy_info["fixed_offset_mean"] = float(
                    np.nanmean(fixed_offset)
                )
            else:
                log.warning(
                    "Fixed constant mode but no fixed scaling available. "
                    "Unexpected state - results may be unreliable."
                )
                anti_degeneracy_info["mode"] = "fixed_constant_fallback"

        elif ad_controller.use_averaged_scaling:
            log.info(
                f"Expanding parameters from auto_averaged mode ({len(popt)} -> "
                f"{2 * n_phi + n_physical})"
            )
            popt_expanded = ad_controller.transform_params_from_constant(popt)

            pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
            pcov_expanded[:n_phi, :n_phi] = np.eye(n_phi) * pcov[0, 0]
            pcov_expanded[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = (
                np.eye(n_phi) * pcov[1, 1]
            )
            pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov[2:, 2:]
            pcov_expanded[2 * n_phi :, :n_phi] = np.tile(pcov[2:, 0:1], (1, n_phi))
            pcov_expanded[:n_phi, 2 * n_phi :] = np.tile(pcov[0:1, 2:], (n_phi, 1))
            pcov_expanded[2 * n_phi :, n_phi : 2 * n_phi] = np.tile(
                pcov[2:, 1:2], (1, n_phi)
            )
            pcov_expanded[n_phi : 2 * n_phi, 2 * n_phi :] = np.tile(
                pcov[1:2, 2:], (n_phi, 1)
            )

            popt = popt_expanded
            pcov = pcov_expanded
            log.info(f"Expanded to {len(popt)} per-angle parameters")
            anti_degeneracy_info["mode"] = "auto_averaged"
            anti_degeneracy_info["original_n_params"] = 2 + n_physical
            anti_degeneracy_info["expanded_n_params"] = len(popt)

        elif ad_controller.use_fourier:
            log.info(
                f"Expanding parameters from Fourier mode ({len(popt)} -> "
                f"{2 * n_phi + n_physical})"
            )
            popt_expanded = ad_controller.transform_params_from_fourier(popt)

            assert ad_controller.fourier is not None
            n_coeffs = ad_controller.fourier.n_coeffs_per_param
            B_contrast = ad_controller.fourier.get_basis_matrix()  # (n_phi, n_coeffs)
            B_offset = ad_controller.fourier.get_basis_matrix()
            assert B_contrast is not None and B_offset is not None

            pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
            pcov_contrast = pcov[:n_coeffs, :n_coeffs]
            pcov_expanded[:n_phi, :n_phi] = B_contrast @ pcov_contrast @ B_contrast.T
            pcov_offset = pcov[n_coeffs : 2 * n_coeffs, n_coeffs : 2 * n_coeffs]
            pcov_expanded[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = (
                B_offset @ pcov_offset @ B_offset.T
            )
            pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov[
                2 * n_coeffs :, 2 * n_coeffs :
            ]
            pcov_expanded[2 * n_phi :, :n_phi] = (
                pcov[2 * n_coeffs :, :n_coeffs] @ B_contrast.T
            )
            pcov_expanded[:n_phi, 2 * n_phi :] = (
                B_contrast @ pcov[:n_coeffs, 2 * n_coeffs :]
            )
            pcov_expanded[2 * n_phi :, n_phi : 2 * n_phi] = (
                pcov[2 * n_coeffs :, n_coeffs : 2 * n_coeffs] @ B_offset.T
            )
            pcov_expanded[n_phi : 2 * n_phi, 2 * n_phi :] = (
                B_offset @ pcov[n_coeffs : 2 * n_coeffs, 2 * n_coeffs :]
            )

            popt = popt_expanded
            pcov = pcov_expanded
            log.info(f"Expanded to {len(popt)} per-angle parameters")
            anti_degeneracy_info["mode"] = "fourier"
            anti_degeneracy_info["fourier_order"] = ad_controller.fourier.order
            anti_degeneracy_info["original_n_params"] = 2 * n_coeffs + n_physical
            anti_degeneracy_info["expanded_n_params"] = len(popt)

        # Add diagnostics to info
        anti_degeneracy_info["controller_diagnostics"] = ad_controller.get_diagnostics()

    # Prepare info dict
    info = {
        "success": success,
        "message": message,
        "nfev": nfev,
        "nit": nit,
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "cost_reduction": cost_reduction,
        "optimization_time": optimization_time,
        "method": "stratified_least_squares",
    }
    if anti_degeneracy_info:
        info["anti_degeneracy"] = anti_degeneracy_info

    # Check for shear collapse in laminar_flow mode
    is_laminar_flow = "gamma_dot_t0" in physical_param_names
    if is_laminar_flow:
        # BUG-4: Use actual unique phi count from stratified data, not .chunks
        n_phi_check = (
            len(set(stratified_data.phi_flat.tolist()))
            if hasattr(stratified_data, "phi_flat")
            else 1
        )
        n_phi = n_phi_check
        if len(popt) > 2 * n_phi + 3:
            gamma_dot_t0_idx = 2 * n_phi + 3
            gamma_dot_t0_value = popt[gamma_dot_t0_idx]
            if abs(gamma_dot_t0_value) < 1e-5:
                log.warning("=" * 80)
                log.warning("SHEAR COLLAPSE WARNING")
                log.warning(
                    f"gamma_dot_t0 = {gamma_dot_t0_value:.2e} s^-1 is effectively zero"
                )
                log.warning(
                    "The model has effectively collapsed to static_isotropic mode."
                )
                log.warning(
                    "RECOMMENDED: Use phi_filtering for angles near 0 and 90 deg"
                )
                log.warning("=" * 80)
                info["shear_collapse_warning"] = {
                    "gamma_dot_t0": float(gamma_dot_t0_value),
                    "threshold": 1e-5,
                    "message": "Shear contribution effectively zero",
                }

    return popt, pcov, info
