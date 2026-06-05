"""Fit Computation Utilities for NLSQ Results.

This module provides functions for computing theoretical fits from NLSQ
optimization results. Extracted from cli/commands.py for better organization.

Extracted from cli/commands.py as part of refactoring (Dec 2025).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core.jax_backend import compute_g2_scaled
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


# Performance Optimization (Spec 006 - FR-007, FR-007a): Vectorized computation
def compute_g2_batch(
    physical_params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_angles: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> jnp.ndarray:
    """Compute g2 for all phi angles in a single vectorized operation.

    Performance Optimization (Spec 006 - FR-007):
    Uses jax.vmap to compute g2 for all angles in parallel instead of
    sequential Python loop. Expected speedup: 10-20x for post-fitting.

    Parameters
    ----------
    physical_params : jnp.ndarray
        Physical parameters array
    t1 : jnp.ndarray
        t1 time values, shape (n_t1,)
    t2 : jnp.ndarray
        t2 time values, shape (n_t2,)
    phi_angles : jnp.ndarray
        Phi angles in radians, shape (n_phi,)
    q : float
        Wave vector magnitude
    L : float
        Sample-to-detector distance
    dt : float
        Time step
    contrast : float
        Contrast parameter (default 1.0 for raw computation)
    offset : float
        Offset parameter (default 1.0 for raw computation)

    Returns
    -------
    jnp.ndarray
        g2 values, shape (n_phi, n_t1, n_t2)
    """
    n_t1 = len(t1)
    n_t2 = len(t2)

    # Define single-angle computation
    def compute_single_angle(phi_val):
        g2 = compute_g2_scaled(
            params=physical_params,
            t1=t1,
            t2=t2,
            phi=jnp.array([phi_val]),
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )
        # Reshape to ensure consistent (n_t1, n_t2) output
        # compute_g2_scaled may return different shapes, so flatten and reshape
        return g2.reshape(n_t1, n_t2)

    # Note: vmap wrapper is recreated per call since the closure captures varying params.
    # This is acceptable for post-processing (not in optimization hot path).
    compute_all_angles = jax.vmap(compute_single_angle)
    return compute_all_angles(phi_angles)


def compute_g2_batch_with_per_angle_scaling(
    physical_params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_angles: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
    contrasts: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """Compute g2 with per-angle contrast/offset in single vectorized operation.

    Performance Optimization (Spec 006 - FR-007a):
    Extends compute_g2_batch for per-angle scaling parameters.

    Parameters
    ----------
    physical_params : jnp.ndarray
        Physical parameters array
    t1, t2 : jnp.ndarray
        Time values
    phi_angles : jnp.ndarray
        Phi angles in radians, shape (n_phi,)
    q, L, dt : float
        Experimental parameters
    contrasts : jnp.ndarray
        Per-angle contrasts, shape (n_phi,)
    offsets : jnp.ndarray
        Per-angle offsets, shape (n_phi,)

    Returns
    -------
    jnp.ndarray
        g2 values with scaling applied, shape (n_phi, n_t1, n_t2)
    """
    n_t1 = len(t1)
    n_t2 = len(t2)

    def compute_single_angle_scaled(phi_val, contrast_val, offset_val):
        g2 = compute_g2_scaled(
            params=physical_params,
            t1=t1,
            t2=t2,
            phi=jnp.array([phi_val]),
            q=q,
            L=L,
            contrast=contrast_val,
            offset=offset_val,
            dt=dt,
        )
        # Reshape to ensure consistent (n_t1, n_t2) output
        return g2.reshape(n_t1, n_t2)

    # Note: vmap wrapper is recreated per call since the closure captures varying params.
    # This is acceptable for post-processing (not in optimization hot path).
    compute_all_angles = jax.vmap(compute_single_angle_scaled, in_axes=(0, 0, 0))
    return compute_all_angles(phi_angles, contrasts, offsets)


def solve_lstsq_batch(
    theory_batch: jnp.ndarray,
    exp_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch least squares solving for all angles.

    Performance Optimization (Spec 006 - FR-008):
    Vectorized least squares using jax.vmap for all angles simultaneously.

    Parameters
    ----------
    theory_batch : jnp.ndarray
        Theory values flattened, shape (n_phi, n_t1 * n_t2)
    exp_batch : jnp.ndarray
        Experimental values flattened, shape (n_phi, n_t1 * n_t2)

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (contrasts, offsets) each shape (n_phi,)
    """

    def solve_single(theory_flat, exp_flat):
        A = jnp.column_stack([theory_flat, jnp.ones_like(theory_flat)])
        solution, _, _, _ = jnp.linalg.lstsq(A, exp_flat, rcond=None)
        return solution[0], solution[1]  # contrast, offset

    solve_all = jax.vmap(solve_single, in_axes=(0, 0))
    contrasts, offsets = solve_all(theory_batch, exp_batch)
    return contrasts, offsets


def normalize_analysis_mode(
    mode: str | None,
    n_params: int,
    n_angles: int,
) -> str:
    """Resolve the analysis mode, inferring from parameter counts if needed.

    Parameters
    ----------
    mode : str | None
        Explicit mode, or None to infer.
    n_params : int
        Number of parameters.
    n_angles : int
        Number of angles.

    Returns
    -------
    str
        Normalized mode: ``"static_anisotropic"``, ``"static_isotropic"``,
        or ``"laminar_flow"``.

    Notes
    -----
    The 3-physical-parameter signature is shared by ``static_isotropic`` and
    ``static_anisotropic``; these cannot be distinguished from the parameter
    count alone, so the angle-resolved ``static_anisotropic`` is the default.
    """
    if mode:
        mode_lower = mode.lower()
        if mode_lower == "static_isotropic":
            return "static_isotropic"
        if mode_lower == "static_anisotropic":
            return "static_anisotropic"
        if mode_lower == "laminar_flow":
            return "laminar_flow"

    # Infer from parameter counts (legacy scalar vs per-angle layout).
    # The 3-physical-param signature is shared by static_isotropic and
    # static_anisotropic; we cannot distinguish them from parameter count
    # alone, so we return the angle-resolved variant as the default.
    candidates = {
        "static_anisotropic": 3,
        "laminar_flow": 7,
    }
    for candidate_mode, n_phys in candidates.items():
        if n_params in {n_phys + 2, 2 * n_angles + n_phys}:
            return candidate_mode

    logger.debug(
        "Unable to infer analysis_mode from params=%s angles=%s; defaulting to static_anisotropic",
        n_params,
        n_angles,
    )
    return "static_anisotropic"


def get_physical_param_count(analysis_mode: AnalysisMode) -> int:
    """Get the number of physical parameters for an analysis mode.

    Parameters
    ----------
    analysis_mode : AnalysisMode
        One of ``"static_anisotropic"``, ``"static_isotropic"``, or
        ``"laminar_flow"``.

    Returns
    -------
    int
        Number of physical parameters (3 for static modes, 7 for
        ``laminar_flow``).

    Raises
    ------
    ValueError
        If the mode is unknown.
    """
    if analysis_mode in ("static_anisotropic", "static_isotropic"):
        return 3  # D0, alpha, D_offset
    elif analysis_mode == "laminar_flow":
        return 7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
    else:
        raise ValueError(
            f"Unknown analysis_mode: '{analysis_mode}'. "
            "Expected 'static_anisotropic', 'static_isotropic', or 'laminar_flow'"
        )


def extract_parameters_from_result(
    parameters: np.ndarray,
    n_angles: int,
    analysis_mode: AnalysisMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Extract contrast, offset, and physical parameters from result.

    Handles both per-angle and scalar parameter layouts.

    Parameters
    ----------
    parameters : np.ndarray
        Full parameter array from optimization.
    n_angles : int
        Number of phi angles.
    analysis_mode : AnalysisMode
        One of ``"static_anisotropic"``, ``"static_isotropic"``, or
        ``"laminar_flow"``.

    Returns
    -------
    tuple
        ``(contrasts, offsets, physical_params, scalar_expansion_used)``. When
        the solver returned scalar contrast/offset, they are broadcast across
        ``n_angles`` and ``scalar_expansion_used`` is ``True``.

    Raises
    ------
    ValueError
        If the parameter count matches neither the per-angle layout
        (``2 * n_angles + n_physical``) nor the scalar layout
        (``n_physical + 2``).
    """
    n_params = len(parameters)
    n_physical = get_physical_param_count(analysis_mode)
    expected_per_angle = 2 * n_angles + n_physical

    scalar_expansion = False

    if n_params == expected_per_angle:
        # Per-angle layout: [contrast_0, ..., contrast_N, offset_0, ..., offset_N, physical...]
        contrasts = parameters[0:n_angles]
        offsets = parameters[n_angles : 2 * n_angles]
        physical_params = parameters[2 * n_angles :]
    elif n_params == (n_physical + 2):
        # Scalar layout: [contrast, offset, physical...]
        logger.warning(
            "Solver returned scalar contrast/offset (parameter count %d). Expanding "
            "scalars across %d filtered angles for result saving.",
            n_params,
            n_angles,
        )
        scalar_expansion = True
        scalar_contrast = float(parameters[0])
        scalar_offset = float(parameters[1])
        contrasts = np.full(n_angles, scalar_contrast, dtype=float)
        offsets = np.full(n_angles, scalar_offset, dtype=float)
        physical_params = parameters[2:]
    else:
        raise ValueError(
            f"Parameter count mismatch! Expected {expected_per_angle} "
            f"(2x{n_angles} scaling + {n_physical} physical), got {n_params}. "
            f"Per-angle scaling is REQUIRED in v2.4.0+"
        )

    return contrasts, offsets, physical_params, scalar_expansion


def compute_theoretical_fits(
    result: Any,
    data: dict[str, Any],
    metadata: dict[str, Any],
    *,
    analysis_mode: AnalysisMode | None = None,
    include_solver_surface: bool = True,
) -> dict[str, Any]:
    """Compute theoretical fits with per-angle least squares scaling.

    Generates theoretical correlation functions using optimized parameters,
    then applies per-angle scaling (contrast, offset) via least-squares fitting
    to match experimental intensities.

    Parameters
    ----------
    result : Any
        NLSQ optimization result with physical parameters.
    data : dict[str, Any]
        Experimental data with ``phi_angles_list``, ``c2_exp``, ``t1``, ``t2``.
    metadata : dict[str, Any]
        Metadata with ``L``, ``dt``, ``q`` for theoretical computation.
    analysis_mode : AnalysisMode | None, optional
        Optional analysis-mode override.
    include_solver_surface : bool, optional
        Whether to include the solver surface in the output.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``"c2_theoretical_raw"`` : raw theoretical fits
          ``(n_angles, n_t1, n_t2)``.
        - ``"c2_theoretical_scaled"`` : scaled fits ``(n_angles, n_t1, n_t2)``.
        - ``"c2_solver_scaled"`` : solver surface (None if not requested).
        - ``"per_angle_scaling"`` : post-hoc lstsq scaling params
          ``(n_angles, 2)``.
        - ``"per_angle_scaling_solver"`` : original solver scaling params.
        - ``"residuals"`` : experiment minus scaled fit
          ``(n_angles, n_t1, n_t2)``.
        - ``"scalar_per_angle_expansion"`` : whether scalar expansion was used.

    Raises
    ------
    ValueError
        If ``dt`` or ``q`` is missing from metadata, or the parameter count is
        invalid.

    Notes
    -----
    The lstsq contrast/offset values may differ from the NLSQ-optimized values:
    lstsq re-fits scaling to the raw theory post-hoc, whereas the NLSQ values
    are jointly optimized with the physical parameters and are authoritative.
    """
    phi_angles = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])
    t1 = np.asarray(data["t1"])
    t2 = np.asarray(data["t2"])

    # Convert 2D meshgrids to 1D if needed
    if t1.ndim == 2:
        t1 = t1[:, 0]
    if t2.ndim == 2:
        t2 = t2[0, :]

    n_params = len(result.parameters)
    n_angles = len(phi_angles)

    # Normalize analysis mode
    normalized_mode = normalize_analysis_mode(
        analysis_mode or getattr(result, "analysis_mode", None),
        n_params,
        n_angles,
    )

    # Extract parameters
    fitted_contrasts, fitted_offsets, physical_params, scalar_expansion = (
        extract_parameters_from_result(result.parameters, n_angles, normalized_mode)
    )

    logger.info(
        f"Per-angle scaling: {n_angles} angles, using FITTED scaling parameters from NLSQ optimization"
    )
    logger.debug(
        f"Extracted fitted parameters - "
        f"contrasts: mean={np.nanmean(fitted_contrasts):.4f}, "
        f"offsets: mean={np.nanmean(fitted_offsets):.4f}"
    )

    # Extract metadata
    L = metadata["L"]
    dt_value = metadata.get("dt")
    if dt_value is not None:
        dt = float(dt_value)
    else:
        # dt is required for the J(t1,t2) numerical integration used by
        # compute_g2_scaled().  A wrong dt produces incorrect theory curves and
        # misleading post-fit visualisations.  Raise rather than silently fall
        # back to an arbitrary 0.1 s default.
        raise ValueError(
            "dt (frame exposure time) is required for compute_theoretical_fits() "
            "but was not found in metadata. Pass metadata with a valid 'dt' key."
        )
    q = metadata["q"]

    if q is None:
        raise ValueError("q (wavevector) is required but was not found")

    logger.info(
        f"Computing theoretical fits for {len(phi_angles)} angles using L={L:.1f} AA, q={q:.6f} AA^-1"
    )

    # Performance Optimization (Spec 006 - FR-007, FR-008):
    # Vectorized computation replaces sequential per-angle loop.
    # Expected speedup: 10-20x for post-fitting analysis.

    # Convert to JAX arrays
    t1_jax = jnp.array(t1)
    t2_jax = jnp.array(t2)
    phi_jax = jnp.array(phi_angles)
    params_jax = jnp.array(physical_params)

    # Compute RAW theory for ALL angles at once (FR-007)
    c2_theoretical_raw = compute_g2_batch(
        physical_params=params_jax,
        t1=t1_jax,
        t2=t2_jax,
        phi_angles=phi_jax,
        q=float(q),
        L=float(L),
        dt=float(dt),
        contrast=1.0,
        offset=1.0,
    )
    c2_theoretical_raw = np.asarray(c2_theoretical_raw)  # Shape: (n_angles, n_t1, n_t2)

    # Compute solver surface for ALL angles at once (FR-007a) if requested
    if include_solver_surface:
        c2_solver_surface = compute_g2_batch_with_per_angle_scaling(
            physical_params=params_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi_angles=phi_jax,
            q=float(q),
            L=float(L),
            dt=float(dt),
            contrasts=jnp.array(fitted_contrasts),
            offsets=jnp.array(fitted_offsets),
        )
        c2_solver_surface = np.asarray(c2_solver_surface)
    else:
        c2_solver_surface = None

    # Batch least-squares scaling (FR-008)
    # Flatten theory and exp for batch lstsq: shape (n_angles, n_t1 * n_t2)
    theory_batch_flat = jnp.array(c2_theoretical_raw.reshape(n_angles, -1))
    exp_batch_flat = jnp.array(c2_exp.reshape(n_angles, -1))

    # Solve all angles at once
    contrasts_lstsq, offsets_lstsq = solve_lstsq_batch(theory_batch_flat, exp_batch_flat)
    contrasts_lstsq = np.asarray(contrasts_lstsq)
    offsets_lstsq = np.asarray(offsets_lstsq)

    # Apply scaling: c2_scaled = contrast * c2_raw + offset
    # Broadcasting: (n_angles, 1, 1) * (n_angles, n_t1, n_t2) + (n_angles, 1, 1)
    c2_theoretical_fitted = (
        contrasts_lstsq[:, None, None] * c2_theoretical_raw + offsets_lstsq[:, None, None]
    )

    # Build per-angle scaling array
    per_angle_scaling = np.column_stack((contrasts_lstsq, offsets_lstsq))
    solver_scaling = np.column_stack((fitted_contrasts, fitted_offsets))

    # Log statistics
    logger.debug(
        f"Batch lstsq - contrasts: mean={np.nanmean(contrasts_lstsq):.4f}, "
        f"offsets: mean={np.nanmean(offsets_lstsq):.4f}"
    )
    logger.info(
        "Note: lstsq contrast/offset values may differ from NLSQ-optimized values. "
        "lstsq re-fits scaling to raw theory (contrast=1, offset=1) post-hoc; "
        "NLSQ values are authoritative as they are jointly optimized with physical parameters."
    )

    residuals = c2_exp - c2_theoretical_fitted

    logger.info(f"Computed theoretical fits for {len(phi_angles)} angles")

    return {
        "c2_theoretical_raw": c2_theoretical_raw,
        "c2_theoretical_scaled": c2_theoretical_fitted,
        "c2_solver_scaled": c2_solver_surface,
        "per_angle_scaling": per_angle_scaling,
        "per_angle_scaling_solver": solver_scaling,
        "residuals": residuals,
        "scalar_per_angle_expansion": scalar_expansion,
    }
