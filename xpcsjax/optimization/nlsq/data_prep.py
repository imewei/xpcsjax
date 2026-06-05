"""Data Preparation Utilities for NLSQ Optimization.

This module provides data preparation functions extracted from wrapper.py
to improve code organization and reduce complexity.

Extracted from wrapper.py as part of refactoring (Dec 2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PreparedData:
    """Container for prepared optimization data.

    Attributes
    ----------
    xdata : np.ndarray
        Flattened independent-variable data.
    ydata : np.ndarray
        Flattened dependent-variable data (observations).
    n_data : int
        Total number of data points.
    n_phi : int
        Number of unique phi angles.
    phi_unique : np.ndarray
        Unique phi angle values.
    """

    xdata: np.ndarray
    ydata: np.ndarray
    n_data: int
    n_phi: int
    phi_unique: np.ndarray


@dataclass
class ExpandedParameters:
    """Container for expanded per-angle parameters.

    Attributes
    ----------
    params : np.ndarray
        Expanded parameter array.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Expanded bounds tuple ``(lower, upper)``, or None.
    n_params : int
        Total number of parameters.
    n_physical : int
        Number of physical parameters.
    n_angles : int
        Number of angles.
    """

    params: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None
    n_params: int
    n_physical: int
    n_angles: int


def expand_per_angle_parameters(
    compact_params: np.ndarray,
    compact_bounds: tuple[np.ndarray, np.ndarray] | None,
    n_angles: int,
    n_physical: int,
    logger: Any = None,
) -> ExpandedParameters:
    """Expand compact parameters to per-angle format.

    When per_angle_scaling=True with N angles, parameters are structured as:
    - N contrast parameters (one per angle)
    - N offset parameters (one per angle)
    - n_physical physical parameters

    Input (compact): ``[contrast, offset, physical_params...]``
    Output (expanded):
    ``[c0, c1, ..., cN-1, o0, o1, ..., oN-1, physical_params...]``

    Parameters
    ----------
    compact_params : np.ndarray
        Compact parameter array (``n_physical + 2`` elements).
    compact_bounds : tuple[np.ndarray, np.ndarray] | None
        Compact bounds tuple, or None.
    n_angles : int
        Number of phi angles.
    n_physical : int
        Number of physical parameters.
    logger : Any, optional
        Optional logger for diagnostics.

    Returns
    -------
    ExpandedParameters
        Per-angle parameters and bounds.

    Raises
    ------
    ValueError
        If the parameter count does not match the expected
        ``n_physical + 2``.
    """
    expected_compact = n_physical + 2
    if len(compact_params) != expected_compact:
        raise ValueError(
            f"Parameter count mismatch for per-angle scaling: "
            f"got {len(compact_params)}, expected {expected_compact} "
            f"({n_physical} physical + 2 scaling). "
            f"For {n_angles} angles, will expand to {n_physical + 2 * n_angles} parameters."
        )

    # Extract base scaling and physical parameters
    # compact_params ordering: [contrast, offset, physical_params...]
    base_contrast = compact_params[0]
    base_offset = compact_params[1]
    physical_params = compact_params[2:]

    if logger:
        logger.info("Expanding scaling parameters for per-angle scaling:")
        logger.info(f"  Angles: {n_angles}")
        logger.info(f"  Physical parameters: {n_physical}")
        logger.info(f"  Base scaling: contrast={base_contrast:.4f}, offset={base_offset:.4f}")

    # Expand to per-angle
    contrast_per_angle = np.full(n_angles, base_contrast)
    offset_per_angle = np.full(n_angles, base_offset)

    # Concatenate: [contrast_per_angle, offset_per_angle, physical_params]
    expanded_params = np.concatenate([contrast_per_angle, offset_per_angle, physical_params])

    # Expand bounds if provided
    expanded_bounds = None
    if compact_bounds is not None:
        lower, upper = compact_bounds
        lower_contrast = lower[0]
        upper_contrast = upper[0]
        lower_offset = lower[1]
        upper_offset = upper[1]
        lower_physical = lower[2:]
        upper_physical = upper[2:]

        expanded_lower = np.concatenate(
            [
                np.full(n_angles, lower_contrast),
                np.full(n_angles, lower_offset),
                lower_physical,
            ]
        )
        expanded_upper = np.concatenate(
            [
                np.full(n_angles, upper_contrast),
                np.full(n_angles, upper_offset),
                upper_physical,
            ]
        )
        expanded_bounds = (expanded_lower, expanded_upper)

        if logger:
            logger.info(f"  Bounds expanded to {len(expanded_lower)} parameters")

    if logger:
        logger.info(f"  Expanded to {len(expanded_params)} parameters:")
        logger.info(f"    - Contrast per angle: {n_angles} (indices 0 to {n_angles - 1})")
        logger.info(
            f"    - Offset per angle: {n_angles} (indices {n_angles} to {2 * n_angles - 1})"
        )
        logger.info(
            f"    - Physical: {n_physical} (indices {2 * n_angles} to {2 * n_angles + n_physical - 1})"
        )

    return ExpandedParameters(
        params=expanded_params,
        bounds=expanded_bounds,
        n_params=len(expanded_params),
        n_physical=n_physical,
        n_angles=n_angles,
    )


def validate_bounds(
    bounds: tuple[np.ndarray, np.ndarray] | None,
    n_params: int,
    logger: Any = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate parameter bounds.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray] | None
        Bounds tuple ``(lower, upper)``, or None.
    n_params : int
        Expected number of parameters.
    logger : Any, optional
        Optional logger for diagnostics.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Validated bounds (as float64 arrays), or None.

    Raises
    ------
    ValueError
        If the bounds dimensions mismatch ``n_params`` or if any
        ``lower > upper``. Equal bounds (fixed parameters) are allowed.
    """
    if bounds is None:
        return None

    lower, upper = bounds

    if len(lower) != n_params or len(upper) != n_params:
        raise ValueError(
            f"Bounds dimension mismatch: expected {n_params}, "
            f"got lower={len(lower)}, upper={len(upper)}"
        )

    # Check for invalid bounds (lower > upper); equal bounds are allowed
    # for parameters that are fixed (lower == upper == fixed_value).
    invalid_mask = lower > upper
    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        raise ValueError(
            f"Invalid bounds at indices {invalid_indices}: "
            f"lower > upper. Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}"
        )

    return (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def validate_initial_params(
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any = None,
) -> np.ndarray:
    """Validate and clip initial parameters to bounds.

    Parameters
    ----------
    params : np.ndarray
        Initial parameter guess.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds, or None.
    logger : Any, optional
        Optional logger for diagnostics.

    Returns
    -------
    np.ndarray
        Validated parameters, clipped to bounds if needed.
    """
    params = np.asarray(params, dtype=float)

    if bounds is None:
        return params

    lower, upper = bounds
    clipped: np.ndarray = np.clip(params, lower, upper)

    if not np.allclose(params, clipped):
        n_clipped = np.sum(~np.isclose(params, clipped))
        if logger:
            logger.warning(f"Clipped {n_clipped} initial parameters to bounds")

    return clipped


def convert_bounds_to_nlsq_format(
    bounds: tuple[np.ndarray, np.ndarray] | tuple[list, list] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert bounds to NLSQ-compatible format.

    NLSQ expects bounds as ``(lower_array, upper_array)`` with float64 dtype.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray] | tuple[list, list] | None
        Input bounds in various formats.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Bounds as ``(lower, upper)`` float64 numpy arrays, or None.
    """
    if bounds is None:
        return None

    lower, upper = bounds

    # Convert to numpy arrays with float64 dtype
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    return (lower, upper)


def build_parameter_labels(
    per_angle_scaling: bool,
    n_phi: int,
    physical_param_names: list[str],
) -> list[str]:
    """Build human-readable parameter labels.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle scaling is enabled.
    n_phi : int
        Number of phi angles.
    physical_param_names : list[str]
        Names of physical parameters.

    Returns
    -------
    list[str]
        List of parameter labels.

    See Also
    --------
    xpcsjax.optimization.nlsq.parameter_utils.build_parameter_labels
        The single canonical implementation this function delegates to.
    """
    # Delegate to the single canonical implementation in parameter_utils. The two
    # copies previously diverged on the non-per-angle case (this one emitted
    # contrast/offset; parameter_utils omitted them), silently mislabeling
    # diagnostics depending on which copy a caller reached. Lazy import avoids a
    # module-load cycle.
    from xpcsjax.optimization.nlsq.parameter_utils import (
        build_parameter_labels as _canonical,
    )

    return _canonical(per_angle_scaling, n_phi, physical_param_names)


def classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    atol: float = 1e-9,
) -> list[str]:
    """Classify parameter status relative to bounds.

    Parameters
    ----------
    values : np.ndarray
        Parameter values.
    lower : np.ndarray | None
        Lower bounds, or None.
    upper : np.ndarray | None
        Upper bounds, or None.
    atol : float, optional
        Absolute tolerance for bound comparison.

    Returns
    -------
    list[str]
        Status for each parameter: ``"active"``, ``"at_lower_bound"``, or
        ``"at_upper_bound"``. All ``"active"`` when bounds are None.
    """
    if lower is None or upper is None:
        return ["active"] * len(values)

    statuses: list[str] = []
    for value, lo, hi in zip(values, lower, upper, strict=False):
        if np.isclose(value, lo, atol=atol * (1.0 + abs(lo)), rtol=0.0):
            statuses.append("at_lower_bound")
        elif np.isclose(value, hi, atol=atol * (1.0 + abs(hi)), rtol=0.0):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses
