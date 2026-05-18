"""Angle filtering utilities for homodyne XPCS analysis.

This module provides functions for filtering phi angles based on target ranges,
with support for wrap-around at the ±180° boundary. Used by CLI commands for
both optimization and plotting workflows.

Extracted from cli/commands.py for better modularity.
"""

from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_angle_to_symmetric_range(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize angle(s) to [-180°, 180°] range.

    The horizontal flow direction is defined as 0°. Angles are normalized
    to be symmetric around 0° in the range [-180°, 180°].

    Physical Interpretation
    -----------------------
    In XPCS experiments with flow, the flow direction is typically set as 0°
    (horizontal reference). Angles are measured relative to this reference.
    Normalizing to [-180°, 180°] provides a natural symmetric representation
    where positive angles are counterclockwise and negative angles are
    clockwise from the flow direction.

    Normalization Rules
    -------------------
    - If 180° < φ < 360°: φ_norm = φ - 360°
      (e.g., 210° → -150°)
    - If -360° < φ < -180°: φ_norm = φ + 360°
      (e.g., -210° → 150°)
    - If -180° ≤ φ ≤ 180°: φ_norm = φ (no change)

    Parameters
    ----------
    angle : float or np.ndarray
        Angle(s) in degrees. Can be scalar (float) or array (np.ndarray).

    Returns
    -------
    float or np.ndarray
        Normalized angle(s) in range [-180°, 180°]. Returns scalar if input
        is scalar, array if input is array.

    Examples
    --------
    >>> normalize_angle_to_symmetric_range(210.0)
    -150.0
    >>> normalize_angle_to_symmetric_range(-210.0)
    150.0
    >>> normalize_angle_to_symmetric_range(np.array([0, 90, 210, -210]))
    array([  0.,  90., -150.,  150.])
    >>> normalize_angle_to_symmetric_range(np.array([180, -180, 360]))
    array([180., -180.,   0.])
    """
    angle_array = np.asarray(angle)
    normalized = angle_array % 360
    normalized = np.where(normalized > 180, normalized - 360, normalized)

    # Return scalar if input was scalar
    if np.isscalar(angle):
        return float(normalized)
    return normalized


def angle_in_range(angle: float, min_angle: float, max_angle: float) -> bool:
    """Check if angle is in range, accounting for wrap-around at ±180°.

    This function handles both normal ranges (where min_angle ≤ max_angle)
    and wrapped ranges that span the ±180° boundary (where min_angle > max_angle).

    Wrapped Range Logic
    -------------------
    When a range spans the ±180° boundary after normalization, the comparison
    logic changes:
    - Normal range [85°, 95°]: 85° ≤ angle ≤ 95°
    - Wrapped range [170°, -170°]: angle ≥ 170° OR angle ≤ -170°

    Example: User specifies range [170°, 190°]
    - After normalization: [170°, -170°] (min > max, wrapped)
    - Angle 175 matches: 175 >= 170 (yes)
    - Angle -175 matches: -175 <= -170 (yes)
    - Angle 0 does not match: 0 < 170 and 0 > -170 (no)

    Parameters
    ----------
    angle : float
        Angle to check (should be normalized to [-180°, 180°])
    min_angle : float
        Minimum angle of range (normalized to [-180°, 180°])
    max_angle : float
        Maximum angle of range (normalized to [-180°, 180°])

    Returns
    -------
    bool
        True if angle is in range, False otherwise

    Examples
    --------
    >>> angle_in_range(90.0, 85.0, 95.0)  # Normal range
    True
    >>> angle_in_range(175.0, 170.0, -170.0)  # Wrapped range
    True
    >>> angle_in_range(-175.0, 170.0, -170.0)  # Wrapped range
    True
    >>> angle_in_range(0.0, 170.0, -170.0)  # Outside wrapped range
    False
    """
    if min_angle <= max_angle:
        # Normal range (doesn't span ±180° boundary)
        return min_angle <= angle <= max_angle
    else:
        # Wrapped range (spans ±180° boundary)
        # Angle matches if it's >= min_angle OR <= max_angle
        return angle >= min_angle or angle <= max_angle


def apply_angle_filtering(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Core angle filtering logic shared by optimization and plotting.

    Filters phi angles and corresponding C2 data based on target_ranges
    specified in configuration. Uses OR logic across ranges: an angle is
    selected if it falls within ANY of the specified ranges.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles in degrees, shape (n_phi,)
    c2_exp : np.ndarray
        Experimental correlation data, shape (n_phi, n_t1, n_t2)
    config : dict
        Configuration dictionary with phi_filtering section

    Returns
    -------
    filtered_indices : list of int
        Indices of angles that matched target ranges
    filtered_phi_angles : np.ndarray
        Filtered phi angles array, shape (n_matched,)
    filtered_c2_exp : np.ndarray
        Filtered C2 data array, shape (n_matched, n_t1, n_t2)

    Notes
    -----
    - Returns all angles (unfiltered) if phi_filtering.enabled is False
    - Returns all angles with warning if no target_ranges specified
    - Returns all angles with warning if no angles match target ranges
    - Angle matching uses wrap-aware range checking (handles ±180° boundary)
    - Normal range [85°, 95°]: 85° ≤ angle ≤ 95°
    - Wrapped range [170°, -170°]: angle ≥ 170° OR angle ≤ -170°
    - Angles matching multiple ranges are only included once
    """
    # Get phi_filtering configuration
    phi_filtering_config = config.get("phi_filtering", {})

    if not phi_filtering_config.get("enabled", False):
        # Filtering disabled - return all angles
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Get target ranges
    target_ranges = phi_filtering_config.get("target_ranges", [])
    if not target_ranges:
        # No ranges specified - return all angles with warning
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Filter angles based on target ranges (OR logic)
    # Uses wrap-aware range checking to handle ranges spanning ±180° boundary
    filtered_indices = []
    for i, angle in enumerate(phi_angles):
        for range_spec in target_ranges:
            min_angle = range_spec.get("min_angle", -180.0)
            max_angle = range_spec.get("max_angle", 180.0)
            if angle_in_range(angle, min_angle, max_angle):
                filtered_indices.append(i)
                break  # Angle matches this range, no need to check other ranges

    if not filtered_indices:
        # No matches - return all angles with warning
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Apply filtering
    # Convert list to numpy array for JAX compatibility
    # (JAX arrays don't accept Python list indexing)
    filtered_indices_array = np.array(filtered_indices)
    filtered_phi_angles = phi_angles[filtered_indices_array]
    filtered_c2_exp = c2_exp[filtered_indices_array]

    return filtered_indices, filtered_phi_angles, filtered_c2_exp


def apply_angle_filtering_for_optimization(
    data: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """Apply angle filtering to data before optimization.

    This function filters phi angles and corresponding C2 data based on the
    phi_filtering configuration before passing data to optimization methods
    (NLSQ or MCMC). It creates a filtered copy of the data dictionary while
    preserving all other keys unchanged.

    Parameters
    ----------
    data : dict
        Full data dictionary with all angles, containing keys:
        - phi_angles_list: np.ndarray of phi angles (n_phi,)
        - c2_exp: np.ndarray of correlation data (n_phi, n_t1, n_t2)
        - wavevector_q_list: np.ndarray (preserved unchanged)
        - t1: np.ndarray (preserved unchanged)
        - t2: np.ndarray (preserved unchanged)
    config : ConfigManager or dict
        Configuration manager with phi_filtering settings

    Returns
    -------
    dict
        Filtered data dictionary with same structure as input but with:
        - phi_angles_list: Filtered to selected angles only
        - c2_exp: First dimension sliced to match selected angles
        - All other keys: Unchanged from input

    Notes
    -----
    Edge Case Handling:
    - If phi_filtering.enabled is False: Returns unfiltered data (DEBUG log)
    - If target_ranges is empty: Returns unfiltered data (WARNING log)
    - If no angles match: Returns unfiltered data (WARNING log)
    """
    import time

    start_time = time.perf_counter()

    # Extract required arrays
    phi_angles = np.asarray(data.get("phi_angles_list", []))
    c2_exp = np.asarray(data.get("c2_exp", []))

    if len(phi_angles) == 0 or len(c2_exp) == 0:
        logger.warning("No phi angles or C2 data available, cannot apply filtering")
        return data

    # Validate angles are in reasonable range (data quality check)
    angles_too_large = phi_angles[np.abs(phi_angles) > 360]
    if len(angles_too_large) > 0:
        logger.warning(
            f"Found {len(angles_too_large)} angle(s) with |phi| > 360 deg: {angles_too_large}. "
            f"This may indicate data loading issues, unit confusion (radians vs degrees), "
            f"or instrument malfunction. Angles will be normalized to [-180 deg, 180 deg] range.",
        )

    # Normalize phi angles to [-180°, 180°] range (flow direction at 0°)
    original_phi_angles = phi_angles.copy()
    normalized_result = normalize_angle_to_symmetric_range(phi_angles)
    # Cast to ndarray since we pass ndarray input (function returns float | ndarray)
    phi_angles = np.asarray(normalized_result)
    logger.info(
        "Normalized phi angles to [-180, 180] deg range (flow direction at 0 deg)"
    )
    logger.debug(f"Original angles: {original_phi_angles}")
    logger.debug(f"Normalized angles: {phi_angles}")

    # Get config dict (handle both ConfigManager and dict types)
    config_dict = config.get_config() if hasattr(config, "get_config") else config

    # Check if filtering is enabled
    phi_filtering_config = config_dict.get("phi_filtering", {})
    if not phi_filtering_config.get("enabled", False):
        logger.debug("Phi filtering not enabled, using all angles for optimization")
        # Return data with normalized angles even when filtering disabled
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Check for target_ranges
    target_ranges = phi_filtering_config.get("target_ranges", [])
    if not target_ranges:
        logger.warning(
            "Phi filtering enabled but no target_ranges specified, using all angles",
        )
        # Return data with normalized angles
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Normalize target_ranges to [-180°, 180°] for consistency
    normalized_ranges = []
    for range_spec in target_ranges:
        min_angle = range_spec.get("min_angle", -180)
        max_angle = range_spec.get("max_angle", 180)
        normalized_min = normalize_angle_to_symmetric_range(min_angle)
        normalized_max = normalize_angle_to_symmetric_range(max_angle)
        normalized_ranges.append(
            {
                "min_angle": normalized_min,
                "max_angle": normalized_max,
                "description": range_spec.get("description", ""),
            },
        )
        logger.debug(
            f"Normalized range [{min_angle} deg, {max_angle} deg] -> [{normalized_min} deg, {normalized_max} deg]"
        )

    # Apply filtering with normalized angles and ranges
    normalized_config = config_dict.copy()
    normalized_config["phi_filtering"] = phi_filtering_config.copy()
    normalized_config["phi_filtering"]["target_ranges"] = normalized_ranges

    filtered_indices, filtered_phi_angles, filtered_c2_exp = apply_angle_filtering(
        phi_angles, c2_exp, normalized_config
    )

    # Check if any angles were filtered
    if len(filtered_indices) == 0 or len(filtered_indices) == len(phi_angles):
        if len(filtered_indices) == 0:
            logger.warning("No angles matched phi_filtering criteria, using all angles")
            # Return data with normalized angles
            normalized_data = data.copy()
            normalized_data["phi_angles_list"] = phi_angles
            return normalized_data
        # All angles matched - no filtering needed
        pass

    # Create filtered data dictionary
    filtered_data = data.copy()
    filtered_data["phi_angles_list"] = filtered_phi_angles
    filtered_data["c2_exp"] = filtered_c2_exp

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Angle filtering completed in {elapsed_ms:.3f}ms")
    logger.info(
        f"Angle filtering for optimization: {len(filtered_indices)} angles selected "
        f"from {len(phi_angles)} total angles"
    )
    logger.info(f"Selected angles: {filtered_phi_angles.tolist()}")

    return filtered_data


def apply_angle_filtering_for_plot(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering to select specific angles for plotting.

    This is a wrapper around apply_angle_filtering() that extracts the
    configuration from the data dictionary and adds plot-specific logging.

    This filters the loaded data (which contains ALL angles) to show only
    the angles specified in phi_filtering configuration.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles in degrees
    c2_exp : np.ndarray
        Experimental correlation data
    data : dict
        Data dictionary containing 'config' key with phi_filtering settings

    Returns
    -------
    tuple
        (filtered_indices, filtered_phi_angles, filtered_c2_exp)

    Notes
    -----
    Uses shared apply_angle_filtering() function for consistent filtering
    logic with optimization workflow.
    """
    # Check if filtering config is available in data dict
    config = data.get("config", None)
    if config is None:
        # No config available - plot all angles
        logger.debug("No config available for angle filtering, plotting all angles")
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Call shared filtering function
    filtered_indices, filtered_phi_angles, filtered_c2_exp = apply_angle_filtering(
        phi_angles,
        c2_exp,
        config,
    )

    # Add plot-specific logging
    phi_filtering_config = config.get("phi_filtering", {})

    if not phi_filtering_config.get("enabled", False):
        logger.debug("Phi filtering not enabled, plotting all angles")
    elif not phi_filtering_config.get("target_ranges", []):
        logger.warning(
            "Phi filtering enabled but no target_ranges specified, plotting all angles",
        )
    elif not filtered_indices or len(filtered_indices) == len(phi_angles):
        if len(filtered_indices) == 0:
            logger.warning("No angles matched target ranges, plotting all angles")
        # else: all angles matched, no special logging needed
    else:
        logger.info(
            f"Angle filtering applied: {len(filtered_indices)} angles selected "
            f"from {len(phi_angles)} total angles",
        )

    return filtered_indices, filtered_phi_angles, filtered_c2_exp
