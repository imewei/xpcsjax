"""Parameter transformation utilities for NLSQ optimization.

This module extracts shear transform logic from nlsq_wrapper.py
to reduce file size and improve maintainability.

Extracted from nlsq_wrapper.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Parameter name aliases for backwards compatibility
PARAMETER_NAME_ALIASES = {
    # Legacy names -> canonical names
    "gamma_dot_0": "gamma_dot_t0",
    "gamma_dot_t_0": "gamma_dot_t0",
    "gamma_dot_offset": "gamma_dot_t_offset",
    "phi_0": "phi0",
}

# Default shear parameter scales for laminar flow mode
DEFAULT_SHEAR_X_SCALE = {
    "gamma_dot_t0": 524.0,
    "beta": 4.0,
    "gamma_dot_t_offset": 771.0,
}


def normalize_param_key(name: str | None) -> str:
    """Normalize parameter name using canonical aliases.

    Parameters
    ----------
    name : str | None
        Parameter name to normalize.

    Returns
    -------
    str
        Canonical parameter name.
    """
    if not name:
        return ""
    key = str(name).strip()
    return PARAMETER_NAME_ALIASES.get(key, key)


def normalize_x_scale_map(raw_map: Any) -> dict[str, float]:
    """Normalize parameter scaling map.

    Parameters
    ----------
    raw_map : Any
        Raw scaling map (dict or other).

    Returns
    -------
    dict[str, float]
        Normalized scaling map with canonical keys.
    """
    if not isinstance(raw_map, dict):
        return {}
    normalized: dict[str, float] = {}
    for raw_key, raw_value in raw_map.items():
        key = normalize_param_key(raw_key)
        if not key:
            continue
        try:
            normalized[key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return normalized


def build_per_parameter_x_scale(
    per_angle_scaling: bool,
    n_angles: int,
    physical_param_names: list[str],
    analysis_mode: str,
    override_map: dict[str, float],
) -> np.ndarray | None:
    """Build per-parameter scale array for optimization.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle scaling is enabled.
    n_angles : int
        Number of phi angles.
    physical_param_names : list[str]
        List of physical parameter names.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").
    override_map : dict[str, float]
        User overrides for parameter scales.

    Returns
    -------
    np.ndarray | None
        Scale array or None if all scales are 1.0.
    """
    effective_physical: dict[str, float] = dict.fromkeys(physical_param_names, 1.0)
    if analysis_mode == "laminar_flow":
        for alias_key, scale in DEFAULT_SHEAR_X_SCALE.items():
            canonical = normalize_param_key(alias_key)
            if canonical in effective_physical:
                effective_physical[canonical] = scale
    for key, value in override_map.items():
        canonical = normalize_param_key(key)
        if canonical in effective_physical:
            effective_physical[canonical] = value

    contrast_scale = float(override_map.get("contrast", 1.0))
    offset_scale = float(override_map.get("offset", 1.0))

    has_nonunity = (
        any(abs(scale - 1.0) > 1e-12 for scale in effective_physical.values())
        or abs(contrast_scale - 1.0) > 1e-12
        or abs(offset_scale - 1.0) > 1e-12
    )
    if not has_nonunity:
        return None

    scales: list[float] = []
    if per_angle_scaling:
        if n_angles <= 0:
            return None
        scales.extend([contrast_scale] * n_angles)
        scales.extend([offset_scale] * n_angles)
    else:
        scales.extend([contrast_scale, offset_scale])

    for name in physical_param_names:
        scales.append(effective_physical.get(name, 1.0))

    return np.asarray(scales, dtype=float)


def format_x_scale_for_log(value: Any) -> str:
    """Format x_scale value for logging.

    Parameters
    ----------
    value : Any
        Scale value to format.

    Returns
    -------
    str
        Formatted string.
    """
    if isinstance(value, np.ndarray):
        return f"array(len={value.size})"
    return str(value)


def parse_shear_transform_config(config: Any | None) -> dict[str, Any]:
    """Parse shear transform configuration.

    Parameters
    ----------
    config : Any | None
        Configuration dict or None.

    Returns
    -------
    dict[str, Any]
        Parsed configuration with defaults.
    """
    if not isinstance(config, dict):
        return {
            "enable_gamma_dot_log": False,
            "enable_beta_centering": False,
            "beta_reference": 0.0,
        }
    return {
        "enable_gamma_dot_log": bool(config.get("enable_gamma_dot_log", False)),
        "enable_beta_centering": bool(config.get("enable_beta_centering", False)),
        "beta_reference": float(config.get("beta_reference", 0.0)),
    }


def build_physical_index_map(
    per_angle_scaling: bool,
    n_angles: int,
    physical_param_names: list[str],
) -> dict[str, int]:
    """Build mapping from parameter names to indices.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle scaling is enabled.
    n_angles : int
        Number of phi angles.
    physical_param_names : list[str]
        List of physical parameter names.

    Returns
    -------
    dict[str, int]
        Mapping from parameter name to index in parameter vector.
    """
    start = 2 * n_angles if per_angle_scaling else 2
    return {name: start + idx for idx, name in enumerate(physical_param_names)}


def apply_forward_shear_transforms_to_vector(
    params: np.ndarray,
    index_map: dict[str, int],
    transform_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply forward shear transforms to parameter vector.

    Transforms parameters from physical space to solver space:
    - gamma_dot_t0 -> log(gamma_dot_t0) if enable_gamma_dot_log
    - beta -> beta - beta_reference if enable_beta_centering

    Parameters
    ----------
    params : np.ndarray
        Parameter vector in physical space.
    index_map : dict[str, int]
        Mapping from parameter names to indices.
    transform_cfg : dict[str, Any]
        Transform configuration.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        (transformed_params, transform_state)
    """
    vector = np.asarray(params, dtype=float).copy()
    state: dict[str, Any] = {
        "gamma_log_idx": None,
        "beta_center_idx": None,
        "beta_reference": float(transform_cfg.get("beta_reference", 0.0)),
    }

    if transform_cfg.get("enable_gamma_dot_log", False):
        # Try canonical name first, fallback to old name for backwards compatibility
        idx = index_map.get("gamma_dot_t0") or index_map.get("gamma_dot_0")
        if idx is not None:
            value = vector[idx]
            if value <= 0:
                raise ValueError(
                    "gamma_dot_t0 must be > 0 when enable_gamma_dot_log is true"
                )
            vector[idx] = np.log(value)
            state["gamma_log_idx"] = idx

    if transform_cfg.get("enable_beta_centering", False):
        idx = index_map.get("beta")
        if idx is not None:
            vector[idx] = vector[idx] - state["beta_reference"]
            state["beta_center_idx"] = idx

    if state["gamma_log_idx"] is None and state["beta_center_idx"] is None:
        return np.asarray(params, dtype=float).copy(), {}

    return vector, state


def apply_forward_shear_transforms_to_bounds(
    bounds: tuple[np.ndarray, np.ndarray] | None,
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply forward shear transforms to parameter bounds.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray] | None
        (lower, upper) bounds in physical space.
    state : dict[str, Any]
        Transform state from apply_forward_shear_transforms_to_vector.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Transformed bounds or None.
    """
    if not bounds or not state:
        return bounds
    lower, upper = (
        np.asarray(bounds[0], dtype=float).copy(),
        np.asarray(bounds[1], dtype=float).copy(),
    )
    gamma_idx = state.get("gamma_log_idx")
    if gamma_idx is not None:
        if lower[gamma_idx] <= 0 or upper[gamma_idx] <= 0:
            raise ValueError(
                "gamma_dot_t0 bounds must be > 0 when enable_gamma_dot_log is true"
            )
        lower[gamma_idx] = np.log(lower[gamma_idx])
        upper[gamma_idx] = np.log(upper[gamma_idx])
    beta_idx = state.get("beta_center_idx")
    if beta_idx is not None:
        beta_ref = state.get("beta_reference", 0.0)
        lower[beta_idx] = lower[beta_idx] - beta_ref
        upper[beta_idx] = upper[beta_idx] - beta_ref
    return (lower, upper)


def apply_inverse_shear_transforms_to_vector(
    params: np.ndarray,
    state: dict[str, Any] | None,
) -> np.ndarray:
    """Apply inverse shear transforms to parameter vector.

    Transforms parameters from solver space back to physical space.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector in solver space.
    state : dict[str, Any] | None
        Transform state from apply_forward_shear_transforms_to_vector.

    Returns
    -------
    np.ndarray
        Parameter vector in physical space.
    """
    if not state:
        return params
    vector = np.asarray(params, dtype=float).copy()
    gamma_idx = state.get("gamma_log_idx")
    if gamma_idx is not None:
        vector[gamma_idx] = np.exp(vector[gamma_idx])
    beta_idx = state.get("beta_center_idx")
    if beta_idx is not None:
        vector[beta_idx] = vector[beta_idx] + state.get("beta_reference", 0.0)
    return vector


def adjust_covariance_for_transforms(
    covariance: np.ndarray,
    transformed_params: np.ndarray,
    physical_params: np.ndarray,
    state: dict[str, Any] | None,
) -> np.ndarray:
    """Adjust covariance matrix for parameter transforms.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix in solver space.
    transformed_params : np.ndarray
        Parameters in solver space.
    physical_params : np.ndarray
        Parameters in physical space.
    state : dict[str, Any] | None
        Transform state.

    Returns
    -------
    np.ndarray
        Covariance matrix in physical space.
    """
    if not state or covariance.size == 0:
        return covariance
    adjusted = np.asarray(covariance, dtype=float).copy()
    gamma_idx = state.get("gamma_log_idx")
    if gamma_idx is not None:
        scale = physical_params[gamma_idx]
        adjusted[gamma_idx, :] *= scale
        adjusted[:, gamma_idx] *= scale
    # beta centering derivative is 1, so covariance unchanged
    return adjusted


def wrap_model_function_with_transforms(
    model_fn: Any,
    state: dict[str, Any] | None,
) -> Any:
    """Wrap model function to apply inverse transforms to parameters.

    Parameters
    ----------
    model_fn : callable
        Original model function.
    state : dict[str, Any] | None
        Transform state.

    Returns
    -------
    callable
        Wrapped model function (or original if no transforms).
    """
    if not state:
        return model_fn

    if not callable(model_fn):
        return model_fn

    def wrapped_model(xdata: np.ndarray, *solver_params: float) -> np.ndarray:
        physical = apply_inverse_shear_transforms_to_vector(
            np.asarray(solver_params), state
        )
        result: np.ndarray = model_fn(xdata, *physical)
        return result

    # Preserve helpful attributes for downstream logging/diagnostics
    for attr in ["n_phi", "n_angles", "per_angle_scaling"]:
        if hasattr(model_fn, attr):
            setattr(wrapped_model, attr, getattr(model_fn, attr))
    return wrapped_model


def wrap_stratified_function_with_transforms(
    residual_fn: Any,
    state: dict[str, Any] | None,
) -> Any:
    """Wrap stratified residual function with transforms.

    Parameters
    ----------
    residual_fn : Any
        Original stratified residual function.
    state : dict[str, Any] | None
        Transform state.

    Returns
    -------
    Any
        Wrapped function (or original if no transforms).
    """
    if not state:
        return residual_fn

    class _TransformedStratified:
        def __init__(self, base_fn: Any, transform_state: dict[str, Any]):
            self._base_fn = base_fn
            self._state = transform_state

        def __call__(self, params: np.ndarray) -> np.ndarray:
            physical = apply_inverse_shear_transforms_to_vector(params, self._state)
            result: np.ndarray = self._base_fn(physical)
            return result

        def __getattr__(self, item: str) -> Any:
            return getattr(self._base_fn, item)

    return _TransformedStratified(residual_fn, state)
