"""Parameter Space Configuration for NLSQ
==========================================

Defines the ParameterSpace class for loading parameter bounds from YAML
configuration files. This enables config-driven NLSQ initialization
without hardcoded bounds.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.types import PARAMETER_NAME_MAPPING
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds for NLSQ optimization.

    This class encapsulates all information needed to define the parameter
    space for NLSQ optimization, including parameter bounds
    loaded from configuration files.

    Attributes
    ----------
    model_type : str
        Model type: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'
    parameter_names : list[str]
        Canonical parameter names (after name mapping)
    bounds : dict[str, tuple[float, float]]
        Parameter bounds: {param_name: (min, max)}
    units : dict[str, str]
        Parameter units: {param_name: unit_string}

    Examples
    --------
    >>> # From config dict
    >>> config = {
    ...     'parameter_space': {
    ...         'model': 'static_anisotropic',
    ...         'bounds': [
    ...             {'name': 'D0', 'min': 100.0, 'max': 1e5},
    ...             {'name': 'alpha', 'min': -2.0, 'max': 2.0}
    ...         ]
    ...     }
    ... }
    >>> param_space = ParameterSpace.from_config(config)
    >>> param_space.get_bounds('D0')
    (100.0, 100000.0)
    """

    model_type: str
    parameter_names: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        analysis_mode: str | None = None,
    ) -> "ParameterSpace":
        """Load ParameterSpace from configuration dictionary.

        This class method constructs a ParameterSpace instance from a YAML
        configuration dict, handling missing values gracefully and integrating
        with the existing ParameterManager for name mapping and defaults.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary (typically loaded from YAML)
        analysis_mode : str, optional
            Analysis mode ('static_anisotropic', 'static_isotropic', or
            'laminar_flow'). Auto-detected from config if not provided.

        Returns
        -------
        ParameterSpace
            Configured parameter space instance

        Raises
        ------
        ValueError
            If parameter_space section is malformed or missing required fields

        Examples
        --------
        >>> config = {'parameter_space': {'model': 'static_anisotropic', 'bounds': [...]}}
        >>> param_space = ParameterSpace.from_config(config)
        >>> param_space.model_type
        'static_anisotropic'

        Notes
        -----
        - Uses ParameterManager for name mapping (gamma_dot_0 → gamma_dot_t0)
        - Falls back to package defaults if config is incomplete
        - Logs warnings for missing or invalid config values
        """
        # Extract parameter_space section
        param_space_config = config_dict.get("parameter_space", {})

        # Determine model type
        if analysis_mode is None:
            # Try to get from config
            analysis_mode = (
                param_space_config.get("model")
                or config_dict.get("analysis_mode")
                or "laminar_flow"
            )

        model_type = analysis_mode.lower()

        # Initialize ParameterManager for name mapping and defaults
        param_manager = ParameterManager(config_dict, analysis_mode=model_type)

        # Get parameter names (use ParameterManager to respect active_parameters)
        parameter_names = param_manager.get_active_parameters()

        # Parse bounds from config
        bounds_dict: dict[str, tuple[float, float]] = {}
        units_dict: dict[str, str] = {}

        config_bounds = param_space_config.get("bounds", [])
        if not isinstance(config_bounds, list):
            logger.warning(
                "parameter_space.bounds must be a list, using package defaults"
            )
            config_bounds = []

        # Build lookup dict from config bounds
        config_bounds_lookup: dict[str, dict[str, Any]] = {}
        for bound_entry in config_bounds:
            if not isinstance(bound_entry, dict):
                continue

            param_name = bound_entry.get("name")
            if not param_name or not isinstance(param_name, str):
                continue

            # Apply name mapping. ``.get(name, name)`` returns the mapped str
            # or falls back to the original ``name`` — never None. Coerce so
            # mypy doesn't lose the str invariant through ``dict[str, str].get``.
            canonical_name: str = str(
                PARAMETER_NAME_MAPPING.get(param_name, param_name)
            )
            config_bounds_lookup[canonical_name] = bound_entry

        # Load bounds for each parameter
        # Also load bounds for contrast and offset scaling parameters
        params_to_load = list(parameter_names) + ["contrast", "offset"]

        for param_name in params_to_load:
            # Skip if already processed (avoid duplicates)
            if param_name in bounds_dict:
                continue

            # Get config entry (if exists)
            config_entry = config_bounds_lookup.get(param_name, {})

            # Extract bounds (with fallback to ParameterManager defaults)
            if "min" in config_entry and "max" in config_entry:
                min_val = float(config_entry["min"])
                max_val = float(config_entry["max"])
            else:
                # Fallback to ParameterManager defaults
                default_bounds = param_manager.get_parameter_bounds([param_name])
                if default_bounds:
                    min_val = default_bounds[0]["min"]
                    max_val = default_bounds[0]["max"]
                    logger.debug(
                        f"Using default bounds for '{param_name}': [{min_val}, {max_val}]"
                    )
                else:
                    # Ultimate fallback: use registry bounds if known
                    from xpcsjax.config.parameter_registry import ParameterRegistry

                    try:
                        info = ParameterRegistry().get_param_info(param_name)
                        min_val, max_val = info.lower_bound, info.upper_bound
                        logger.debug(
                            f"Using registry bounds for '{param_name}': [{min_val}, {max_val}]"
                        )
                    except KeyError:
                        raise KeyError(
                            f"Parameter '{param_name}' is not registered in "
                            f"ParameterRegistry. Register it in "
                            f"xpcsjax/config/parameter_registry.py before use."
                        ) from None

            bounds_dict[param_name] = (min_val, max_val)

            # Extract unit (optional)
            unit = config_entry.get("unit", "")
            if unit:
                units_dict[param_name] = unit

        # Log summary
        logger.info(
            f"Loaded ParameterSpace: model={model_type}, "
            f"n_params={len(parameter_names)}, "
            f"parameters={parameter_names}"
        )

        return cls(
            model_type=model_type,
            parameter_names=parameter_names,
            bounds=bounds_dict,
            units=units_dict,
        )

    @classmethod
    def from_defaults(
        cls,
        analysis_mode: str = "laminar_flow",
    ) -> "ParameterSpace":
        """Create ParameterSpace with package defaults (no config file).

        This method creates a ParameterSpace using only the hardcoded
        defaults from ParameterManager, useful when no config file is
        available or for testing.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: 'static_anisotropic', 'static_isotropic', or 'laminar_flow'

        Returns
        -------
        ParameterSpace
            Parameter space with default bounds

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static_anisotropic')
        >>> param_space.parameter_names
        ['D0', 'alpha', 'D_offset']
        """
        logger.info(
            f"Creating ParameterSpace from package defaults (mode={analysis_mode})"
        )

        # Create empty config and let from_config handle defaults
        empty_config: dict[str, Any] = {"analysis_mode": analysis_mode}

        return cls.from_config(empty_config, analysis_mode=analysis_mode)

    def copy(self) -> "ParameterSpace":
        """Return a shallow copy safe for localized mutations."""

        return ParameterSpace(
            model_type=self.model_type,
            parameter_names=self.parameter_names.copy(),
            bounds=self.bounds.copy(),
            units=self.units.copy(),
        )

    def drop_parameters(self, names: set[str]) -> "ParameterSpace":
        """Return a copy with specific parameters removed."""

        if not names:
            return self.copy()

        filtered_names = [name for name in self.parameter_names if name not in names]
        filtered_bounds = {k: v for k, v in self.bounds.items() if k not in names}
        filtered_units = {k: v for k, v in self.units.items() if k not in names}

        return ParameterSpace(
            model_type=self.model_type,
            parameter_names=filtered_names,
            bounds=filtered_bounds,
            units=filtered_units,
        )

    def get_bounds(self, param_name: str) -> tuple[float, float]:
        """Get bounds for a specific parameter.

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        tuple[float, float]
            (min_value, max_value)

        Raises
        ------
        KeyError
            If parameter not found in parameter space
        """
        if param_name not in self.bounds:
            raise KeyError(
                f"Parameter '{param_name}' not in parameter space. "
                f"Available: {list(self.bounds.keys())}"
            )
        return self.bounds[param_name]

    def get_bounds_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds as numpy arrays (for optimization).

        Returns
        -------
        lower_bounds : np.ndarray
            Array of lower bounds (in parameter_names order)
        upper_bounds : np.ndarray
            Array of upper bounds (in parameter_names order)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static_anisotropic')
        >>> lower, upper = param_space.get_bounds_array()
        >>> lower.shape
        (3,)
        """
        lower = np.array([self.bounds[name][0] for name in self.parameter_names])
        upper = np.array([self.bounds[name][1] for name in self.parameter_names])
        return lower, upper

    def validate_values(
        self, values: dict[str, float], tolerance: float = 1e-10
    ) -> tuple[bool, list[str]]:
        """Validate parameter values against bounds.

        Parameters
        ----------
        values : dict[str, float]
            Parameter values to validate
        tolerance : float
            Tolerance for bounds checking

        Returns
        -------
        is_valid : bool
            True if all values are within bounds
        violations : list[str]
            List of violation messages (empty if valid)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static_anisotropic')
        >>> values = {'D0': 1000.0, 'alpha': -1.2, 'D_offset': 0.0}
        >>> is_valid, violations = param_space.validate_values(values)
        >>> is_valid
        True
        """
        violations = []

        for param_name, value in values.items():
            if param_name not in self.bounds:
                violations.append(
                    f"Unknown parameter '{param_name}' (not in parameter space)"
                )
                continue

            min_val, max_val = self.bounds[param_name]

            if value < min_val - tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} < min ({min_val:.3e}) "
                    f"by {min_val - value:.3e}"
                )
            elif value > max_val + tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} > max ({max_val:.3e}) "
                    f"by {value - max_val:.3e}"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def get_single_angle_geometry_config(self) -> dict[str, float]:
        """Return heuristic geometry config for single-angle diffusion reparameterization.

        Derives log-space center and delta location parameters from the D0 and
        D_offset bounds midpoints when those parameters are present. Falls back
        to sensible defaults when either parameter is absent.
        """

        d0_bounds = self.bounds.get("D0", (100.0, 1e5))
        d_offset_bounds = self.bounds.get("D_offset", (-1e5, 1e5))

        if "D0" not in self.bounds or "D_offset" not in self.bounds:
            return {
                "enabled": True,
                "log_center_loc": 8.0,
                "log_center_scale": 1.0,
                "delta_loc": 0.0,
                "delta_scale": 1.0,
                "delta_floor": 1e-3,
            }

        # Use bounds midpoints as heuristic location estimates
        d0_mid = (d0_bounds[0] + d0_bounds[1]) / 2.0
        d_offset_mid = (d_offset_bounds[0] + d_offset_bounds[1]) / 2.0
        d0_half_width = (d0_bounds[1] - d0_bounds[0]) / 2.0

        center_mu = max(d0_mid + d_offset_mid, 1e-6)
        center_sigma = max(d0_half_width, 1.0)

        log_center_loc = float(np.log(center_mu))
        log_center_scale = float(max(0.25, np.log1p(center_sigma / center_mu)))

        target_delta = float(np.clip(d0_mid / center_mu, 1e-3, 5.0))
        delta_loc = (
            float(np.log(np.expm1(target_delta))) if target_delta >= 1e-3 else -5.0
        )
        delta_scale = float(max(0.5, d0_half_width / center_mu))

        return {
            "enabled": True,
            "log_center_loc": log_center_loc,
            "log_center_scale": log_center_scale,
            "delta_loc": delta_loc,
            "delta_scale": delta_scale,
            "delta_floor": 1e-3,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParameterSpace(model={self.model_type}, "
            f"n_params={len(self.parameter_names)}, "
            f"params={self.parameter_names})"
        )

    def clamp_to_open_interval(
        self, param_name: str, value: float, epsilon: float = 1e-6
    ) -> float:
        """Clamp parameter value to strictly inside bounds (open interval).

        TruncatedNormal transforms require values strictly inside (min, max)
        - not equal to the boundaries. This method ensures values are at least epsilon
        away from both bounds.

        Parameters
        ----------
        param_name : str
            Parameter name
        value : float
            Value to clamp
        epsilon : float, default 1e-6
            Minimum distance from boundaries

        Returns
        -------
        float
            Clamped value strictly inside (min + epsilon, max - epsilon)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static_anisotropic')
        >>> # If offset bounds are [0.5, 1.5] and value equals 0.5 (boundary violation)
        >>> clamped = param_space.clamp_to_open_interval('offset', 0.5)
        >>> # Returns 0.500001 (0.5 + 1e-6), strictly inside bounds
        >>> clamped > 0.5 and clamped < 1.5
        True
        """
        if param_name not in self.bounds:
            raise KeyError(
                f"Parameter '{param_name}' not in parameter space. "
                f"Available: {list(self.bounds.keys())}"
            )

        min_val, max_val = self.bounds[param_name]

        # Ensure epsilon doesn't exceed half the interval width
        interval_width = max_val - min_val
        safe_epsilon = min(epsilon, interval_width / 10.0)

        # Clamp to open interval: (min + epsilon, max - epsilon)
        # Step 1: Clamp value to be at least min_val + epsilon
        value_clamped_min = max(value, min_val + safe_epsilon)
        # Step 2: Clamp value to be at most max_val - epsilon
        clamped_value = min(value_clamped_min, max_val - safe_epsilon)

        return float(clamped_value)

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"ParameterSpace: {self.model_type} model"]
        lines.append(f"  Parameters ({len(self.parameter_names)}):")

        for param_name in self.parameter_names:
            min_val, max_val = self.bounds[param_name]
            unit = self.units.get(param_name, "")

            lines.append(
                f"    {param_name:20s}: "
                f"[{min_val:10.3e}, {max_val:10.3e}] "
                f"{unit}"
            )

        return "\n".join(lines)
