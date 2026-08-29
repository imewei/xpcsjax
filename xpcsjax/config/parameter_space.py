"""Parameter space configuration for NLSQ.

Defines the ParameterSpace class for loading parameter bounds from YAML
configuration files. This enables config-driven NLSQ initialization
without hardcoded bounds.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.config.types import PARAMETER_NAME_MAPPING, coerce_finite_float
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
        analysis_mode: AnalysisMode | None = None,
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
        # Extract parameter_space section. ``or {}`` (not the .get default) is
        # required: PyYAML parses a blank ``parameter_space:`` block to None,
        # and dict.get(key, default) only substitutes default when key is
        # absent, not when present with value None.
        param_space_config = config_dict.get("parameter_space") or {}

        # Determine model type
        if analysis_mode is None:
            # Try to get from config
            analysis_mode = AnalysisMode.parse(
                str(
                    param_space_config.get("model")
                    or config_dict.get("analysis_mode")
                    or "laminar_flow"
                ),
                allow_bare_static=True,
            )

        model_type = analysis_mode.lower()

        # Initialize ParameterManager for name mapping and defaults
        param_manager = ParameterManager(config_dict, analysis_mode=analysis_mode)

        # Get parameter names (use ParameterManager to respect active_parameters)
        parameter_names = param_manager.get_active_parameters()

        # Parse bounds from config
        bounds_dict: dict[str, tuple[float, float]] = {}
        units_dict: dict[str, str] = {}

        config_bounds = param_space_config.get("bounds", [])
        if not isinstance(config_bounds, list):
            logger.warning("parameter_space.bounds must be a list, using package defaults")
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
            #
            # NOTE: an alias/canonical-name collision here (e.g. both
            # "gamma_dot_0" and "gamma_dot_t0" in the same bounds list) is
            # already caught upstream by ParameterManager._load_config_bounds
            # -- ``ParameterManager(config_dict, ...)`` above (this method's
            # first statement) parses the identical ``parameter_space.bounds``
            # list and raises before this loop ever runs, so an explicit
            # duplicate guard here would be unreachable dead code.
            canonical_name: str = str(PARAMETER_NAME_MAPPING.get(param_name, param_name))
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
                min_val = coerce_finite_float(
                    config_entry["min"], context=f"parameter_space.bounds[{param_name!r}].min"
                )
                max_val = coerce_finite_float(
                    config_entry["max"], context=f"parameter_space.bounds[{param_name!r}].max"
                )
            else:
                # Fallback to ParameterManager defaults. get_parameter_bounds
                # with a single-name list either returns a 1-element list or
                # raises KeyError — it never returns an empty/falsy list — so
                # there is no third fallback tier to reach here.
                default_bounds = param_manager.get_parameter_bounds([param_name])
                min_val = default_bounds[0]["min"]
                max_val = default_bounds[0]["max"]
                logger.debug(f"Using default bounds for '{param_name}': [{min_val}, {max_val}]")

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
        analysis_mode: AnalysisMode = AnalysisMode.LAMINAR_FLOW,
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
        logger.info(f"Creating ParameterSpace from package defaults (mode={analysis_mode})")

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
                violations.append(f"Unknown parameter '{param_name}' (not in parameter space)")
                continue

            min_val, max_val = self.bounds[param_name]

            if np.isnan(value):
                violations.append(f"{param_name} = {value} is NaN (not a valid value)")
            elif value < min_val - tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} < min ({min_val:.3e}) by {min_val - value:.3e}"
                )
            elif value > max_val + tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} > max ({max_val:.3e}) by {value - max_val:.3e}"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"ParameterSpace(model={self.model_type}, "
            f"n_params={len(self.parameter_names)}, "
            f"params={self.parameter_names})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"ParameterSpace: {self.model_type} model"]
        lines.append(f"  Parameters ({len(self.parameter_names)}):")

        for param_name in self.parameter_names:
            min_val, max_val = self.bounds[param_name]
            unit = self.units.get(param_name, "")

            lines.append(f"    {param_name:20s}: [{min_val:10.3e}, {max_val:10.3e}] {unit}")

        return "\n".join(lines)
