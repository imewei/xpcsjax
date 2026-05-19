"""Parameter Space Configuration for MCMC/CMC
============================================

Defines the ParameterSpace class for loading parameter bounds and prior
distributions from YAML configuration files. This enables config-driven
MCMC initialization without hardcoded priors.

This module is part of the v2.1.0 MCMC simplification implementation.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.types import PARAMETER_NAME_MAPPING
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

_BETA_DEFAULT_CONC = 2.0
_SINGLE_ANGLE_CONTRAST_BOUNDS = (0.3, 0.7)
_SINGLE_ANGLE_OFFSET_BOUNDS = (0.9, 1.1)
_SINGLE_ANGLE_SCALAR_SIGMA = 0.05


def _compute_beta_concentrations(
    mu: float,
    sigma: float,
    min_val: float,
    max_val: float,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """Derive Beta concentration parameters on a scaled interval.

    Parameters
    ----------
    mu, sigma : float
        Original prior mean/std defined over [min_val, max_val]
    min_val, max_val : float
        Parameter bounds (finite, min < max)
    epsilon : float
        Minimum offset used to keep normalized mean inside (0, 1)

    Returns
    -------
    tuple[float, float]
        (alpha, beta) concentrations for Beta(alpha, beta)
    """

    width = max_val - min_val
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("BetaScaled priors require finite, positive interval width")

    # Normalize statistics to [0, 1]
    mean_norm = (mu - min_val) / width
    mean_norm = float(np.clip(mean_norm, epsilon, 1.0 - epsilon))

    sigma_norm = abs(sigma) / width if width > 0 else 0.0
    variance_norm = sigma_norm**2

    # Maximum admissible variance for Beta distribution at given mean
    max_variance = mean_norm * (1.0 - mean_norm)

    if variance_norm <= 0.0 or variance_norm >= max_variance:
        # Fallback to gentle prior centered in the interval
        return _BETA_DEFAULT_CONC, _BETA_DEFAULT_CONC

    alpha_plus_beta = (mean_norm * (1.0 - mean_norm) / variance_norm) - 1.0
    if alpha_plus_beta <= 0.0:
        return _BETA_DEFAULT_CONC, _BETA_DEFAULT_CONC

    alpha = mean_norm * alpha_plus_beta
    beta = (1.0 - mean_norm) * alpha_plus_beta

    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        return _BETA_DEFAULT_CONC, _BETA_DEFAULT_CONC

    return float(alpha), float(beta)


@dataclass
class PriorDistribution:
    """Prior distribution specification for a parameter.

    Attributes
    ----------
    dist_type : str
        Distribution type: 'Normal', 'TruncatedNormal', 'Uniform', 'LogNormal'
    mu : float
        Mean (location parameter)
    sigma : float
        Standard deviation (scale parameter)
    min_val : float
        Minimum bound (for truncated distributions)
    max_val : float
        Maximum bound (for truncated distributions)
    """

    dist_type: str  # 'Normal', 'TruncatedNormal', 'Uniform', 'LogNormal', 'BetaScaled'
    mu: float = 0.0
    sigma: float = 1.0
    min_val: float = -np.inf
    max_val: float = np.inf
    alpha: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        """Validate distribution parameters."""
        if self.dist_type not in [
            "Normal",
            "TruncatedNormal",
            "Uniform",
            "LogNormal",
            "BetaScaled",
        ]:
            logger.warning(
                f"Unknown distribution type '{self.dist_type}', defaulting to TruncatedNormal"
            )
            self.dist_type = "TruncatedNormal"

        # Validate bounds
        if self.min_val >= self.max_val:
            raise ValueError(
                f"Invalid bounds: min_val ({self.min_val}) >= max_val ({self.max_val})"
            )

        # For TruncatedNormal/Uniform, bounds must be finite
        if self.dist_type in ["TruncatedNormal", "Uniform", "BetaScaled"]:
            if np.isinf(self.min_val) or np.isinf(self.max_val):
                raise ValueError(
                    f"{self.dist_type} requires finite bounds, got [{self.min_val}, {self.max_val}]"
                )
        if self.dist_type == "BetaScaled":
            if self.alpha is None or self.beta is None:
                self.alpha, self.beta = _compute_beta_concentrations(
                    self.mu,
                    self.sigma,
                    self.min_val,
                    self.max_val,
                )
            if self.alpha <= 0 or self.beta <= 0:
                raise ValueError("BetaScaled concentration parameters must be positive")

    def to_distribution_kwargs(self) -> dict[str, Any]:
        """Convert to generic distribution constructor kwargs.

        Returns
        -------
        dict
            Keyword arguments for distribution constructors (loc/scale/low/high/
            concentration1/concentration0 conventions; consumer-agnostic).
        """
        if self.dist_type == "Normal":
            return {"loc": self.mu, "scale": self.sigma}
        elif self.dist_type == "TruncatedNormal":
            return {
                "loc": self.mu,
                "scale": self.sigma,
                "low": self.min_val,
                "high": self.max_val,
            }
        elif self.dist_type == "Uniform":
            return {"low": self.min_val, "high": self.max_val}
        elif self.dist_type == "LogNormal":
            return {"loc": self.mu, "scale": self.sigma}
        elif self.dist_type == "BetaScaled":
            return {
                "concentration1": self.alpha,
                "concentration0": self.beta,
                "low": self.min_val,
                "high": self.max_val,
            }
        else:
            # Fallback to TruncatedNormal
            return {
                "loc": self.mu,
                "scale": self.sigma,
                "low": self.min_val,
                "high": self.max_val,
            }


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds and prior distributions.

    This class encapsulates all information needed to define the parameter
    space for MCMC/CMC optimization, including parameter bounds and prior
    distributions loaded from configuration files.

    Attributes
    ----------
    model_type : str
        Model type: 'static' or 'laminar_flow'
    parameter_names : list[str]
        Canonical parameter names (after name mapping)
    bounds : dict[str, tuple[float, float]]
        Parameter bounds: {param_name: (min, max)}
    priors : dict[str, PriorDistribution]
        Prior distributions: {param_name: PriorDistribution}
    units : dict[str, str]
        Parameter units: {param_name: unit_string}

    Examples
    --------
    >>> # From config dict
    >>> config = {
    ...     'parameter_space': {
    ...         'model': 'static',
    ...         'bounds': [
    ...             {'name': 'D0', 'min': 100.0, 'max': 1e5,
    ...              'prior_mu': 1000.0, 'prior_sigma': 1000.0, 'type': 'TruncatedNormal'},
    ...             {'name': 'alpha', 'min': -2.0, 'max': 2.0,
    ...              'prior_mu': -1.2, 'prior_sigma': 0.3, 'type': 'Normal'}
    ...         ]
    ...     }
    ... }
    >>> param_space = ParameterSpace.from_config(config)
    >>> param_space.get_bounds('D0')
    (100.0, 100000.0)
    >>> prior = param_space.get_prior('D0')
    >>> prior.dist_type
    'TruncatedNormal'
    """

    model_type: str
    parameter_names: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    priors: dict[str, PriorDistribution] = field(default_factory=dict)
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
            Analysis mode ('static' or 'laminar_flow'). Auto-detected from
            config if not provided.

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
        >>> config = {'parameter_space': {'model': 'static', 'bounds': [...]}}
        >>> param_space = ParameterSpace.from_config(config)
        >>> param_space.model_type
        'static'

        Notes
        -----
        - Uses ParameterManager for name mapping (gamma_dot_0 → gamma_dot_t0)
        - Falls back to package defaults if config is incomplete
        - Validates all bounds and prior distribution parameters
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
        priors_dict: dict[str, PriorDistribution] = {}
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

        # Load bounds and priors for each parameter
        # Also include contrast and offset (scaling parameters) for MCMC per-angle initialization
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
                        min_val, max_val = 0.0, 1.0
                        logger.warning(
                            f"No bounds found for '{param_name}', using [0.0, 1.0]"
                        )

            bounds_dict[param_name] = (min_val, max_val)

            # Extract prior distribution
            prior_mu = config_entry.get("prior_mu", (min_val + max_val) / 2.0)
            prior_sigma = config_entry.get("prior_sigma", (max_val - min_val) / 4.0)
            dist_type = config_entry.get("type", "TruncatedNormal")

            # Create PriorDistribution object
            try:
                prior = PriorDistribution(
                    dist_type=dist_type,
                    mu=float(prior_mu),
                    sigma=float(prior_sigma),
                    min_val=min_val,
                    max_val=max_val,
                )
                priors_dict[param_name] = prior
            except ValueError as e:
                logger.warning(
                    f"Invalid prior for '{param_name}': {e}. Using default TruncatedNormal."
                )
                # Fallback prior
                priors_dict[param_name] = PriorDistribution(
                    dist_type="TruncatedNormal",
                    mu=(min_val + max_val) / 2.0,
                    sigma=(max_val - min_val) / 4.0,
                    min_val=min_val,
                    max_val=max_val,
                )

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
            priors=priors_dict,
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
            Analysis mode: 'static' or 'laminar_flow'

        Returns
        -------
        ParameterSpace
            Parameter space with default bounds and wide priors

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
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
            priors=self.priors.copy(),
            units=self.units.copy(),
        )

    def drop_parameters(self, names: set[str]) -> "ParameterSpace":
        """Return a copy with specific parameters removed."""

        if not names:
            return self.copy()

        filtered_names = [name for name in self.parameter_names if name not in names]
        filtered_bounds = {k: v for k, v in self.bounds.items() if k not in names}
        filtered_priors = {k: v for k, v in self.priors.items() if k not in names}
        filtered_units = {k: v for k, v in self.units.items() if k not in names}

        return ParameterSpace(
            model_type=self.model_type,
            parameter_names=filtered_names,
            bounds=filtered_bounds,
            priors=filtered_priors,
            units=filtered_units,
        )

    def with_prior_overrides(
        self, overrides: dict[str, PriorDistribution]
    ) -> "ParameterSpace":
        """Return a copy with select priors replaced."""

        if not overrides:
            return self.copy()

        cloned = self.copy()
        for name, prior in overrides.items():
            if name not in cloned.priors:
                raise KeyError(f"Cannot override prior for unknown parameter '{name}'")
            cloned.priors[name] = prior
        return cloned

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

    def get_prior(self, param_name: str) -> PriorDistribution:
        """Get prior distribution for a specific parameter.

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        PriorDistribution
            Prior distribution specification

        Raises
        ------
        KeyError
            If parameter not found in parameter space
        """
        if param_name not in self.priors:
            raise KeyError(
                f"Parameter '{param_name}' not in parameter space. "
                f"Available: {list(self.priors.keys())}"
            )
        return self.priors[param_name]

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
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> lower, upper = param_space.get_bounds_array()
        >>> lower.shape
        (3,)
        """
        lower = np.array([self.bounds[name][0] for name in self.parameter_names])
        upper = np.array([self.bounds[name][1] for name in self.parameter_names])
        return lower, upper

    def get_prior_means(self) -> np.ndarray:
        """Get prior means as numpy array (for initialization).

        Returns
        -------
        np.ndarray
            Array of prior means (in parameter_names order)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> means = param_space.get_prior_means()
        >>> means.shape
        (3,)
        """
        return np.array([self.priors[name].mu for name in self.parameter_names])

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
        >>> param_space = ParameterSpace.from_defaults('static')
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

    def convert_to_beta_scaled_priors(self) -> "ParameterSpace":
        """Return a copy of the ParameterSpace with BetaScaled priors on bounded params."""

        new_priors: dict[str, PriorDistribution] = {}
        for param_name, prior in self.priors.items():
            has_finite_bounds = np.isfinite(prior.min_val) and np.isfinite(
                prior.max_val
            )
            if (
                prior.dist_type in {"TruncatedNormal", "Uniform", "BetaScaled"}
                and has_finite_bounds
            ):
                alpha, beta = _compute_beta_concentrations(
                    prior.mu,
                    prior.sigma,
                    prior.min_val,
                    prior.max_val,
                )
                new_priors[param_name] = PriorDistribution(
                    dist_type="BetaScaled",
                    mu=prior.mu,
                    sigma=prior.sigma,
                    min_val=prior.min_val,
                    max_val=prior.max_val,
                    alpha=alpha,
                    beta=beta,
                )
                logger.debug(
                    "Converted %s prior to BetaScaled on [%s, %s] (alpha=%.3f, beta=%.3f)",
                    param_name,
                    prior.min_val,
                    prior.max_val,
                    alpha,
                    beta,
                )
            else:
                # Leave unbounded priors untouched (Normal / LogNormal)
                new_priors[param_name] = prior

        return ParameterSpace(
            model_type=self.model_type,
            parameter_names=self.parameter_names.copy(),
            bounds=self.bounds.copy(),
            priors=new_priors,
            units=self.units.copy(),
        )

    def convert_to_beta_priors(self) -> "ParameterSpace":
        """Backward compatible alias for BetaScaled conversion."""

        return self.convert_to_beta_scaled_priors()

    def get_single_angle_fallback_prior(self, param_name: str) -> PriorDistribution:
        """Return a gentle BetaScaled prior for single-angle stabilization.

        Parameters
        ----------
        param_name : str
            Target parameter (e.g., 'D0', 'alpha', 'D_offset').

        Returns
        -------
        PriorDistribution
            BetaScaled prior centered within the configured bounds.
        """

        if param_name not in self.bounds:
            raise KeyError(
                f"Parameter '{param_name}' not available for single-angle fallback"
            )

        min_val, max_val = self.bounds[param_name]
        center = (min_val + max_val) / 2.0
        sigma = (max_val - min_val) / 4.0

        return PriorDistribution(
            dist_type="BetaScaled",
            mu=center,
            sigma=sigma,
            min_val=min_val,
            max_val=max_val,
            alpha=_BETA_DEFAULT_CONC,
            beta=_BETA_DEFAULT_CONC,
        )

    def with_single_angle_stabilization(
        self,
        *,
        enable_beta_fallback: bool = False,
    ) -> "ParameterSpace":
        """Return a copy with priors tightened for the single-angle regime."""

        new_bounds = self.bounds.copy()
        new_priors = self.priors.copy()

        def _tighten_scalar(
            name: str,
            bounds: tuple[float, float],
        ) -> None:
            min_val, max_val = bounds
            new_bounds[name] = bounds
            base_prior = self.priors.get(name)
            mu = base_prior.mu if base_prior else (min_val + max_val) / 2.0
            mu = float(np.clip(mu, min_val + 1e-6, max_val - 1e-6))
            new_priors[name] = PriorDistribution(
                dist_type="TruncatedNormal",
                mu=mu,
                sigma=_SINGLE_ANGLE_SCALAR_SIGMA,
                min_val=min_val,
                max_val=max_val,
            )

        _tighten_scalar("contrast", _SINGLE_ANGLE_CONTRAST_BOUNDS)
        _tighten_scalar("offset", _SINGLE_ANGLE_OFFSET_BOUNDS)

        if enable_beta_fallback:
            for param_name in ("D0", "alpha", "D_offset"):
                if param_name in self.bounds:
                    new_priors[param_name] = self.get_single_angle_fallback_prior(
                        param_name
                    )

        return ParameterSpace(
            model_type=self.model_type,
            parameter_names=self.parameter_names.copy(),
            bounds=new_bounds,
            priors=new_priors,
            units=self.units.copy(),
        )

    def get_single_angle_geometry_config(self) -> dict[str, float]:
        """Return heuristic priors for single-angle diffusion reparameterization."""

        try:
            d0_prior = self.get_prior("D0")
            d_offset_prior = self.get_prior("D_offset")
        except KeyError:
            return {
                "enabled": True,
                "log_center_loc": 8.0,
                "log_center_scale": 1.0,
                "delta_loc": 0.0,
                "delta_scale": 1.0,
                "delta_floor": 1e-3,
            }

        d0_bounds = self.bounds.get("D0", (100.0, 1e5))
        d_offset_bounds = self.bounds.get("D_offset", (-1e5, 1e5))
        center_mu = d0_prior.mu + d_offset_prior.mu
        if center_mu <= 0:
            center_mu = max(
                1e-6,
                (d0_bounds[0] + d0_bounds[1] + d_offset_bounds[0] + d_offset_bounds[1])
                / 4.0,
            )
        center_sigma = abs(d0_prior.sigma) + abs(d_offset_prior.sigma)
        if not np.isfinite(center_sigma) or center_sigma <= 0:
            center_sigma = max(d0_bounds[1] - d0_bounds[0], 1.0)

        log_center_loc = float(np.log(max(center_mu, 1e-6)))
        log_center_scale = float(
            max(0.25, np.log1p(center_sigma / max(center_mu, 1e-6)))
        )

        target_delta = d0_prior.mu / max(center_mu, 1e-6)
        target_delta = float(np.clip(target_delta, 1e-3, 5.0))
        delta_loc = (
            float(np.log(np.expm1(target_delta))) if target_delta >= 1e-3 else -5.0
        )
        delta_scale = float(max(0.5, abs(d0_prior.sigma) / max(center_mu, 1e-6)))

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
        >>> param_space = ParameterSpace.from_defaults('static')
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
            prior = self.priors[param_name]
            unit = self.units.get(param_name, "")

            lines.append(
                f"    {param_name:20s}: "
                f"[{min_val:10.3e}, {max_val:10.3e}] "
                f"{prior.dist_type}(mu={prior.mu:.3e}, sigma={prior.sigma:.3e}) "
                f"{unit}"
            )

        return "\n".join(lines)
