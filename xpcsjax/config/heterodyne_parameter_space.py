"""Parameter space definition with prior distributions for Bayesian inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.config.heterodyne_parameter_names import (
    ALL_PARAM_NAMES,
    ALL_PARAM_NAMES_WITH_SCALING,
    SCALING_PARAMS,
)
from xpcsjax.config.parameter_registry import DEFAULT_REGISTRY
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    import jax.numpy as jnp

logger = get_logger(__name__)


class PriorType(Enum):
    """Available prior distribution types."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    LOGNORMAL = "lognormal"
    HALFNORMAL = "halfnormal"
    EXPONENTIAL = "exponential"
    BETA_SCALED = "beta_scaled"


@dataclass
class PriorDistribution:
    """Prior distribution specification for a parameter."""

    prior_type: PriorType
    params: dict[str, float] = field(default_factory=dict)

    @classmethod
    def uniform(cls, low: float, high: float) -> PriorDistribution:
        """Create uniform prior."""
        return cls(PriorType.UNIFORM, {"low": low, "high": high})

    @classmethod
    def normal(cls, loc: float, scale: float) -> PriorDistribution:
        """Create normal (Gaussian) prior."""
        return cls(PriorType.NORMAL, {"loc": loc, "scale": scale})

    @classmethod
    def lognormal(cls, loc: float, scale: float) -> PriorDistribution:
        """Create log-normal prior (for positive parameters)."""
        return cls(PriorType.LOGNORMAL, {"loc": loc, "scale": scale})

    @classmethod
    def halfnormal(cls, scale: float) -> PriorDistribution:
        """Create half-normal prior (for positive parameters)."""
        return cls(PriorType.HALFNORMAL, {"scale": scale})

    @classmethod
    def truncated_normal(
        cls,
        loc: float,
        scale: float,
        low: float,
        high: float,
    ) -> PriorDistribution:
        """Create truncated normal prior (bounded Gaussian)."""
        return cls(
            PriorType.TRUNCATED_NORMAL,
            {"loc": loc, "scale": scale, "low": low, "high": high},
        )

    @classmethod
    def beta_scaled(
        cls,
        low: float,
        high: float,
        concentration1: float,
        concentration2: float,
    ) -> PriorDistribution:
        """Create a Beta prior scaled to [low, high].

        The distribution is Beta(concentration1, concentration2) affine-transformed
        to the interval [low, high]. This is useful for bounded parameters where
        you want to express a prior belief about the shape within the bounds.

        Args:
            low: Lower bound of the support.
            high: Upper bound of the support.
            concentration1: First concentration parameter (alpha > 0).
            concentration2: Second concentration parameter (beta > 0).

        Returns:
            PriorDistribution with BETA_SCALED type.
        """
        return cls(
            PriorType.BETA_SCALED,
            {
                "low": low,
                "high": high,
                "concentration1": concentration1,
                "concentration2": concentration2,
            },
        )

    # NOTE: The upstream heterodyne `to_numpyro()` method (which converted
    # PriorDistribution → NumPyro distribution objects) is intentionally
    # omitted in xpcsjax. CMC / Bayesian sampling is permanently out of
    # scope (spec §15.1). PriorDistribution is retained as metadata only.


@dataclass
class ParameterSpace:
    """Complete parameter space for heterodyne model optimization.

    Manages parameter values, bounds, vary flags, and priors.
    """

    values: dict[str, float] = field(default_factory=dict)
    vary: dict[str, bool] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    priors: dict[str, PriorDistribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize with defaults from registry."""
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            info = DEFAULT_REGISTRY[name]
            if name not in self.values:
                self.values[name] = info.default
            if name not in self.vary:
                self.vary[name] = info.vary_default
            if name not in self.bounds:
                self.bounds[name] = (info.min_bound, info.max_bound)
            if name not in self.priors:
                self.priors[name] = _default_prior(name, info)

    @property
    def n_total(self) -> int:
        """Total number of parameters."""
        return len(ALL_PARAM_NAMES)

    @property
    def n_varying(self) -> int:
        """Number of parameters that vary in optimization."""
        return len(self.varying_names)

    @property
    def varying_names(self) -> list[str]:
        """Names of parameters that vary (physics + scaling)."""
        return [
            name for name in ALL_PARAM_NAMES_WITH_SCALING if self.vary.get(name, False)
        ]

    @property
    def fixed_names(self) -> list[str]:
        """Names of parameters that are fixed."""
        return [
            name
            for name in ALL_PARAM_NAMES_WITH_SCALING
            if not self.vary.get(name, False)
        ]

    @property
    def varying_physics_names(self) -> list[str]:
        """Names of varying physics parameters (excludes scaling)."""
        return [name for name in ALL_PARAM_NAMES if self.vary.get(name, False)]

    @property
    def scaling_values(self) -> dict[str, float]:
        """Get contrast and offset values."""
        return {name: self.values[name] for name in SCALING_PARAMS}

    def get_initial_array(self) -> np.ndarray:
        """Get initial values as numpy array in canonical order.

        Returns:
            Array of shape (14,) with parameter values
        """
        return np.array([self.values[name] for name in ALL_PARAM_NAMES])

    def to_config(self) -> dict[str, Any]:
        """Serialize this space to a dict compatible with :meth:`from_config`.

        Produces the ``initial_parameters`` flat-format understood by
        :func:`_apply_initial_parameters`.  Bounds and priors are not
        serialized — workers rebuild them from the registry defaults.
        Only values and ``active_parameters`` (vary flags) are round-tripped.

        Returns:
            Config dict that ``from_config()`` can reconstruct into an
            equivalent ParameterSpace (same values and varying_names).
        """
        return {
            "initial_parameters": {
                "parameter_names": list(ALL_PARAM_NAMES_WITH_SCALING),
                "values": [
                    float(self.values[name]) for name in ALL_PARAM_NAMES_WITH_SCALING
                ],
                "active_parameters": list(self.varying_names),
            }
        }

    def get_bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds as numpy arrays.

        Returns:
            (lower_bounds, upper_bounds) each of shape (14,)
        """
        lower = np.array([self.bounds[name][0] for name in ALL_PARAM_NAMES])
        upper = np.array([self.bounds[name][1] for name in ALL_PARAM_NAMES])
        return lower, upper

    def get_vary_mask(self) -> np.ndarray:
        """Get boolean mask for varying parameters.

        Returns:
            Boolean array of shape (14,)
        """
        return np.array([self.vary[name] for name in ALL_PARAM_NAMES])

    def array_to_dict(self, arr: np.ndarray | jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary.

        Args:
            arr: Array of shape (14,)

        Returns:
            Dict mapping parameter names to values
        """
        return {name: float(arr[i]) for i, name in enumerate(ALL_PARAM_NAMES)}

    def update_from_dict(self, params: dict[str, float]) -> None:
        """Update parameter values from dictionary.

        Args:
            params: Dict with parameter names as keys

        Raises:
            ValueError: If a key doesn't match any known parameter
        """
        for name, value in params.items():
            if name not in self.values:
                raise ValueError(
                    f"Unknown parameter '{name}'. "
                    f"Valid parameters: {list(ALL_PARAM_NAMES)}"
                )
            self.values[name] = value

    def validate(self) -> list[str]:
        """Validate parameter space configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for name in ALL_PARAM_NAMES:
            value = self.values.get(name)
            bounds = self.bounds.get(name)

            if value is None:
                errors.append(f"Missing value for {name}")
                continue

            if bounds is None:
                errors.append(f"Missing bounds for {name}")
                continue

            low, high = bounds
            if not (low <= value <= high):
                errors.append(f"{name}={value} outside bounds [{low}, {high}]")

        return errors

    def convert_to_beta_priors(self) -> None:
        """Convert all TruncatedNormal priors to BetaScaled priors.

        For each parameter whose prior is TRUNCATED_NORMAL, this method
        computes equivalent Beta concentration parameters via the method
        of moments and replaces the prior in-place with a BETA_SCALED
        distribution over the same bounds.

        Parameters with other prior types are left unchanged.
        """
        for name, prior in self.priors.items():
            if prior.prior_type != PriorType.TRUNCATED_NORMAL:
                continue

            loc = prior.params["loc"]
            scale = prior.params["scale"]
            low, high = self.bounds[name]

            conc1, conc2 = _compute_beta_concentrations(loc, scale, low, high)
            self.priors[name] = PriorDistribution.beta_scaled(
                low,
                high,
                conc1,
                conc2,
            )
            logger.debug(
                "Converted %s prior: TruncatedNormal(loc=%.4g, scale=%.4g) "
                "-> BetaScaled(conc1=%.4g, conc2=%.4g) on [%.4g, %.4g]",
                name,
                loc,
                scale,
                conc1,
                conc2,
                low,
                high,
            )

    def with_single_angle_stabilization(self) -> ParameterSpace:
        """Return a new ParameterSpace with tightened bounds for single-angle analysis.

        Narrows contrast bounds to [value-0.2, value+0.2] and offset bounds
        to [value-0.1, value+0.1], clamped to the original bounds.

        Returns:
            A new ParameterSpace with tightened scaling bounds.
        """
        new = ParameterSpace(
            values=deepcopy(self.values),
            vary=deepcopy(self.vary),
            bounds=deepcopy(self.bounds),
            priors=deepcopy(self.priors),
        )

        # Tighten contrast bounds
        if "contrast" in new.bounds:
            low, high = new.bounds["contrast"]
            val = new.values["contrast"]
            new_low = max(low, val - 0.2)
            new_high = min(high, val + 0.2)
            new.bounds["contrast"] = (new_low, new_high)
            logger.debug(
                "Single-angle stabilization: contrast bounds [%.4g, %.4g] -> [%.4g, %.4g]",
                low,
                high,
                new_low,
                new_high,
            )

        # Tighten offset bounds
        if "offset" in new.bounds:
            low, high = new.bounds["offset"]
            val = new.values["offset"]
            new_low = max(low, val - 0.1)
            new_high = min(high, val + 0.1)
            new.bounds["offset"] = (new_low, new_high)
            logger.debug(
                "Single-angle stabilization: offset bounds [%.4g, %.4g] -> [%.4g, %.4g]",
                low,
                high,
                new_low,
                new_high,
            )

        return new

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParameterSpace:
        """Create ParameterSpace from configuration dictionary.

        Supports two input formats (homodyne parity):

        1. **Grouped format** (preferred) — ``parameters.{group}.{param}``::

               parameters:
                 reference:
                   D0_ref:
                     value: 5000.0
                     min: 200.0
                     max: 50000.0
                     vary: true

        2. **Flat format** — ``initial_parameters.parameter_names`` + ``values``::

               initial_parameters:
                 parameter_names: [D0_ref, alpha_ref]
                 values: [5000.0, 0.5]
                 active_parameters: [D0_ref]   # optional vary subset

        When both are present, grouped format takes precedence (it is applied
        second so its values overwrite flat-format values).

        Args:
            config: Config dict with 'parameters' and/or 'initial_parameters'
                sections.

        Returns:
            Configured ParameterSpace
        """
        space = cls()

        # --- Flat format: initial_parameters (homodyne parity) ---------------
        _apply_initial_parameters(space, config)

        # --- Grouped format: parameters.{group}.{param} (primary) -----------
        params_config = config.get("parameters", {})

        group_map = {
            "reference": ["D0_ref", "alpha_ref", "D_offset_ref"],
            "sample": ["D0_sample", "alpha_sample", "D_offset_sample"],
            "velocity": ["v0", "beta", "v_offset"],
            "fraction": ["f0", "f1", "f2", "f3"],
            "angle": ["phi0"],
            "scaling": ["contrast", "offset"],
        }

        for group_name, param_names in group_map.items():
            group_config = params_config.get(group_name, {})

            # Check for unknown keys in this group
            known_params = set(param_names)
            for ck in group_config:
                if ck not in known_params:
                    raise ValueError(
                        f"Unknown parameter key '{ck}' in group '{group_name}'. "
                        f"Valid keys: {param_names}"
                    )

            for param_name in param_names:
                # Direct key match only — no substring matching
                if param_name not in group_config:
                    continue

                pconfig = group_config[param_name]
                if isinstance(pconfig, dict):
                    reg_info = DEFAULT_REGISTRY[param_name]
                    if "value" in pconfig:
                        new_val = pconfig["value"]
                        if new_val != reg_info.default:
                            logger.debug(
                                "Config overrides %s value: %.6g -> %.6g",
                                param_name,
                                reg_info.default,
                                new_val,
                            )
                        space.values[param_name] = new_val
                    if "min" in pconfig and "max" in pconfig:
                        new_bounds = (pconfig["min"], pconfig["max"])
                        if (
                            new_bounds[0] != reg_info.min_bound
                            or new_bounds[1] != reg_info.max_bound
                        ):
                            logger.debug(
                                "Config overrides %s bounds: [%.4g, %.4g] -> [%.4g, %.4g]",
                                param_name,
                                reg_info.min_bound,
                                reg_info.max_bound,
                                new_bounds[0],
                                new_bounds[1],
                            )
                        space.bounds[param_name] = new_bounds
                    if "vary" in pconfig:
                        new_vary = pconfig["vary"]
                        if new_vary != reg_info.vary_default:
                            logger.debug(
                                "Config overrides %s vary: %s -> %s",
                                param_name,
                                reg_info.vary_default,
                                new_vary,
                            )
                        space.vary[param_name] = new_vary
                    if "prior" in pconfig:
                        prior_type_str = pconfig["prior"]
                        prior_params = pconfig.get("prior_params", {})
                        space.priors[param_name] = _build_prior(
                            param_name,
                            prior_type_str,
                            prior_params,
                            space.bounds[param_name],
                        )

        space._config_dict: dict[str, Any] = config  # type: ignore[attr-defined]
        return space


def _apply_initial_parameters(space: ParameterSpace, config: dict[str, Any]) -> None:
    """Apply ``initial_parameters`` flat-format values to *space*.

    Homodyne parity: supports::

        initial_parameters:
          parameter_names: [D0_ref, alpha_ref, ...]
          values: [5000.0, 0.5, ...]
          active_parameters: [D0_ref]   # optional: only these vary

    Args:
        space: ParameterSpace to modify in-place.
        config: Full configuration dictionary.
    """
    from xpcsjax.config.types import PARAMETER_NAME_MAPPING

    initial = config.get("initial_parameters", {})
    if not initial or not isinstance(initial, dict):
        return

    param_names_raw = initial.get("parameter_names")
    param_values = initial.get("values")

    if (
        not param_names_raw
        or not isinstance(param_names_raw, list)
        or param_values is None
        or not isinstance(param_values, list)
    ):
        return

    # Apply name mapping for legacy/alias names
    param_names = [PARAMETER_NAME_MAPPING.get(str(n), str(n)) for n in param_names_raw]

    if len(param_names) != len(param_values):
        logger.warning(
            "initial_parameters: parameter_names (%d) and values (%d) length mismatch; "
            "skipping flat-format override",
            len(param_names),
            len(param_values),
        )
        return

    for name, value in zip(param_names, param_values, strict=True):
        if name in space.values:
            space.values[name] = float(value)
            logger.debug(
                "initial_parameters: set %s = %.6g (flat-format override)", name, value
            )
        else:
            logger.warning("initial_parameters: unknown parameter '%s', skipping", name)

    # active_parameters: if provided, only these parameters vary
    active_raw = initial.get("active_parameters")
    if active_raw and isinstance(active_raw, list):
        active_names = {PARAMETER_NAME_MAPPING.get(str(n), str(n)) for n in active_raw}
        for name in ALL_PARAM_NAMES_WITH_SCALING:
            if name in active_names:
                space.vary[name] = True
            elif name in space.vary:
                space.vary[name] = False
        logger.debug(
            "initial_parameters: active_parameters set %d params to vary",
            len(active_names),
        )


# Default TruncatedNormal prior specifications: (loc, scale)
# All parameters use TruncatedNormal priors truncated to their registry bounds.
# IMPORTANT: must stay in sync with parameter_registry.py prior_mean/prior_std
# (CLAUDE.md rule #9 — dual prior system).  See tests/unit/test_prior_sanity.py
# for the contract test that enforces this.
_DEFAULT_PRIOR_SPECS: dict[str, tuple[float, float]] = {
    "D0_ref": (1e4, 1e4),  # widened from 5e3 → 1e4 (see registry comment)
    "alpha_ref": (0.0, 1.0),
    "D_offset_ref": (0.0, 1e3),
    "D0_sample": (1e4, 1e4),  # widened from 5e3 → 1e4 (see registry comment)
    "alpha_sample": (0.0, 1.0),
    "D_offset_sample": (0.0, 1e3),
    "v0": (1e3, 1000.0),  # widened from 500 → 1000 (see registry comment)
    "beta": (0.0, 1.0),
    "v_offset": (0.0, 25.0),
    "f0": (0.5, 0.25),
    "f1": (0.0, 5.0),
    "f2": (0.0, 1e3),
    "f3": (0.0, 0.5),
    "phi0": (0.0, 5.0),
    "contrast": (0.5, 0.25),
    "offset": (1.0, 0.25),
}


def _compute_beta_concentrations(
    mean: float,
    std: float,
    low: float,
    high: float,
) -> tuple[float, float]:
    """Compute Beta concentration parameters from desired mean and std on [low, high].

    Uses the method of moments to find (alpha, beta) such that a
    Beta(alpha, beta) distribution scaled to [low, high] has the
    specified mean and standard deviation.

    The standard Beta(alpha, beta) on [0, 1] has:
        mu_01 = alpha / (alpha + beta)
        var_01 = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

    We map the desired mean/std from [low, high] to [0, 1]:
        mu_01 = (mean - low) / (high - low)
        var_01 = (std / (high - low))^2

    Then solve for alpha, beta via method of moments:
        alpha = mu_01 * ((mu_01 * (1 - mu_01)) / var_01 - 1)
        beta = (1 - mu_01) * ((mu_01 * (1 - mu_01)) / var_01 - 1)

    Args:
        mean: Desired mean on [low, high].
        std: Desired standard deviation on [low, high].
        low: Lower bound.
        high: Upper bound.

    Returns:
        Tuple (concentration1, concentration2) both > 0.

    Raises:
        ValueError: If the mean is outside [low, high] or std is too large
            for a valid Beta distribution.
    """
    if high <= low:
        raise ValueError(f"high ({high}) must be > low ({low})")
    if not (low <= mean <= high):
        raise ValueError(f"mean ({mean}) must be in [{low}, {high}]")

    range_width = high - low
    mu_01 = (mean - low) / range_width
    var_01 = (std / range_width) ** 2

    # Variance of Beta on [0,1] must be < mu*(1-mu)
    max_var = mu_01 * (1.0 - mu_01)
    if var_01 >= max_var:
        raise ValueError(
            f"std={std} is too large for Beta on [{low}, {high}] with mean={mean}. "
            f"Max std ~ {(max_var**0.5) * range_width:.4e}"
        )

    # Method of moments
    common = mu_01 * (1.0 - mu_01) / var_01 - 1.0
    alpha = mu_01 * common
    beta_param = (1.0 - mu_01) * common

    # Floor to avoid degenerate distributions
    alpha = max(alpha, 0.01)
    beta_param = max(beta_param, 0.01)

    return alpha, beta_param


def _default_prior(
    name: str,
    info: Any,
) -> PriorDistribution:
    """Build the default TruncatedNormal prior for a parameter.

    Args:
        name: Parameter name.
        info: ParameterInfo from the registry.

    Returns:
        TruncatedNormal prior distribution.
    """
    if name in _DEFAULT_PRIOR_SPECS:
        loc, scale = _DEFAULT_PRIOR_SPECS[name]
        return PriorDistribution.truncated_normal(
            loc=loc,
            scale=scale,
            low=info.min_bound,
            high=info.max_bound,
        )
    # Fallback for any unspecified parameter
    return PriorDistribution.uniform(info.min_bound, info.max_bound)


def _build_prior(
    name: str,
    prior_type_str: str,
    prior_params: dict[str, float],
    bounds: tuple[float, float],
) -> PriorDistribution:
    """Build a PriorDistribution from config strings.

    Args:
        name: Parameter name (for error messages)
        prior_type_str: One of "uniform", "normal", "lognormal",
            "halfnormal", "exponential"
        prior_params: Distribution-specific parameters
            (e.g. {"loc": 0, "scale": 1} for normal)
        bounds: (low, high) bounds, used as fallback for uniform

    Returns:
        Configured PriorDistribution
    """
    try:
        prior_type = PriorType(prior_type_str)
    except ValueError:
        valid = [pt.value for pt in PriorType]
        raise ValueError(
            f"Unknown prior type '{prior_type_str}' for parameter '{name}'. "
            f"Valid types: {valid}"
        ) from None

    if prior_type == PriorType.UNIFORM:
        low = prior_params.get("low", bounds[0])
        high = prior_params.get("high", bounds[1])
        return PriorDistribution.uniform(low, high)
    elif prior_type == PriorType.NORMAL:
        loc = prior_params.get("loc", (bounds[0] + bounds[1]) / 2)
        scale = prior_params.get("scale", (bounds[1] - bounds[0]) / 4)
        return PriorDistribution.normal(loc, scale)
    elif prior_type == PriorType.TRUNCATED_NORMAL:
        loc = prior_params.get("loc", (bounds[0] + bounds[1]) / 2)
        scale = prior_params.get("scale", (bounds[1] - bounds[0]) / 4)
        low = prior_params.get("low", bounds[0])
        high = prior_params.get("high", bounds[1])
        return PriorDistribution.truncated_normal(loc, scale, low, high)
    elif prior_type == PriorType.LOGNORMAL:
        loc = prior_params.get("loc", 0.0)
        scale = prior_params.get("scale", 1.0)
        return PriorDistribution.lognormal(loc, scale)
    elif prior_type == PriorType.HALFNORMAL:
        scale = prior_params.get("scale", 1.0)
        return PriorDistribution.halfnormal(scale)
    elif prior_type == PriorType.EXPONENTIAL:
        return PriorDistribution(PriorType.EXPONENTIAL, prior_params)
    elif prior_type == PriorType.BETA_SCALED:
        low = prior_params.get("low", bounds[0])
        high = prior_params.get("high", bounds[1])
        conc1 = prior_params.get("concentration1", 2.0)
        conc2 = prior_params.get("concentration2", 2.0)
        return PriorDistribution.beta_scaled(low, high, conc1, conc2)
    else:
        raise ValueError(f"Unhandled prior type: {prior_type}")


def clamp_to_open_interval(
    value: float,
    low: float,
    high: float,
    epsilon: float = 1e-6,
) -> float:
    """Clamp value to the open interval (low+epsilon, high-epsilon).

    Useful for Beta distribution parameters that must be strictly
    within their support bounds.

    Args:
        value: Value to clamp.
        low: Lower bound of the closed interval.
        high: Upper bound of the closed interval.
        epsilon: Margin to inset from the bounds.

    Returns:
        Clamped value in (low+epsilon, high-epsilon).
    """
    return max(low + epsilon, min(value, high - epsilon))
