"""Parameter space definition with bounds for heterodyne NLSQ optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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


@dataclass
class ParameterSpace:
    """Complete parameter space for heterodyne model optimization.

    Manages parameter values, bounds, vary flags, and priors.
    """

    values: dict[str, float] = field(default_factory=dict)
    vary: dict[str, bool] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

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

        # Stash the original config dict on the instance so callers can
        # round-trip back to YAML. mypy doesn't allow a type annotation on a
        # non-self assignment (the ``space._config_dict: ...`` form is
        # reserved for ``self.<attr>``), so drop the inline annotation and
        # keep the existing attr-defined ignore.
        space._config_dict = config  # type: ignore[attr-defined]
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
