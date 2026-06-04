"""Model class hierarchy for heterodyne correlation analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode, get_registry
from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne

# Canonical 14-parameter heterodyne names, sourced from xpcsjax's
# parameter registry so the ordering tracks Tasks 24/25 renames
# (``v_beta``, ``phi0_het``) instead of the upstream heterodyne names
# (``beta``, ``phi0``) that would collide with the homodyne flow params.
ALL_PARAM_NAMES: tuple[str, ...] = tuple(get_registry().get_param_names(AnalysisMode.TWO_COMPONENT))

if TYPE_CHECKING:
    pass


class HeterodyneModelBase(ABC):
    """Abstract base class for heterodyne models."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of model parameters."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in order."""
        ...

    @abstractmethod
    def compute_correlation(
        self,
        params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi_angle: float,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute model correlation matrix.

        Args:
            params: Parameter array
            t: Time array
            q: Wavevector
            dt: Time step
            phi_angle: Detector phi angle (degrees)
            contrast: Speckle contrast (beta), default 1.0
            offset: Baseline offset, default 1.0

        Returns:
            Correlation matrix
        """
        ...

    @abstractmethod
    def get_default_params(self) -> np.ndarray:
        """Get default parameter values."""
        ...


@dataclass
class TwoComponentModel(HeterodyneModelBase):
    """Two-component heterodyne correlation model.

    Implements the 14-parameter model:
    - Reference transport (3): D0_ref, alpha_ref, D_offset_ref
    - Sample transport (3): D0_sample, alpha_sample, D_offset_sample
    - Velocity (3): v0, v_beta, v_offset
    - Fraction (4): f0, f1, f2, f3
    - Angle (1): phi0_het
    """

    _defaults: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default parameter values."""
        if not self._defaults:
            self._defaults = {
                "D0_ref": 1e4,
                "alpha_ref": 0.0,
                "D_offset_ref": 0.0,
                "D0_sample": 1e4,
                "alpha_sample": 0.0,
                "D_offset_sample": 0.0,
                "v0": 1e3,
                "v_beta": 0.0,
                "v_offset": 0.0,
                "f0": 0.5,
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
                "phi0_het": 0.0,
            }

    @property
    def n_params(self) -> int:
        """Number of parameters (14)."""
        return 14

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names in canonical order."""
        return ALL_PARAM_NAMES

    def compute_correlation(
        self,
        params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi_angle: float,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute two-time heterodyne correlation.

        Args:
            params: Parameter array, shape (14,)
            t: Time array
            q: Scattering wavevector
            dt: Time step
            phi_angle: Detector phi angle (degrees)
            contrast: Speckle contrast (beta), default 1.0
            offset: Baseline offset, default 1.0

        Returns:
            Correlation matrix c2(t1, t2), shape (N, N)
        """
        return compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)  # type: ignore[no-any-return]

    def get_default_params(self) -> np.ndarray:
        """Get default parameter values as array."""
        return np.array([self._defaults[name] for name in ALL_PARAM_NAMES])

    def params_to_dict(self, params: np.ndarray | jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary.

        Args:
            params: Parameter array, shape (14,)

        Returns:
            Dict mapping names to values
        """
        return {name: float(params[i]) for i, name in enumerate(ALL_PARAM_NAMES)}

    def dict_to_params(self, param_dict: dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array.

        Args:
            param_dict: Dict with parameter names as keys

        Returns:
            Parameter array, shape (14,)
        """
        return np.array([param_dict.get(name, self._defaults[name]) for name in ALL_PARAM_NAMES])

    def compute_g1_reference(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
        q: float,
    ) -> jnp.ndarray:
        """Compute reference g1 correlation only (1D visualization helper).

        .. note::
            Uses pointwise g1(t) = exp(-q²J(t)), which does not represent
            the two-time integral physics. For production correlation, use
            compute_correlation which uses the integral formulation.

        Args:
            params: Full parameter array
            t: Time array
            q: Wavevector

        Returns:
            g1_ref array
        """
        D0, alpha, offset = params[0], params[1], params[2]
        # Use jnp.where instead of jnp.maximum to preserve gradients at the
        # t=0 floor (jnp.maximum zeros the gradient when t < 1e-10).
        t_safe = jnp.where(t > 1e-10, t, 1e-10)
        J = D0 * jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0) + offset
        # Physical positivity: jnp.maximum gives subgradient 0.5 at J=0,
        # allowing offset gradient to pass through the boundary.
        J = jnp.maximum(J, 0.0)
        return jnp.exp(-q * q * J)

    def compute_g1_sample(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
        q: float,
    ) -> jnp.ndarray:
        """Compute sample g1 correlation only (1D visualization helper).

        .. note::
            Uses pointwise g1(t) = exp(-q²J(t)), which does not represent
            the two-time integral physics. For production correlation, use
            compute_correlation which uses the integral formulation.

        Args:
            params: Full parameter array
            t: Time array
            q: Wavevector

        Returns:
            g1_sample array
        """
        D0, alpha, offset = params[3], params[4], params[5]
        # Use jnp.where instead of jnp.maximum to preserve gradients at the
        # t=0 floor (jnp.maximum zeros the gradient when t < 1e-10).
        t_safe = jnp.where(t > 1e-10, t, 1e-10)
        J = D0 * jnp.where(t > 0, jnp.power(t_safe, alpha), 0.0) + offset
        # Physical positivity: jnp.maximum gives subgradient 0.5 at J=0,
        # allowing offset gradient to pass through the boundary.
        J = jnp.maximum(J, 0.0)
        return jnp.exp(-q * q * J)

    def compute_fraction(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute sample fraction only.

        Args:
            params: Full parameter array
            t: Time array

        Returns:
            f_sample array in [0, 1]
        """
        f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
        exponent = jnp.clip(f1 * (t - f2), -100, 100)
        return jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)


@dataclass
class ReducedModel(HeterodyneModelBase):
    """Reduced heterodyne model with a subset of active parameters.

    Inactive parameters are held fixed at their canonical default values.
    Useful for simplified analysis modes (e.g., reference-only diffusion).

    Args:
        _active_params: Ordered tuple of parameter names that are free to vary.
    """

    _active_params: tuple[str, ...]

    # Full default values for all 14 parameters (canonical defaults)
    _FULL_DEFAULTS: dict[str, float] = field(
        default_factory=lambda: {
            "D0_ref": 1e4,
            "alpha_ref": 0.0,
            "D_offset_ref": 0.0,
            "D0_sample": 1e4,
            "alpha_sample": 0.0,
            "D_offset_sample": 0.0,
            "v0": 1e3,
            "v_beta": 0.0,
            "v_offset": 0.0,
            "f0": 0.5,
            "f1": 0.0,
            "f2": 0.0,
            "f3": 0.0,
            "phi0_het": 0.0,
        }
    )

    def __post_init__(self) -> None:
        """Validate active params and precompute expansion constants."""
        invalid = [n for n in self._active_params if n not in ALL_PARAM_NAMES]
        if invalid:
            raise ValueError(
                f"Unknown parameter names: {invalid}. Valid names: {list(ALL_PARAM_NAMES)}"
            )
        # Precompute template and index mapping for _expand_to_full
        object.__setattr__(
            self,
            "_template",
            jnp.array([self._FULL_DEFAULTS[name] for name in ALL_PARAM_NAMES]),
        )
        idx_list = [ALL_PARAM_NAMES.index(name) for name in self._active_params]
        object.__setattr__(self, "_active_indices", tuple(idx_list))
        object.__setattr__(
            self,
            "_active_indices_array",
            jnp.array(idx_list, dtype=jnp.int32),
        )

    @property
    def n_params(self) -> int:
        """Number of active (free) parameters."""
        return len(self._active_params)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Active parameter names in order."""
        return self._active_params

    def get_default_params(self) -> np.ndarray:
        """Get default values for active parameters only."""
        return np.array([self._FULL_DEFAULTS[name] for name in self._active_params])

    def _expand_to_full(self, params: jnp.ndarray) -> jnp.ndarray:
        """Expand active-parameter array to full 14-element array.

        Uses precomputed template and index mapping for efficiency.
        Inactive parameters retain their canonical defaults.

        Args:
            params: Active-parameter array, shape (n_params,)

        Returns:
            Full parameter array, shape (14,)
        """
        return self._template.at[self._active_indices_array].set(params)  # type: ignore[attr-defined,no-any-return]

    def compute_correlation(
        self,
        params: jnp.ndarray,
        t: jnp.ndarray,
        q: float,
        dt: float,
        phi_angle: float,
        contrast: float = 1.0,
        offset: float = 1.0,
    ) -> jnp.ndarray:
        """Compute model correlation from reduced parameter set.

        Inactive parameters are held at canonical defaults.

        Args:
            params: Active-parameter array, shape (n_params,)
            t: Time array
            q: Scattering wavevector
            dt: Time step
            phi_angle: Detector phi angle (degrees)
            contrast: Speckle contrast (beta), default 1.0
            offset: Baseline offset, default 1.0

        Returns:
            Correlation matrix c2(t1, t2), shape (N, N)
        """
        full_params = self._expand_to_full(params)
        return compute_c2_heterodyne(full_params, t, q, dt, phi_angle, contrast, offset)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Analysis mode registry
# ---------------------------------------------------------------------------

ANALYSIS_MODES: dict[str, tuple[str, ...]] = {
    "static_ref": ("D0_ref", "alpha_ref", "D_offset_ref"),
    "static_both": (
        "D0_ref",
        "alpha_ref",
        "D_offset_ref",
        "D0_sample",
        "alpha_sample",
        "D_offset_sample",
    ),
    "two_component": ALL_PARAM_NAMES,
}


def create_model(mode: str) -> HeterodyneModelBase:
    """Factory function that returns a model for the requested analysis mode.

    Args:
        mode: One of ``"static_ref"``, ``"static_both"``, ``"two_component"``.

    Returns:
        ``TwoComponentModel`` for ``"two_component"``;
        ``ReducedModel`` for all other recognised modes.

    Raises:
        ValueError: If *mode* is not a recognised analysis mode.
    """
    if mode not in ANALYSIS_MODES:
        valid = ", ".join(sorted(ANALYSIS_MODES))
        raise ValueError(f"Unknown analysis mode '{mode}'. Valid modes: {valid}")
    if mode == "two_component":
        return TwoComponentModel()
    return ReducedModel(_active_params=ANALYSIS_MODES[mode])


# Default model instance
DEFAULT_MODEL = TwoComponentModel()
