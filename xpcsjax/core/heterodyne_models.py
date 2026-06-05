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
        """Compute the model two-time correlation matrix.

        Parameters
        ----------
        params : jnp.ndarray
            Parameter array.
        t : jnp.ndarray
            Time array.
        q : float
            Scattering wavevector magnitude.
        dt : float
            Time step.
        phi_angle : float
            Detector phi angle in degrees.
        contrast : float, optional
            Speckle contrast (the kernel-internal ``beta``), default ``1.0``.
        offset : float, optional
            Baseline offset, default ``1.0``.

        Returns
        -------
        jnp.ndarray
            Correlation matrix.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> np.ndarray:
        """Return the default parameter values."""
        ...


@dataclass
class TwoComponentModel(HeterodyneModelBase):
    """Two-component heterodyne correlation model.

    Implements the canonical 14-parameter model, in registry order
    ``[D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample,
    D_offset_sample, v0, v_beta, v_offset, f0, f1, f2, f3, phi0_het]``:

    - Reference transport (3): ``D0_ref``, ``alpha_ref``, ``D_offset_ref``
    - Sample transport (3): ``D0_sample``, ``alpha_sample``, ``D_offset_sample``
    - Velocity (3): ``v0``, ``v_beta``, ``v_offset``
    - Fraction (4): ``f0``, ``f1``, ``f2``, ``f3``
    - Angle (1): ``phi0_het``

    Notes
    -----
    The parameter registry (``xpcsjax.config.parameter_registry``) is the
    authoritative source for names, ordering, and bounds. The config-facing
    names ``v_beta`` / ``phi0_het`` map to the kernel-internal ``beta`` /
    ``phi0``; both name the same quantities.

    Examples
    --------
    >>> model = TwoComponentModel()
    >>> params = model.get_default_params()
    >>> c2 = model.compute_correlation(params, t, q=0.01, dt=0.1, phi_angle=45.0)
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
        """Compute the two-time heterodyne correlation matrix.

        Parameters
        ----------
        params : jnp.ndarray
            Parameter array, shape ``(14,)`` in canonical registry order.
        t : jnp.ndarray
            Time array.
        q : float
            Scattering wavevector magnitude.
        dt : float
            Time step.
        phi_angle : float
            Detector phi angle in degrees.
        contrast : float, optional
            Speckle contrast (the kernel-internal ``beta``), default ``1.0``.
        offset : float, optional
            Baseline offset, default ``1.0``.

        Returns
        -------
        jnp.ndarray
            Correlation matrix ``c2(t1, t2)``, shape ``(N, N)``.
        """
        return compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast, offset)  # type: ignore[no-any-return]

    def get_default_params(self) -> np.ndarray:
        """Return the default parameter values as an array."""
        return np.array([self._defaults[name] for name in ALL_PARAM_NAMES])

    def params_to_dict(self, params: np.ndarray | jnp.ndarray) -> dict[str, float]:
        """Convert a parameter array to a name-keyed dictionary.

        Parameters
        ----------
        params : np.ndarray or jnp.ndarray
            Parameter array, shape ``(14,)``.

        Returns
        -------
        dict of str to float
            Mapping from canonical parameter names to values.
        """
        return {name: float(params[i]) for i, name in enumerate(ALL_PARAM_NAMES)}

    def dict_to_params(self, param_dict: dict[str, float]) -> np.ndarray:
        """Convert a name-keyed parameter dictionary to an array.

        Missing names fall back to the canonical default value.

        Parameters
        ----------
        param_dict : dict of str to float
            Mapping with parameter names as keys.

        Returns
        -------
        np.ndarray
            Parameter array, shape ``(14,)`` in canonical registry order.
        """
        return np.array([param_dict.get(name, self._defaults[name]) for name in ALL_PARAM_NAMES])

    def compute_g1_reference(
        self,
        params: np.ndarray | jnp.ndarray,
        t: jnp.ndarray,
        q: float,
    ) -> jnp.ndarray:
        """Compute the reference g1 correlation only (1D visualization helper).

        Parameters
        ----------
        params : np.ndarray or jnp.ndarray
            Full parameter array (the reference transport triple is used).
        t : jnp.ndarray
            Time array.
        q : float
            Scattering wavevector magnitude.

        Returns
        -------
        jnp.ndarray
            Reference ``g1`` array.

        Notes
        -----
        Uses the pointwise form ``g1(t) = exp(-q**2 J(t))``, which does not
        represent the two-time integral physics. For production correlation
        use :meth:`compute_correlation`, which uses the integral formulation.
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
        """Compute the sample g1 correlation only (1D visualization helper).

        Parameters
        ----------
        params : np.ndarray or jnp.ndarray
            Full parameter array (the sample transport triple is used).
        t : jnp.ndarray
            Time array.
        q : float
            Scattering wavevector magnitude.

        Returns
        -------
        jnp.ndarray
            Sample ``g1`` array.

        Notes
        -----
        Uses the pointwise form ``g1(t) = exp(-q**2 J(t))``, which does not
        represent the two-time integral physics. For production correlation
        use :meth:`compute_correlation`, which uses the integral formulation.
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
        """Compute the sample fraction only.

        Parameters
        ----------
        params : np.ndarray or jnp.ndarray
            Full parameter array (the fraction parameters ``f0..f3`` are used).
        t : jnp.ndarray
            Time array.

        Returns
        -------
        jnp.ndarray
            Sample fraction array clipped to ``[0, 1]``.
        """
        f0, f1, f2, f3 = params[9], params[10], params[11], params[12]
        exponent = jnp.clip(f1 * (t - f2), -100, 100)
        return jnp.clip(f0 * jnp.exp(exponent) + f3, 0.0, 1.0)


@dataclass
class ReducedModel(HeterodyneModelBase):
    """Reduced heterodyne model with a subset of active parameters.

    Inactive parameters are held fixed at their canonical default values.
    Useful for simplified analysis modes (e.g. reference-only diffusion).

    Parameters
    ----------
    _active_params : tuple of str
        Ordered tuple of parameter names that are free to vary.
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
        """Expand an active-parameter array to the full 14-element array.

        Uses a precomputed template and index mapping for efficiency;
        inactive parameters retain their canonical defaults.

        Parameters
        ----------
        params : jnp.ndarray
            Active-parameter array, shape ``(n_params,)``.

        Returns
        -------
        jnp.ndarray
            Full parameter array, shape ``(14,)``.
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
        """Compute the model correlation from the reduced parameter set.

        Inactive parameters are held at their canonical defaults.

        Parameters
        ----------
        params : jnp.ndarray
            Active-parameter array, shape ``(n_params,)``.
        t : jnp.ndarray
            Time array.
        q : float
            Scattering wavevector magnitude.
        dt : float
            Time step.
        phi_angle : float
            Detector phi angle in degrees.
        contrast : float, optional
            Speckle contrast (the kernel-internal ``beta``), default ``1.0``.
        offset : float, optional
            Baseline offset, default ``1.0``.

        Returns
        -------
        jnp.ndarray
            Correlation matrix ``c2(t1, t2)``, shape ``(N, N)``.
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
    """Return a model instance for the requested analysis mode.

    Parameters
    ----------
    mode : str
        One of ``"static_ref"``, ``"static_both"``, or ``"two_component"``.

    Returns
    -------
    HeterodyneModelBase
        A :class:`TwoComponentModel` for ``"two_component"``; a
        :class:`ReducedModel` for all other recognised modes.

    Raises
    ------
    ValueError
        If ``mode`` is not a recognised analysis mode.
    """
    if mode not in ANALYSIS_MODES:
        valid = ", ".join(sorted(ANALYSIS_MODES))
        raise ValueError(f"Unknown analysis mode '{mode}'. Valid modes: {valid}")
    if mode == "two_component":
        return TwoComponentModel()
    return ReducedModel(_active_params=ANALYSIS_MODES[mode])


# Default model instance
DEFAULT_MODEL = TwoComponentModel()
