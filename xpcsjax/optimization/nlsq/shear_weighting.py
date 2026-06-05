"""Shear-Sensitivity Weighting for Anti-Degeneracy Defense.

This module implements angle-dependent loss weighting to prevent gradient
cancellation in the shear term during optimization.

Part of Anti-Degeneracy Defense System v2.9.1.

The Problem
-----------
The shear term gradient is:
    d(g1_shear)/d(gamma_dot_t0) ~ cos(phi0 - phi)

When summed uniformly over all angles:
- Angles near phi0: cos(phi0 - phi) ~ +1 (positive contribution)
- Angles near phi0 +/- 90deg: cos ~ 0 (negligible)
- Angles near phi0 +/- 180deg: cos ~ -1 (negative contribution)

With uniformly distributed angles, positive and negative contributions
CANCEL, leading to near-zero net gradient for gamma_dot_t0. This causes
the shear parameter to collapse to its lower bound.

The Solution
------------
Use angle-dependent loss weighting:

    L = sum_phi w(phi) * sum_tau (g2_model - g2_exp)^2

where w(phi) emphasizes shear-sensitive angles:

    w(phi) = w_min + (1 - w_min) * abs(cos(phi0_current - phi))^alpha

This converts gradient cancellation into a weighted sum where shear-sensitive
angles (parallel/antiparallel to flow) contribute more than perpendicular
angles. All angles still contribute to prevent information loss.

Configuration
-------------
shear_weighting:
    enable: true                    # Enable shear-sensitivity weighting
    min_weight: 0.3                 # Minimum weight (0-1)
    alpha: 1.0                      # Shear sensitivity exponent (1 = linear)
    update_frequency: 1             # Update weights every N outer iterations
    initial_phi0: null              # Initial phi0 guess (null = use config)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.config import safe_float
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jax import Array

logger = get_logger(__name__)


# Performance Optimization (Spec 001 - FR-001, T014): JIT-compiled weight computation
# static_argnums=(4,): `normalize` is a config bool — never traced, prevents spurious
# retrace when the bool's concrete value changes between calls.
@partial(jax.jit, static_argnums=(4,))
def _compute_weights_jax(
    phi_angles: jnp.ndarray,
    phi0: float,
    min_weight: float,
    alpha: float,
    normalize: bool,
) -> jnp.ndarray:
    """JIT-compiled shear weight computation for optimal performance.

    Computes angle-dependent weights that emphasize shear-sensitive angles
    (parallel/antiparallel to flow direction).

    Parameters
    ----------
    phi_angles : jnp.ndarray
        Phi angles in degrees.
    phi0 : float
        Current phi0 estimate in degrees.
    min_weight : float
        Minimum weight for perpendicular angles (0-1).
    alpha : float
        Shear sensitivity exponent.
    normalize : bool
        Whether to normalize weights so mean = 1.

    Returns
    -------
    jnp.ndarray
        Weight array of shape (n_phi,).
    """
    # Convert to radians
    phi0_rad = jnp.radians(phi0)
    phi_rad = jnp.radians(phi_angles)

    # Compute shear sensitivity: |cos(phi0 - phi)|
    # Underflow protection: use jnp.where (gradient-safe) instead of jnp.maximum.
    # phi0 is a traced parameter; jnp.maximum zeros its gradient when cos_factor ≤ 1e-10.
    _cos_abs = jnp.abs(jnp.cos(phi0_rad - phi_rad))
    cos_factor = jnp.where(_cos_abs > 1e-10, _cos_abs, 1e-10)

    # Apply exponent and scale
    # w(phi) = w_min + (1 - w_min) * |cos(phi0 - phi)|^alpha
    weights = min_weight + (1.0 - min_weight) * (cos_factor**alpha)

    # Normalize if enabled using jax.lax.cond for JIT compatibility
    return jax.lax.cond(
        normalize,
        lambda w: w / jnp.mean(w),
        lambda w: w,
        weights,
    )


@dataclass
class ShearWeightingConfig:
    """Configuration for shear-sensitivity weighting.

    Attributes
    ----------
    enable : bool
        Enable shear-sensitivity weighting. Default True.
    min_weight : float
        Minimum weight for perpendicular angles. Range [0, 1]. Default 0.3.
    alpha : float
        Shear sensitivity exponent. Higher = more aggressive weighting.
        Default 1.0 (linear).
    update_frequency : int
        Update weights every N outer iterations. Default 1.
    initial_phi0 : float or None
        Initial phi0 guess in degrees. None = use config or 0.0.
    normalize : bool
        Normalize weights so sum = n_phi. Default True.
    """

    enable: bool = True
    min_weight: float = 0.3
    alpha: float = 1.0
    update_frequency: int = 1
    initial_phi0: float | None = None
    normalize: bool = True

    @classmethod
    def from_config(cls, config: Mapping) -> ShearWeightingConfig:
        """Create from configuration dictionary.

        Parameters
        ----------
        config : Mapping
            Configuration dictionary.

        Returns
        -------
        ShearWeightingConfig
            Configuration object.
        """
        sw_config = config.get("shear_weighting", {})

        return cls(
            enable=sw_config.get("enable", True),
            min_weight=safe_float(sw_config.get("min_weight"), 0.3),
            alpha=safe_float(sw_config.get("alpha"), 1.0),
            update_frequency=int(sw_config.get("update_frequency", 1)),
            initial_phi0=safe_float(sw_config.get("initial_phi0"), 0.0)
            if sw_config.get("initial_phi0") is not None
            else None,
            normalize=sw_config.get("normalize", True),
        )


class ShearSensitivityWeighting:
    """Shear-sensitivity weighted loss for anti-degeneracy defense.

    This class manages angle-dependent weights that emphasize shear-sensitive
    angles during optimization, preventing gradient cancellation.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles in degrees.
    n_physical : int
        Number of physical parameters.
    phi0_index : int
        Index of phi0 in physical parameters (typically 6 for laminar_flow).
    config : ShearWeightingConfig
        Weighting configuration.

    Examples
    --------
    >>> phi_angles = np.array([-30, 0, 30, 60, 90, 120])
    >>> weighter = ShearSensitivityWeighting(phi_angles, n_physical=7, phi0_index=6)
    >>> weights = weighter.get_weights(phi0_current=-5.0)
    >>> # Angles near -5 deg and 175 deg get higher weight
    """

    def __init__(
        self,
        phi_angles: np.ndarray,
        n_physical: int,
        phi0_index: int,
        config: ShearWeightingConfig | None = None,
    ):
        """Initialize the weighter and precompute weights for the initial phi0.

        See the class docstring for the parameter semantics. ``config`` defaults
        to a :class:`ShearWeightingConfig` with library defaults when ``None``.
        """
        self.phi_angles = np.asarray(phi_angles, dtype=np.float64)
        self.n_phi = len(self.phi_angles)
        self.n_physical = n_physical
        self.phi0_index = phi0_index
        self.config = config or ShearWeightingConfig()

        # Current phi0 estimate
        self._phi0_current = self.config.initial_phi0 or 0.0

        # Precomputed weight lookup (per phi index)
        self._weights = self._compute_weights(self._phi0_current)
        self._weights_jax = jnp.asarray(self._weights)

        # Tracking
        self._update_count = 0

        if self.config.enable:
            logger.info(
                f"ShearSensitivityWeighting initialized: "
                f"n_phi={self.n_phi}, min_weight={self.config.min_weight:.2f}, "
                f"alpha={self.config.alpha:.1f}, initial_phi0={self._phi0_current:.1f} deg"
            )

    def _compute_weights(self, phi0: float) -> np.ndarray:
        """Compute angle weights for given phi0.

        Performance Optimization (Spec 001 - FR-001, T015): Uses JIT-compiled
        computation for 2-3x speedup on repeated calls.

        Parameters
        ----------
        phi0 : float
            Current phi0 estimate in degrees.

        Returns
        -------
        np.ndarray
            Weight array of shape (n_phi,).
        """
        # Performance Optimization (Spec 001 - FR-001, T015): Use JIT-compiled version
        result = _compute_weights_jax(
            jnp.asarray(self.phi_angles),
            phi0,
            self.config.min_weight,
            self.config.alpha,
            self.config.normalize,
        )
        return np.asarray(result)

    def update_phi0(self, params: np.ndarray, iteration: int = 0) -> None:
        """Update phi0 estimate from current parameters.

        Parameters
        ----------
        params : np.ndarray
            Current parameter vector. Physical parameters should be at the end.
        iteration : int
            Current iteration number.
        """
        if not self.config.enable:
            return

        # Check if we should update this iteration
        if iteration % self.config.update_frequency != 0:
            return

        # Extract phi0 from parameters
        # Parameter layout: [per_angle_params, physical_params]
        # phi0 is the last physical parameter (index phi0_index from the end of physical)
        n_per_angle = len(params) - self.n_physical
        phi0_idx = n_per_angle + self.phi0_index
        new_phi0 = float(params[phi0_idx])

        # Check if phi0 has changed significantly
        if abs(new_phi0 - self._phi0_current) > 0.1:  # 0.1 degree threshold
            self._phi0_current = new_phi0
            self._weights = self._compute_weights(new_phi0)
            self._weights_jax = jnp.asarray(self._weights)
            self._update_count += 1

            logger.debug(
                f"ShearSensitivityWeighting updated: "
                f"phi0={new_phi0:.2f} deg, weights range=[{self._weights.min():.3f}, "
                f"{self._weights.max():.3f}]"
            )

    def get_weights(self, phi0_current: float | None = None) -> np.ndarray:
        """Get current angle weights.

        Parameters
        ----------
        phi0_current : float, optional
            Override phi0 for weight computation. If None, uses stored value.

        Returns
        -------
        np.ndarray
            Weight array of shape (n_phi,).
        """
        if phi0_current is not None and phi0_current != self._phi0_current:
            return self._compute_weights(phi0_current)
        return self._weights

    def get_weights_jax(self) -> Array:
        """Get current angle weights as JAX array.

        Returns
        -------
        jax.Array
            Weight array of shape (n_phi,).
        """
        return self._weights_jax

    def apply_weights_to_loss(self, residuals: Array, phi_indices: Array) -> Array:
        """Apply angle weights to residuals for loss computation.

        Computes weighted mean squared error:
            L = sum_i w[phi_idx[i]] * residuals[i]^2 / sum_i w[phi_idx[i]]

        Parameters
        ----------
        residuals : jax.Array
            Residuals array of shape (n_data,).
        phi_indices : jax.Array
            Phi index for each data point, shape (n_data,).

        Returns
        -------
        jax.Array
            Weighted loss (scalar).
        """
        if not self.config.enable:
            return jnp.mean(residuals**2) * len(residuals)

        # Lookup weights for each data point
        weights = self._weights_jax[phi_indices.astype(jnp.int32)]

        # Weighted mean squared error
        weighted_residuals_sq = weights * residuals**2
        weighted_loss = jnp.sum(weighted_residuals_sq)

        return weighted_loss

    def compute_weighted_mse(self, residuals: Array, phi_indices: Array) -> Array:
        """Compute weighted MSE (for gradient computation).

        Parameters
        ----------
        residuals : jax.Array
            Residuals array of shape (n_data,).
        phi_indices : jax.Array
            Phi index for each data point, shape (n_data,).

        Returns
        -------
        jax.Array
            Weighted MSE (scalar).
        """
        if not self.config.enable:
            return jnp.mean(residuals**2)

        # Lookup weights for each data point
        weights = self._weights_jax[phi_indices.astype(jnp.int32)]

        # Weighted mean: sum(w * r^2) / sum(w)
        weighted_mse = jnp.sum(weights * residuals**2) / jnp.sum(weights)

        return weighted_mse

    def get_diagnostics(self) -> dict:
        """Get weighting diagnostics.

        Returns
        -------
        dict
            Diagnostic information.
        """
        return {
            "enabled": self.config.enable,
            "min_weight": self.config.min_weight,
            "alpha": self.config.alpha,
            "current_phi0": self._phi0_current,
            "update_count": self._update_count,
            "weights_range": [float(self._weights.min()), float(self._weights.max())],
            "weights_mean": float(self._weights.mean()),
            "weights_std": float(self._weights.std()),
        }

    @property
    def phi0_current(self) -> float:
        """Current phi0 estimate in degrees."""
        return self._phi0_current


def create_shear_weighting(
    phi_angles: np.ndarray,
    n_physical: int,
    config: Mapping | None = None,
    physical_param_names: list[str] | None = None,
) -> ShearSensitivityWeighting | None:
    """Create a shear-weighting instance from config, if enabled.

    Parameters
    ----------
    phi_angles : np.ndarray
        Phi angles in degrees.
    n_physical : int
        Number of physical parameters.
    config : Mapping, optional
        Configuration dictionary.

    Returns
    -------
    ShearSensitivityWeighting or None
        Weighting object if enabled, None otherwise.
    """
    if config is None:
        return None

    sw_config = ShearWeightingConfig.from_config(config)

    if not sw_config.enable:
        logger.debug("Shear-sensitivity weighting disabled by config")
        return None

    # phi0 is typically the last of the 7 physical parameters in laminar_flow
    # Physical params: [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
    if physical_param_names is not None and "phi0" not in physical_param_names:
        logger.debug("phi0 not in physical params -- shear weighting disabled")
        return None
    phi0_index = (
        physical_param_names.index("phi0")
        if physical_param_names is not None and "phi0" in physical_param_names
        else 6
    )

    return ShearSensitivityWeighting(
        phi_angles=phi_angles,
        n_physical=n_physical,
        phi0_index=phi0_index,
        config=sw_config,
    )
