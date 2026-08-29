"""Pre-computed physics factors for efficient correlation computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    pass


@dataclass
class PhysicsFactors:
    """Pre-computed physics factors that do not depend on fit parameters.

    These are computed once from the experimental setup and reused across all
    optimization iterations for efficiency. This is the heterodyne-specific
    ``PhysicsFactors``, distinct from the homodyne
    :class:`xpcsjax.core.physics_factors.PhysicsFactors`.

    Attributes
    ----------
    t : jnp.ndarray
        Time array, shape ``(N,)``.
    q : float
        Scattering wavevector magnitude.
    q_squared : float
        Pre-computed ``q**2``.
    dt : float
        Time step.
    n_times : int
        Number of time points.
    phi_angle : float
        Detector phi angle in degrees.
    """

    # Time arrays
    t: jnp.ndarray  # Time array, shape (N,)

    # Scattering
    q: float  # Wavevector magnitude
    q_squared: float  # q²

    # Temporal
    dt: float  # Time step
    n_times: int  # Number of time points

    # Geometry
    phi_angle: float  # Detector phi angle (degrees)

    def __post_init__(self) -> None:
        """Validate that ``q`` and ``dt`` are finite and strictly positive."""
        if not math.isfinite(self.q) or self.q <= 0:
            raise ValueError(f"q must be finite and positive, got {self.q}")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"dt must be finite and positive, got {self.dt}")


def create_physics_factors(
    n_times: int,
    dt: float,
    q: float,
    phi_angle: float = 0.0,
    t_start: float = 0.0,
) -> PhysicsFactors:
    """Create physics factors from experimental parameters.

    Parameters
    ----------
    n_times : int
        Number of time points.
    dt : float
        Time step.
    q : float
        Scattering wavevector magnitude.
    phi_angle : float, optional
        Detector phi angle in degrees, default ``0.0``.
    t_start : float, optional
        Starting time, default ``0.0``.

    Returns
    -------
    PhysicsFactors
        A populated physics-factors instance.
    """
    # Create time array
    t = jnp.arange(n_times) * dt + t_start

    return PhysicsFactors(
        t=t,
        q=float(q),
        q_squared=float(q * q),
        dt=float(dt),
        n_times=n_times,
        phi_angle=float(phi_angle),
    )
