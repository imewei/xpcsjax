"""Pre-computed physics factors for efficient correlation computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class PhysicsFactors:
    """Pre-computed factors that don't depend on fit parameters.

    These are computed once from experimental setup and reused across
    all optimization iterations for efficiency.
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
        """Validate factors."""
        if self.q <= 0:
            raise ValueError(f"q must be positive, got {self.q}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

    @property
    def time_extent(self) -> float:
        """Total time span."""
        return float(self.t[-1] - self.t[0])

    def get_q_cosine(self, phi0: float = 0.0) -> jnp.ndarray:
        """Get q * cos(phi_total) for cross-term phase.

        Args:
            phi0: Additional angle from fit parameters

        Returns:
            q * cos(phi_angle + phi0) as JAX scalar
        """
        total_phi_rad = jnp.deg2rad(self.phi_angle + phi0)
        return self.q * jnp.cos(total_phi_rad)


def create_physics_factors(
    n_times: int,
    dt: float,
    q: float,
    phi_angle: float = 0.0,
    t_start: float = 0.0,
) -> PhysicsFactors:
    """Create physics factors from experimental parameters.

    Args:
        n_times: Number of time points
        dt: Time step
        q: Scattering wavevector magnitude
        phi_angle: Detector phi angle (degrees)
        t_start: Starting time (default 0)

    Returns:
        PhysicsFactors instance
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


def create_physics_factors_from_config(config: dict) -> PhysicsFactors:
    """Create physics factors from configuration dictionary.

    Reads from ``analyzer_parameters`` (canonical) with fallback to legacy
    ``temporal``/``scattering`` top-level sections for backwards compatibility.

    Args:
        config: Configuration with ``analyzer_parameters`` or legacy
            ``temporal``/``scattering`` sections.

    Returns:
        PhysicsFactors instance
    """
    ap = config.get("analyzer_parameters", {})
    temporal = config.get("temporal", {})
    scattering = config.get("scattering", {})

    # Prefer analyzer_parameters; fall back to legacy sections
    dt = float(ap.get("dt", temporal.get("dt", 1.0)))

    if "start_frame" in ap:
        start_frame = int(ap["start_frame"])
        end_frame = int(ap["end_frame"])
        n_times = end_frame - start_frame + 1
        t_start = dt  # relative time within window: first usable frame at 1×dt
    else:
        n_times = int(temporal.get("time_length", 1000))
        t_start = float(temporal.get("t_start", dt))

    ap_scat = ap.get("scattering", {})
    q = float(ap_scat.get("wavevector_q", scattering.get("wavevector_q", 0.01)))

    logger.debug(
        "Physics factors: n_times=%d, dt=%.4e, q=%.4f, t_start=%.4e",
        n_times,
        dt,
        q,
        float(t_start),
    )
    return create_physics_factors(
        n_times=n_times,
        dt=dt,
        q=q,
        phi_angle=0.0,  # Set per-fit
        t_start=float(t_start),
    )


@dataclass
class CachedMatrices:
    """Cached matrices that depend only on time grid.

    These are expensive to recompute and don't change during fitting.
    """

    # Time difference matrix: |t1 - t2|
    time_diff: jnp.ndarray

    # Age matrix: (t1 + t2) / 2
    mean_time: jnp.ndarray

    # Indices for upper/lower triangular
    triu_indices: tuple[jnp.ndarray, jnp.ndarray]
    tril_indices: tuple[jnp.ndarray, jnp.ndarray]


def create_cached_matrices(factors: PhysicsFactors) -> CachedMatrices:
    """Create cached matrices from physics factors.

    Args:
        factors: PhysicsFactors instance

    Returns:
        CachedMatrices instance
    """
    t1, t2 = jnp.meshgrid(factors.t, factors.t, indexing="ij")
    n = factors.n_times

    return CachedMatrices(
        time_diff=jnp.abs(t1 - t2),
        mean_time=(t1 + t2) / 2,
        triu_indices=jnp.triu_indices(n),
        tril_indices=jnp.tril_indices(n),
    )
