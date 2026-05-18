"""Physics Factors Pre-computation Module
========================================

Pre-computes and caches physics factors for efficient XPCS calculations.
This module implements the medium-term and long-term architectural improvements
identified in the architectural comparison analysis.

The PhysicsFactors class provides:
1. Eager pre-computation of physics factors at initialization
2. Immutable, validated factor storage
3. JIT-compatible tuple representation for functional cores
4. Type safety and comprehensive validation

Physics Factors:
----------------
1. wavevector_q_squared_half_dt = 0.5 * q² * dt
   - Used in diffusion correlation: exp(-factor * D_integral)

2. sinc_prefactor = (qL/2π) * dt
   - Used in shear correlation: sinc(prefactor * cos(φ) * gamma_integral)

These factors are computed once and reused across all correlation calculations,
improving both performance and safety by eliminating repeated computation and
validation.
"""

from dataclasses import dataclass

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import JAX for type hints
try:
    import jax.numpy as jnp
except ImportError:
    jnp = np  # type: ignore[misc]


@dataclass(frozen=True)
class PhysicsFactors:
    """Pre-computed physics factors for XPCS correlation calculations.

    This immutable dataclass stores pre-computed physics factors that depend
    on experimental configuration (q, L, dt) but not on model parameters.
    By pre-computing these once, we:

    1. Improve performance (no repeated computation)
    2. Enhance safety (validate once at creation)
    3. Enable caching (immutable and hashable)
    4. Support JIT compilation (via tuple representation)

    Attributes
    ----------
    wavevector_q : float
        Scattering wave vector magnitude [Å⁻¹]
        Physical range: 10⁻⁴ to 10⁻¹ Å⁻¹ (typical XPCS)
    stator_rotor_gap : float
        Sample-detector distance (characteristic length) [Å]
        Physical range: 10⁵ to 10⁷ Å (1-1000 μm typical)
    dt : float
        Time step between frames [s]
        Physical range: 10⁻³ to 10² s (typical XPCS frame rates)
    wavevector_q_squared_half_dt : float
        Pre-computed factor: 0.5 * q² * dt
        Used in diffusion correlation calculation
    sinc_prefactor : float
        Pre-computed factor: (q * L / 2π) * dt
        Used in shear correlation calculation

    Examples
    --------
    >>> # Basic usage
    >>> factors = PhysicsFactors.from_config(q=0.01, L=2e6, dt=0.1)
    >>> factors.wavevector_q_squared_half_dt
    5e-06

    >>> # JIT-compatible usage
    >>> @jit
    >>> def compute_correlation(params, factors_tuple):
    >>>     q_factor, sinc_factor = factors_tuple
    >>>     # ... use factors in JIT-compiled code
    >>>
    >>> result = compute_correlation(params, factors.to_tuple())

    >>> # Validation
    >>> invalid = PhysicsFactors.from_config(q=-0.01, L=2e6, dt=0.1)
    ValueError: wavevector_q must be positive, got -0.01
    """

    # Configuration parameters
    wavevector_q: float
    stator_rotor_gap: float
    dt: float

    # Pre-computed factors
    wavevector_q_squared_half_dt: float
    sinc_prefactor: float

    def __post_init__(self) -> None:
        """Validate physics factors after initialization."""
        self._validate()

    @classmethod
    def from_config(
        cls,
        q: float,
        L: float,
        dt: float,
        validate: bool = True,
    ) -> "PhysicsFactors":
        """Create PhysicsFactors from experimental configuration.

        This is the recommended way to create PhysicsFactors instances.
        It computes the derived factors and optionally validates all values.

        Parameters
        ----------
        q : float
            Scattering wave vector magnitude [Å⁻¹]
        L : float
            Sample-detector distance (stator_rotor_gap) [Å]
        dt : float
            Time step between frames [s]
        validate : bool, default=True
            Whether to validate parameter ranges

        Returns
        -------
        PhysicsFactors
            Immutable instance with pre-computed factors

        Raises
        ------
        ValueError
            If any parameter is invalid and validate=True

        Examples
        --------
        >>> factors = PhysicsFactors.from_config(
        ...     q=0.01,           # 0.01 Å⁻¹
        ...     L=2e6,            # 2000 μm = 2 mm
        ...     dt=0.1            # 0.1 s = 100 ms
        ... )
        >>> factors.wavevector_q_squared_half_dt
        5e-06
        >>> factors.sinc_prefactor
        318.30988618379064
        """
        # Compute derived factors
        dt_value = dt
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
        sinc_prefactor = 0.5 / np.pi * q * L * dt_value

        # Create instance (validation happens in __post_init__)
        instance = cls(
            wavevector_q=q,
            stator_rotor_gap=L,
            dt=dt_value,
            wavevector_q_squared_half_dt=wavevector_q_squared_half_dt,
            sinc_prefactor=sinc_prefactor,
        )

        logger.debug(f"Created PhysicsFactors: q={q:.6e}, L={L:.6e}, dt={dt_value:.6e}")
        logger.debug(f"  q^2*dt/2 = {wavevector_q_squared_half_dt:.6e}")
        logger.debug(f"  q*L*dt/(2*pi) = {sinc_prefactor:.6e}")

        return instance

    def _validate(self) -> None:
        """Validate physics factors for physical consistency.

        Checks:
        1. All values are positive
        2. Values are finite (not NaN or inf)
        3. Values are within physically reasonable ranges

        Raises
        ------
        ValueError
            If any validation check fails
        """
        # Check positivity
        if self.wavevector_q <= 0:
            raise ValueError(f"wavevector_q must be positive, got {self.wavevector_q}")
        if self.stator_rotor_gap <= 0:
            raise ValueError(
                f"stator_rotor_gap must be positive, got {self.stator_rotor_gap}",
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        # Check finiteness
        if not np.isfinite(self.wavevector_q):
            raise ValueError(f"wavevector_q must be finite, got {self.wavevector_q}")
        if not np.isfinite(self.stator_rotor_gap):
            raise ValueError(
                f"stator_rotor_gap must be finite, got {self.stator_rotor_gap}",
            )
        if not np.isfinite(self.dt):
            raise ValueError(f"dt must be finite, got {self.dt}")

        # Check reasonable ranges (warn but don't fail)
        if self.wavevector_q < 1e-4 or self.wavevector_q > 1.0:
            logger.warning(
                f"wavevector_q = {self.wavevector_q:.6e} A^-1 is outside typical "
                f"XPCS range [1e-4, 1] A^-1",
            )

        if self.stator_rotor_gap < 1e5 or self.stator_rotor_gap > 1e8:
            logger.warning(
                f"stator_rotor_gap = {self.stator_rotor_gap:.6e} A is outside typical "
                f"range [1e5, 1e8] A (10 um - 10 mm)",
            )

        if self.dt < 1e-6 or self.dt > 1e3:
            logger.warning(
                f"dt = {self.dt:.6e} s is outside typical XPCS range [1e-6, 1e3] s",
            )

        # Check derived factors
        if not np.isfinite(self.wavevector_q_squared_half_dt):
            raise ValueError(
                "Computed wavevector_q_squared_half_dt is not finite. "
                "Check q and dt values.",
            )
        if not np.isfinite(self.sinc_prefactor):
            raise ValueError(
                "Computed sinc_prefactor is not finite. Check q, L, and dt values.",
            )

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple for JIT-compatible function calls.

        Returns the two pre-computed factors as a tuple that can be
        unpacked in JIT-compiled functions. This is the recommended
        way to pass factors to functional cores.

        Returns
        -------
        tuple of (float, float)
            (wavevector_q_squared_half_dt, sinc_prefactor)

        Examples
        --------
        >>> factors = PhysicsFactors.from_config(q=0.01, L=2e6, dt=0.1)
        >>> q_factor, sinc_factor = factors.to_tuple()
        >>>
        >>> @jit
        >>> def compute_g1(params, q_factor, sinc_factor):
        >>>     # Use factors in JIT-compiled code
        >>>     g1_diff = jnp.exp(-q_factor * D_integral)
        >>>     # ...
        >>>
        >>> result = compute_g1(params, *factors.to_tuple())
        """
        return (self.wavevector_q_squared_half_dt, self.sinc_prefactor)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization or inspection.

        Returns
        -------
        dict
            Dictionary with all factor values

        Examples
        --------
        >>> factors = PhysicsFactors.from_config(q=0.01, L=2e6, dt=0.1)
        >>> factors.to_dict()
        {
            'wavevector_q': 0.01,
            'stator_rotor_gap': 2000000.0,
            'dt': 0.1,
            'wavevector_q_squared_half_dt': 5e-06,
            'sinc_prefactor': 318.30988618379064
        }
        """
        return {
            "wavevector_q": self.wavevector_q,
            "stator_rotor_gap": self.stator_rotor_gap,
            "dt": self.dt,
            "wavevector_q_squared_half_dt": self.wavevector_q_squared_half_dt,
            "sinc_prefactor": self.sinc_prefactor,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"PhysicsFactors(q={self.wavevector_q:.6e} A^-1, "
            f"L={self.stator_rotor_gap:.6e} A, "
            f"dt={self.dt:.6e} s)"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"PhysicsFactors(\n"
            f"  wavevector_q={self.wavevector_q:.6e},\n"
            f"  stator_rotor_gap={self.stator_rotor_gap:.6e},\n"
            f"  dt={self.dt:.6e},\n"
            f"  wavevector_q_squared_half_dt={self.wavevector_q_squared_half_dt:.6e},\n"
            f"  sinc_prefactor={self.sinc_prefactor:.6e}\n"
            f")"
        )


def create_physics_factors_from_config_dict(config: dict) -> PhysicsFactors:
    """Create PhysicsFactors from a homodyne configuration dictionary.

    Convenience function that extracts the necessary parameters from
    a standard homodyne configuration dictionary.

    Parameters
    ----------
    config : dict
        Homodyne configuration dictionary with structure::

            {
                'analyzer_parameters': {
                    'temporal': {'dt': float},
                    'scattering': {'wavevector_q': float},
                    'geometry': {'stator_rotor_gap': float}
                }
            }

    Returns
    -------
    PhysicsFactors
        Pre-computed physics factors

    Raises
    ------
    KeyError
        If required configuration keys are missing
    ValueError
        If parameter values are invalid

    Examples
    --------
    >>> config = {
    ...     'analyzer_parameters': {
    ...         'temporal': {'dt': 0.1},
    ...         'scattering': {'wavevector_q': 0.01},
    ...         'geometry': {'stator_rotor_gap': 2e6}
    ...     }
    ... }
    >>> factors = create_physics_factors_from_config_dict(config)
    """
    try:
        analyzer_params = config["analyzer_parameters"]
        dt = analyzer_params["temporal"]["dt"]
        q = analyzer_params["scattering"]["wavevector_q"]
        L = analyzer_params["geometry"]["stator_rotor_gap"]

        return PhysicsFactors.from_config(q=q, L=L, dt=dt)

    except KeyError as e:
        raise KeyError(
            f"Missing required configuration key: {e}. "
            f"Expected structure: config['analyzer_parameters']['temporal']['dt'], "
            f"config['analyzer_parameters']['scattering']['wavevector_q'], "
            f"config['analyzer_parameters']['geometry']['stator_rotor_gap']",
        ) from e


__all__ = ["PhysicsFactors", "create_physics_factors_from_config_dict"]
