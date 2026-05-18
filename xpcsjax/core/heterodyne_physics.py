"""Physical constants, parameter bounds, and validation for heterodyne model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class ValidationResult:
    """Result of parameter validation with detailed error reporting.

    Attributes:
        valid: True if all parameters are within bounds.
        violations: List of human-readable violation messages.
        parameters_checked: Number of parameters validated.
        message: Summary message about validation result.
    """

    valid: bool
    violations: list[str] = field(default_factory=list)
    parameters_checked: int = 0
    message: str = ""

    def __str__(self) -> str:
        if self.valid:
            return f"OK {self.message}"
        violations_str = "\n  - ".join(self.violations)
        return f"FAIL {self.message}\n  - {violations_str}"


@dataclass(frozen=True)
class PhysicsConstants:
    """Physical constants for XPCS scattering analysis.

    All values in SI base units unless otherwise noted.
    """

    # Boltzmann constant (J/K)
    k_B: ClassVar[float] = 1.380649e-23

    # Planck constant (J·s)
    h: ClassVar[float] = 6.62607015e-34

    # Speed of light (m/s)
    c: ClassVar[float] = 299792458.0

    # X-ray wavelengths (Å) - common energies
    WAVELENGTH_8KEV: ClassVar[float] = 1.55  # Å
    WAVELENGTH_10KEV: ClassVar[float] = 1.24  # Å
    WAVELENGTH_12KEV: ClassVar[float] = 1.0332  # Å

    # Typical q-ranges (inverse Angstroms)
    Q_MIN_TYPICAL: ClassVar[float] = 1e-4
    Q_MAX_TYPICAL: ClassVar[float] = 1.0

    # Time scales (seconds)
    TIME_MIN_XPCS: ClassVar[float] = 1e-6  # Microsecond resolution
    TIME_MAX_XPCS: ClassVar[float] = 1e3  # Kilosecond measurements

    # Velocity ranges (Å/s) — heterodyne equivalent of homodyne shear rate
    VELOCITY_MIN: ClassVar[float] = 1e-6  # Quasi-static limit
    VELOCITY_MAX: ClassVar[float] = 1e4  # Upper bound for directed flow

    # Fraction parameter ranges
    FRACTION_MIN: ClassVar[float] = 0.0
    FRACTION_MAX: ClassVar[float] = 1.0

    # Numerical stability constants
    EPS: ClassVar[float] = 1e-12  # Avoid division by zero
    MAX_EXP_ARG: ClassVar[float] = 700.0  # Prevent exponential overflow
    MIN_POSITIVE: ClassVar[float] = 1e-100  # Minimum positive value


# Default parameter bounds for heterodyne model
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    # Reference transport
    "D0_ref": (100.0, 1e6),
    "alpha_ref": (-2.0, 2.0),
    "D_offset_ref": (-1e5, 1e5),
    # Sample transport
    "D0_sample": (100.0, 1e6),
    "alpha_sample": (-2.0, 2.0),
    "D_offset_sample": (-1e5, 1e5),
    # Velocity
    "v0": (1e-6, 1e4),
    "v_beta": (-2.0, 2.0),
    "v_offset": (-100.0, 100.0),
    # Fraction
    "f0": (0.0, 1.0),
    "f1": (-10.0, 10.0),
    "f2": (-1e4, 1e4),
    "f3": (0.0, 1.0),
    # Angle
    "phi0_het": (-10.0, 10.0),
}


def get_default_bounds_array() -> tuple[np.ndarray, np.ndarray]:
    """Get default bounds as arrays in canonical parameter order.

    Returns:
        (lower_bounds, upper_bounds) each of shape (14,)
    """
    from xpcsjax.core.heterodyne_models import ALL_PARAM_NAMES

    lower = np.array([PARAMETER_BOUNDS[name][0] for name in ALL_PARAM_NAMES])
    upper = np.array([PARAMETER_BOUNDS[name][1] for name in ALL_PARAM_NAMES])
    return lower, upper


@dataclass(frozen=True)
class TransportPhysics:
    """Physical interpretation of transport parameters.

    Transport coefficient: J(t) = D0 * t^alpha + offset

    Physical regimes based on alpha:
    - alpha = 1.0: Normal (Brownian) diffusion
    - alpha < 1.0: Subdiffusion (crowded/constrained)
    - alpha > 1.0: Superdiffusion (active/directed)
    - alpha = 2.0: Ballistic motion
    """

    # Alpha value regimes
    NORMAL_DIFFUSION: ClassVar[float] = 1.0
    BALLISTIC: ClassVar[float] = 2.0

    @staticmethod
    def interpret_alpha(alpha: float) -> str:
        """Interpret alpha value physically.

        For J(t) = D0 * t^alpha + offset:
        - alpha ≈ 0: constant transport rate (equilibrium)
        - alpha ≈ 1: linearly growing transport (normal diffusion)
        - alpha < 1: sub-linear (subdiffusive)
        - alpha > 1: super-linear (superdiffusive)
        - alpha ≈ 2: quadratic (ballistic)

        Args:
            alpha: Transport rate exponent

        Returns:
            Physical interpretation string
        """
        if abs(alpha) < 0.05:
            return "constant transport (equilibrium)"
        elif alpha < 0:
            return "decelerating transport"
        elif alpha < 0.5:
            return "strongly subdiffusive"
        elif abs(alpha - 1.0) < 0.05:
            return "normal diffusion"
        elif alpha < 1.0:
            return "subdiffusive"
        elif alpha < 1.5:
            return "weakly superdiffusive"
        elif alpha < 2.0:
            return "superdiffusive"
        else:
            return "ballistic/directed"

    @staticmethod
    def diffusion_coefficient(D0: float, alpha: float, t: float = 1.0) -> float:
        """Compute effective diffusion coefficient at time t.

        For J(t) = D0 * t^alpha, the effective D is:
        D_eff = dJ/dt = D0 * alpha * t^(alpha-1)

        Args:
            D0: Transport prefactor
            alpha: Transport exponent
            t: Time point (default 1.0)

        Returns:
            Effective diffusion coefficient
        """
        if t <= 0:
            return 0.0
        return float(D0 * alpha * (t ** (alpha - 1)))


def validate_parameters(
    params: dict[str, float],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> ValidationResult:
    """Validate parameter values against physical bounds.

    Args:
        params: Dictionary mapping parameter names to values.
        bounds: Optional custom bounds. Defaults to PARAMETER_BOUNDS.

    Returns:
        ValidationResult with violations list if any bounds are exceeded.
    """
    if bounds is None:
        bounds = PARAMETER_BOUNDS

    violations: list[str] = []
    checked = 0

    for name, value in params.items():
        if name not in bounds:
            continue
        checked += 1
        lo, hi = bounds[name]
        if not np.isfinite(value):
            violations.append(f"{name}={value} is not finite")
        elif value < lo:
            violations.append(f"{name}={value:.6g} < lower bound {lo:.6g}")
        elif value > hi:
            violations.append(f"{name}={value:.6g} > upper bound {hi:.6g}")

    valid = len(violations) == 0
    n_violations = len(violations)
    message = (
        f"All {checked} parameters within bounds"
        if valid
        else f"{n_violations} violation(s) in {checked} parameters"
    )

    return ValidationResult(
        valid=valid,
        violations=violations,
        parameters_checked=checked,
        message=message,
    )
