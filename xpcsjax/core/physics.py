"""Physical constants and parameter validation for homodyne XPCS analysis.

===========================================================

Centralized physical constants, parameter bounds, and validation functions
for xpcsjax scattering analysis. Provides reference values and constraints
based on experimental physics and numerical stability requirements.

This module establishes the physical framework for all model computations
and ensures parameter values remain within reasonable bounds for stable
numerical computation.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of parameter validation with detailed error reporting.

    Provides comprehensive information about parameter validation
    including which parameters violated bounds and by how much.

    Attributes
    ----------
    valid : bool
        True if all parameters are within bounds
    violations : list of str
        List of human-readable violation messages
    parameters_checked : int
        Number of parameters validated
    message : str
        Summary message about validation result
    """

    valid: bool
    violations: list[str] = field(default_factory=list)
    parameters_checked: int = 0
    message: str = ""

    def __str__(self) -> str:
        """Return a human-readable representation for logging."""
        if self.valid:
            return f"OK {self.message}"
        else:
            violations_str = "\n  - ".join(self.violations)
            return f"FAIL {self.message}\n  - {violations_str}"


class PhysicsConstants:
    """Physical constants and reference values for XPCS analysis.

    These values are based on typical synchrotron X-ray scattering
    experiments and provide reasonable defaults for most analyses.
    """

    # X-ray wavelengths (Angstroms)
    WAVELENGTH_CU_KA = 1.54  # Copper K-alpha
    WAVELENGTH_8KEV = 1.55  # ~8 keV synchrotron
    WAVELENGTH_12KEV = 1.0332  # ~12 keV synchrotron (λ = hc/E = 12398.4/12000 Å)
    WAVELENGTH_15KEV = 0.83  # ~15 keV synchrotron

    # Typical q-ranges (inverse Angstroms)
    Q_MIN_TYPICAL = 1e-4
    Q_MAX_TYPICAL = 1.0

    # Time scales (seconds)
    TIME_MIN_XPCS = 1e-6  # Microsecond resolution
    TIME_MAX_XPCS = 1e3  # Kilosecond measurements

    # Diffusion coefficient ranges (Å²/s)
    DIFFUSION_MIN = 100.0  # Minimum for colloidal systems
    DIFFUSION_MAX = 1e5  # Maximum for fast colloidal systems
    DIFFUSION_TYPICAL = 100.0

    # Shear rate ranges (s⁻¹)
    SHEAR_RATE_MIN = 1e-6  # Quasi-static limit
    SHEAR_RATE_MAX = 0.5  # Upper bound aligned with YAML template
    SHEAR_RATE_TYPICAL = 0.01

    # Angular ranges (degrees) - focused range for laminar flow analysis
    ANGLE_MIN = -10.0
    ANGLE_MAX = 10.0

    # Offset parameter bounds
    DIFFUSION_OFFSET_MIN = -1e5  # Allow negative for jammed/arrested systems
    DIFFUSION_OFFSET_MAX = 1e5  # Maximum positive diffusion offset
    SHEAR_OFFSET_MIN = -0.1  # Minimum shear rate offset (allows small negative)
    SHEAR_OFFSET_MAX = 0.1  # Maximum shear rate offset

    # Numerical stability
    EPS = 1e-12  # Avoid division by zero
    MAX_EXP_ARG = 700.0  # Prevent exponential overflow
    MIN_POSITIVE = 1e-100  # Minimum positive value

    # Physical parameter bounds
    # NOTE: These are reference values. The PRIMARY bounds used by NLSQ
    # are defined in xpcsjax.core.fitting.ParameterSpace
    ALPHA_MIN = -2.0  # Minimum diffusion exponent (tighter for numerical stability)
    ALPHA_MAX = 2.0  # Maximum diffusion exponent
    BETA_MIN = -2.0  # Minimum shear exponent (tighter for numerical stability)
    BETA_MAX = 2.0  # Maximum shear exponent


def parameter_bounds() -> dict[str, list[tuple[float, float]]]:
    """Get standard parameter bounds for all model types.

    Returns
    -------
    dict
        Mapping of model type (``"diffusion"``, ``"shear"``, ``"combined"``) to
        the ordered list of ``(min, max)`` bounds tuples for that model.
    """
    return {
        "diffusion": [
            (PhysicsConstants.DIFFUSION_MIN, PhysicsConstants.DIFFUSION_MAX),  # D0
            (PhysicsConstants.ALPHA_MIN, PhysicsConstants.ALPHA_MAX),  # alpha
            (
                PhysicsConstants.DIFFUSION_OFFSET_MIN,
                PhysicsConstants.DIFFUSION_OFFSET_MAX,
            ),  # D_offset
        ],
        "shear": [
            (
                PhysicsConstants.SHEAR_RATE_MIN,
                PhysicsConstants.SHEAR_RATE_MAX,
            ),  # gamma_dot_t0
            (PhysicsConstants.BETA_MIN, PhysicsConstants.BETA_MAX),  # beta
            (
                PhysicsConstants.SHEAR_OFFSET_MIN,
                PhysicsConstants.SHEAR_OFFSET_MAX,
            ),  # gamma_dot_t_offset
            (PhysicsConstants.ANGLE_MIN, PhysicsConstants.ANGLE_MAX),  # phi0
        ],
        "combined": [
            # Diffusion parameters
            (PhysicsConstants.DIFFUSION_MIN, PhysicsConstants.DIFFUSION_MAX),  # D0
            (PhysicsConstants.ALPHA_MIN, PhysicsConstants.ALPHA_MAX),  # alpha
            (
                PhysicsConstants.DIFFUSION_OFFSET_MIN,
                PhysicsConstants.DIFFUSION_OFFSET_MAX,
            ),  # D_offset
            # Shear parameters
            (
                PhysicsConstants.SHEAR_RATE_MIN,
                PhysicsConstants.SHEAR_RATE_MAX,
            ),  # gamma_dot_t0
            (PhysicsConstants.BETA_MIN, PhysicsConstants.BETA_MAX),  # beta
            (
                PhysicsConstants.SHEAR_OFFSET_MIN,
                PhysicsConstants.SHEAR_OFFSET_MAX,
            ),  # gamma_dot_t_offset
            (PhysicsConstants.ANGLE_MIN, PhysicsConstants.ANGLE_MAX),  # phi0
        ],
    }


def validate_parameters_detailed(
    params: np.ndarray,
    bounds: list[tuple[float, float]],
    param_names: list[str] | None = None,
    tolerance: float = 1e-10,
) -> ValidationResult:
    """Validate parameter values against bounds with detailed error reporting.

    This is the enhanced validation function that provides comprehensive
    information about which parameters violated bounds and by how much.

    Parameters
    ----------
    params : np.ndarray
        Parameter array to validate
    bounds : list of tuple
        List of (min, max) tuples for each parameter
    param_names : list of str, optional
        Names of parameters for better error messages. If None, uses indices.
    tolerance : float
        Tolerance for bounds checking (default: 1e-10)

    Returns
    -------
    ValidationResult
        Detailed validation result with violations list

    Examples
    --------
    >>> params = np.array([100.0, -1.5, 10.0])
    >>> bounds = [(1.0, 1000.0), (-2.0, 2.0), (0.0, 100.0)]
    >>> result = validate_parameters_detailed(params, bounds, ["D0", "alpha", "D_offset"])
    >>> if not result.valid:
    ...     print(result.violations)
    """
    violations = []

    # Check if we're dealing with JAX tracers during gradient computation
    try:
        param_str = str(type(params[0] if hasattr(params, "__getitem__") else params))
        if "Tracer" in param_str or "LinearizeTracer" in param_str:
            # Skip validation during JAX gradient computation
            return ValidationResult(
                valid=True,
                violations=[],
                parameters_checked=0,
                message="Skipped validation for JAX tracers",
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tracer detection during validation skipped: %s", exc)

    # Check parameter count
    if len(params) != len(bounds):
        return ValidationResult(
            valid=False,
            violations=[
                f"Parameter count mismatch: got {len(params)} parameters, "
                f"expected {len(bounds)} bounds",
            ],
            parameters_checked=0,
            message="Parameter count validation failed",
        )

    # Use indices if no names provided
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(params))]

    # Validate each parameter
    validated_count = 0
    for i, (param, (min_val, max_val)) in enumerate(zip(params, bounds, strict=False)):
        # Check if param is a JAX tracer
        try:
            param_type_str = str(type(param))
            if "Tracer" in param_type_str or "LinearizeTracer" in param_type_str:
                continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tracer detection for param failed: %s", exc)

        # Validate concrete numeric values
        try:
            param_val = float(param)
            param_name = param_names[i] if i < len(param_names) else f"param_{i}"

            if not (min_val - tolerance <= param_val <= max_val + tolerance):
                # Calculate violation magnitude
                if param_val < min_val:
                    violation_amount = min_val - param_val
                    direction = "below"
                else:
                    violation_amount = param_val - max_val
                    direction = "above"

                violations.append(
                    f"{param_name} = {param_val:.6e} is {direction} bounds "
                    f"[{min_val:.6e}, {max_val:.6e}] by {violation_amount:.6e}",
                )
            validated_count += 1
        except (TypeError, ValueError):
            # Likely a JAX tracer, skip
            continue

    # Create result
    is_valid = len(violations) == 0
    if is_valid:
        message = f"Validated {validated_count} parameters successfully"
    else:
        message = f"Validation failed: {len(violations)} parameter(s) out of bounds"

    return ValidationResult(
        valid=is_valid,
        violations=violations,
        parameters_checked=validated_count,
        message=message,
    )


def validate_parameters(
    params: np.ndarray,
    bounds: list[tuple[float, float]],
    tolerance: float = 1e-10,
) -> bool:
    """Validate parameter values against bounds with tolerance.

    This is the legacy function that returns just a boolean.
    For detailed validation, use validate_parameters_detailed().

    Parameters
    ----------
    params : np.ndarray
        Parameter array to validate
    bounds : list of tuple
        List of (min, max) tuples for each parameter
    tolerance : float
        Tolerance for bounds checking

    Returns
    -------
    bool
        True if all parameters are within bounds, False otherwise
    """
    # Use the detailed validation and return just the boolean
    result = validate_parameters_detailed(params, bounds, None, tolerance)

    # Log violations if any
    if not result.valid and result.violations:
        for violation in result.violations:
            logger.warning(violation)

    return result.valid


def clip_parameters(
    params: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Clip parameters to stay within bounds.

    Parameters
    ----------
    params : np.ndarray
        Parameter array to clip.
    bounds : list of tuple
        List of ``(min, max)`` tuples, one per parameter.

    Returns
    -------
    np.ndarray
        Clipped parameter array.

    Raises
    ------
    ValueError
        If the parameter count does not match the number of bounds.
    """
    if len(params) != len(bounds):
        raise ValueError(
            f"Parameter count mismatch: got {len(params)}, expected {len(bounds)}",
        )

    clipped = np.zeros_like(params)
    for i, (param, (min_val, max_val)) in enumerate(zip(params, bounds, strict=False)):
        clipped[i] = np.clip(param, min_val, max_val)

        if abs(clipped[i] - param) > 1e-10:
            logger.debug(f"Clipped parameter {i}: {param} -> {clipped[i]}")

    return clipped


def get_default_parameters(model_type: str) -> np.ndarray:
    """Get sensible default parameters for a model type.

    Parameters
    ----------
    model_type : str
        One of ``"diffusion"``, ``"shear"``, or ``"combined"``.

    Returns
    -------
    np.ndarray
        Array of default parameter values for the requested model.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognized.
    """
    defaults = {
        "diffusion": np.array(
            [
                PhysicsConstants.DIFFUSION_TYPICAL,  # D0 = 100 Å²/s
                0.0,  # alpha = 0 (normal diffusion)
                PhysicsConstants.DIFFUSION_TYPICAL / 10,  # D_offset = 10 Å²/s
            ],
        ),
        "shear": np.array(
            [
                PhysicsConstants.SHEAR_RATE_TYPICAL,  # gamma_dot_t0 = 1 s⁻¹
                0.0,  # beta = 0 (constant shear)
                PhysicsConstants.SHEAR_OFFSET_MIN,  # gamma_dot_t_offset (lower bound = -0.1)
                0.0,  # phi0 = 0 degrees
            ],
        ),
        "combined": np.array(
            [
                # Diffusion defaults
                PhysicsConstants.DIFFUSION_TYPICAL,  # D0 = 100 Å²/s
                0.0,  # alpha = 0
                PhysicsConstants.DIFFUSION_TYPICAL / 10,  # D_offset = 10 Å²/s
                # Shear defaults
                PhysicsConstants.SHEAR_RATE_TYPICAL,  # gamma_dot_t0 = 1 s⁻¹
                0.0,  # beta = 0
                PhysicsConstants.SHEAR_OFFSET_MIN,  # gamma_dot_t_offset (lower bound = -0.1)
                0.0,  # phi0 = 0 degrees
            ],
        ),
    }

    if model_type not in defaults:
        raise ValueError(
            f"Unknown model type '{model_type}'. Must be one of {list(defaults.keys())}",
        )

    return defaults[model_type]


def validate_experimental_setup(q: float, L: float, wavelength: float | None = None) -> bool:
    """Validate experimental setup parameters for physical reasonableness.

    Parameters
    ----------
    q : float
        Scattering wave vector magnitude [Å⁻¹].
    L : float
        Sample-detector distance [Å] (checked against the ``[1e5, 1e8]`` Å
        range, i.e. 10 µm to 10 mm).
    wavelength : float, optional
        X-ray wavelength [Å]. Checked against ``[0.1, 10.0]`` Å when provided.

    Returns
    -------
    bool
        True if the setup is physically reasonable; False (with a logged
        warning) if any value is outside its expected range.
    """
    # Check q-range
    if not (PhysicsConstants.Q_MIN_TYPICAL <= q <= PhysicsConstants.Q_MAX_TYPICAL):
        logger.warning(
            f"q-value {q:.2e} A^-1 outside typical range "
            f"[{PhysicsConstants.Q_MIN_TYPICAL:.2e}, {PhysicsConstants.Q_MAX_TYPICAL:.2e}]",
        )
        return False

    # Check detector distance (L is in Angstroms)
    # Typical range: 100,000 A (10 um) to 100,000,000 A (10 mm)
    # Note: 1 A = 1e-10 m, so 1e5 A = 10 um, 1e8 A = 10 mm.
    if not (1e5 <= L <= 1e8):
        logger.warning(
            f"Sample-detector distance {L:.1f} A outside reasonable range [1e5, 1e8] A (10 um to 10 mm)",
        )
        return False

    # Check wavelength if provided
    if wavelength is not None:
        if not (0.1 <= wavelength <= 10.0):
            logger.warning(
                f"X-ray wavelength {wavelength:.2f} A outside reasonable range [0.1, 10.0]",
            )
            return False

    return True


def estimate_correlation_time(D0: float, alpha: float, q: float) -> float:
    """Estimate characteristic correlation time for diffusion process.

    For normal diffusion (alpha=0): τ ≈ 1/(q²D₀).
    For anomalous diffusion the scaling is more complex; a rough correction
    factor ``(1 + |alpha|)`` is applied.

    Parameters
    ----------
    D0 : float
        Reference diffusion coefficient [Å²/s].
    alpha : float
        Diffusion exponent. ``|alpha| < 1e-12`` selects the normal-diffusion
        branch.
    q : float
        Scattering wave vector [Å⁻¹].

    Returns
    -------
    float
        Estimated correlation time [s], or ``inf`` when ``D0 <= 0``.
    """
    # P2-R6-02: Use epsilon tolerance instead of exact float equality.
    # NLSQ parameters are rarely exactly 0.0; alpha=1e-15 from an
    # optimiser would incorrectly take the anomalous branch.
    if abs(alpha) < 1e-12:
        # Normal diffusion
        return 1.0 / (q**2 * D0) if D0 > 0 else np.inf
    else:
        # Anomalous diffusion - approximate scaling
        # This is a rough estimate for experimental planning
        base_time = 1.0 / (q**2 * D0) if D0 > 0 else np.inf
        return base_time * (1.0 + abs(alpha))  # Rough correction


def get_parameter_info(model_type: str) -> dict[str, Any]:
    """Get comprehensive parameter information for a model type.

    Parameters
    ----------
    model_type : str
        One of ``"diffusion"``, ``"shear"``, or ``"combined"``.

    Returns
    -------
    dict
        Parameter metadata: ``names``, ``descriptions``, ``physical_meaning``,
        ``bounds``, ``defaults``, and ``n_parameters``.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognized.
    """
    info: dict[str, dict[str, list[str]]] = {
        "diffusion": {
            "names": ["D0", "alpha", "D_offset"],
            "descriptions": [
                "Reference diffusion coefficient (Å²/s)",
                "Diffusion time-dependence exponent (-)",
                "Baseline diffusion coefficient (Å²/s)",
            ],
            "physical_meaning": [
                "Characteristic mobility scale",
                "0=normal, >0=super-diffusion, <0=sub-diffusion",
                "Residual diffusion at t=0",
            ],
        },
        "shear": {
            "names": ["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
            "descriptions": [
                "Reference shear rate (s⁻¹)",
                "Shear rate time-dependence exponent (-)",
                "Baseline shear rate (s⁻¹)",
                "Flow direction angle (degrees)",
            ],
            "physical_meaning": [
                "Characteristic shear rate scale",
                "0=constant, >0=accelerating, <0=decelerating",
                "Residual shear rate at t=0",
                "Preferred flow direction",
            ],
        },
        "combined": {
            "names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
            "descriptions": [
                "Reference diffusion coefficient (Å²/s)",
                "Diffusion time-dependence exponent (-)",
                "Baseline diffusion coefficient (Å²/s)",
                "Reference shear rate (s⁻¹)",
                "Shear rate time-dependence exponent (-)",
                "Baseline shear rate (s⁻¹)",
                "Flow direction angle (degrees)",
            ],
            "physical_meaning": [
                "Characteristic mobility scale",
                "0=normal, >0=super-diffusion, <0=sub-diffusion",
                "Residual diffusion at t=0",
                "Characteristic shear rate scale",
                "0=constant, >0=accelerating, <0=decelerating",
                "Residual shear rate at t=0",
                "Preferred flow direction",
            ],
        },
    }

    if model_type not in info:
        raise ValueError(f"Unknown model type '{model_type}'")

    # Add common information
    result: dict[str, Any] = info[model_type].copy()
    result.update(
        {
            "bounds": parameter_bounds()[model_type],
            "defaults": get_default_parameters(model_type).tolist(),
            "n_parameters": len(info[model_type]["names"]),
        },
    )

    return result


# Export main functions and constants
__all__ = [
    "PhysicsConstants",
    "parameter_bounds",
    "validate_parameters",
    "clip_parameters",
    "get_default_parameters",
    "validate_experimental_setup",
    "estimate_correlation_time",
    "get_parameter_info",
]
