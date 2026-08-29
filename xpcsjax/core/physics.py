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
        except (TypeError, ValueError) as e:
            if "Tracer" in str(type(param)) or "LinearizeTracer" in str(type(param)):
                # Genuinely a JAX tracer that slipped past the check above.
                continue
            logger.debug(
                "Could not validate parameter %s (value=%r): %s",
                param_names[i] if i < len(param_names) else f"param_{i}",
                param,
                e,
            )
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


# Export main functions and constants
__all__ = [
    "PhysicsConstants",
    "ValidationResult",
    "validate_parameters",
    "validate_parameters_detailed",
]
