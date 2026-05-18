"""Physics-Based Parameter Validators for XPCS Analysis

Registry-driven validation of physics parameters based on theoretical
understanding of XPCS and soft matter dynamics.

This module provides:
- PHYSICS_CONSTRAINTS: Declarative constraint definitions per parameter
- validate_single_parameter(): Check one parameter against constraints
- validate_cross_parameter(): Check inter-parameter relationships
- PhysicsViolation: Named tuple for constraint violations

Each constraint is defined with:
- condition: Lambda function returning True if violation detected
- message: Human-readable explanation
- severity: error/warning/info
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# Severity levels for constraint violations
class ConstraintSeverity:
    """Severity levels for physics constraint violations."""

    ERROR = "error"  # Physically impossible
    WARNING = "warning"  # Unusual but possible
    INFO = "info"  # Noteworthy observation


@dataclass
class PhysicsViolation:
    """A physics constraint violation."""

    param: str
    value: float
    message: str
    severity: str

    def format(self) -> str:
        """Format violation as string."""
        return f"{self.param} = {self.value:.3e}: {self.message} [{self.severity}]"


@dataclass
class ConstraintRule:
    """A single constraint rule for a parameter."""

    condition: Callable[[float], bool]  # Returns True if violated
    message: str
    severity: str


# Physics constraints registry: param_name -> list of rules
PHYSICS_CONSTRAINTS: dict[str, list[ConstraintRule]] = {
    "D0": [
        ConstraintRule(
            condition=lambda v: v <= 0,
            message="non-positive diffusion coefficient (physically impossible)",
            severity=ConstraintSeverity.ERROR,
        ),
        ConstraintRule(
            condition=lambda v: v > 1e7,
            message="extremely large diffusion coefficient (check units: nm²/s expected)",
            severity=ConstraintSeverity.WARNING,
        ),
    ],
    "alpha": [
        ConstraintRule(
            condition=lambda v: v < -1.5,
            message="very strongly subdiffusive (α < -1.5 extremely rare)",
            severity=ConstraintSeverity.WARNING,
        ),
        ConstraintRule(
            condition=lambda v: v > 1.0,
            message="strongly superdiffusive (α > 1 rare, ballistic/active systems only)",
            severity=ConstraintSeverity.WARNING,
        ),
        ConstraintRule(
            condition=lambda v: -0.1 < v < 0.1,
            message="near-normal diffusion (α ≈ 0, standard Brownian motion)",
            severity=ConstraintSeverity.INFO,
        ),
    ],
    "D_offset": [
        ConstraintRule(
            condition=lambda v: v < 0,
            message="negative offset (check if this is intended)",
            severity=ConstraintSeverity.WARNING,
        ),
    ],
    "gamma_dot_t0": [
        ConstraintRule(
            condition=lambda v: v < 0,
            message="negative shear rate (physically impossible, use positive value)",
            severity=ConstraintSeverity.ERROR,
        ),
        ConstraintRule(
            condition=lambda v: v > 0.5,
            message="very high shear rate (check units: s⁻¹ expected)",
            severity=ConstraintSeverity.WARNING,
        ),
        ConstraintRule(
            condition=lambda v: 0 < v < 1e-6,
            message="very low shear rate (approaching quasi-static limit)",
            severity=ConstraintSeverity.INFO,
        ),
    ],
    "beta": [
        ConstraintRule(
            condition=lambda v: v < -2.0 or v > 2.0,
            message="time exponent outside typical range [-2, 2]",
            severity=ConstraintSeverity.WARNING,
        ),
    ],
    "gamma_dot_t_offset": [
        ConstraintRule(
            condition=lambda v: abs(v) > 0.1,
            message="shear rate offset outside [-0.1, 0.1] (check units: s⁻¹ expected)",
            severity=ConstraintSeverity.WARNING,
        ),
    ],
    "phi0": [
        ConstraintRule(
            condition=lambda v: abs(v) > 10.0,
            message="flow angle outside [-10, 10] degrees (check alignment)",
            severity=ConstraintSeverity.INFO,
        ),
    ],
    "contrast": [
        ConstraintRule(
            condition=lambda v: v <= 0 or v > 1.0,
            message="contrast outside physical range (0, 1]",
            severity=ConstraintSeverity.ERROR,
        ),
        ConstraintRule(
            condition=lambda v: 0 < v < 0.1,
            message="very low contrast (check signal quality)",
            severity=ConstraintSeverity.WARNING,
        ),
    ],
    "offset": [
        ConstraintRule(
            condition=lambda v: v <= 0,
            message="non-positive baseline (physically impossible)",
            severity=ConstraintSeverity.ERROR,
        ),
    ],
}

# Severity priority mapping for filtering
SEVERITY_PRIORITY = {
    ConstraintSeverity.ERROR: 3,
    ConstraintSeverity.WARNING: 2,
    ConstraintSeverity.INFO: 1,
}


def validate_single_parameter(
    param: str,
    value: float,
    min_severity: str = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate a single parameter against physics constraints.

    Args:
        param: Parameter name
        value: Parameter value
        min_severity: Minimum severity to report

    Returns:
        List of violations found
    """
    violations: list[PhysicsViolation] = []
    min_priority = SEVERITY_PRIORITY.get(min_severity, 2)

    if param not in PHYSICS_CONSTRAINTS:
        return violations

    for rule in PHYSICS_CONSTRAINTS[param]:
        if SEVERITY_PRIORITY[rule.severity] >= min_priority:
            try:
                if rule.condition(value):
                    violations.append(
                        PhysicsViolation(
                            param=param,
                            value=value,
                            message=rule.message,
                            severity=rule.severity,
                        )
                    )
            except (TypeError, ValueError):
                # Skip if condition can't be evaluated
                pass

    return violations


def validate_cross_parameter_constraints(
    params: dict[str, float],
    min_severity: str = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate cross-parameter physics constraints.

    Args:
        params: Dictionary of parameter name -> value
        min_severity: Minimum severity to report

    Returns:
        List of violations found
    """
    violations: list[PhysicsViolation] = []
    min_priority = SEVERITY_PRIORITY.get(min_severity, 2)

    # D_offset vs D0 overfitting check
    if all(k in params for k in ["D0", "alpha", "D_offset"]):
        D0, D_offset = params["D0"], params["D_offset"]
        if D0 > 0 and D_offset > 0.5 * D0:
            if SEVERITY_PRIORITY[ConstraintSeverity.INFO] >= min_priority:
                ratio = D_offset / D0
                violations.append(
                    PhysicsViolation(
                        param="D_offset",
                        value=D_offset,
                        message=f"offset is {ratio:.1%} of D0 (may indicate overfitting)",
                        severity=ConstraintSeverity.INFO,
                    )
                )

    return violations


def validate_all_parameters(
    params: dict[str, float],
    min_severity: str = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate all parameters against physics constraints.

    Args:
        params: Dictionary of parameter name -> value
        min_severity: Minimum severity to report

    Returns:
        List of all violations found
    """
    violations: list[PhysicsViolation] = []

    # Single parameter constraints
    for param, value in params.items():
        violations.extend(validate_single_parameter(param, value, min_severity))

    # Cross-parameter constraints
    violations.extend(validate_cross_parameter_constraints(params, min_severity))

    return violations


def get_constraint_summary() -> dict[str, Any]:
    """Get summary of all defined constraints.

    Returns:
        Dictionary with constraint counts and parameter coverage
    """
    return {
        "parameters_covered": list(PHYSICS_CONSTRAINTS.keys()),
        "total_constraints": sum(len(rules) for rules in PHYSICS_CONSTRAINTS.values()),
        "by_parameter": {
            param: len(rules) for param, rules in PHYSICS_CONSTRAINTS.items()
        },
    }


__all__ = [
    "ConstraintSeverity",
    "ConstraintRule",
    "PhysicsViolation",
    "PHYSICS_CONSTRAINTS",
    "SEVERITY_PRIORITY",
    "validate_single_parameter",
    "validate_cross_parameter_constraints",
    "validate_all_parameters",
    "get_constraint_summary",
]
