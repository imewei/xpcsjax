r"""Physics-based parameter validators for homodyne XPCS analysis.

Registry-driven validation of physics parameters based on the theoretical
understanding of XPCS and soft-matter dynamics. The constraints encoded here
complement the hard bounds in
:mod:`xpcsjax.config.parameter_registry` (the single source of truth for
parameter names and bounds); these rules flag values that are physically
suspect or impossible even when they fall inside the registry bounds.

This module provides:

- :data:`PHYSICS_CONSTRAINTS` -- declarative per-parameter constraint rules.
- :func:`validate_single_parameter` -- check one parameter against its rules.
- :func:`validate_cross_parameter_constraints` -- check inter-parameter
  relationships.
- :func:`validate_all_parameters` -- run both single- and cross-parameter
  checks.
- :class:`PhysicsViolation` -- record of a triggered constraint.

Each :class:`ConstraintRule` is defined with:

- ``condition`` -- callable returning ``True`` when the value violates the rule.
- ``message`` -- human-readable explanation.
- ``severity`` -- one of ``error`` / ``warning`` / ``info``.

See Also
--------
xpcsjax.config.heterodyne_physics_validators : Sibling validators for the
    14-parameter heterodyne (``two_component``) model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


# Severity levels for constraint violations
class ConstraintSeverity(StrEnum):
    """Severity levels for physics constraint violations.

    StrEnum (a str subclass): members compare equal to their string values, so
    existing ``== "error"`` checks and f-string formatting are unchanged, while
    static checkers now treat the set as closed. Mirrors the heterodyne
    ConstraintSeverity in heterodyne_physics_validators.py (previously this homodyne
    copy was a plain class with bare string constants).
    """

    ERROR = "error"  # Physically impossible
    WARNING = "warning"  # Unusual but possible
    INFO = "info"  # Noteworthy observation


@dataclass
class PhysicsViolation:
    """A single triggered physics constraint violation.

    Attributes
    ----------
    param : str
        Name of the offending parameter.
    value : float
        The value that triggered the violation.
    message : str
        Human-readable explanation of why the value is suspect.
    severity : ConstraintSeverity
        Severity of the violation (``error`` / ``warning`` / ``info``).
    """

    param: str
    value: float
    message: str
    severity: ConstraintSeverity

    def format(self) -> str:
        """Render the violation as a single human-readable line."""
        return f"{self.param} = {self.value:.3e}: {self.message} [{self.severity}]"


@dataclass
class ConstraintRule:
    """A single physics constraint rule for one parameter.

    Attributes
    ----------
    condition : collections.abc.Callable
        Predicate over the parameter value; returns ``True`` when the value
        violates the rule.
    message : str
        Human-readable explanation attached to a triggered violation.
    severity : ConstraintSeverity
        Severity assigned to a triggered violation.
    """

    condition: Callable[[float], bool]  # Returns True if violated
    message: str
    severity: ConstraintSeverity


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
    min_severity: ConstraintSeverity = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate a single parameter against its physics constraints.

    Parameters
    ----------
    param : str
        Parameter name. Names absent from :data:`PHYSICS_CONSTRAINTS` have no
        rules and produce no violations.
    value : float
        Parameter value to check.
    min_severity : ConstraintSeverity, optional
        Minimum severity to report; lower-severity rules are skipped.

    Returns
    -------
    list of PhysicsViolation
        Violations triggered for this parameter, possibly empty.

    Notes
    -----
    A rule whose ``condition`` raises :class:`TypeError` or :class:`ValueError`
    (e.g. a non-numeric value) is silently skipped rather than propagated.
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
    min_severity: ConstraintSeverity = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate constraints that span multiple parameters.

    Currently checks for an over-large ``D_offset`` relative to ``D0`` (an
    offset exceeding half of ``D0``), reported at ``info`` severity as a
    possible overfitting signal.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to value.
    min_severity : ConstraintSeverity, optional
        Minimum severity to report.

    Returns
    -------
    list of PhysicsViolation
        Cross-parameter violations triggered, possibly empty.
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
    min_severity: ConstraintSeverity = ConstraintSeverity.WARNING,
) -> list[PhysicsViolation]:
    """Validate all parameters against single- and cross-parameter constraints.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to value.
    min_severity : ConstraintSeverity, optional
        Minimum severity to report.

    Returns
    -------
    list of PhysicsViolation
        Every triggered violation, single-parameter first then
        cross-parameter (unordered by severity).

    See Also
    --------
    validate_single_parameter : Per-parameter checks.
    validate_cross_parameter_constraints : Inter-parameter checks.
    """
    violations: list[PhysicsViolation] = []

    # Single parameter constraints
    for param, value in params.items():
        violations.extend(validate_single_parameter(param, value, min_severity))

    # Cross-parameter constraints
    violations.extend(validate_cross_parameter_constraints(params, min_severity))

    return violations


def get_constraint_summary() -> dict[str, Any]:
    """Summarize the defined constraint registry.

    Returns
    -------
    dict
        Keys ``parameters_covered`` (list of parameter names with rules),
        ``total_constraints`` (rule count across all parameters), and
        ``by_parameter`` (per-parameter rule count).
    """
    return {
        "parameters_covered": list(PHYSICS_CONSTRAINTS.keys()),
        "total_constraints": sum(len(rules) for rules in PHYSICS_CONSTRAINTS.values()),
        "by_parameter": {param: len(rules) for param, rules in PHYSICS_CONSTRAINTS.items()},
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
