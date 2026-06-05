"""Physics constraint validators for heterodyne (``two_component``) parameters.

Registry-driven sanity checks for the 14-parameter heterodyne model. These
rules flag physically suspect or impossible parameter values (and a few
cross-parameter relationships) that the hard bounds in
:mod:`xpcsjax.config.parameter_registry` do not by themselves catch. Diffusion
coefficients are interpreted in Ångström units (Å) consistent with the
registry; velocities are reported in Å/s.

This module provides:

- :data:`PHYSICS_CONSTRAINTS` -- declarative per-parameter constraint rules.
- :func:`validate_single_parameter` / :func:`validate_cross_parameter_constraints`
  / :func:`validate_all_parameters` -- the staged check entry points.
- :func:`validate_parameters` -- convenience wrapper returning a
  :class:`ValidationResult` from an array or dict of parameters.
- :func:`validate_time_integral_safety` -- numerical-safety check on the
  ``D0 * t**alpha`` time integral.
- :func:`validate_correlation_inputs` -- shape / finiteness / monotonicity
  check on correlation-matrix inputs.

See Also
--------
xpcsjax.config.physics_validators : Sibling validators for the homodyne models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES

if TYPE_CHECKING:
    pass


class ConstraintSeverity(Enum):
    """Severity level for physics constraint violations."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class PhysicsViolation:
    """A single triggered physics constraint violation.

    Attributes
    ----------
    parameter : str
        Name of the offending parameter (or a composite label such as
        ``"f0+f3"`` for cross-parameter checks).
    value : float or None
        The value that triggered the violation.
    message : str
        Human-readable explanation, typically including the value.
    severity : ConstraintSeverity
        Severity of the violation.
    """

    parameter: str
    value: float | None
    message: str
    severity: ConstraintSeverity


@dataclass
class ValidationResult:
    """Outcome of a parameter validation, partitioned by severity.

    Truthy when :attr:`is_valid` is ``True`` (no errors), via
    :meth:`__bool__`.

    Attributes
    ----------
    is_valid : bool
        ``True`` when no ``error``-severity violations were found.
    errors : list of str
        Messages for ``error``-severity violations.
    warnings : list of str
        Messages for ``warning``-severity violations.
    info : list of str
        Messages for ``info``-severity violations.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    info: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return :attr:`is_valid` so the result is truthy when valid."""
        return self.is_valid


@dataclass(frozen=True)
class ConstraintRule:
    """A single physics constraint rule for one parameter.

    Attributes
    ----------
    check : collections.abc.Callable
        Predicate over the parameter value; returns ``True`` when the value
        violates the rule.
    message : str
        Human-readable explanation attached to a triggered violation.
    severity : ConstraintSeverity
        Severity assigned to a triggered violation.
    """

    check: Callable[[float], bool]
    message: str
    severity: ConstraintSeverity


PHYSICS_CONSTRAINTS: dict[str, list[ConstraintRule]] = {
    "D0_ref": [
        ConstraintRule(lambda v: v < 0, "must be non-negative", ConstraintSeverity.ERROR),
        ConstraintRule(
            lambda v: v < 1e-12 and v >= 0,
            "near zero; may cause degenerate diffusion",
            ConstraintSeverity.WARNING,
        ),
        ConstraintRule(
            lambda v: v > 1e5,
            "unusually large diffusion coefficient",
            ConstraintSeverity.WARNING,
        ),
    ],
    "D0_sample": [
        ConstraintRule(lambda v: v < 0, "must be non-negative", ConstraintSeverity.ERROR),
        ConstraintRule(
            lambda v: v < 1e-12 and v >= 0,
            "near zero; may cause degenerate diffusion",
            ConstraintSeverity.WARNING,
        ),
        ConstraintRule(
            lambda v: v > 1e5,
            "unusually large diffusion coefficient",
            ConstraintSeverity.WARNING,
        ),
    ],
    "alpha_ref": [
        ConstraintRule(
            lambda v: v < -1.5,
            "strongly subdiffusive (alpha < -1.5)",
            ConstraintSeverity.WARNING,
        ),
        ConstraintRule(lambda v: v > 1.0, "superdiffusive regime", ConstraintSeverity.INFO),
        ConstraintRule(
            lambda v: abs(v) > 2,
            "unusual magnitude (|alpha| > 2)",
            ConstraintSeverity.WARNING,
        ),
    ],
    "alpha_sample": [
        ConstraintRule(
            lambda v: v < -1.5,
            "strongly subdiffusive (alpha < -1.5)",
            ConstraintSeverity.WARNING,
        ),
        ConstraintRule(lambda v: v > 1.0, "superdiffusive regime", ConstraintSeverity.INFO),
        ConstraintRule(
            lambda v: abs(v) > 2,
            "unusual magnitude (|alpha| > 2)",
            ConstraintSeverity.WARNING,
        ),
    ],
    "v0": [
        ConstraintRule(lambda v: v < 0, "negative velocity", ConstraintSeverity.WARNING),
        ConstraintRule(lambda v: v > 1e3, "large velocity (> 1e3 Å/s)", ConstraintSeverity.WARNING),
    ],
    "f0": [
        ConstraintRule(lambda v: not (0 <= v <= 1), "must be in [0, 1]", ConstraintSeverity.ERROR),
    ],
    "f3": [
        ConstraintRule(lambda v: not (0 <= v <= 1), "must be in [0, 1]", ConstraintSeverity.ERROR),
    ],
    "f1": [
        ConstraintRule(
            lambda v: abs(v) > 5,
            "large magnitude; fraction may change rapidly",
            ConstraintSeverity.WARNING,
        ),
    ],
    "beta": [
        ConstraintRule(
            lambda v: abs(v) > 2,
            "unusual magnitude (|beta| > 2)",
            ConstraintSeverity.WARNING,
        ),
    ],
}


def validate_single_parameter(
    param: str,
    value: float,
    min_severity: ConstraintSeverity = ConstraintSeverity.INFO,
) -> list[PhysicsViolation]:
    """Validate a single parameter against its physics constraints.

    Parameters
    ----------
    param : str
        Parameter name. Names with no rules in :data:`PHYSICS_CONSTRAINTS`
        produce no violations.
    value : float
        Parameter value to check.
    min_severity : ConstraintSeverity, optional
        Minimum severity to include. ``INFO`` includes all, ``WARNING``
        includes warnings and errors, ``ERROR`` includes only errors.

    Returns
    -------
    list of PhysicsViolation
        Violations triggered for this parameter, possibly empty.
    """
    severity_order = {
        ConstraintSeverity.INFO: 0,
        ConstraintSeverity.WARNING: 1,
        ConstraintSeverity.ERROR: 2,
    }
    min_level = severity_order[min_severity]

    violations: list[PhysicsViolation] = []
    rules = PHYSICS_CONSTRAINTS.get(param, [])

    for rule in rules:
        if severity_order[rule.severity] < min_level:
            continue
        if rule.check(value):
            violations.append(
                PhysicsViolation(
                    parameter=param,
                    value=value,
                    message=f"{param}={value:.3e}: {rule.message}",
                    severity=rule.severity,
                )
            )

    return violations


def validate_cross_parameter_constraints(
    params: dict[str, float],
    min_severity: ConstraintSeverity = ConstraintSeverity.INFO,
) -> list[PhysicsViolation]:
    """Validate constraints that span multiple parameters.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to value.
    min_severity : ConstraintSeverity, optional
        Minimum severity to include.

    Returns
    -------
    list of PhysicsViolation
        Cross-parameter violations triggered, possibly empty.

    Notes
    -----
    The cross-parameter checks are:

    - ``f0 + f3 > 1`` (``error``): total fraction exceeds unity.
    - ``D_offset_ref / D0_ref > 0.5`` (``warning``): offset dominates diffusion.
    - ``D_offset_sample / D0_sample > 0.5`` (``warning``): offset dominates
      diffusion.
    - ``v0 <= 0`` (``info``): the two-component model expects a positive
      velocity.
    """
    severity_order = {
        ConstraintSeverity.INFO: 0,
        ConstraintSeverity.WARNING: 1,
        ConstraintSeverity.ERROR: 2,
    }
    min_level = severity_order[min_severity]
    violations: list[PhysicsViolation] = []

    # f0 + f3 > 1
    if "f0" in params and "f3" in params:
        total = params["f0"] + params["f3"]
        if total > 1.0 and severity_order[ConstraintSeverity.ERROR] >= min_level:
            violations.append(
                PhysicsViolation(
                    parameter="f0+f3",
                    value=total,
                    message=f"f0 + f3 = {total:.3f} > 1; total fraction exceeds unity",
                    severity=ConstraintSeverity.ERROR,
                )
            )

    # D_offset_ref / D0_ref ratio
    if "D_offset_ref" in params and "D0_ref" in params and params["D0_ref"] > 0:
        ratio = params["D_offset_ref"] / params["D0_ref"]
        if ratio > 0.5 and severity_order[ConstraintSeverity.WARNING] >= min_level:
            violations.append(
                PhysicsViolation(
                    parameter="D_offset_ref/D0_ref",
                    value=ratio,
                    message=f"D_offset_ref/D0_ref = {ratio:.3f} > 0.5; offset dominates diffusion",
                    severity=ConstraintSeverity.WARNING,
                )
            )

    # D_offset_sample / D0_sample ratio
    if "D_offset_sample" in params and "D0_sample" in params and params["D0_sample"] > 0:
        ratio = params["D_offset_sample"] / params["D0_sample"]
        if ratio > 0.5 and severity_order[ConstraintSeverity.WARNING] >= min_level:
            violations.append(
                PhysicsViolation(
                    parameter="D_offset_sample/D0_sample",
                    value=ratio,
                    message=f"D_offset_sample/D0_sample = {ratio:.3f} > 0.5; offset dominates diffusion",
                    severity=ConstraintSeverity.WARNING,
                )
            )

    # v0 positive check (informational for two_component context)
    if (
        "v0" in params
        and params["v0"] <= 0
        and severity_order[ConstraintSeverity.INFO] >= min_level
    ):
        violations.append(
            PhysicsViolation(
                parameter="v0",
                value=params["v0"],
                message=f"v0={params['v0']:.3e} is non-positive; two-component model requires positive velocity",
                severity=ConstraintSeverity.INFO,
            )
        )

    return violations


def validate_all_parameters(
    params: dict[str, float],
    min_severity: ConstraintSeverity = ConstraintSeverity.INFO,
) -> list[PhysicsViolation]:
    """Validate all parameters against single- and cross-parameter constraints.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to value.
    min_severity : ConstraintSeverity, optional
        Minimum severity to include.

    Returns
    -------
    list of PhysicsViolation
        Every triggered violation, sorted by severity (errors first, then
        warnings, then info).

    See Also
    --------
    validate_single_parameter : Per-parameter checks.
    validate_cross_parameter_constraints : Inter-parameter checks.
    """
    violations: list[PhysicsViolation] = []

    # Single-parameter constraints
    for param, value in params.items():
        violations.extend(validate_single_parameter(param, value, min_severity))

    # Cross-parameter constraints
    violations.extend(validate_cross_parameter_constraints(params, min_severity))

    # Sort: errors first, then warnings, then info
    severity_order = {
        ConstraintSeverity.ERROR: 0,
        ConstraintSeverity.WARNING: 1,
        ConstraintSeverity.INFO: 2,
    }
    violations.sort(key=lambda v: severity_order[v.severity])

    return violations


def validate_parameters(params: np.ndarray | dict[str, float]) -> ValidationResult:
    """Validate heterodyne model parameters against physical constraints.

    Convenience wrapper over :func:`validate_all_parameters` that accepts the
    packed parameter array directly and returns a severity-partitioned
    :class:`ValidationResult`.

    Parameters
    ----------
    params : numpy.ndarray or dict
        Either a length-14 array (positionally aligned with
        :data:`~xpcsjax.config.heterodyne_parameter_names.ALL_PARAM_NAMES`) or
        a dict mapping parameter names to values.

    Returns
    -------
    ValidationResult
        Result with ``errors``, ``warnings``, and ``info`` message lists. An
        array whose length is not 14 yields an invalid result with a single
        descriptive error.
    """
    if isinstance(params, np.ndarray):
        if len(params) != 14:
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected 14 parameters, got {len(params)}"],
                warnings=[],
            )
        param_dict = {name: float(params[i]) for i, name in enumerate(ALL_PARAM_NAMES)}
    else:
        param_dict = dict(params)

    # Use the new severity-stratified system
    violations = validate_all_parameters(param_dict)

    errors = [v.message for v in violations if v.severity == ConstraintSeverity.ERROR]
    warnings = [v.message for v in violations if v.severity == ConstraintSeverity.WARNING]
    info = [v.message for v in violations if v.severity == ConstraintSeverity.INFO]

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
    )


def validate_time_integral_safety(
    alpha: float,
    t_min: float,
    t_max: float,
) -> ValidationResult:
    r"""Check the ``D0 * t**alpha`` time integral for numerical hazards.

    For :math:`J(t) = D_0\,t^{\alpha}`, the integral from 0 to :math:`T`
    needs care when ``alpha < 0`` (singularity at :math:`t = 0`) or ``alpha``
    is large (potential overflow of :math:`t^{\alpha}`).

    Parameters
    ----------
    alpha : float
        Diffusion-exponent value.
    t_min : float
        Minimum time. Must be ``> 0`` when ``alpha < 0``.
    t_max : float
        Maximum time.

    Returns
    -------
    ValidationResult
        Invalid (with an error) when ``alpha < 0`` and ``t_min <= 0``;
        otherwise valid, possibly carrying instability/overflow warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if alpha < 0 and t_min <= 0:
        errors.append(f"alpha={alpha:.3f} < 0 requires t_min > 0, got t_min={t_min}")

    if alpha < -1:
        warnings.append(f"alpha={alpha:.3f} < -1 may cause numerical instability near t=0")

    if alpha > 3:
        # t^alpha can overflow for large t
        try:
            power_val = t_max**alpha
            if power_val > 1e15:
                warnings.append(f"t_max^alpha = {t_max}^{alpha} = {power_val:.2e} may overflow")
        except OverflowError:
            warnings.append(f"t_max^alpha = {t_max}^{alpha:.1f} overflows float range")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_correlation_inputs(
    t1: np.ndarray,
    t2: np.ndarray,
    c2_data: np.ndarray,
) -> ValidationResult:
    """Validate correlation-matrix inputs for shape, finiteness, and range.

    Parameters
    ----------
    t1 : numpy.ndarray
        First time axis; must be strictly increasing.
    t2 : numpy.ndarray
        Second time axis; must be strictly increasing.
    c2_data : numpy.ndarray
        Correlation data; expected shape ``(len(t1), len(t2))``.

    Returns
    -------
    ValidationResult
        Errors for shape mismatch, NaN/inf entries, or non-monotonic time
        axes; warnings for out-of-range correlation values (negative or
        ``> 2``).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Shape checks
    expected_shape = (len(t1), len(t2))
    if c2_data.shape != expected_shape:
        errors.append(
            f"c2_data shape {c2_data.shape} doesn't match time grids ({len(t1)}, {len(t2)})"
        )

    # NaN/Inf checks
    nan_count = np.sum(np.isnan(c2_data))
    if nan_count > 0:
        errors.append(f"c2_data contains {nan_count} NaN values")

    inf_count = np.sum(np.isinf(c2_data))
    if inf_count > 0:
        errors.append(f"c2_data contains {inf_count} infinite values")

    # Value range checks
    if np.any(c2_data < 0):
        warnings.append("c2_data contains negative values (unusual for correlation)")

    if np.any(c2_data > 2):
        warnings.append("c2_data contains values > 2 (unusual for normalized correlation)")

    # Monotonicity of time axes
    if not np.all(np.diff(t1) > 0):
        errors.append("t1 must be strictly increasing")

    if not np.all(np.diff(t2) > 0):
        errors.append("t2 must be strictly increasing")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
