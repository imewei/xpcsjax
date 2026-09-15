"""Shared physics-validation primitives for homodyne and heterodyne constraint tables.

Single source of truth for the severity enum, the violation/rule dataclasses,
the severity-priority ordering, and the non-finite-value predicate that
:mod:`xpcsjax.config.physics_validators` (homodyne) and
:mod:`xpcsjax.config.heterodyne_physics_validators` (``two_component``) each
build their own ``PHYSICS_CONSTRAINTS`` table on top of. Each sibling module
keeps its own constraint table and its own differing defaults (e.g.
``validate_single_parameter``'s ``min_severity`` default) -- this module only
consolidates the machinery that was previously copy-drifted between them.

Field names follow the homodyne module (``PhysicsViolation.param``,
``ConstraintRule.condition``); the heterodyne module exposes ``.parameter`` /
``.check`` as read-only property aliases for one release rather than breaking
its existing external readers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class ConstraintSeverity(StrEnum):
    """Severity levels for physics constraint violations.

    ``StrEnum`` (a ``str`` subclass): members compare equal to their string
    values, so existing ``== "error"`` checks and f-string formatting keep
    working, while static checkers treat the set as closed.
    """

    ERROR = "error"  # Physically impossible
    WARNING = "warning"  # Unusual but possible
    INFO = "info"  # Noteworthy observation


# Severity priority for ">=min_severity" threshold filtering. Higher = more
# severe. Ordinal only -- the exact integers are not part of the contract,
# just their relative order.
SEVERITY_PRIORITY: dict[ConstraintSeverity, int] = {
    ConstraintSeverity.ERROR: 3,
    ConstraintSeverity.WARNING: 2,
    ConstraintSeverity.INFO: 1,
}


def is_non_finite(value: float) -> bool:
    """Return ``True`` for ``NaN`` / ``±inf``; ``False`` for finite or non-numeric.

    IEEE-754 makes every relational comparison with ``NaN`` return ``False``,
    so a constraint rule's ``<``/``>``/``<=``/``>=`` predicate would silently
    *accept* a ``NaN``; this lets ``validate_single_parameter`` flag it
    explicitly instead. A non-numeric ``value`` returns ``False`` so it falls
    through to the caller's own ``except`` handling.
    """
    try:
        return not math.isfinite(value)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PhysicsViolation:
    """A single triggered physics constraint violation.

    Attributes
    ----------
    param : str
        Name of the offending parameter (or a composite label such as
        ``"f0+f3"`` for cross-parameter checks).
    value : float or None
        The value that triggered the violation.
    message : str
        Human-readable explanation, typically including the value.
    severity : ConstraintSeverity
        Severity of the violation.
    """

    param: str
    value: float | None
    message: str
    severity: ConstraintSeverity

    def format(self) -> str:
        """Render the violation as a single human-readable line."""
        return f"{self.param} = {self.value:.3e}: {self.message} [{self.severity}]"


@dataclass(frozen=True)
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

    condition: Callable[[float], bool]
    message: str
    severity: ConstraintSeverity


__all__ = [
    "ConstraintSeverity",
    "SEVERITY_PRIORITY",
    "is_non_finite",
    "PhysicsViolation",
    "ConstraintRule",
]
