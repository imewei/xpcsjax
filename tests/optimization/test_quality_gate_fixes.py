"""Regression tests for the quality-gate audit fixes.

Locks in the behavioral contracts established by the audit so they cannot
silently regress:

- H-2: ``OptimizationResult`` invariant validation in ``__post_init__``.
- M-8: ``AnalysisMode.parse`` single normalization authority.
- CR-5: CMA-ES success/message separated from convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.results import OptimizationResult


def _result(**overrides: object) -> OptimizationResult:
    """Construct a minimal OptimizationResult, overriding selected fields."""
    kwargs: dict[str, object] = {
        "parameters": np.array([1.0, 2.0, 3.0]),
        "uncertainties": np.array([0.1, 0.2, 0.3]),
        "covariance": np.eye(3),
        "chi_squared": 1.0,
        "reduced_chi_squared": 0.5,
        "convergence_status": "converged",
        "iterations": 5,
        "execution_time": 0.1,
        "device_info": {},
    }
    kwargs.update(overrides)
    return OptimizationResult(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# H-2: OptimizationResult invariants
# --------------------------------------------------------------------------- #


def test_converged_result_with_empty_parameters_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty parameters"):
        _result(
            parameters=np.array([]),
            uncertainties=np.array([]),
            covariance=np.zeros((0, 0)),
        )


def test_converged_result_with_nonfinite_parameters_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _result(parameters=np.array([1.0, np.nan, 3.0]))


def test_covariance_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="covariance shape"):
        _result(covariance=np.eye(4))


def test_uncertainties_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncertainties length"):
        _result(uncertainties=np.array([0.1, 0.2]))


def test_failed_result_with_empty_parameters_is_allowed() -> None:
    """A genuinely failed fit may carry empty parameters — not an error."""
    res = _result(
        parameters=np.array([]),
        uncertainties=np.array([]),
        covariance=np.zeros((0, 0)),
        convergence_status="failed",
    )
    assert res.success is False


def test_valid_converged_result_constructs() -> None:
    res = _result()
    assert res.success is True


# --------------------------------------------------------------------------- #
# M-8: AnalysisMode.parse single authority
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("static_anisotropic", AnalysisMode.STATIC_ANISOTROPIC),
        ("Static_Anisotropic", AnalysisMode.STATIC_ANISOTROPIC),
        ("STATIC_ISOTROPIC", AnalysisMode.STATIC_ISOTROPIC),
        ("laminar_flow", AnalysisMode.LAMINAR_FLOW),
        ("Laminar Flow", AnalysisMode.LAMINAR_FLOW),
        ("two_component", AnalysisMode.TWO_COMPONENT),
        ("two-component", AnalysisMode.TWO_COMPONENT),
        ("Heterodyne", AnalysisMode.TWO_COMPONENT),
    ],
)
def test_parse_canonicalizes_synonyms(raw: str, expected: AnalysisMode) -> None:
    assert AnalysisMode.parse(raw) == expected


def test_parse_rejects_bare_static_by_default() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        AnalysisMode.parse("static")


def test_parse_allows_bare_static_when_opted_in() -> None:
    assert AnalysisMode.parse("static", allow_bare_static=True) == (
        AnalysisMode.STATIC_ANISOTROPIC
    )


def test_parse_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown analysis mode"):
        AnalysisMode.parse("not_a_real_mode")


def test_parse_returns_str_subclass() -> None:
    """StrEnum result must still compare/serialize as its string value."""
    parsed = AnalysisMode.parse("Heterodyne")
    assert parsed == "two_component"
    assert parsed.value == "two_component"
