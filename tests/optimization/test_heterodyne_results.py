"""Smoke tests for the heterodyne NLSQResult dataclass + result helpers.

``xpcsjax/optimization/nlsq/heterodyne_results.py`` defines the result shape
every heterodyne fitter builds — fitted parameters, covariance, uncertainties,
chi², convergence reason, metadata. Downstream consumers (logger, post-fit
diagnostics, multi-phi result aggregation) read these fields directly.

If the dataclass shape drifts (a renamed field, a flipped optional default,
a missing factory) the consequence is silent: results still build, but
they're missing or mis-keyed. These tests fence the contract.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult


def _minimal_result() -> NLSQResult:
    """The smallest legal NLSQResult — just the four required fields."""
    return NLSQResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        parameter_names=["D0", "alpha", "D_offset"],
        success=True,
        message="converged",
    )


def test_required_fields_round_trip() -> None:
    """parameters, parameter_names, success, message are required positional
    args. Other fields default to None/empty."""
    r = _minimal_result()
    assert r.parameters.shape == (3,)
    assert r.parameter_names == ["D0", "alpha", "D_offset"]
    assert r.success is True
    assert r.message == "converged"

    # Defaults
    assert r.uncertainties is None
    assert r.covariance is None
    assert r.final_cost is None
    assert r.reduced_chi_squared is None
    assert r.n_iterations == 0
    assert r.n_function_evals == 0


def test_metadata_is_per_instance_dict() -> None:
    """Two results must not share the same metadata dict (field defaults to
    a fresh dict via ``field(default_factory=dict)``). A regression to
    ``= {}`` would alias the dict across all instances."""
    r1 = _minimal_result()
    r2 = _minimal_result()

    r1.metadata["foo"] = "bar"
    assert "foo" not in r2.metadata, (
        "metadata is shared across instances — likely a class-level "
        "default; the dataclass needs field(default_factory=dict)"
    )


def test_optional_arrays_round_trip() -> None:
    """The optional uncertainty / covariance / residuals fields accept
    arrays and round-trip them unchanged."""
    cov = np.eye(3, dtype=np.float64) * 0.1
    sigma = np.array([0.01, 0.02, 0.03])
    resid = np.array([1.0, -1.0, 0.5])

    r = NLSQResult(
        parameters=np.array([1.0, 2.0, 3.0]),
        parameter_names=["a", "b", "c"],
        success=True,
        message="ok",
        uncertainties=sigma,
        covariance=cov,
        residuals=resid,
        final_cost=0.5,
        reduced_chi_squared=1.2,
        n_iterations=42,
        n_function_evals=84,
        convergence_reason="ftol_reached",
    )

    np.testing.assert_array_equal(r.uncertainties, sigma)
    np.testing.assert_array_equal(r.covariance, cov)
    np.testing.assert_array_equal(r.residuals, resid)
    assert r.final_cost == 0.5
    assert r.reduced_chi_squared == 1.2
    assert r.n_iterations == 42
    assert r.n_function_evals == 84
    assert r.convergence_reason == "ftol_reached"


def test_failure_result_message_preserved() -> None:
    """When success=False the message must carry the diagnostic — a regression
    to silently empty/truncated messages would hide convergence failures."""
    r = NLSQResult(
        parameters=np.array([np.nan, np.nan, np.nan]),
        parameter_names=["D0", "alpha", "D_offset"],
        success=False,
        message="Tier standard failed after 3 retries",
    )
    assert r.success is False
    assert "Tier" in r.message and "retries" in r.message
