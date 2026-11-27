"""Covariance reported by the sequential per-angle heterodyne path."""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq import heterodyne_core
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult

_N_T = 12
_N_ROWS = (_N_T - 1) * (_N_T - 2)  # off-diagonal residual rows, t = 0 excluded


def _weights(n_observed: int) -> np.ndarray:
    """Weights keeping ``n_observed`` of the residual rows."""
    weights = np.zeros((_N_T, _N_T))
    rows, cols = np.nonzero(~np.eye(_N_T, dtype=bool))
    keep = (rows > 0) & (cols > 0)
    weights[rows[keep][:n_observed], cols[keep][:n_observed]] = 1.0
    return weights


@pytest.mark.parametrize("n_observed", [60, 10], ids=["some-dof", "no-dof"])
def test_masked_pixels_are_not_degrees_of_freedom(monkeypatch, n_observed):
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=_N_T)
    p = model.param_manager.n_varying

    def solve(_self, jax_residual_fn, initial_params, n_data, **_kwargs):  # called by keyword
        assert n_data == _N_ROWS
        return NLSQResult(
            parameters=np.asarray(initial_params),
            parameter_names=list(model.param_manager.varying_names),
            success=True,
            message="ok",
            covariance=np.eye(p),
            uncertainties=np.ones(p),
            final_cost=0.5,
        )

    monkeypatch.setattr(heterodyne_core.NLSQAdapter, "fit_jax", solve)

    def fit(weights):
        return heterodyne_core._fit_local(model, c2[0], float(phi[0]), NLSQConfig(), weights, True)

    unmasked, result = fit(None), fit(_weights(n_observed))

    if n_observed > p:
        ratio = (_N_ROWS - p) / (n_observed - p)
        np.testing.assert_allclose(result.uncertainties, np.sqrt(ratio))
        np.testing.assert_allclose(result.reduced_chi_squared, unmasked.reduced_chi_squared * ratio)
    else:
        assert np.all(np.isnan(result.uncertainties))
        assert result.metadata["covariance_is_placeholder"] is True
        assert np.isnan(result.reduced_chi_squared)


@pytest.mark.parametrize("n_observed", [60, 10], ids=["some-dof", "no-dof"])
def test_aggregate_reduced_chi2_counts_only_observations(n_observed):
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=_N_T)
    p = model.param_manager.n_varying
    solve = NLSQResult(
        parameters=np.asarray(model.param_manager.get_initial_values()),
        parameter_names=list(model.param_manager.varying_names),
        success=True,
        message="ok",
        final_cost=0.5,
    )

    unmasked, result = (
        heterodyne_core._aggregate_individual_results([solve], model, phi, c2, 0.0, weights=w)
        for w in (None, _weights(n_observed))
    )

    if n_observed > p:
        np.testing.assert_allclose(
            result.reduced_chi_squared,
            unmasked.reduced_chi_squared * (_N_ROWS - p) / (n_observed - p),
        )
    else:
        assert np.isnan(result.reduced_chi_squared)
        assert result.quality_flag == "unknown"


def test_aggregate_without_any_covariance_reports_unknown_physics_sigmas():
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=_N_T)
    p = model.param_manager.n_varying
    solve = NLSQResult(
        parameters=np.asarray(model.param_manager.get_initial_values()),
        parameter_names=list(model.param_manager.varying_names),
        success=True,
        message="ok",
        covariance=np.full((p, p), np.nan),
        final_cost=0.5,
    )

    result = heterodyne_core._aggregate_individual_results([solve], model, phi, c2, 0.0)

    physics = result.uncertainties[2:]  # [contrast | offset | 14 physics]
    assert np.all(np.isnan(physics[model.param_manager.varying_indices]))
    assert result.nlsq_diagnostics["covariance_is_placeholder"] is True
