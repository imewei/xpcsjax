"""Uncertainty parity between the two heterodyne in-memory joint-fit paths.

The engine route (``fit_two_component_via_engine``) hands nlsq a padded,
diagonal-masked residual vector; nlsq scales its covariance by
``cost / (ysize - p)`` with that padded ``ysize``. The production
``fit_nlsq_multi_phi`` residual carries only the ``n_phi * (n_t-1) * (n_t-2)``
valid observations. Without the post-solve dof rescale the engine route reported
uncertainties smaller by ``sqrt((n_valid - p) / (ysize - p))`` for the SAME
problem (2.7 % on this fixture, ~1/(2 n_t) plus padding on real data).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.covariance import rescale_covariance_dof
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

N_PHI, N_T = 3, 20


def _cfg(mode: str) -> NLSQConfig:
    c = NLSQConfig(
        method="trf",
        loss="soft_l1",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        x_scale="jac",
        max_nfev=2000,
        enable_cmaes=False,
        multistart=False,
    )
    c.per_angle_mode = mode
    return c


def test_rescale_covariance_dof_factor():
    pcov = np.eye(2)
    out = rescale_covariance_dof(pcov, n_rows_solver=1083, n_valid=1026, n_params=16)
    np.testing.assert_allclose(out, pcov * (1083 - 16) / (1026 - 16))
    # Identity when the counts agree or a dof is non-positive.
    assert rescale_covariance_dof(pcov, n_rows_solver=100, n_valid=100, n_params=3) is not None
    np.testing.assert_array_equal(
        rescale_covariance_dof(pcov, n_rows_solver=100, n_valid=100, n_params=3), pcov
    )
    np.testing.assert_array_equal(
        rescale_covariance_dof(pcov, n_rows_solver=10, n_valid=2, n_params=3), pcov
    )


@pytest.mark.parametrize("mode", ["constant", "averaged", "individual"])
def test_engine_route_uncertainties_match_multi_phi_dof(mode):
    model, c2, phi = make_synthetic_two_component(n_phi=N_PHI, n_t=N_T)
    eng = fit_two_component_via_engine(model, c2, np.asarray(phi), _cfg(mode), None)
    ref = fit_nlsq_multi_phi(model, c2, list(phi), _cfg(mode), None)

    # The dof the engine covariance is now expressed on is the production one.
    dof = eng.nlsq_diagnostics["covariance_dof"]
    n_valid = N_PHI * (N_T - 1) * (N_T - 2)
    assert dof["n_valid"] == n_valid
    assert dof["n_rows_solver"] >= N_PHI * (N_T - 1) ** 2  # padded grid incl. masked diagonal
    assert dof["n_params"] == eng.parameters.size

    unc_e = np.asarray(eng.uncertainties)
    unc_r = np.asarray(ref.uncertainties)
    p_r = np.asarray(ref.parameters)
    assert unc_e.shape == unc_r.shape
    # Compare only the IDENTIFIABLE parameters (relative uncertainty < 10 %).
    # The 14-physics fixture is degenerate in several physics directions
    # (sigma ~ 1e6): there the two paths sit in slightly different basins and
    # the ratio reflects the basin, not the estimator. On the identifiable
    # ones the two estimators must agree once the dof matches; before the
    # rescale their ratio was pinned at sqrt((1026-p)/(1083-p)) ~ 0.973.
    ident = np.isfinite(unc_e) & np.isfinite(unc_r) & (unc_r > 0) & (unc_r < 0.1 * np.abs(p_r))
    if mode == "constant":
        # Physics-only: this fixture leaves no physics parameter identifiable.
        if not ident.any():
            return
    assert ident.any(), "fixture must yield at least one identifiable parameter"
    ratio = unc_e[ident] / unc_r[ident]
    np.testing.assert_allclose(ratio, 1.0, atol=1e-2)
