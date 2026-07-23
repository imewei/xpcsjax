"""Regression: ``_fit_local`` reports reduced chi^2 on the ``(n-1)*(n-2)`` DOF.

The heterodyne residual mask (``_offdiag_indices``) excludes BOTH the t=0
boundary row/column AND the diagonal, so the valid residual count is
``(n-1)*(n-2)`` — the convention ``_compute_per_angle_chi2`` already uses.
``_fit_local`` / ``_fit_cmaes`` used to compute ``c2.size - n_matrix`` (the
diagonal-only ``N^2 - N``), inflating the DOF and skewing ``reduced_chi_squared``.

This reconstructs sigma^2 and SSR exactly the way the code does from the fit
result, then asserts the reported reduced chi^2 matches the ``(n-1)*(n-2)`` DOF
value and NOT the old ``N^2 - N`` value. It checks arithmetic, not fit quality,
so the result doesn't depend on which solver basin was reached (uses the
actual ``final_cost`` the solve produced).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import _fit_local

from ._heterodyne_fixtures import make_synthetic_two_component


def _sigma2_far_lag(c2_matrix: np.ndarray) -> float:
    """Re-derive the far-lag photon-noise estimate the fitters use."""
    n = c2_matrix.shape[0]
    row_idx = np.arange(n)
    lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
    far_vals = c2_matrix[lag_mat >= n // 2]
    return float(np.var(far_vals)) if far_vals.size > 1 else 0.0


def test_fit_local_reduced_chi2_uses_n_minus_1_n_minus_2_dof() -> None:
    n_t = 12
    model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=n_t)
    config = NLSQConfig(enable_cmaes=False)

    result = _fit_local(
        model, c2[0], float(phi[0]), config, weights=None, use_nlsq_library=True, angle_idx=0
    )

    assert result.final_cost is not None
    sigma2 = _sigma2_far_lag(np.asarray(c2[0]))
    if sigma2 <= 1e-12:
        pytest.skip("far-lag noise estimate degenerate; chi^2 correction not applied")

    ssr = 2.0 * float(result.final_cost)
    n_varying = model.param_manager.n_varying

    dof_correct = max((n_t - 1) * (n_t - 2) - n_varying, 1)
    dof_buggy = max(n_t * n_t - n_t - n_varying, 1)
    chi2_correct = ssr / (sigma2 * dof_correct)
    chi2_buggy = ssr / (sigma2 * dof_buggy)

    assert dof_correct != dof_buggy  # sanity: the two conventions differ here
    assert result.reduced_chi_squared == pytest.approx(chi2_correct, rel=1e-9), (
        "reduced_chi_squared must use (n-1)*(n-2) DOF, matching _compute_per_angle_chi2"
    )
    assert result.reduced_chi_squared != pytest.approx(chi2_buggy, rel=1e-9), (
        "reduced_chi_squared still uses the buggy N^2 - N (diagonal-only) DOF"
    )
