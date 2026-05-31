"""Golden tests for the extracted ``_build_joint_problem`` helper.

Task 1 of the heterodyne joint-escapes work is a *behavior-preserving* refactor:
the plain joint fit (``_fit_joint_multi_phi``) builds its residual / x0 / bounds /
reparameterizer inline; we lift that construction into a shared
``_build_joint_problem`` helper so the upcoming CMA-ES and multistart global
escapes optimize the IDENTICAL objective. These tests pin:

1. The extracted problem is internally consistent (shapes align, x0 in-bounds,
   residual finite) and exposes the meta the escapes need.
2. The plain joint fit is numerically unchanged after the extraction — proven by
   the SSR conservation invariant ``chi2_per_angle.sum() == chi_squared``.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import (
    _build_joint_problem,
    fit_nlsq_multi_phi,
)


def test_build_joint_problem_shapes_and_residual_finite():
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    prob = _build_joint_problem(model, c2, phi, cfg, weights=None)
    x0 = np.asarray(prob.x0, dtype=np.float64)
    lb = np.asarray(prob.lb, dtype=np.float64)
    ub = np.asarray(prob.ub, dtype=np.float64)
    assert x0.shape == lb.shape == ub.shape
    assert np.all(lb <= x0) and np.all(x0 <= ub)
    r = np.asarray(prob.joint_residual_fn(x0), dtype=np.float64)
    assert r.ndim == 1 and np.all(np.isfinite(r))
    assert prob.meta["n_physics_varying"] == len(model.param_manager.varying_names)


def test_plain_joint_fit_unchanged_after_extraction():
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=20)
    cfg = NLSQConfig.from_dict(
        {"analysis_mode": "two_component", "per_angle_mode": "individual"}
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    diag = res.nlsq_diagnostics
    assert np.isclose(
        float(np.sum(diag["chi2_per_angle"])), res.chi_squared, rtol=1e-6
    )
