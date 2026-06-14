"""Phase 5 — averaged uses EXPANDED constrained-model DOF (2*n_phi + n_physics)."""
from __future__ import annotations

import numpy as np


def _fit(mode, n_phi):
    from tests.optimization.test_phase5_standard_resolver import _fit as f
    return f(mode, n_phi)


def test_averaged_dof_basis_is_expanded():
    res = _fit("auto", n_phi=4)  # averaged
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "averaged"
    # The decision: averaged reduced-chi2/covariance uses 2*n_phi + n_physics, NOT
    # the optimizer count (n_physics + 2). Documented + stamped for auditability.
    assert diag.get("reduced_chi2_dof_basis") == "expanded_constrained_model"
    assert int(diag.get("n_dof_effective")) == 2 * 4 + 7  # 15


def test_individual_dof_basis_is_optimizer():
    res = _fit("individual", n_phi=4)
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "individual"
    # individual: optimizer DOF == expanded DOF, both 2*n_phi + n_physics.
    assert int(diag.get("n_dof_effective")) == 2 * 4 + 7


def test_constant_dof_basis_is_physics_only():
    from tests.optimization.test_phase5_standard_resolver import _laminar_cfg
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_phi, n_t = 4, 40
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    cfg = _laminar_cfg("constant", n_t)
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0))
    data = {"phi_angles_list": phi, "c2_exp": c2, "t1": t, "t2": t,
            "wavevector_q_list": np.array([0.0237])}
    res = fit_nlsq(data, cfg)
    diag = dict(res.nlsq_diagnostics or {})
    assert int(diag.get("n_dof_effective")) == 7  # physics only
