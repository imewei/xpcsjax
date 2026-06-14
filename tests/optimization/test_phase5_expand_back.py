"""Phase 5 — averaged/constant results expand back to the dense per-angle layout."""
from __future__ import annotations

import numpy as np


def test_averaged_result_is_dense_per_angle():
    from tests.optimization.test_phase5_standard_resolver import _fit

    n_phi = 4
    res = _fit("auto", n_phi=n_phi)  # averaged
    params = np.asarray(res.parameters, dtype=np.float64)
    # Dense scaling-first layout: [c_0..c_{n-1}, o_0..o_{n-1}, *7 physics]
    assert params.shape[0] == 2 * n_phi + 7
    # averaged broadcast: all contrasts equal, all offsets equal
    contrasts = params[:n_phi]
    offsets = params[n_phi : 2 * n_phi]
    np.testing.assert_allclose(contrasts, np.full(n_phi, contrasts[0]), rtol=1e-12)
    np.testing.assert_allclose(offsets, np.full(n_phi, offsets[0]), rtol=1e-12)
    # covariance/uncertainties are dense too
    assert np.asarray(res.uncertainties).shape[0] == 2 * n_phi + 7


def test_constant_result_is_dense_per_angle():
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
    params = np.asarray(res.parameters, dtype=np.float64)
    assert params.shape[0] == 2 * n_phi + 7  # dense, scaling frozen but surfaced
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "constant"
    assert int(diag.get("n_optimized")) == 0
