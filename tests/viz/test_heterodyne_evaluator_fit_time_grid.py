"""The heterodyne branch of ``_evaluate_c2_per_angle`` must evaluate the
FIT-TIME model.

The heterodyne fit runs the stateful ``HeterodyneModel`` whose time axis is
``arange(n) * dt + t_start`` (``t_start = dt`` by default), synced to the data
length. The evaluator used to feed the loader's ``t1`` (origin 0) to the adapter
kernel, shifting every plotted surface by one ``dt`` relative to what was fit —
invisible on 1000-frame data, wrong on short grids. Pin the fix on a short grid
where the shift is decisive.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data
from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
from xpcsjax.service.fit import run_fit
from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle, _unpack_heterodyne_scaling


def test_heterodyne_evaluator_matches_fit_time_model_on_short_grid():
    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=12)
    n_t = data["c2_exp"].shape[1]
    dt = float(cfg.config["analyzer_parameters"]["dt"])
    t = np.arange(n_t, dtype=np.float64) * dt  # loader contract: origin 0
    data["t1"], data["t2"] = t, t.copy()
    cfg.config["analyzer_parameters"]["geometry"] = {"stator_rotor_gap": 2_000_000.0}

    res = run_fit(cfg, data)
    model = cfg.get_model()
    config = cfg.get_config()
    phi = np.asarray(data["phi_angles_list"], dtype=np.float64)

    # Oracle: the exact object/grid the fit ran (optimization/nlsq/__init__.py).
    hm = HeterodyneModel.from_config(cfg.config)
    hm.sync_time_axis(np.arange(n_t, dtype=np.float64))
    contrasts, offsets, physical, _ = _unpack_heterodyne_scaling(
        model, res, n_phi_expected=len(phi)
    )
    full = hm.param_manager.expand_varying_to_full(np.asarray(physical, dtype=np.float64))

    mask = np.ones((n_t, n_t), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    np.fill_diagonal(mask, False)
    for i, phi_deg in enumerate(phi):
        viz = np.asarray(_evaluate_c2_per_angle(model, res, data, config, float(phi_deg), i))
        oracle = np.asarray(
            hm.compute_correlation(
                phi_angle=float(phi_deg),
                params=full,
                contrast=float(contrasts[i]),
                offset=float(offsets[i]),
                angle_idx=i,
            )
        )
        np.testing.assert_allclose(viz, oracle, rtol=0, atol=1e-12)
        # And the plotted surface reproduces the data the fit converged on
        # (synthetic noise 5e-4): the one-dt shift gave rms ~ 7e-2 here.
        rms = np.sqrt(np.mean((viz - data["c2_exp"][i])[mask] ** 2))
        assert rms < 2e-3, rms
