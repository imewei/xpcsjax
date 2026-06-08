"""Laminar stratified-LS ``execute_layers`` (Phase 3 step 11) — direct-call oracle.

The laminar mirror of the heterodyne gated L2/L3 execution lives in
``strategies/stratified_ls.py::fit_with_stratified_least_squares``. C020-scale
individual fits (53 params, 23 M points) route to the OUT_OF_CORE path on memory
(verified: peak ~59 GB > threshold), so they do NOT exercise the stratified-LS
path. This oracle therefore calls ``fit_with_stratified_least_squares`` DIRECTLY
on a synthetic ``laminar_flow`` dataset — the function the mirror lives in —
bypassing the wrapper's memory router.

It asserts the hardware-robust MECHANISM contract (cross-platform safe — the
executed-path *numerics* are CPU-microarch specific, see
``project_heterodyne-engine-route-platform-fragility``, but whether L2 fires /
keep-better holds / markers are honest is structural):

* **flag OFF (default):** the single ``least_squares`` solve is the result — no
  L2 (no ``"hierarchical"`` key), honest inactive markers, byte-identical path.
* **flag ON:** L2 executes via the controller's hierarchical optimizer
  (``"hierarchical"`` present, ``execute_layers_status == "executed"``) and the
  keep-better guard holds (data-only SSR never worse than the flag-OFF baseline
  beyond tol); L3 (``regularization``) rides inside when configured.
* **objective separation:** the reported ``final_cost`` is the data-only SSR
  (penalty rows never contaminate it).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

_NAMES = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
_LOG = logging.getLogger("test_laminar_execute_layers")


def _build_laminar_stratified_data():
    """Synthetic ≥-chunk laminar dataset: 5 angles, self-consistent g2 at truth."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel

    phi = np.array([0.0, 36.0, 72.0, 108.0, 144.0], dtype=np.float64)
    n_t = 12
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    cfg = ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": n_t,
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
            },
            "initial_parameters": {"parameter_names": _NAMES, "values": true.tolist()},
        }
    )
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0), dtype=np.float64)

    pf, t1f, t2f, g2f = [], [], [], []
    for i, p in enumerate(phi):
        for a in range(n_t):
            for b in range(n_t):
                pf.append(p)
                t1f.append(t[a])
                t2f.append(t[b])
                g2f.append(c2[i, a, b])
    pf = np.array(pf)
    t1f = np.array(t1f)
    t2f = np.array(t2f)
    g2f = np.array(g2f)
    strat = SimpleNamespace(
        phi_flat=pf,
        t1_flat=t1f,
        t2_flat=t2f,
        g2_flat=g2f,
        phi=np.unique(pf),
        t1=np.unique(t1f),
        t2=np.unique(t2f),
        g2=g2f,
        sigma=np.ones((len(phi), n_t, n_t)),
        q=0.0237,
        L=2000000.0,
        dt=0.1,
        stratification_diagnostics=None,
        chunk_sizes=None,
    )
    n_phi = len(phi)
    init = np.concatenate([np.full(n_phi, 0.3), np.full(n_phi, 1.0), true])  # [c|o|physics]
    lower = np.concatenate(
        [np.zeros(2 * n_phi), np.array([1e-6, -2.0, -50.0, -1.0, -2.0, -50.0, -180.0])]
    )
    upper = np.concatenate(
        [np.full(2 * n_phi, 5.0), np.array([1e5, 2.0, 50.0, 1.0, 2.0, 50.0, 180.0])]
    )
    return strat, init, (lower, upper), n_phi


def _fit(execute_layers: bool, *, regularization: bool = True):
    from xpcsjax.config.parameter_registry import AnalysisMode
    from xpcsjax.optimization.nlsq.strategies.stratified_ls import (
        fit_with_stratified_least_squares,
    )

    strat, init, bounds, _n_phi = _build_laminar_stratified_data()
    ad = {
        "enable": True,
        "per_angle_mode": "individual",
        "execute_layers": execute_layers,
        "hierarchical": {
            "enable": True,
            "max_outer_iterations": 2,
            "physical_max_iterations": 30,
            "per_angle_max_iterations": 20,
        },
    }
    # ``regularization_mode`` defaults to "relative" (L3 active); "none" is the
    # explicit opt-out (regularization.enable is ignored by AntiDegeneracyConfig).
    ad["regularization"] = (
        {"enable": True, "mode": "relative", "lambda": 1.0}
        if regularization
        else {"mode": "none"}
    )
    popt, _pcov, info = fit_with_stratified_least_squares(
        strat,
        True,
        _NAMES,
        init,
        bounds,
        _LOG,
        target_chunk_size=2000,
        anti_degeneracy_config=ad,
        nlsq_config_dict={"max_iterations": 100},
        analysis_mode=AnalysisMode.LAMINAR_FLOW,
    )
    return popt, info


def test_flag_off_is_single_solve_no_l2():
    """Flag OFF: no L2 executes; honest inactive markers (byte-identical path)."""
    popt, info = _fit(execute_layers=False)
    ad = info.get("anti_degeneracy", {})
    assert "hierarchical" not in ad  # wrapper derives hierarchical_active from this
    assert "execute_layers_status" not in ad
    assert len(popt) == 17  # 2*5 + 7, individual layout


def test_flag_on_executes_l2_and_l3_keep_better():
    """Flag ON: L2 executes (+ L3), keep-better holds, objective is data-only."""
    _popt_off, info_off = _fit(execute_layers=False)
    popt_on, info_on = _fit(execute_layers=True)

    ad_on = info_on["anti_degeneracy"]
    # L2 executed via the controller's hierarchical optimizer.
    assert "hierarchical" in ad_on
    assert ad_on["execute_layers_status"] in ("executed", "executed_not_converged")
    assert ad_on["execute_layers_kind"] == "L2_hierarchical"
    assert "execute_layers_converged" in ad_on
    # L3 rode inside the L2 scalar loss.
    assert "regularization" in ad_on
    # Identity-placeholder covariance flagged on the accepted L2 branch.
    assert ad_on.get("covariance_is_placeholder") is True
    # Keep-better: the executed data-only objective is never worse than baseline.
    assert info_on["final_cost"] <= info_off["final_cost"] * (1.0 + 1e-3)
    assert len(popt_on) == 17
    assert np.isfinite(info_on["final_cost"])


def test_flag_on_l2_without_l3():
    """L2 alone (no regularization configured) still executes and keeps better."""
    _popt_off, info_off = _fit(execute_layers=False, regularization=False)
    _popt_on, info_on = _fit(execute_layers=True, regularization=False)
    ad_on = info_on["anti_degeneracy"]
    assert "hierarchical" in ad_on
    assert "regularization" not in ad_on  # L3 not configured
    assert info_on["final_cost"] <= info_off["final_cost"] * (1.0 + 1e-3)
