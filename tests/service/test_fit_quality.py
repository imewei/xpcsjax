"""Tests for the mode-agnostic NRMSE metric (xpcsjax/service/fit_quality.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import xpcsjax.service.fit_quality as fq
from xpcsjax.optimization.nlsq.results import OptimizationResult


def _result(*, sigma_is_default: bool = False) -> OptimizationResult:
    return OptimizationResult(
        parameters=np.array([1.0, 2.0]),
        uncertainties=np.array([0.1, 0.1]),
        covariance=np.eye(2) * 0.01,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.0,
        device_info={},
        sigma_is_default=sigma_is_default,
    )


def _cm(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        get_config=lambda: {"analysis_mode": mode},
        get_model=lambda: object(),
        config={"analysis_mode": mode},
    )


def test_nrmse_matches_closed_form_over_masked_support(monkeypatch):
    """nrmse == sqrt(SSR/n_valid) / std(data) over the t>0, off-diagonal mask,
    accumulated across angles."""
    rng = np.random.default_rng(0)
    n_phi, n_t = 2, 6
    c2 = 1.0 + rng.normal(0.0, 0.1, size=(n_phi, n_t, n_t))
    fit = c2 + rng.normal(0.0, 0.02, size=c2.shape)
    # Poison the excluded cells: they must not affect the metric.
    c2[:, 0, :] = 1e6
    c2[:, :, 0] = -1e6
    for i in range(n_phi):
        np.fill_diagonal(c2[i], 1e9)

    # The lazy import inside compute_fit_quality reads from viz; redirect it.
    import xpcsjax.viz.nlsq_plots as viz

    monkeypatch.setattr(
        viz, "_evaluate_c2_per_angle", lambda m, r, d, c, phi, phi_index=None: fit[phi_index]
    )

    data = {"c2_exp": c2, "phi_angles_list": np.array([0.0, 90.0])}
    block = fq.compute_fit_quality(_result(), data, _cm("laminar_flow"))

    mask = np.ones((n_t, n_t), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    np.fill_diagonal(mask, False)
    obs = np.concatenate([c2[i][mask] for i in range(n_phi)])
    res = np.concatenate([(c2[i] - fit[i])[mask] for i in range(n_phi)])
    expected = np.sqrt(np.mean(res**2)) / np.std(obs)

    assert block["n_valid"] == n_phi * (n_t - 1) * (n_t - 2)
    assert block["n_angles_evaluated"] == n_phi
    assert block["mask"] == fq.MASK_DESCRIPTION
    np.testing.assert_allclose(block["nrmse"], expected, rtol=1e-12)
    np.testing.assert_allclose(block["c2_std"], np.std(obs), rtol=1e-12)


def test_failed_angle_is_skipped_not_fatal(monkeypatch):
    import xpcsjax.viz.nlsq_plots as viz

    c2 = np.ones((2, 5, 5))

    def _eval(m, r, d, c, phi, phi_index=None):
        if phi_index == 1:
            raise RuntimeError("boom")
        return c2[0] + 0.1

    monkeypatch.setattr(viz, "_evaluate_c2_per_angle", _eval)
    data = {"c2_exp": c2, "phi_angles_list": np.array([0.0, 90.0])}
    block = fq.compute_fit_quality(_result(), data, _cm("laminar_flow"))
    assert block["n_angles_failed"] == 1
    assert block["n_angles_evaluated"] == 1
    # Constant data -> zero spread -> nrmse undefined (NaN), never a fake finite.
    assert np.isnan(block["nrmse"])
    np.testing.assert_allclose(block["rms_residual"], 0.1)


@pytest.mark.parametrize(
    ("mode", "sigma_default", "expected"),
    [
        ("two_component", False, "far_lag_estimate"),
        ("laminar_flow", True, "none_unweighted"),
        ("laminar_flow", False, "data"),
        ("static_isotropic", True, "none_unweighted"),
    ],
)
def test_sigma_source_vocabulary(mode, sigma_default, expected):
    assert fq.sigma_source(_result(sigma_is_default=sigma_default), mode) == expected


def test_sigma_source_stratified_default_is_constant_placeholder():
    res = _result(sigma_is_default=True)
    res.stratification_diagnostics = object()  # any non-None marker
    assert fq.sigma_source(res, "laminar_flow") == "default_constant_0.01"


def test_attach_never_raises_on_broken_config_manager():
    res = _result()
    fq.attach_fit_quality(res, {"c2_exp": None}, SimpleNamespace())  # no get_config
    assert res.nlsq_diagnostics is None or "fit_quality" not in res.nlsq_diagnostics


def test_run_fit_attaches_fit_quality_two_component_end_to_end():
    """Real two_component fit through the service seam carries a finite nrmse."""
    from tests.optimization._heterodyne_fixtures import make_cfgmgr_and_data
    from xpcsjax.service.fit import run_fit

    cfg, data = make_cfgmgr_and_data(n_phi=3, n_t=12)
    # The shared fixture carries only what the fit needs; the per-angle model
    # evaluator (shared with the plots) also needs the loader's time grids and
    # the geometry block, exactly as a real loaded dataset provides them.
    n_t = data["c2_exp"].shape[1]
    t = np.arange(n_t, dtype=np.float64) * cfg.config["analyzer_parameters"]["dt"]
    data["t1"], data["t2"] = t, t.copy()  # loader contract: 1D [0, dt, 2dt, ...]
    cfg.config["analyzer_parameters"]["geometry"] = {"stator_rotor_gap": 2_000_000.0}
    res = run_fit(cfg, data)
    block = res.nlsq_diagnostics["fit_quality"]
    assert block["sigma_source"] == "far_lag_estimate"
    assert block["n_angles_evaluated"] == 3
    assert np.isfinite(block["nrmse"])
    # Self-consistent synthetic data (noise 5e-4 on c2 ~ 1.0-1.3): misfit is a
    # small fraction of the data spread.
    assert 0.0 < block["nrmse"] < 0.05


def test_run_fit_attaches_fit_quality_laminar_flow_end_to_end():
    """Real laminar_flow (homodyne) fit through the service seam: same block,
    same mask, ``sigma_source`` names the default-constant sigma."""
    from tests.optimization.test_phase5_standard_resolver import _laminar_cfg
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.service.fit import run_fit

    n_phi, n_t = 3, 12
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    cfg = _laminar_cfg("auto", n_t)
    # Synthetic data must match what the fit sees on real data: the loader's
    # preprocessing applies the diagonal correction (data/preprocessing.py,
    # ``apply_diagonal_correction: True`` default) and the wrapper's model
    # applies the same correction, and the data time grid must be the model
    # grid the c2 was generated on (HomodyneModel: linspace(0, dt*(n_t-1))).
    from xpcsjax.core.diagonal_correction import apply_diagonal_correction

    dt = float(cfg.config["analyzer_parameters"]["dt"])
    t = np.arange(n_t, dtype=np.float64) * dt
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    c2 = np.asarray(HomodyneModel(cfg.config).compute_c2(true, phi, contrast=0.3, offset=1.0))
    c2 = c2 + np.random.default_rng(3).normal(0.0, 5e-4, size=c2.shape)
    c2 = np.stack([np.asarray(apply_diagonal_correction(c2[i])) for i in range(n_phi)])
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),
    }
    res = run_fit(cfg, data)
    block = res.nlsq_diagnostics["fit_quality"]
    assert block["sigma_source"] == "none_unweighted"
    assert block["n_angles_evaluated"] == n_phi
    assert block["n_valid"] == n_phi * (n_t - 1) * (n_t - 2)
    assert np.isfinite(block["nrmse"])
    # Consistent synthetic data: misfit is a small fraction of the data spread.
    assert 0.0 < block["nrmse"] < 0.05
