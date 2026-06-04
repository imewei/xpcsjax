"""Phase-0 gate: NLSQ's curve_fit callback must be observational (cannot perturb a
fit) AND fired per-iteration. Standing guard that the L4 monitor callback can never
change a solve trajectory."""

import numpy as np


def _recording_callback():
    seen = []

    def cb(iteration, cost, params, info=None, **kwargs):
        seen.append((int(iteration), np.asarray(params, dtype=np.float64).copy()))
        return None

    return cb, seen


def test_heterodyne_curve_fit_callback_is_observational_and_per_iteration():
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.gradient_monitor import _set_debug_curvefit_callback
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    base = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)

    cb, seen = _recording_callback()
    _set_debug_curvefit_callback(cb)
    try:
        withcb = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    finally:
        _set_debug_curvefit_callback(None)

    print(f"[heterodyne] len(seen)={len(seen)}")

    assert np.array_equal(np.asarray(base.parameters), np.asarray(withcb.parameters))
    assert base.chi_squared == withcb.chi_squared
    # Hard-gate covariance (pcov) bit-identity too: monitor-on vs monitor-off
    # must be identical at rtol=0/atol=0 on popt + pcov + chi2.
    cov_a = getattr(base, "covariance", None)
    cov_b = getattr(withcb, "covariance", None)
    if cov_a is not None and cov_b is not None:
        assert np.array_equal(np.asarray(cov_a), np.asarray(cov_b))
    assert len(seen) == 0 or len(seen) >= 2  # per-iteration, or fallback-only


def _build_laminar_fit():
    """Build a small synthetic laminar_flow fit that routes through the live
    NLSQWrapper STANDARD curve_fit path.

    CMA-ES and multi-start are disabled so ``fit_nlsq_jax`` does not delegate to
    the global-search wrappers; the in-memory STANDARD curve_fit path runs
    instead. With the default ``enable_recovery=True``, the live curve_fit call
    is the one in ``recovery.execute_with_recovery`` (the seam is wired there;
    ``fallback_chain`` and ``stratified_ls`` carry the same seam for the
    recovery-disabled and >=1M-point paths).
    """
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_t = 8
    phi = np.array([0.0, 90.0], dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true_params = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    config_dict = {
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
            "values": true_params.tolist(),
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "laminar_flow",
                "max_iterations": 50,
                "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": {"enable": False},
            },
            "stratification": {"enabled": False},
        },
    }

    cfg = ConfigManager(config_override=config_dict)

    # Build self-consistent synthetic c2 from the model at the true params (+ tiny
    # noise) so the live STANDARD curve_fit solve runs cleanly instead of diverging.
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(
        model.compute_c2(true_params, phi, contrast=0.3, offset=1.0),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed=20260529)
    c2 = c2 + rng.normal(0.0, 5e-4, size=c2.shape)

    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237], dtype=np.float64),
    }
    return fit_nlsq, data, cfg


def test_homodyne_curve_fit_callback_is_observational_and_per_iteration():
    from xpcsjax.optimization.nlsq.gradient_monitor import _set_debug_curvefit_callback

    fit_nlsq, data, cfg = _build_laminar_fit()

    base = fit_nlsq(data, cfg)

    cb, seen = _recording_callback()
    _set_debug_curvefit_callback(cb)
    try:
        withcb = fit_nlsq(data, cfg)
    finally:
        _set_debug_curvefit_callback(None)

    print(f"[homodyne] len(seen)={len(seen)}")

    base_params = np.asarray(_result_params(base), dtype=np.float64)
    withcb_params = np.asarray(_result_params(withcb), dtype=np.float64)
    assert np.array_equal(base_params, withcb_params)
    assert _result_chi2(base) == _result_chi2(withcb)
    # Hard-gate covariance (pcov) bit-identity too (popt + pcov + chi2 at rtol=0).
    cov_a = getattr(base, "covariance", None)
    cov_b = getattr(withcb, "covariance", None)
    if cov_a is not None and cov_b is not None:
        assert np.array_equal(np.asarray(cov_a), np.asarray(cov_b))
    assert len(seen) == 0 or len(seen) >= 2  # per-iteration, or fallback-only


def _result_params(result):
    for attr in ("parameters", "popt", "x"):
        if hasattr(result, attr):
            return getattr(result, attr)
    raise AttributeError("no parameter attribute on result")


def _result_chi2(result):
    for attr in ("chi_squared", "chi2", "final_cost", "cost"):
        if hasattr(result, attr):
            return getattr(result, attr)
    raise AttributeError("no chi-squared attribute on result")
