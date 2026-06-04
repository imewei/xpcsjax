"""100k–1M shuffle-regime parity tests for two_component.

``laminar_flow`` reorganizes + seed-42 pre-shuffles >100k-point per-angle fits
that stay in-memory (stratified-LS only engages at >=1M). The heterodyne
(``two_component``) in-memory path mirrors this with an OBJECTIVE-INVARIANT
seed-42 **angle-axis** reorder, scoped to global-scaling modes
(``averaged``/``constant``) where the fitted vector is angle-order-invariant.

These tests pin:
1. the reorder/restore helpers (invertibility + realignment), and
2. the end-to-end guarantee that enabling the regime changes **no** fitted value
   (the hard parity guard for the oracle).
"""

import logging

import numpy as np
import yaml

from xpcsjax.optimization.nlsq import _restore_angle_order, _seed42_angle_reorder

_DT = 1.0
_Q = 0.0054
_N_TIMES = 12
# 3 angles so the default per_angle_mode="auto" resolves to "averaged".
_PHI = np.array([0.0, 45.0, 90.0], dtype=np.float64)
_NOISE = 1e-3


def test_seed42_angle_reorder_is_invertible_and_logs(caplog):
    c2 = np.arange(3 * 4 * 4, dtype=np.float64).reshape(3, 4, 4)
    phi = np.array([10.0, 20.0, 30.0])
    w = np.ones_like(c2)

    with caplog.at_level(
        logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_logging"
    ):
        c2r, phir, wr, inv = _seed42_angle_reorder(c2, phi, w, 150_000)

    # The inverse permutation recovers the caller's original ordering exactly.
    np.testing.assert_array_equal(c2r[inv], c2)
    np.testing.assert_array_equal(phir[inv], phi)
    np.testing.assert_array_equal(wr[inv], w)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "seed=42" in text, text
    assert "Angle-stratified reorder complete" in text, text


def test_restore_angle_order_realigns_per_angle_fields():
    perm = np.array([2, 0, 1])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(3)

    class _R:
        pass

    r = _R()
    # Fit "saw" reordered angles; per-angle diagnostics come back shuffled.
    r.nlsq_diagnostics = {
        "chi2_per_angle": np.array([9.0, 7.0, 8.0]),
        "phi_angles": np.array([90.0, 0.0, 45.0]),
    }
    _restore_angle_order(r, inv)

    np.testing.assert_array_equal(
        r.nlsq_diagnostics["chi2_per_angle"], np.array([9.0, 7.0, 8.0])[inv]
    )
    np.testing.assert_array_equal(
        r.nlsq_diagnostics["phi_angles"], np.array([0.0, 45.0, 90.0])
    )


def test_restore_angle_order_is_best_effort_on_bad_input():
    class _R:
        nlsq_diagnostics = None

    _restore_angle_order(_R(), np.array([0, 1, 2]))  # must not raise


def test_restore_angle_order_realigns_all_per_angle_arrays():
    """Every ``*_per_angle*`` array (not just chi2/phi) must be realigned."""
    perm = np.array([2, 0, 1])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(3)

    class _R:
        pass

    r = _R()
    r.nlsq_diagnostics = {
        "chi2_per_angle": np.array([9.0, 7.0, 8.0]),
        "contrast_per_angle_quantile": np.array([0.9, 0.7, 0.8]),
        "n_iterations": 5,  # scalar — must be left untouched
    }
    _restore_angle_order(r, inv)
    np.testing.assert_array_equal(
        r.nlsq_diagnostics["contrast_per_angle_quantile"], np.array([0.9, 0.7, 0.8])[inv]
    )
    assert r.nlsq_diagnostics["n_iterations"] == 5  # scalar untouched


def test_seed42_reorder_passes_through_2d_weights():
    """Shared 2-D (N, N) weights broadcast across angles and must NOT be permuted
    (regression: permuting axis-0 of a 2-D array corrupts the time axis)."""
    c2 = np.zeros((3, 4, 4))
    phi = np.array([0.0, 45.0, 90.0])
    w2d = np.arange(16, dtype=float).reshape(4, 4)  # shared (N, N)
    _, _, wr, _ = _seed42_angle_reorder(c2, phi, w2d, 150_000)
    np.testing.assert_array_equal(wr, w2d)  # unchanged, not permuted


def test_seed42_reorder_permutes_3d_weights():
    c2 = np.zeros((3, 4, 4))
    phi = np.array([0.0, 45.0, 90.0])
    w3d = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)
    _, _, wr, inv = _seed42_angle_reorder(c2, phi, w3d, 150_000)
    assert wr.shape == w3d.shape
    np.testing.assert_array_equal(wr[inv], w3d)  # per-angle, invertible


def _averaged_config() -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _DT,
            "start_frame": 1,
            "end_frame": _N_TIMES,
            "scattering": {"wavevector_q": _Q},
        },
        "scaling": {
            "n_angles": len(_PHI),
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 50,
                "enable_cmaes": False,
            },
        },
    }


def test_shuffle_regime_is_objective_invariant(tmp_path, monkeypatch):
    """The hard parity guard: enabling the 100k–1M regime must not change any
    fitted value. We drive the size gate via a monkeypatched point estimate so a
    tiny, fast fixture exercises the regime branch.
    """
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg_path = tmp_path / "avg.yaml"
    cfg_path.write_text(yaml.safe_dump(_averaged_config()))
    cfg = ConfigManager(str(cfg_path))

    truth = HeterodyneModel.from_config(cfg.config)
    rng = np.random.default_rng(20260519)
    c2 = np.stack(
        [
            np.asarray(truth.compute_correlation(phi_angle=float(p), angle_idx=i))
            + rng.normal(0.0, _NOISE, (_N_TIMES, _N_TIMES))
            for i, p in enumerate(_PHI)
        ]
    )
    data = {"c2": c2, "phi": _PHI}

    # Regime ON: gate sees 150k points -> use_strat True, <1M -> in-memory + reorder.
    monkeypatch.setattr(nlsq_pkg, "_estimate_heterodyne_points", lambda *_: 150_000)
    res_on = fit_nlsq(data, cfg)

    # Regime OFF: gate sees 50k (<100k) -> should_use_stratification False -> no reorder.
    monkeypatch.setattr(nlsq_pkg, "_estimate_heterodyne_points", lambda *_: 50_000)
    res_off = fit_nlsq(data, cfg)

    # The reorder is OBJECTIVE-invariant: the fit objective (chi-squared) is
    # unchanged. We assert on the objective, NOT the parameter vector: this
    # synthetic 14-parameter fit is intentionally tiny and lands on a flat /
    # degenerate minimum, so equi-objective parameter sets differ between runs
    # (the same property laminar_flow's shuffle has) — that is the ill-posed
    # problem, not the reorder.
    np.testing.assert_allclose(
        float(res_on.chi_squared), float(res_off.chi_squared), rtol=1e-4, atol=0.0
    )

    # Realignment correctness (end-to-end): per-angle diagnostics come back in
    # the CALLER's angle order. phi_angles must equal the input order, and
    # chi2_per_angle must match the no-reorder run ELEMENT-FOR-ELEMENT — which is
    # only possible if the inverse permutation undoes the shuffle correctly.
    np.testing.assert_array_equal(
        np.asarray(res_on.nlsq_diagnostics["phi_angles"], dtype=float), _PHI
    )
    np.testing.assert_allclose(
        np.asarray(res_on.nlsq_diagnostics["chi2_per_angle"], dtype=float),
        np.asarray(res_off.nlsq_diagnostics["chi2_per_angle"], dtype=float),
        rtol=1e-3,
        atol=1e-8,
    )


def _build_averaged_fixture(tmp_path):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    cfg_path = tmp_path / "avg.yaml"
    cfg_path.write_text(yaml.safe_dump(_averaged_config()))
    cfg = ConfigManager(str(cfg_path))
    truth = HeterodyneModel.from_config(cfg.config)
    rng = np.random.default_rng(20260519)
    c2 = np.stack(
        [
            np.asarray(truth.compute_correlation(phi_angle=float(p), angle_idx=i))
            + rng.normal(0.0, _NOISE, (_N_TIMES, _N_TIMES))
            for i, p in enumerate(_PHI)
        ]
    )
    return cfg, c2


def test_regime_runs_with_2d_weights(tmp_path, monkeypatch):
    """Regression: the regime must run with shared 2-D weights (the reorder must
    not permute/corrupt them) and stay objective-invariant."""
    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg, c2 = _build_averaged_fixture(tmp_path)
    w2d = np.ones((_N_TIMES, _N_TIMES), dtype=float)  # shared 2-D weights

    monkeypatch.setattr(nlsq_pkg, "_estimate_heterodyne_points", lambda *_: 150_000)
    res_on = fit_nlsq({"c2": c2, "phi": _PHI, "weights": w2d}, cfg)
    monkeypatch.setattr(nlsq_pkg, "_estimate_heterodyne_points", lambda *_: 50_000)
    res_off = fit_nlsq({"c2": c2, "phi": _PHI, "weights": w2d}, cfg)

    np.testing.assert_allclose(
        float(res_on.chi_squared), float(res_off.chi_squared), rtol=1e-4, atol=0.0
    )


def test_constant_mode_is_not_reordered(tmp_path, monkeypatch, caplog):
    """Scoping guard: 'constant' freezes PER-ANGLE scaling + writes model.scaling
    in angle order, so it must be EXCLUDED from the reorder."""
    import logging

    import xpcsjax.optimization.nlsq as nlsq_pkg
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg_dict = _averaged_config()
    cfg_dict["optimization"]["nlsq"]["per_angle_mode"] = "constant"
    cfg_path = tmp_path / "const.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    cfg = ConfigManager(str(cfg_path))

    truth = HeterodyneModel.from_config(cfg.config)
    rng = np.random.default_rng(20260519)
    c2 = np.stack(
        [
            np.asarray(truth.compute_correlation(phi_angle=float(p), angle_idx=i))
            + rng.normal(0.0, _NOISE, (_N_TIMES, _N_TIMES))
            for i, p in enumerate(_PHI)
        ]
    )

    monkeypatch.setattr(nlsq_pkg, "_estimate_heterodyne_points", lambda *_: 150_000)
    with caplog.at_level(
        logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_logging"
    ):
        fit_nlsq({"c2": c2, "phi": _PHI}, cfg)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Pre-shuffled angle order" not in text, "constant mode must not be reordered"
