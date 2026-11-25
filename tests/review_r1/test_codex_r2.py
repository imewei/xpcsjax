"""Regression tests for Codex review round 2 (R2-A*, R2-C*, countered disputes)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
    fit_heterodyne_stratified_least_squares as fit_strat,
)

pytestmark = [pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian:UserWarning")]


def test_r2_a1_duplicate_angles_keep_physics(monkeypatch):
    model, c2, phi = make_synthetic_two_component(n_phi=4, n_t=12)
    phi = np.array(phi, copy=True)
    phi[-1] = phi[0]

    def solve(self, *, initial_params, **kwargs):
        p = np.array(initial_params, copy=True)
        p[:6] = [0.2, 0.3, 0.4, 0.9, 1.0, 1.1]
        return NLSQResult(p, self._parameter_names, True, "ok")

    monkeypatch.setattr(NLSQAdapter, "fit", solve)
    cfg = NLSQConfig(per_angle_mode="individual")
    order = np.argsort(phi, kind="stable")
    a = fit_strat(model=model, c2=c2, phi=phi, weights=None, config=cfg)
    b = fit_strat(model=model, c2=c2[order], phi=phi[order], weights=None, config=cfg)
    np.testing.assert_allclose(a.parameters[-14:], b.parameters[-14:])


def test_r2_a2_shared_weights_survive_phi_sort(monkeypatch):
    from tests.review_r1.test_codex_opthet import _patch_exact_fit, _unsorted_two_angle

    model, c2, phi = _unsorted_two_angle()
    seen: dict = {}
    _patch_exact_fit(monkeypatch, seen)
    result = fit_strat(
        model=model,
        c2=c2,
        phi=phi,
        weights=np.ones((1, *c2.shape[1:])),
        config=NLSQConfig(per_angle_mode="individual"),
    )
    np.testing.assert_allclose(result.parameters[2:4], [1.1, 0.9])


def test_r2_a3_individual_covariance_ignores_penalty_rows():
    def solve(mode):
        m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
        cfg = NLSQConfig(
            per_angle_mode="individual",
            regularization_mode=mode,
            group_variance_lambda=0.0,
            loss="linear",
        )
        return fit_nlsq_multi_phi(m, c2, phi, cfg)

    a, b = solve("adaptive"), solve("none")
    np.testing.assert_allclose(a.chi_squared, b.chi_squared, rtol=1e-9)
    ratio = float(np.median(np.diag(a.covariance) / np.diag(b.covariance)))
    assert abs(ratio - 1.0) < 1e-3, ratio


def test_r2_a6_fixed_slot_does_not_consume_dof():
    from tests.optimization.test_fixed_parameters_integration import _config, _synthetic_data
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    sequential = {"optimization": {"stratification": {"force_sequential_fallback": True}}}
    for mode, adapter, extra in [
        ("laminar_flow", False, sequential),
        ("static_isotropic", True, {}),
    ]:
        data = _synthetic_data(mode)
        cfg = ConfigManager(
            config_override=_config(mode, fixed_parameters={"D_offset": 37.5}, extra_top=extra)
        )
        r = fit_nlsq_jax(data, cfg, use_adapter=adapter)
        expected_dof = data["g2"].size - len(r.parameters) + 1
        np.testing.assert_allclose(
            r.reduced_chi_squared, r.chi_squared / expected_dof, rtol=1e-13, atol=0
        )


def test_r2_a7_sigma_masked_rows_are_not_observations():
    from types import SimpleNamespace

    from xpcsjax.optimization.nlsq.strategies.stratified_ls import _n_contributing_rows

    t = np.arange(4.0)
    t1, t2 = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    sigma = np.ones((2, 4, 4))
    sigma[0, 0, 1] = 0.0  # one masked off-diagonal cell
    sigma[1, 2, 2] = 0.0  # a masked DIAGONAL cell must not be subtracted twice
    data = SimpleNamespace(t1_flat=np.tile(t1, 2), t2_flat=np.tile(t2, 2), sigma=sigma)
    assert _n_contributing_rows(data) == 2 * 12 - 1


def test_r2_a8_integer_symmetrization_promotes_before_addition(tmp_path):
    from tests.review_r1.test_codex_data import _bare_loader, _half, _write_aps_u

    path = tmp_path / "integer.h5"
    _write_aps_u(path, [_half(200, dtype=np.uint8)], [0.0])
    for config in ({}, {"data_filtering": {"enabled": True, "quality_threshold": 0.9}}):
        r = _bare_loader(config)._load_from_hdf(str(path))
        assert np.asarray(r["c2_exp"])[0, 0, 0] == 200


def test_r2_a10_npz_has_one_phi_per_surface():
    import io

    from tests.review_r1.test_codex_data import _bare_loader, _qc_data
    from xpcsjax.data.npz_cache import load_from_cache
    from xpcsjax.data.xpcs_loader import XPCSDataFormatError

    data = _qc_data()
    data.update(
        c2_exp=np.repeat(data["c2_exp"], 2, axis=0),
        wavevector_q_list=np.array([0.01, 0.01]),
        phi_angles_list=np.array([[0.0, 45.0], [90.0, 135.0]]),
    )
    buf = io.BytesIO()
    np.savez(buf, **data)
    buf.seek(0)
    with pytest.raises(XPCSDataFormatError):
        load_from_cache(_bare_loader(), buf)


def _preview_call(cfg):
    from pathlib import Path
    from unittest.mock import patch

    from xpcsjax.cli.plot_families import simulated
    from xpcsjax.config import ConfigManager

    n = cfg["analyzer_parameters"]["end_frame"]
    with (
        patch.object(simulated, "_evaluate_model_c2", return_value=np.ones((n, n))) as spy,
        patch("xpcsjax.viz.plot_simulated_data"),
    ):
        simulated._plot_simulated_from_config(
            ConfigManager(config_override=cfg), 0.3, 1.0, "0", Path("."), None
        )
    return spy.call_args


def test_r2_a12_and_config_b5_two_component_preview_uses_fit_inputs():
    cfg = {
        "analysis_mode": "two_component",
        "scattering": {"wavevector_q": 0.037},
        "parameters": {"velocity": {"v0": {"value": 42}}},
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 4,
            "scattering": {"phi_angles": [0.0]},
        },
    }
    call = _preview_call(cfg)
    assert call.kwargs["q"] == 0.037  # nested phi_angles must not hide the top-level q
    assert call.args[1][6] == 42  # grouped value the fit reads, not the bounds midpoint


def test_r2_c1_reordered_active_list_keeps_registry_order():
    cfg = {
        "analysis_mode": "static_isotropic",
        "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 2},
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [100.0, 0.8, 0.25],
            "active_parameters": ["alpha", "D0", "D_offset"],
        },
    }
    np.testing.assert_allclose(_preview_call(cfg).args[1], [100.0, 0.8, 0.25])


def test_r2_a13_newline_path_with_non_bmp_char_round_trips(tmp_path):
    import yaml

    from xpcsjax.cli.config_template import generate_config

    path = "scan \U0001f600\n.h5"
    out = generate_config("static_isotropic", tmp_path / "cfg.yaml", data_path=path)
    assert yaml.safe_load(out.read_text())["experimental_data"]["file_path"] == path


def test_r2_c3_config_save_preserves_symlink(tmp_path):
    from types import SimpleNamespace

    from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog

    target = tmp_path / "shared.yaml"
    target.write_text("before: true\n")
    link = tmp_path / "dataset.yaml"
    link.symlink_to(target)
    editor = SimpleNamespace(
        _path=link, _load_error=None, _editor=SimpleNamespace(toPlainText=lambda: "after: true\n")
    )
    ConfigTextEditorDialog.save(editor)
    assert link.is_symlink()
    assert target.read_text() == "after: true\n"


def test_r2_c4_cache_replay_does_not_accumulate():
    import io
    import json

    from tests.review_r1.test_codex_data import _bare_loader, _qc_data
    from xpcsjax.data.npz_cache import load_from_cache

    loader = _bare_loader()
    buf = io.BytesIO()
    np.savez(buf, **_qc_data(), cache_metadata_json=json.dumps({"load_degradations": ["fallback"]}))
    raw = buf.getvalue()
    for _ in range(2):
        load_from_cache(loader, io.BytesIO(raw))
    assert len(loader.load_degradations) == 1


def test_r2_data_b17_schema_validation_reports_instead_of_raising():
    from xpcsjax.data.config import validate_config_schema

    report = validate_config_schema(
        {
            "experimental_data": {"data_folder_path": ".", "data_file_name": "x.hdf"},
            "analyzer_parameters": {"start_frame": 1, "end_frame": 4, "dt": "bad"},
        }
    )
    assert not report.is_valid


def test_r2_ioviz_b10_singleton_axis_is_a_clear_error():
    pytest.importorskip("datashader")
    from xpcsjax.viz.datashader_backend import DatashaderRenderer

    with pytest.raises(ValueError, match="at least 2 coordinates"):
        DatashaderRenderer().rasterize_heatmap(
            np.full((3, 1), 1.2), np.array([0.0]), np.arange(3.0), vmin=1.0, vmax=1.5
        )


def test_r2_optstrat_b16_standard_executor_accepts_unbounded():
    import logging

    from xpcsjax.optimization.nlsq.strategies.executors import get_executor

    x = np.arange(1.0, 5.0)
    result = get_executor("standard").execute(
        lambda xb, a: a * xb, x, 2 * x, np.array([1.0]), None, "linear", "jac", logging.getLogger()
    )
    np.testing.assert_allclose(np.asarray(result.popt), [2.0], rtol=1e-6)


# ----------------------------------------------------------------- round 3 (R3-*)
@pytest.mark.parametrize("singleton", [False, True])
def test_r3_a2_engine_broadcasts_shared_weight_slab(monkeypatch, singleton):
    from tests.review_r1.test_codex_opthet import _patch_exact_fit, _unsorted_two_angle
    from xpcsjax.optimization.nlsq.heterodyne_engine_route import fit_two_component_via_engine

    model, c2, phi = _unsorted_two_angle()
    _patch_exact_fit(monkeypatch, {})
    cfg = NLSQConfig(per_angle_mode="individual")
    weights = np.ones_like(c2[:1] if singleton else c2[0])
    result = fit_two_component_via_engine(model, c2, phi, cfg, weights=weights)
    assert result.chi_squared < 1e-12


def test_r3_a3_row_count_uses_engine_thresholds():
    from types import SimpleNamespace

    from xpcsjax.optimization.nlsq.strategies.stratified_ls import _n_contributing_rows

    t = np.arange(4.0)
    t1, t2 = (a.ravel() for a in np.meshgrid(t, t, indexing="ij"))
    sigma = np.ones((1, 4, 4))
    sigma[0, 0, 1] = 1e-12  # below the engine's 1e-10 sigma mask
    assert _n_contributing_rows(SimpleNamespace(t1_flat=t1, t2_flat=t2, sigma=sigma)) == 11


def test_r3_a5_phi_row_vector_is_rejected():
    import io

    from tests.review_r1.test_codex_data import _bare_loader, _qc_data
    from xpcsjax.data.npz_cache import load_from_cache
    from xpcsjax.data.xpcs_loader import XPCSDataFormatError

    data = _qc_data()
    data.update(
        c2_exp=np.repeat(data["c2_exp"], 2, axis=0),
        wavevector_q_list=np.array([0.01, 0.01]),
        phi_angles_list=np.array([[0.0, 45.0]]),
    )
    buf = io.BytesIO()
    np.savez(buf, **data)
    buf.seek(0)
    with pytest.raises(XPCSDataFormatError):
        load_from_cache(_bare_loader(), buf)


@pytest.mark.parametrize("path", ["scan\u0085test\n.h5", "scan\x7ftest\n.h5"])
def test_r3_a6_control_characters_in_newline_path_round_trip(tmp_path, path):
    import yaml

    from xpcsjax.cli.config_template import generate_config

    out = generate_config("static_isotropic", tmp_path / "cfg.yaml", data_path=path)
    assert yaml.safe_load(out.read_text())["experimental_data"]["file_path"] == path


def test_r3_a8_zero_data_dof_reports_placeholder_covariance():
    def run(mode):
        m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=4)
        m.param_manager.set_vary("D_offset_ref", False)
        m.param_manager.set_vary("D_offset_sample", False)
        cfg = NLSQConfig(
            per_angle_mode="individual",
            regularization_mode=mode,
            group_variance_lambda=0.0,
            loss="linear",
            max_nfev=100,
        )
        return fit_nlsq_multi_phi(m, c2, phi, cfg)

    a, b = run("adaptive"), run("none")
    assert b.nlsq_diagnostics["covariance_is_placeholder"]
    assert a.nlsq_diagnostics["covariance_is_placeholder"]


def test_r3_a9_schema_validation_reports_overflowing_number():
    from xpcsjax.data.config import validate_config_schema

    report = validate_config_schema(
        {
            "experimental_data": {"data_folder_path": ".", "data_file_name": "x.hdf"},
            "analyzer_parameters": {"start_frame": 1, "end_frame": 4, "dt": 10**400},
        }
    )
    assert not report.is_valid


def test_r3_c1_stateful_plot_model_keeps_its_own_grid_at_fit_q():
    from copy import deepcopy
    from types import SimpleNamespace

    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

    cfg = {
        "analysis_mode": "static_isotropic",
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 4,
            "scattering": {"wavevector_q": 0.0054},
            "geometry": {"stator_rotor_gap": 1.0},
        },
    }
    model = HomodyneModel(cfg)
    plot_cfg = deepcopy(cfg)
    del plot_cfg["analyzer_parameters"]["start_frame"]
    del plot_cfg["analyzer_parameters"]["end_frame"]
    result = SimpleNamespace(parameters=np.array([0.2, 1.0, 100.0, 0.8, 0.25]))
    surface = _evaluate_c2_per_angle(model, result, {"wavevector_q_list": [0.005]}, plot_cfg, 0.0)
    assert surface.shape == (4, 4) and np.isfinite(surface).all()
    measured_cfg = deepcopy(cfg)
    measured_cfg["analyzer_parameters"]["scattering"]["wavevector_q"] = 0.005
    expected = HomodyneModel(measured_cfg).compute_c2_single_angle(
        result.parameters[2:], 0.0, 0.2, 1.0
    )
    np.testing.assert_allclose(surface, np.asarray(expected))


def test_r3_c2_standard_executor_reports_backend_verdict():
    import logging
    from unittest.mock import patch

    from xpcsjax.optimization.nlsq.strategies import executors

    real = executors.curve_fit

    def short_budget(*args, **kwargs):
        kwargs["max_nfev"] = 1
        return real(*args, **kwargs)

    x = np.arange(1.0, 5.0)
    with patch.object(executors, "curve_fit", short_budget):
        r = executors.get_executor("standard").execute(
            lambda xb, a: a * xb,
            x,
            2 * x,
            np.array([1.0]),
            None,
            "linear",
            "jac",
            logging.getLogger(),
        )
    assert r.info["success"] is False
