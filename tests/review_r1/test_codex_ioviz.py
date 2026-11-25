"""Red tests for Codex review round 1, shard ioviz (xpcsjax/io, utils, viz)."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def multi_angle_data() -> dict[str, Any]:
    n_phi, n = 4, 64
    t = np.arange(n) * 0.1
    t1g, t2g = np.meshgrid(t, t, indexing="ij")
    phi = np.array([0.0, 45.0, 90.0, 135.0])
    c2 = np.stack([1.0 + 0.2 * np.exp(-np.abs(t1g - t2g) / 2.0)] * n_phi)
    return {"c2_exp": c2, "phi_angles_list": phi, "t1": t, "t2": t, "dt": 0.1}


@pytest.fixture
def homodyne_config() -> dict[str, Any]:
    return {
        "analyzer_parameters": {
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 64},
            "scattering": {"wavevector_q": 0.0054},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "static_isotropic",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [100.0, -0.5, 0.0],
        },
    }


@pytest.fixture
def homodyne_model(homodyne_config):
    from xpcsjax.core.homodyne_model import HomodyneModel

    return HomodyneModel(homodyne_config)


@pytest.fixture
def homodyne_result():
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    return OptimizationResult(
        parameters=np.array([0.2, 1.0, 100.0, -0.5, 0.0]),
        uncertainties=np.array([0.01, 0.005, 5.0, 0.05, 0.01]),
        covariance=np.eye(5) * 0.01,
        chi_squared=12.5,
        reduced_chi_squared=0.906,
        convergence_status="converged",
        iterations=42,
        execution_time=1.234,
        device_info={"platform": "cpu"},
    )


def test_ioviz_b1(homodyne_model, homodyne_result):
    from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

    phys = np.array([100.0, -0.5, 0.0])
    # Scaling block is in np.unique(phi) order (0 deg, 90 deg); data rows are [90, 0].
    result = replace(
        homodyne_result,
        parameters=np.r_[0.2, 0.4, 1.0, 1.3, phys],
        uncertainties=np.ones(7),
        covariance=np.eye(7),
    )
    data = {"phi_angles_list": np.array([90.0, 0.0])}
    c2 = _evaluate_c2_per_angle(homodyne_model, result, data, {}, 90.0, phi_index=0)
    expected = homodyne_model.compute_c2_single_angle(phys, 90.0, 0.4, 1.3)
    np.testing.assert_allclose(c2, np.asarray(expected))


def test_ioviz_b2(homodyne_config, homodyne_result):
    import jax.numpy as jnp

    from xpcsjax.core.models import make_model
    from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle

    model = make_model(homodyne_config)  # CombinedModel, as ConfigManager.get_model returns
    t = np.arange(64) * 0.1
    # Loader picked the closest bin (0.0050); config requested 0.0054. The fit uses the bin.
    data = {"t1": t, "t2": t, "wavevector_q_list": np.array([0.0050])}
    c2 = _evaluate_c2_per_angle(model, homodyne_result, data, homodyne_config, 0.0)
    tj = jnp.asarray(t)
    expected = model.compute_g2(
        jnp.asarray([100.0, -0.5, 0.0]),
        tj,
        tj,
        jnp.asarray([0.0]),
        0.0050,
        2_000_000.0,
        0.2,
        1.0,
        0.1,
    )[0]
    np.testing.assert_allclose(c2, np.asarray(expected))


def test_ioviz_b3(tmp_path):
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.service.persist import save_results_npz

    result = OptimizationResult(
        parameters=np.ones(33),
        uncertainties=np.ones(33),
        covariance=np.eye(33),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
        nlsq_diagnostics={"diagnostic_covariance": np.eye(33)},
    )
    assert save_results_npz(result, tmp_path).is_file()


def test_ioviz_b4(tmp_path, homodyne_model, homodyne_result, multi_angle_data, homodyne_config):
    from xpcsjax.viz import generate_nlsq_plots

    result = replace(homodyne_result, uncertainties=np.array(np.nan))
    generate_nlsq_plots(
        homodyne_model,
        result,
        multi_angle_data,
        homodyne_config,
        tmp_path,
        plots=(),
        parallel=False,
    )
    path = tmp_path / "simulated_data" / "simulation_config_fitted.json"
    assert path.is_file()
    json.loads(path.read_text())


def _restore_xpcsjax_logger():
    lg = logging.getLogger("xpcsjax")
    return lg, list(lg.handlers), lg.propagate, lg.level


def test_ioviz_b5(tmp_path):
    from xpcsjax.utils.logging import configure_logging

    lg, handlers, propagate, level = _restore_xpcsjax_logger()
    (tmp_path / "analysis.log").mkdir()
    config = {
        "console": {"enabled": True},
        "file": {"enabled": True, "path": str(tmp_path), "filename": "analysis.log"},
    }
    try:
        assert configure_logging(config) is None
    finally:
        lg.handlers[:] = handlers
        lg.propagate = propagate
        lg.setLevel(level)


def test_ioviz_b6():
    from xpcsjax.utils.logging import ContextFilter, log_context

    with log_context(run_id="job"), log_context(phase="fit"):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
        ContextFilter().filter(record)
        assert (record.run_id, record.phase) == ("job", "fit")
