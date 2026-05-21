"""Shared fixtures for viz tests."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _agg_backend() -> None:
    matplotlib.use("Agg")


@pytest.fixture
def synthetic_single_angle_data() -> dict[str, Any]:
    n = 64
    dt = 0.1
    t = np.arange(n) * dt
    t1g, t2g = np.meshgrid(t, t, indexing="ij")
    tau = 2.0
    c2 = 1.0 + 0.2 * np.exp(-np.abs(t1g - t2g) / tau)
    rng = np.random.default_rng(seed=42)
    c2 = c2 + rng.normal(scale=0.01, size=c2.shape)
    return {"c2_exp": c2, "t": t, "dt": dt}


@pytest.fixture
def synthetic_multi_angle_data() -> dict[str, Any]:
    n_phi, n = 4, 64
    dt = 0.1
    t = np.arange(n) * dt
    t1g, t2g = np.meshgrid(t, t, indexing="ij")
    phi_angles = np.array([0.0, 45.0, 90.0, 135.0])
    rng = np.random.default_rng(seed=42)
    c2_exp = np.empty((n_phi, n, n))
    for i, phi in enumerate(phi_angles):
        tau = 2.0 + 0.5 * np.sin(np.deg2rad(phi))
        c2_exp[i] = 1.0 + 0.2 * np.exp(-np.abs(t1g - t2g) / tau)
        c2_exp[i] += rng.normal(scale=0.01, size=(n, n))
    return {"c2_exp": c2_exp, "phi_angles_list": phi_angles, "t1": t, "t2": t, "dt": dt}


@pytest.fixture
def minimal_homodyne_config() -> dict[str, Any]:
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
def homodyne_model(minimal_homodyne_config):
    from xpcsjax.core.homodyne_model import HomodyneModel

    return HomodyneModel(minimal_homodyne_config)


@pytest.fixture
def heterodyne_model():
    from xpcsjax.core.heterodyne_model import HeterodyneModel

    return HeterodyneModel()


@pytest.fixture
def converged_homodyne_result():
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    n_params = 5
    return OptimizationResult(
        parameters=np.array([0.2, 1.0, 100.0, -0.5, 0.0]),
        uncertainties=np.array([0.01, 0.005, 5.0, 0.05, 0.01]),
        covariance=np.eye(n_params) * 0.01,
        chi_squared=12.5,
        reduced_chi_squared=0.906,
        convergence_status="converged",
        iterations=42,
        execution_time=1.234,
        device_info={"platform": "cpu"},
    )
