"""Shared pytest fixtures for the parity test suite."""

import numpy as np
import pytest


@pytest.fixture
def tiny_laminar_config_and_data():
    """Small synthetic laminar_flow fit that routes through the live NLSQWrapper
    STANDARD curve_fit path.

    Construction mirrors ``_build_laminar_fit()`` in
    ``tests/optimization/test_l4_callback_observational.py`` exactly — same data
    shapes, same config keys, same seed — so that this fixture is a drop-in for
    any test that needs a reproducible laminar (config, data) pair without
    inventing new data shapes.
    """
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel

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
    return cfg, data
