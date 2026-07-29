"""End-to-end result-assembly tests: a tied config's OptimizationResult must
report the full 14-physics block (D0_ref == D0_sample exactly) with mirrored
covariance/uncertainty, on every wired in-memory joint-fit path.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES

_D0_REF_IDX = list(ALL_PARAM_NAMES).index("D0_ref")
_D0_SAMPLE_IDX = list(ALL_PARAM_NAMES).index("D0_sample")

_N_TIMES = 40
_DT = 1.0
_Q = 0.0054
_NOISE_SIGMA = 1e-3


def _tied_config_dict(phi_angles: np.ndarray, per_angle_mode: str) -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _DT,
            "start_frame": 1,
            "end_frame": _N_TIMES,
            "scattering": {"wavevector_q": _Q},
        },
        "scaling": {
            "n_angles": len(phi_angles),
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "initial_parameters": {
            "tied_parameters": {
                "D0_ref": "D0_sample",
                "alpha_ref": "alpha_sample",
                "D_offset_ref": "D_offset_sample",
            },
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 50,
                "enable_cmaes": False,
                "anti_degeneracy": {"per_angle_mode": per_angle_mode},
            },
        },
    }


def _build_synthetic_c2(model, phi_angles):
    rng = np.random.default_rng(seed=20260729)
    c2_stack = np.empty((len(phi_angles), _N_TIMES, _N_TIMES), dtype=np.float64)
    for i, phi in enumerate(phi_angles):
        c2 = np.asarray(model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        c2_stack[i] = c2 + rng.normal(0.0, _NOISE_SIGMA, size=c2.shape)
    return c2_stack


def _run_tied_fit(tmp_path, phi_angles, per_angle_mode):
    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg_path = tmp_path / "tied.yaml"
    cfg_path.write_text(yaml.safe_dump(_tied_config_dict(phi_angles, per_angle_mode)))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)
    data = {"c2": c2, "phi": phi_angles}
    return fit_nlsq(data, cfg)


def _assert_tied_result_shape(result):
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag, "nlsq_diagnostics must record the tied_parameters map"
    params = np.asarray(result.parameters, dtype=np.float64)
    assert params.size >= 14, f"expected physics block full-length 14, got total {params.size}"
    # locate the physics block: it's always 14 contiguous entries at the
    # position recorded by n_physics (or, for constant mode, the whole vector)
    if result.n_physics is not None:
        physics = result.physics_parameters
    else:
        physics = params
    assert physics.size == 14
    assert physics[_D0_REF_IDX] == physics[_D0_SAMPLE_IDX]
    cov = np.asarray(result.covariance, dtype=np.float64)
    unc = np.asarray(result.uncertainties, dtype=np.float64)
    # find the tied pair's position within the full vector for the cov/unc check
    if result.n_physics is not None:
        offset = params.size - 14
    else:
        offset = 0
    ref_pos, sample_pos = offset + _D0_REF_IDX, offset + _D0_SAMPLE_IDX
    if np.isfinite(unc[sample_pos]):
        assert unc[ref_pos] == unc[sample_pos]
        assert cov[ref_pos, ref_pos] == cov[sample_pos, sample_pos]


def test_averaged_mode_tied_fit_reports_full_physics(tmp_path):
    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)  # n_phi=3 -> averaged
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    _assert_tied_result_shape(result)


def test_individual_mode_tied_fit_reports_full_physics(tmp_path):
    phi_angles = np.array([0.0, 90.0], dtype=np.float64)  # n_phi=2 -> individual
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    _assert_tied_result_shape(result)
