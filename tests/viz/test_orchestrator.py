"""End-to-end tests for generate_nlsq_plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import generate_nlsq_plots


def _phi_filename(phi: float, prefix: str = "c2_heatmaps") -> str:
    return f"{prefix}_phi_{phi:.1f}deg.png"


def test_orchestrator_writes_all_files_homodyne(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
    )
    for phi in synthetic_multi_angle_data["phi_angles_list"]:
        assert (tmp_path / _phi_filename(phi, "c2_heatmaps")).exists()
        assert (tmp_path / _phi_filename(phi, "residuals")).exists()
        assert (tmp_path / "simulated_data" / f"simulated_c2_fitted_phi_{phi:.1f}deg.png").exists()
    assert (tmp_path / "simulated_data" / "c2_fitted_data.npz").exists()
    assert (tmp_path / "simulated_data" / "simulation_config_fitted.json").exists()


def test_orchestrator_npz_shapes_consistent(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
    )
    npz = np.load(tmp_path / "simulated_data" / "c2_fitted_data.npz")
    n_phi = synthetic_multi_angle_data["phi_angles_list"].size
    n = synthetic_multi_angle_data["t1"].size
    assert npz["c2_exp"].shape == (n_phi, n, n)
    assert npz["c2_fitted"].shape == (n_phi, n, n)
    assert npz["residuals"].shape == (n_phi, n, n)
    assert float(npz["t1"][0]) == pytest.approx(0.0)


def test_orchestrator_plots_filter(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
        plots=("comparison",),
    )
    for phi in synthetic_multi_angle_data["phi_angles_list"]:
        assert (tmp_path / _phi_filename(phi, "c2_heatmaps")).exists()
        assert not (tmp_path / _phi_filename(phi, "residuals")).exists()


def test_orchestrator_closes_all_figures(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    starting = len(plt.get_fignums())
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
    )
    assert len(plt.get_fignums()) == starting


def test_orchestrator_heterodyne_raises_notimplemented(
    tmp_path,
    heterodyne_model,
    synthetic_multi_angle_data,
):
    """Per Spec Amendment 3: heterodyne deferred. Orchestrator raises early
    via isinstance check, BEFORE attempting per-angle compute. Cleaner than
    catching NotImplementedError from _evaluate_c2_per_angle.
    """
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    n = heterodyne_model.get_default_parameters().shape[0]
    result = OptimizationResult(
        parameters=np.asarray(heterodyne_model.get_default_parameters()),
        uncertainties=np.ones(n) * 0.01,
        covariance=np.eye(n),
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )
    config = {
        "analyzer_parameters": {
            "dt": 0.1,
            "scattering": {"wavevector_q": 0.0054},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "heterodyne",
    }
    with pytest.raises(NotImplementedError, match="heterodyne"):
        generate_nlsq_plots(
            model=heterodyne_model,
            result=result,
            data=synthetic_multi_angle_data,
            config=config,
            output_dir=tmp_path,
        )
