"""End-to-end tests for generate_nlsq_plots."""

from __future__ import annotations

import hashlib
from pathlib import Path

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


def test_unknown_plot_family_raises(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    with pytest.raises(ValueError, match="Unknown plot families"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
            plots=("comparison", "bogus"),
        )


def test_invalid_compression_raises(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    with pytest.raises(ValueError, match="compression"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
            compression="brotli",  # type: ignore[arg-type]
        )


def test_missing_c2_exp_raises(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    bad_data = {k: v for k, v in synthetic_multi_angle_data.items() if k != "c2_exp"}
    with pytest.raises(ValueError, match="c2_exp"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=bad_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
        )


def test_c2_exp_shape_mismatch_raises(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    bad_data = dict(synthetic_multi_angle_data)
    bad_data["c2_exp"] = bad_data["c2_exp"][:, :10, :10]
    with pytest.raises(ValueError, match="c2_exp.shape"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=bad_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
        )


def test_missing_q_raises(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
):
    bad_config = {"analyzer_parameters": {"dt": 0.1}}
    with pytest.raises(ValueError, match="wavevector_q"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=bad_config,
            output_dir=tmp_path,
        )


def test_unsupported_model_raises(
    tmp_path,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    class FakeModel:
        pass

    with pytest.raises(TypeError, match="Unsupported model type"):
        generate_nlsq_plots(
            model=FakeModel(),  # type: ignore[arg-type]
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
        )


def test_orchestrator_fail_soft_on_bad_angle(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
    monkeypatch,
):
    """One angle's compute failure leaves NaN; others render; NPZ still written."""
    from xpcsjax.viz import nlsq_plots as mod

    call_count = {"n": 0}
    real = mod._evaluate_c2_per_angle

    def flaky(model, result, data, config, phi_deg):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated compute failure")
        return real(model, result, data, config, phi_deg)

    monkeypatch.setattr(mod, "_evaluate_c2_per_angle", flaky)

    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
    )
    loaded = np.load(tmp_path / "simulated_data" / "c2_fitted_data.npz")
    assert np.all(np.isnan(loaded["c2_fitted"][1]))
    for i in [0, 2, 3]:
        assert np.all(np.isfinite(loaded["c2_fitted"][i]))


def _png_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parallel_produces_identical_pngs_to_sequential(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
):
    seq_dir = tmp_path / "seq"
    par_dir = tmp_path / "par"
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=seq_dir,
        parallel=False,
    )
    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=par_dir,
        parallel=True,
    )
    for phi in synthetic_multi_angle_data["phi_angles_list"]:
        s = seq_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
        p = par_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
        assert s.exists() and p.exists()
        assert _png_sha256(s) == _png_sha256(p)


def test_parallel_fallback_on_pool_failure(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
    monkeypatch,
):
    import multiprocessing

    def broken_get_context(method=None):
        raise OSError("simulated spawn failure")

    monkeypatch.setattr(multiprocessing, "get_context", broken_get_context)

    generate_nlsq_plots(
        model=homodyne_model,
        result=converged_homodyne_result,
        data=synthetic_multi_angle_data,
        config=minimal_homodyne_config,
        output_dir=tmp_path,
        parallel=True,
    )
    for phi in synthetic_multi_angle_data["phi_angles_list"]:
        assert (tmp_path / f"c2_heatmaps_phi_{phi:.1f}deg.png").exists()


def test_use_datashader_without_install_falls_back(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
    monkeypatch,
    caplog,
):
    """If datashader is missing, orchestrator warns and uses matplotlib."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xpcsjax.viz.datashader_backend":
            raise ImportError("simulated missing datashader")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level("WARNING", logger="xpcsjax.viz.nlsq_plots"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
            use_datashader=True,
        )
    assert any("viz-fast" in r.message for r in caplog.records)
    for phi in synthetic_multi_angle_data["phi_angles_list"]:
        assert (tmp_path / f"c2_heatmaps_phi_{phi:.1f}deg.png").exists()
