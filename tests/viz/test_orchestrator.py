"""End-to-end tests for generate_nlsq_plots."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import generate_nlsq_plots


def _phi_filename(phi_idx: int, phi: float, prefix: str = "c2_heatmaps") -> str:
    # Filename format includes the angle index so .1f-equal angles can't collide.
    return f"{prefix}_phi_{phi_idx:03d}_{phi:.3f}deg.png"


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
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        assert (tmp_path / _phi_filename(i, phi, "c2_heatmaps")).exists()
        assert (tmp_path / _phi_filename(i, phi, "residuals")).exists()
        assert (
            tmp_path / "simulated_data" / _phi_filename(i, phi, "simulated_c2_fitted")
        ).exists()
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
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        assert (tmp_path / _phi_filename(i, phi, "c2_heatmaps")).exists()
        assert not (tmp_path / _phi_filename(i, phi, "residuals")).exists()


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


def test_orchestrator_heterodyne_writes_all_files(
    tmp_path,
    heterodyne_model,
    converged_heterodyne_result,
    synthetic_multi_angle_data,
):
    """Heterodyne now produces plots + NPZ + JSON via per-angle scaling reconstruction."""
    config = {
        "analyzer_parameters": {
            "dt": 0.1,
            "scattering": {"wavevector_q": 0.0054},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "heterodyne",
    }
    generate_nlsq_plots(
        model=heterodyne_model,
        result=converged_heterodyne_result,
        data=synthetic_multi_angle_data,
        config=config,
        output_dir=tmp_path,
    )
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        assert (tmp_path / _phi_filename(i, phi, "c2_heatmaps")).exists()
        assert (tmp_path / _phi_filename(i, phi, "residuals")).exists()
    assert (tmp_path / "simulated_data" / "c2_fitted_data.npz").exists()
    assert (tmp_path / "simulated_data" / "simulation_config_fitted.json").exists()


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
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        s = seq_dir / _phi_filename(i, phi, "c2_heatmaps")
        p = par_dir / _phi_filename(i, phi, "c2_heatmaps")
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
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        assert (tmp_path / _phi_filename(i, phi, "c2_heatmaps")).exists()


def test_use_datashader_without_install_falls_back(
    tmp_path,
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
    monkeypatch,
    caplog,
):
    """If datashader is missing, orchestrator warns and uses matplotlib.

    The orchestrator checks the module-level ``DATASHADER_AVAILABLE`` flag
    (set once at import time). To simulate "datashader not installed" we
    flip the flag directly — patching ``builtins.__import__`` would fire
    too late because the probe import already ran during module load.
    """
    from xpcsjax.viz import nlsq_plots as mod

    monkeypatch.setattr(mod, "DATASHADER_AVAILABLE", False)

    with caplog.at_level("WARNING", logger="xpcsjax.viz.nlsq_plots"):
        generate_nlsq_plots(
            model=homodyne_model,
            result=converged_homodyne_result,
            data=synthetic_multi_angle_data,
            config=minimal_homodyne_config,
            output_dir=tmp_path,
            use_datashader=True,
            parallel=False,  # sequential matplotlib fallback
        )
    assert any("viz-fast" in r.message for r in caplog.records)
    for i, phi in enumerate(synthetic_multi_angle_data["phi_angles_list"]):
        assert (tmp_path / _phi_filename(i, phi, "c2_heatmaps")).exists()


def test_homodyne_model_no_legacy_plot_methods():
    """plot_simulated_data and _generate_heatmap_plots were removed in favor of xpcsjax.viz."""
    from xpcsjax.core.homodyne_model import HomodyneModel

    assert not hasattr(HomodyneModel, "plot_simulated_data"), (
        "HomodyneModel.plot_simulated_data was removed — use xpcsjax.viz.plot_simulated_data instead"
    )
    assert not hasattr(HomodyneModel, "_generate_heatmap_plots"), (
        "HomodyneModel._generate_heatmap_plots was removed — use xpcsjax.viz.generate_nlsq_plots instead"
    )
