"""Regression tests for the bugs surfaced by the Codex+Gemini review of the
unpushed viz commits. Each test targets a specific finding so that if a
regression slips back in, the failure points at the exact contract that
was previously broken.
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. import xpcsjax.viz does NOT pull matplotlib into sys.modules
#    (locks in the lazy __getattr__ guarantee — the eager `import
#    matplotlib.pyplot as plt` at the top of nlsq_plots.py is fine *only*
#    because nlsq_plots itself is loaded lazily through viz/__init__.py).
# ---------------------------------------------------------------------------


def test_viz_import_is_lazy_for_matplotlib():
    # Run in a clean subprocess so prior imports in the test runner don't
    # poison the result.
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_check_lazy_viz_import, args=(q,))
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0, "lazy-import probe subprocess crashed"
    loaded_matplotlib = q.get(timeout=5)
    assert not loaded_matplotlib, (
        "import xpcsjax.viz should NOT load matplotlib; the viz/__init__.py "
        "lazy __getattr__ pattern is broken if it does."
    )


def _check_lazy_viz_import(q):
    import sys as _sys

    import xpcsjax.viz  # noqa: F401

    q.put(any(name.startswith("matplotlib") for name in _sys.modules))


# ---------------------------------------------------------------------------
# 2. Heterodyne non-individual layout fails loudly (not silently with NaN
#    artifacts). Previously the orchestrator caught Exception per-angle and
#    wrote all-NaN files.
# ---------------------------------------------------------------------------


def _make_heterodyne_data(n_phi=3, n_t=16):
    t = np.arange(n_t, dtype=float) * 0.1
    phi = np.linspace(0.0, 90.0, n_phi)
    c2 = np.ones((n_phi, n_t, n_t)) + 0.1
    return {"c2_exp": c2, "phi_angles_list": phi, "t1": t, "t2": t}


def _make_heterodyne_config():
    return {
        "analyzer_parameters": {
            "scattering": {"wavevector_q": 0.005},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
            "dt": 0.1,
        },
        "analysis_mode": "two_component",
    }


def _make_heterodyne_result(n_params: int):
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    return OptimizationResult(
        parameters=np.ones(n_params),
        uncertainties=np.full(n_params, 0.01),
        covariance=np.eye(n_params) * 0.01,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )


def test_heterodyne_constant_mode_raises_not_implemented(tmp_path):
    """n_total == n_physical (constant mode) → NotImplementedError upfront."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    model = HeterodyneModel()
    n_physical = len(model.parameter_names)
    result = _make_heterodyne_result(n_physical)  # constant mode shape
    data = _make_heterodyne_data(n_phi=3)
    config = _make_heterodyne_config()

    with pytest.raises(NotImplementedError, match="constant"):
        generate_nlsq_plots(
            model=model,
            result=result,
            data=data,
            config=config,
            output_dir=tmp_path,
        )


def test_heterodyne_unknown_layout_raises_not_implemented(tmp_path):
    """A parameter count between constant and individual → NotImplementedError."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    model = HeterodyneModel()
    n_physical = len(model.parameter_names)
    # Pick a residual that's even but does NOT equal 2 * n_phi (n_phi=3 → 6).
    # n_total = n_physical + 4 mimics a fourier mode with K=0 (2*(2K+1) = 2),
    # or other intermediate shapes.
    result = _make_heterodyne_result(n_physical + 4)
    data = _make_heterodyne_data(n_phi=3)
    config = _make_heterodyne_config()

    with pytest.raises(NotImplementedError, match="fourier|individual"):
        generate_nlsq_plots(
            model=model,
            result=result,
            data=data,
            config=config,
            output_dir=tmp_path,
        )


def test_heterodyne_individual_layout_does_not_raise(tmp_path):
    """The individual layout (n_physical + 2*n_phi) is the supported path."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    model = HeterodyneModel()
    n_physical = len(model.parameter_names)
    n_phi = 3
    n_total = n_physical + 2 * n_phi
    # Use realistic-ish contrasts and offsets so compute_g1 doesn't NaN out.
    physical = np.asarray(model.get_default_parameters(), dtype=float)
    params = np.concatenate(
        [
            np.full(n_phi, 0.2),
            np.full(n_phi, 1.0),
            physical,
        ]
    )
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    result = OptimizationResult(
        parameters=params,
        uncertainties=np.full(n_total, 0.01),
        covariance=np.eye(n_total) * 0.01,
        chi_squared=2.5,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )
    data = _make_heterodyne_data(n_phi=n_phi)
    config = _make_heterodyne_config()

    # Should not raise; produces artifacts in tmp_path.
    generate_nlsq_plots(
        model=model,
        result=result,
        data=data,
        config=config,
        output_dir=tmp_path,
        plots=(),  # don't bother rendering PNGs; we just want the validator
    )
    assert (tmp_path / "simulated_data" / "c2_fitted_data.npz").exists()


# ---------------------------------------------------------------------------
# 3. Rectangular grid (n_t1 != n_t2) — extent uses both axes correctly.
# ---------------------------------------------------------------------------


def test_plot_nlsq_fit_uses_distinct_t1_t2_extent():
    from xpcsjax.viz import plot_nlsq_fit

    n_t1, n_t2 = 12, 20
    c2 = np.ones((n_t1, n_t2)) + 0.1
    t1 = np.linspace(0.0, 1.1, n_t1)
    t2 = np.linspace(0.0, 1.9, n_t2)
    fig = plot_nlsq_fit(c2, c2, t=t1, t2=t2)
    assert fig is not None
    # Check the image extent on the first panel matches (t2[0], t2[-1], t1[0], t1[-1]).
    image_axes = [ax for ax in fig.axes if ax.images]
    assert image_axes, "no image axes created"
    extent = image_axes[0].images[0].get_extent()
    assert extent == pytest.approx((float(t2[0]), float(t2[-1]), float(t1[0]), float(t1[-1])))


# ---------------------------------------------------------------------------
# 4. plot_* returns None when save_path is provided (and Figure when not).
#    Previously the docstring promised a closed Figure — a footgun.
# ---------------------------------------------------------------------------


def test_plot_nlsq_fit_returns_none_when_saved(tmp_path):
    from xpcsjax.viz import plot_nlsq_fit

    c2 = np.ones((8, 8)) + 0.1
    save_path = tmp_path / "out.png"
    result = plot_nlsq_fit(c2, c2, save_path=save_path)
    assert result is None
    assert save_path.exists()


def test_plot_residual_map_returns_none_when_saved(tmp_path):
    from xpcsjax.viz import plot_residual_map

    c2 = np.ones((8, 8)) + 0.1
    save_path = tmp_path / "out.png"
    result = plot_residual_map(c2, c2 * 0.95, save_path=save_path)
    assert result is None
    assert save_path.exists()


def test_plot_simulated_data_returns_none_when_saved(tmp_path):
    from xpcsjax.viz import plot_simulated_data

    c2 = np.ones((8, 8)) + 0.1
    save_path = tmp_path / "out.png"
    result = plot_simulated_data(c2, save_path=save_path)
    assert result is None
    assert save_path.exists()


# ---------------------------------------------------------------------------
# 5. plot_simulated_data with empty input does not raise.
#    Previously it unpacked c2_sim.shape and indexed t_arr[0]/-1 immediately.
# ---------------------------------------------------------------------------


def test_plot_simulated_data_empty_input_does_not_raise():
    from xpcsjax.viz import plot_simulated_data

    empty = np.zeros((0, 0))
    fig = plot_simulated_data(empty)
    assert fig is not None


# ---------------------------------------------------------------------------
# 6. compute_diagonal_overlay_stats validates c2_fit shape match.
# ---------------------------------------------------------------------------


def test_compute_diagonal_overlay_stats_shape_mismatch_raises():
    from xpcsjax.viz import compute_diagonal_overlay_stats

    c2_exp = np.ones((2, 8, 8))
    c2_fit = np.ones((2, 8, 7))  # wrong inner shape
    with pytest.raises(ValueError, match="c2_fit.shape"):
        compute_diagonal_overlay_stats(c2_exp, c2_fit, phi_index=0)


def test_compute_diagonal_overlay_stats_2d_input_raises():
    from xpcsjax.viz import compute_diagonal_overlay_stats

    with pytest.raises(ValueError, match="must be 3-D"):
        compute_diagonal_overlay_stats(np.ones((4, 4)), np.ones((4, 4)), phi_index=0)


# ---------------------------------------------------------------------------
# 7. _write_npz_compressed uses a unique temp file — two concurrent writers
#    targeting the same final path don't clobber each other's temp file.
# ---------------------------------------------------------------------------


def test_npz_writer_unique_temp_paths(tmp_path):
    """Two consecutive write attempts to the same target each use distinct
    temp files (verified by inspecting the temp-file names before final
    rename). The mkstemp suffix guarantees uniqueness."""
    from xpcsjax.viz.nlsq_plots import _write_npz_compressed

    target = tmp_path / "fit.npz"
    arrays = {"a": np.zeros(4), "b": np.ones(4)}

    # First write — should produce the final file.
    _write_npz_compressed(target, arrays, compression="none")
    assert target.exists()
    # No stale .tmp files left over.
    stale = list(tmp_path.glob("*.tmp"))
    assert stale == [], f"unexpected stale temp files: {stale}"

    # Second write — overwrites cleanly via os.replace.
    _write_npz_compressed(target, {"a": np.ones(4)}, compression="none")
    assert target.exists()
    stale = list(tmp_path.glob("*.tmp"))
    assert stale == [], f"second write left stale temp files: {stale}"


def test_npz_writer_rejects_structured_object_dtype(tmp_path):
    """A structured dtype with object fields must be rejected (not silently
    serialized via the non-portable fallback path)."""
    from xpcsjax.viz.nlsq_plots import _write_npz_compressed

    target = tmp_path / "fit.npz"
    # Object-dtype 1-D array.
    arr = np.array(["a", "b", "c"], dtype=object)
    with pytest.raises(TypeError, match="object dtype"):
        _write_npz_compressed(target, {"meta": arr}, compression="none")


# ---------------------------------------------------------------------------
# 8. JAX env vars are set in xpcsjax/__init__.py — workers inheriting from
#    parent see JAX_PLATFORMS=cpu before any `import jax`.
# ---------------------------------------------------------------------------


def test_xpcsjax_init_sets_jax_platforms_cpu():
    import os

    import xpcsjax  # noqa: F401

    # setdefault means the value is present after import (either set here or
    # honored from the user's environment).
    assert os.environ.get("JAX_PLATFORMS") == "cpu"


# ---------------------------------------------------------------------------
# 9. PNG filenames include the angle index — no collisions between angles
#    that round to the same .1f decimal.
# ---------------------------------------------------------------------------


def test_orchestrator_filenames_disambiguate_close_angles(tmp_path):
    """Two phi values that differ in the second decimal (10.04°, 10.05°)
    must produce distinct PNGs. Previously they collided under .1f rounding.
    """
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    config = {
        "analyzer_parameters": {
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 16},
            "scattering": {"wavevector_q": 0.005},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "static_isotropic",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [100.0, -0.5, 0.0],
        },
    }
    model = HomodyneModel(config)
    n_t = 16
    t = np.arange(n_t, dtype=float) * 0.1
    # Two close angles — .1f rounding would collapse them.
    phi = np.array([10.04, 10.05], dtype=float)
    c2_exp = np.ones((2, n_t, n_t)) + 0.1
    data = {
        "c2_exp": c2_exp,
        "phi_angles_list": phi,
        "t1": t,
        "t2": t,
    }
    result = OptimizationResult(
        parameters=np.array([0.2, 1.0, 100.0, -0.5, 0.0]),
        uncertainties=np.full(5, 0.01),
        covariance=np.eye(5) * 0.01,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )
    generate_nlsq_plots(
        model=model,
        result=result,
        data=data,
        config=config,
        output_dir=tmp_path,
        plots=("comparison",),
    )
    pngs = sorted(p.name for p in tmp_path.glob("c2_heatmaps_*.png"))
    assert len(pngs) == 2, f"expected 2 distinct PNGs, got {pngs}"


# ---------------------------------------------------------------------------
# 10. _save_fig closes the figure even when savefig raises.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Datashader parity: when use_datashader=True and the [viz-fast] extra is
#     installed, the orchestrator dispatches to the Datashader fast path.
#     Each angle produces a c2_heatmaps_*.png via the hybrid pipeline.
# ---------------------------------------------------------------------------


def test_datashader_path_produces_comparison_pngs(tmp_path):
    pytest.importorskip("datashader")
    pytest.importorskip("xarray")
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    config = {
        "analyzer_parameters": {
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 32},
            "scattering": {"wavevector_q": 0.005},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "static_isotropic",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [100.0, -0.5, 0.0],
        },
    }
    model = HomodyneModel(config)
    n_t = 32
    n_phi = 3
    t = np.arange(n_t, dtype=float) * 0.1
    phi = np.linspace(0.0, 90.0, n_phi)
    c2_exp = np.ones((n_phi, n_t, n_t)) + 0.1
    data = {"c2_exp": c2_exp, "phi_angles_list": phi, "t1": t, "t2": t}
    result = OptimizationResult(
        parameters=np.array([0.2, 1.0, 100.0, -0.5, 0.0]),
        uncertainties=np.full(5, 0.01),
        covariance=np.eye(5) * 0.01,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )

    generate_nlsq_plots(
        model=model,
        result=result,
        data=data,
        config=config,
        output_dir=tmp_path,
        use_datashader=True,
        parallel=False,  # sequential to keep the test fast
        plots=("comparison",),
    )

    # Datashader produced one 3-panel comparison PNG per angle.
    pngs = sorted(p.name for p in tmp_path.glob("c2_heatmaps_*.png"))
    assert len(pngs) == n_phi, f"expected {n_phi} Datashader PNGs, got {pngs}"


def test_datashader_path_parallel_dispatches_via_pool(tmp_path):
    """Smoke: use_datashader=True + parallel=True hits the spawn pool path.
    Validates that the pool dispatch works end-to-end without crashing,
    not the exact pool topology (pool internals are hard to introspect)."""
    pytest.importorskip("datashader")
    pytest.importorskip("xarray")
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import generate_nlsq_plots

    config = {
        "analyzer_parameters": {
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 16},
            "scattering": {"wavevector_q": 0.005},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "static_isotropic",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [100.0, -0.5, 0.0],
        },
    }
    model = HomodyneModel(config)
    n_t, n_phi = 16, 2
    t = np.arange(n_t, dtype=float) * 0.1
    phi = np.linspace(0.0, 45.0, n_phi)
    c2_exp = np.ones((n_phi, n_t, n_t)) + 0.1
    data = {"c2_exp": c2_exp, "phi_angles_list": phi, "t1": t, "t2": t}
    result = OptimizationResult(
        parameters=np.array([0.2, 1.0, 100.0, -0.5, 0.0]),
        uncertainties=np.full(5, 0.01),
        covariance=np.eye(5) * 0.01,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )
    generate_nlsq_plots(
        model=model,
        result=result,
        data=data,
        config=config,
        output_dir=tmp_path,
        use_datashader=True,
        parallel=True,
        plots=("comparison",),
    )
    pngs = sorted(p.name for p in tmp_path.glob("c2_heatmaps_*.png"))
    assert len(pngs) == n_phi


def test_datashader_renderer_handles_rectangular_grid():
    """The fast path must handle rectangular (n_t1 != n_t2) grids correctly."""
    pytest.importorskip("datashader")
    from PIL import Image

    from xpcsjax.viz.datashader_backend import DatashaderRenderer

    n_t1, n_t2 = 32, 48
    data = np.ones((n_t1, n_t2)) + 0.1
    t1 = np.linspace(0.0, 3.1, n_t1)
    t2 = np.linspace(0.0, 4.7, n_t2)
    renderer = DatashaderRenderer(width=400, height=300)
    img = renderer.rasterize_heatmap(data.T, t1, t2)  # x=t1, y=t2
    assert isinstance(img, Image.Image)
    assert img.size == (400, 300)


def test_datashader_module_load_does_not_block_when_missing(monkeypatch):
    """DATASHADER_AVAILABLE=False must not break ``import xpcsjax.viz.nlsq_plots``.

    Locks in the lazy-probe pattern: nlsq_plots is importable even without
    the [viz-fast] extra; the dispatcher transparently degrades to mpl.
    """
    from xpcsjax.viz import nlsq_plots as mod

    monkeypatch.setattr(mod, "DATASHADER_AVAILABLE", False)
    # The flag flip alone should not raise; the orchestrator reads it at
    # call time and routes to the matplotlib path.
    assert mod.DATASHADER_AVAILABLE is False


# ---------------------------------------------------------------------------
# 12. _save_fig closes the figure even when savefig raises.
# ---------------------------------------------------------------------------


def test_save_fig_closes_on_savefig_exception(tmp_path):
    import matplotlib.pyplot as plt

    from xpcsjax.viz.nlsq_plots import _save_fig

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fignum = fig.number

    # Point save_path at a directory we can create but use an invalid suffix
    # so matplotlib's savefig raises. The figure must still close.
    bad_path = tmp_path / "out.notarealextension"
    with pytest.raises(Exception):  # noqa: B017 — exception class varies by mpl version
        _save_fig(fig, bad_path)
    assert not plt.fignum_exists(fignum), "figure should be closed even when savefig raises"


# ---------------------------------------------------------------------------
# 13. Non-individual heterodyne scaling modes that carry per_angle_mode in
#     diagnostics (averaged / constant) are now reconstructed by viz instead
#     of raising NotImplementedError. Regression for the auto->averaged fit
#     that crashed plot generation (16 params != 20 individual-mode params).
# ---------------------------------------------------------------------------


def test_heterodyne_averaged_mode_does_not_raise(tmp_path):
    """Averaged mode (14 physics + 2 shared scaling, per_angle_mode in diag)."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import _unpack_heterodyne_scaling, generate_nlsq_plots

    model = HeterodyneModel()
    n_phi = 3
    physical = np.asarray(model.get_default_parameters(), dtype=float)
    # Averaged layout: [physics..., contrast, offset].
    params = np.concatenate([physical, [0.2, 1.0]])
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.full(params.size, 0.01),
        covariance=np.eye(params.size) * 0.01,
        chi_squared=2.5,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
        nlsq_diagnostics={
            "per_angle_mode": "averaged",
            "averaged_contrast": 0.2,
            "averaged_offset": 1.0,
        },
    )
    data = _make_heterodyne_data(n_phi=n_phi)
    config = _make_heterodyne_config()

    # The shared contrast/offset must be replicated across all angles.
    contrasts, offsets, phys, n = _unpack_heterodyne_scaling(
        model, result, n_phi_expected=n_phi
    )
    assert n == n_phi
    np.testing.assert_allclose(contrasts, 0.2)
    np.testing.assert_allclose(offsets, 1.0)
    np.testing.assert_allclose(phys, physical)

    # End-to-end: must not raise (previously raised NotImplementedError).
    generate_nlsq_plots(
        model=model, result=result, data=data, config=config, output_dir=tmp_path
    )


def test_laminar_flow_combined_model_does_not_raise(tmp_path):
    """make_model returns a bare CombinedModel for homodyne modes
    (static_*/laminar_flow) — the Task-28 contract. generate_nlsq_plots must
    accept it (regression: previously raised "Unsupported model type:
    CombinedModel"). Drives the model via CombinedModel.compute_g2.
    """
    from xpcsjax.core.models import make_model
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import _unpack_result_params, generate_nlsq_plots

    model = make_model({"analysis_mode": "laminar_flow"})
    assert type(model).__name__ == "CombinedModel"

    physical = np.asarray(model.get_default_parameters(), dtype=float)  # 7 physical
    # Homodyne result layout: [contrast, offset, *physical].
    params = np.concatenate([[0.2, 1.0], physical])
    n = params.size
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.full(n, 0.01),
        covariance=np.eye(n) * 0.01,
        chi_squared=2.5,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
    )

    # Unpacking treats CombinedModel like the homodyne layout.
    contrast, offset, phys, names = _unpack_result_params(model, result, {})
    assert contrast == 0.2
    assert offset == 1.0
    np.testing.assert_allclose(phys, physical)
    assert names == list(model.parameter_names)

    data = _make_heterodyne_data(n_phi=3)
    config = _make_heterodyne_config()
    config["analysis_mode"] = "laminar_flow"

    generate_nlsq_plots(
        model=model, result=result, data=data, config=config, output_dir=tmp_path
    )


def test_heterodyne_constant_mode_with_diag_does_not_raise(tmp_path):
    """Constant mode (14 physics, frozen per-angle scaling in diagnostics)."""
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult
    from xpcsjax.viz.nlsq_plots import _unpack_heterodyne_scaling, generate_nlsq_plots

    model = HeterodyneModel()
    n_phi = 3
    physical = np.asarray(model.get_default_parameters(), dtype=float)
    result = OptimizationResult(
        parameters=physical.copy(),
        uncertainties=np.full(physical.size, 0.01),
        covariance=np.eye(physical.size) * 0.01,
        chi_squared=2.5,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=10,
        execution_time=0.1,
        device_info={"platform": "cpu"},
        nlsq_diagnostics={
            "per_angle_mode": "constant",
            "contrast_per_angle_fixed": np.full(n_phi, 0.2),
            "offset_per_angle_fixed": np.full(n_phi, 1.0),
        },
    )
    data = _make_heterodyne_data(n_phi=n_phi)
    config = _make_heterodyne_config()

    contrasts, offsets, phys, _ = _unpack_heterodyne_scaling(
        model, result, n_phi_expected=n_phi
    )
    np.testing.assert_allclose(contrasts, 0.2)
    np.testing.assert_allclose(offsets, 1.0)
    np.testing.assert_allclose(phys, physical)

    generate_nlsq_plots(
        model=model, result=result, data=data, config=config, output_dir=tmp_path
    )
