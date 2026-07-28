"""Unit tests for low-level plot functions and helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import (
    _evaluate_c2_per_angle,
    _save_fig,
    _unpack_result_params,
    plot_nlsq_fit,
    plot_residual_map,
    plot_simulated_data,
)


def test_save_fig_with_none_is_noop() -> None:
    fig, _ = plt.subplots()
    _save_fig(fig, None)
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_save_fig_writes_and_closes(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    save_path = tmp_path / "test.png"
    n = fig.number
    _save_fig(fig, save_path)
    assert save_path.exists()
    with open(save_path, "rb") as f:
        assert f.read(4) == b"\x89PNG"
    assert not plt.fignum_exists(n)


def test_save_fig_creates_parent_dirs(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    save_path = tmp_path / "nested" / "dir" / "test.png"
    _save_fig(fig, save_path)
    assert save_path.exists()


def test_unpack_homodyne(
    homodyne_model, converged_homodyne_result, minimal_homodyne_config
) -> None:
    contrast, offset, physical_params, param_names = _unpack_result_params(
        homodyne_model, converged_homodyne_result, minimal_homodyne_config
    )
    assert contrast == pytest.approx(0.2)
    assert offset == pytest.approx(1.0)
    assert physical_params.shape == (3,)
    np.testing.assert_array_almost_equal(physical_params, [100.0, -0.5, 0.0])
    assert len(param_names) == 3


def test_unpack_heterodyne_per_angle_layout(
    heterodyne_model,
    converged_heterodyne_result,
) -> None:
    """Per-angle layout: [c_0..N-1, o_0..N-1, 14 physical]."""
    config = {"analyzer_parameters": {"dt": 0.1}}
    contrast, offset, physical_params, names = _unpack_result_params(
        heterodyne_model, converged_heterodyne_result, config
    )
    # Returns mean contrast/offset across angles (~0.2/~1.0 from fixture)
    assert contrast == pytest.approx(0.2)
    assert offset == pytest.approx(1.0)
    # Physical params: 14 registry-ordered values
    assert physical_params.shape == (14,)
    assert len(names) == 14


def test_unpack_unsupported_model_raises() -> None:
    class FakeModel:
        pass

    with pytest.raises(TypeError, match="Unsupported model type"):
        _unpack_result_params(FakeModel(), None, {})  # type: ignore[arg-type]


def test_unpack_homodyne_short_params_raises(
    homodyne_model,
    minimal_homodyne_config,
) -> None:
    """Homodyne result with <3 params should raise ValueError."""
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    bad = OptimizationResult(
        parameters=np.array([0.2, 1.0]),  # only 2 params, missing physical
        uncertainties=np.ones(2),
        covariance=np.eye(2),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )
    # 2 params with a >=3-physical homodyne model leaves a negative scaling
    # block (2 - n_physical < 0), which _homodyne_scaling_arrays rejects.
    with pytest.raises(ValueError, match="non-negative even count"):
        _unpack_result_params(homodyne_model, bad, minimal_homodyne_config)


def test_unpack_homodyne_per_angle_layout_offset_not_misread() -> None:
    """Regression: homodyne per-angle results are ``[c_0..N-1, o_0..N-1, phys]``.

    The fit uses per-angle scaling by default, so a laminar result for 3 angles
    is ``[c0,c1,c2, o0,o1,o2, D0,alpha,D_offset,gd0,beta,gdoff,phi0]`` (13). The
    viz must infer n_phi from the 7 physical params, NOT assume the scalar
    ``[contrast, offset, *physical]`` layout — doing the latter read ``offset``
    as ``c1`` (so offset == contrast) and shifted scaling values into the
    physical block, rendering the fitted c2 surface flat.
    """
    from xpcsjax.core.models import make_model
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    model = make_model({"analysis_mode": "laminar_flow"})
    n_physical = len(model.parameter_names)  # 7
    contrasts = np.array([0.30, 0.31, 0.32])
    offsets = np.array([1.10, 1.11, 1.12])
    physical = np.asarray(model.get_default_parameters(), dtype=float)
    params = np.concatenate([contrasts, offsets, physical])  # 3 + 3 + 7 = 13

    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(params.size),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )

    contrast, offset, physical_params, names = _unpack_result_params(model, result, {})
    # offset must be the mean OFFSET (~1.11), never the mean contrast (~0.31).
    assert offset == pytest.approx(offsets.mean())
    assert contrast == pytest.approx(contrasts.mean())
    assert offset != pytest.approx(contrast)
    # Physical block must be exactly the 7 physical params (not shifted/padded).
    assert physical_params.size == n_physical
    np.testing.assert_allclose(physical_params, physical)
    assert list(names) == list(model.parameter_names)


def test_unpack_heterodyne_size_mismatch_raises(heterodyne_model) -> None:
    """Heterodyne result with wrong param count should raise ValueError."""
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    bad = OptimizationResult(
        parameters=np.arange(5, dtype=float),  # 5 - 14 = -9, definitely < 0
        uncertainties=np.ones(5),
        covariance=np.eye(5),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=1,
        execution_time=0.1,
        device_info={},
    )
    config = {"analyzer_parameters": {"dt": 0.1}}
    with pytest.raises(ValueError, match=r"per-angle layout"):
        _unpack_result_params(heterodyne_model, bad, config)


def test_evaluate_homodyne_2d_finite(
    homodyne_model,
    converged_homodyne_result,
    synthetic_multi_angle_data,
    minimal_homodyne_config,
) -> None:
    data = synthetic_multi_angle_data
    c2 = _evaluate_c2_per_angle(
        homodyne_model,
        converged_homodyne_result,
        data,
        minimal_homodyne_config,
        phi_deg=45.0,
    )
    assert c2.ndim == 2
    assert c2.shape == (data["t1"].size, data["t2"].size)
    assert np.all(np.isfinite(c2))


def test_evaluate_heterodyne_returns_2d_finite(
    heterodyne_model,
    converged_heterodyne_result,
    synthetic_multi_angle_data,
) -> None:
    """Heterodyne path returns a real c2 surface in the expected [1.0, 1.5] range."""
    config = {
        "analyzer_parameters": {
            "dt": 0.1,
            "scattering": {"wavevector_q": 0.0054},
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "heterodyne",
    }
    c2 = _evaluate_c2_per_angle(
        heterodyne_model,
        converged_heterodyne_result,
        synthetic_multi_angle_data,
        config,
        phi_deg=45.0,
    )
    assert c2.ndim == 2
    assert c2.shape == (
        synthetic_multi_angle_data["t1"].size,
        synthetic_multi_angle_data["t2"].size,
    )
    assert np.all(np.isfinite(c2))
    # c2 = offset + contrast * g1². With offset=1.0, contrast=0.2, g1² in [0,1],
    # c2 should be in [1.0, 1.2].
    assert 0.95 < float(np.nanmean(c2)) < 1.25


def test_evaluate_unsupported_raises() -> None:
    class FakeModel:
        pass

    with pytest.raises(TypeError, match="Unsupported model type"):
        _evaluate_c2_per_angle(FakeModel(), None, {}, {}, phi_deg=0.0)  # type: ignore[arg-type]


def test_resolve_phi_index_prefers_explicit_index() -> None:
    """Duplicate-valued phi entries must resolve by loop index, not first match."""
    from xpcsjax.viz.nlsq_plots import _resolve_phi_index

    data = {"phi_angles_list": np.array([0.0, 45.0, 45.0, 90.0])}
    # Explicit index wins even when the nominal value is duplicated.
    assert _resolve_phi_index(data, 45.0, phi_index=2) == 2
    assert _resolve_phi_index(data, 45.0, phi_index=1) == 1
    # Fallback (no index) picks the first matching value — the legacy path.
    assert _resolve_phi_index(data, 45.0, phi_index=None) == 1
    with pytest.raises(ValueError, match="not found"):
        _resolve_phi_index(data, 12.5, phi_index=None)


def test_evaluate_heterodyne_duplicate_phi_uses_own_scaling(
    heterodyne_model,
    synthetic_multi_angle_data,
) -> None:
    """Two angles sharing a phi value each render with THEIR OWN contrast/offset.

    Regression for the index re-derivation bug: ``np.isclose`` returned the
    FIRST duplicate, so every later duplicate-valued angle silently rendered
    with the first duplicate's scaling. c2 = offset[i] + contrast[i] * g1²; for
    a shared phi the g1² surface is identical, so any difference must come from
    the per-index scaling.
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    t = synthetic_multi_angle_data["t1"]
    # phi_angles_list has index 1 and index 2 sharing the value 45.0.
    phi_angles = np.array([0.0, 45.0, 45.0, 90.0])
    n_phi = phi_angles.size
    data = {
        "c2_exp": np.empty((n_phi, t.size, t.size)),
        "phi_angles_list": phi_angles,
        "t1": t,
        "t2": t,
        "dt": 0.1,
    }
    het = HeterodyneModel()
    physical = np.asarray(het.get_default_parameters(), dtype=float)
    # Distinct scaling for the duplicate pair: idx1 -> (0.2, 1.0), idx2 -> (0.5, 2.0).
    contrasts = np.array([0.2, 0.2, 0.5, 0.2])
    offsets = np.array([1.0, 1.0, 2.0, 1.0])
    result = OptimizationResult(
        parameters=np.concatenate([contrasts, offsets, physical]),
        uncertainties=np.ones(2 * n_phi + physical.size) * 0.01,
        covariance=np.eye(2 * n_phi + physical.size) * 0.01,
        chi_squared=2.5,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=1,
        execution_time=1.0,
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

    c2_idx1 = _evaluate_c2_per_angle(
        heterodyne_model, result, data, config, phi_deg=45.0, phi_index=1
    )
    c2_idx2 = _evaluate_c2_per_angle(
        heterodyne_model, result, data, config, phi_deg=45.0, phi_index=2
    )

    # The two duplicate angles must render differently (buggy code made them equal).
    assert not np.allclose(c2_idx1, c2_idx2)
    # Recovering g1² from each surface using ITS OWN scaling must agree — proving
    # each render used its own (contrast, offset), not the first duplicate's.
    g1_from_1 = (c2_idx1 - 1.0) / 0.2
    g1_from_2 = (c2_idx2 - 2.0) / 0.5
    assert np.allclose(g1_from_1, g1_from_2)


def test_evaluate_null_scattering_section_raises_valueerror(
    heterodyne_model,
    converged_heterodyne_result,
    synthetic_multi_angle_data,
) -> None:
    """A null (present-but-None) config section degrades to the intended ValueError.

    ``analyzer_parameters: {scattering: null}`` in YAML yields ``{"scattering":
    None}``; the old ``.get("scattering", {})`` returned None and the chained
    ``.get(...)`` raised an opaque AttributeError instead of the clear ValueError.
    """
    config = {
        "analyzer_parameters": {
            "dt": 0.1,
            "scattering": None,
            "geometry": {"stator_rotor_gap": 2_000_000.0},
        },
        "analysis_mode": "heterodyne",
    }
    with pytest.raises(ValueError, match="wavevector_q"):
        _evaluate_c2_per_angle(
            heterodyne_model,
            converged_heterodyne_result,
            synthetic_multi_angle_data,
            config,
            phi_deg=45.0,
        )


def test_evaluate_homodyne_null_scattering_section_raises_valueerror() -> None:
    """The homodyne branch's null-config-section guard (mirrors the heterodyne
    guard pinned by ``test_evaluate_null_scattering_section_raises_valueerror``
    above, but was previously only exercised on the heterodyne side).

    Reaching this guard requires a model WITHOUT ``compute_c2_single_angle``
    (the bare ``CombinedModel`` that ``make_model`` returns for
    static_*/laminar_flow) -- ``HomodyneModel`` has that method and takes the
    early-return branch instead, never touching the guard under test.
    """
    from xpcsjax.core.models import make_model
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    model = make_model({"analysis_mode": "laminar_flow"})
    assert type(model).__name__ == "CombinedModel"
    assert not hasattr(model, "compute_c2_single_angle")

    physical = np.asarray(model.get_default_parameters(), dtype=float)
    params = np.concatenate([[0.2, 1.0], physical])
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
    )
    n = 8
    t = np.arange(n, dtype=float) * 0.1
    data = {"t1": t, "t2": t}
    config = {
        "analyzer_parameters": {"scattering": None, "dt": 0.1},
        "analysis_mode": "laminar_flow",
    }

    with pytest.raises(ValueError, match="wavevector_q"):
        _evaluate_c2_per_angle(model, result, data, config, phi_deg=45.0)


def test_plot_nlsq_fit_three_image_axes(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_nlsq_fit(
        d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], phi_deg=45.0, reduced_chi_squared=0.906
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 3
    plt.close(fig)


def test_plot_nlsq_fit_suptitle_chi_squared(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_nlsq_fit(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], reduced_chi_squared=0.906)
    suptitle = fig._suptitle.get_text() if fig._suptitle else ""
    assert "0.906" in suptitle
    plt.close(fig)


def test_plot_nlsq_fit_shared_color_scale(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_nlsq_fit(d["c2_exp"], d["c2_exp"] * 1.05, t=d["t"])
    image_axes = [ax for ax in fig.axes if ax.images]
    assert image_axes[0].images[0].norm.vmin == image_axes[1].images[0].norm.vmin
    assert image_axes[0].images[0].norm.vmax == image_axes[1].images[0].norm.vmax
    plt.close(fig)


def test_plot_nlsq_fit_residual_cmap_is_rdbu(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_nlsq_fit(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"])
    image_axes = [ax for ax in fig.axes if ax.images]
    assert image_axes[2].images[0].get_cmap().name in {"RdBu_r", "RdBu"}
    plt.close(fig)


def test_plot_nlsq_fit_save_path_writes_png(
    synthetic_single_angle_data,
    tmp_path: Path,
) -> None:
    d = synthetic_single_angle_data
    save_path = tmp_path / "fit.png"
    plot_nlsq_fit(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], save_path=save_path)
    assert save_path.exists()
    with open(save_path, "rb") as f:
        assert f.read(4) == b"\x89PNG"


def test_plot_nlsq_fit_accepts_t_none(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_nlsq_fit(d["c2_exp"], d["c2_exp"] * 0.95, t=None)
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_plot_residual_map_four_main_axes(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_residual_map(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], phi_deg=45.0)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_residual_map_histogram_normal_overlay(
    synthetic_single_angle_data,
) -> None:
    d = synthetic_single_angle_data
    fig = plot_residual_map(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"])
    hist_axes = [ax for ax in fig.axes if "Distribution" in ax.get_title()]
    assert len(hist_axes) == 1
    legend = hist_axes[0].get_legend()
    assert legend is not None
    label = legend.get_texts()[0].get_text()
    assert "Normal" in label and "μ" in label and "σ" in label
    plt.close(fig)


def test_plot_residual_map_all_nan_residuals(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    exp_nan = np.full_like(d["c2_exp"], np.nan)
    fig = plot_residual_map(exp_nan, exp_nan, t=d["t"])
    plt.close(fig)


def test_plot_residual_map_save_path_writes_png(
    synthetic_single_angle_data, tmp_path: Path
) -> None:
    d = synthetic_single_angle_data
    save_path = tmp_path / "residuals.png"
    plot_residual_map(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], save_path=save_path)
    assert save_path.exists()
    with open(save_path, "rb") as f:
        assert f.read(4) == b"\x89PNG"


def test_plot_simulated_data_single_image_axis(synthetic_single_angle_data) -> None:
    d = synthetic_single_angle_data
    fig = plot_simulated_data(
        d["c2_exp"],
        t=d["t"],
        phi_deg=45.0,
        contrast=0.2,
        offset=1.0,
        analysis_mode="static_isotropic",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 1
    plt.close(fig)


def test_plot_simulated_data_annotation_ignores_single_inf(
    synthetic_single_angle_data,
) -> None:
    # Regression: nanmean/nanmin/nanmax only skip NaN, not inf — a single inf
    # value must not poison the Mean/Range annotation text.
    d = synthetic_single_angle_data
    c2_sim = d["c2_exp"].copy()
    c2_sim[0, 0] = np.inf
    finite = c2_sim[np.isfinite(c2_sim)]
    fig = plot_simulated_data(c2_sim, t=d["t"])
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any(f"Mean: {np.mean(finite):.4f}" in t for t in texts)
    plt.close(fig)


def test_plot_simulated_data_save_path_writes_png(
    synthetic_single_angle_data,
    tmp_path: Path,
) -> None:
    d = synthetic_single_angle_data
    save_path = tmp_path / "sim.png"
    plot_simulated_data(d["c2_exp"], t=d["t"], save_path=save_path)
    assert save_path.exists()
    with open(save_path, "rb") as f:
        assert f.read(4) == b"\x89PNG"


# pytest-mpl snapshot tests return the Figure for image comparison (the plugin
# contract). Without ``--mpl`` they run as plain smoke tests, so pytest emits a
# PytestReturnNotNoneWarning for the returned Figure; suppress it per-test (the
# global return-None check stays active for every other test).
@pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="plot_nlsq_fit_baseline.png",
    tolerance=0.5,
    style="default",
)
def test_plot_nlsq_fit_snapshot(synthetic_single_angle_data):
    d = synthetic_single_angle_data
    return plot_nlsq_fit(
        d["c2_exp"],
        d["c2_exp"] * 0.95,
        t=d["t"],
        phi_deg=45.0,
        reduced_chi_squared=0.906,
    )


@pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="plot_residual_map_baseline.png",
    tolerance=0.5,
    style="default",
)
def test_plot_residual_map_snapshot(synthetic_single_angle_data):
    d = synthetic_single_angle_data
    return plot_residual_map(d["c2_exp"], d["c2_exp"] * 0.95, t=d["t"], phi_deg=45.0)


@pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="plot_simulated_data_baseline.png",
    tolerance=0.5,
    style="default",
)
def test_plot_simulated_data_snapshot(synthetic_single_angle_data):
    d = synthetic_single_angle_data
    return plot_simulated_data(
        d["c2_exp"],
        t=d["t"],
        phi_deg=45.0,
        contrast=0.2,
        offset=1.0,
        analysis_mode="static_isotropic",
    )
