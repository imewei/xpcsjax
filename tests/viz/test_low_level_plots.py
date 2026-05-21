"""Unit tests for low-level plot functions and helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import _save_fig, _unpack_result_params


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


def test_unpack_heterodyne_keeps_full_vector(heterodyne_model) -> None:
    """Heterodyne registry has 14 names without 'contrast'/'offset' slots, so
    the strict contract introduced in Task 4 raises rather than silently
    falling back to params[0]/[1]. This regression-guards that strict path
    instead of the previous quietly-wrong behavior.
    """
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    result = OptimizationResult(
        parameters=np.arange(14, dtype=float) * 0.1,
        uncertainties=np.ones(14) * 0.01,
        covariance=np.eye(14),
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        convergence_status="converged",
        iterations=10,
        execution_time=1.0,
        device_info={"platform": "cpu"},
    )
    config = {"analyzer_parameters": {"dt": 0.1}}
    with pytest.raises(ValueError, match="'contrast' and/or 'offset'"):
        _unpack_result_params(heterodyne_model, result, config)


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
    with pytest.raises(ValueError, match="needs >=3"):
        _unpack_result_params(homodyne_model, bad, minimal_homodyne_config)


def test_unpack_heterodyne_size_mismatch_raises(heterodyne_model) -> None:
    """Heterodyne result with wrong param count should raise ValueError."""
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    bad = OptimizationResult(
        parameters=np.arange(5, dtype=float),  # 5 params, heterodyne expects 14
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
    with pytest.raises(ValueError, match="expects 14 params"):
        _unpack_result_params(heterodyne_model, bad, config)
