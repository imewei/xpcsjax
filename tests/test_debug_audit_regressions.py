"""Regression tests for the 2026-06-10 whole-codebase debug audit.

Each test pins a specific confirmed-and-fixed defect so it cannot silently
regress. Tests are deliberately lightweight (pure functions / small inputs) and
reference the finding they guard in the docstring.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_get_group_indices_scaling_resolves() -> None:
    """Audit [26]: get_group_indices('scaling') must resolve, not KeyError.

    The 'scaling' group (contrast, offset) lives in the 16-element with-scaling
    array at indices 14, 15; physics-group indices are unchanged.
    """
    from xpcsjax.config.heterodyne_parameter_names import get_group_indices

    assert get_group_indices("scaling") == (14, 15)
    # Physics groups keep their canonical 0-based positions.
    assert get_group_indices("reference") == (0, 1, 2)
    assert get_group_indices("angle") == (13,)


def test_combine_angle_results_excludes_nonfinite_covariance() -> None:
    """Audit [5]: a non-finite per-angle covariance must not poison the combined
    inverse-variance covariance (it is already zero-weighted in the params)."""
    from xpcsjax.optimization.nlsq.strategies.sequential import combine_angle_results

    per_angle = [
        {
            "success": True,
            "parameters": np.array([1.0, 2.0]),
            "covariance": np.diag([0.1, 0.2]),
            "n_points": 100,
            "cost": 1.0,
        },
        {
            "success": True,
            "parameters": np.array([1.1, 2.1]),
            "covariance": np.diag([np.nan, np.inf]),  # failed solve
            "n_points": 100,
            "cost": 1.0,
        },
    ]

    params, cov, _cost = combine_angle_results(per_angle, weighting="inverse_variance")

    assert np.all(np.isfinite(params)), "params poisoned by non-finite angle"
    assert np.all(np.isfinite(cov)), "combined covariance poisoned by non-finite angle"


def test_parameter_manager_two_component_default_params() -> None:
    """Audit [8]: the active-parameter fallback for two_component must return the
    heterodyne parameter set, not the laminar_flow set."""
    from xpcsjax.config.parameter_manager import ParameterManager

    pm = ParameterManager({}, "two_component")
    params = pm._get_default_active_parameters()

    assert "D0_ref" in params
    assert "phi0_het" in params
    # Must NOT fall through to the laminar_flow parameter list.
    assert "gamma_dot_t0" not in params


def test_absent_analysis_mode_resolves_to_isotropic_consistently() -> None:
    """Audit [25]: with analysis_mode absent, the .analysis_mode property and the
    cached ParameterManager must agree (canonical default: static_isotropic)."""
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.config.parameter_registry import AnalysisMode

    cm = ConfigManager(config_override={"analyzer_parameters": {}})  # no analysis_mode key

    assert cm.analysis_mode == AnalysisMode.STATIC_ISOTROPIC
    pm_mode = str(cm._get_parameter_manager().analysis_mode).lower()
    assert "isotropic" in pm_mode
    assert "anisotropic" not in pm_mode



def _make_aps_old_hdf5(path: str, n_pairs: int = 6, msize: int = 8) -> None:
    """Write a minimal APS-old-format HDF5 file the loader can parse."""
    import h5py

    with h5py.File(path, "w") as f:
        f.create_dataset("xpcs/dqlist", data=np.linspace(0.01, 0.05, n_pairs).reshape(1, n_pairs))
        f.create_dataset("xpcs/dphilist", data=np.linspace(0.0, 150.0, n_pairs).reshape(1, n_pairs))
        grp = f.create_group("exchange/C2T_all")
        half = np.ones((msize, msize), dtype=np.float64)
        for i in range(n_pairs):
            grp.create_dataset(str(i + 1), data=half)


@pytest.mark.parametrize("quality_enabled", [True, False])
def test_aps_old_zero_selection_raises(tmp_path, monkeypatch, quality_enabled) -> None:
    """Audit [6] + Codex follow-up: an empty (q,phi) selection must fail loudly on
    BOTH the quality-filtered and the phi-only APS-old load paths, rather than flow
    downstream as a malformed empty c2 stack."""
    pytest.importorskip("h5py")
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    hdf = tmp_path / "aps_old.h5"
    _make_aps_old_hdf5(str(hdf))

    data_filtering: dict = {"enabled": True}
    if quality_enabled:
        data_filtering["quality_filtering"] = {"enabled": True}
    config = {
        "analysis_mode": "static_isotropic",
        "experimental_data": {
            "data_folder_path": str(tmp_path),
            "data_file_name": "aps_old.h5",
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 8,
            "scattering": {"wavevector_q": 0.03},
        },
        "data_filtering": data_filtering,
    }
    loader = XPCSDataLoader(config_dict=config, configure_logging=False)
    # Force "everything filtered out" so the selection collapses to empty,
    # independent of the q/phi/quality filter config details.
    monkeypatch.setattr(loader, "_get_selected_indices", lambda *a, **k: np.array([], dtype=int))

    with pytest.raises(ValueError, match=r"zero \(q,phi\) pairs"):
        loader._load_aps_old_format(str(hdf))


def test_cache_hit_rate_is_a_true_hit_rate(tmp_path, monkeypatch) -> None:
    """Audit [20] (double-check follow-up): cache_hit_rate must be a true
    hits/(hits+misses) fraction, not (#resident keys)/(hits+puts).

    Before the fix, misses were never counted and the numerator was the
    memory-cache size, so the metric could not express the fraction of accesses
    served from cache (and mis-classified the bottleneck type).
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from xpcsjax.data.performance_engine import MultiLevelCache

    cache = MultiLevelCache(memory_cache_mb=64)

    # Cold: no accesses yet -> a neutral 0-access hit rate, no division blow-up.
    stats = cache.get_cache_stats()
    assert stats["hits"] == 0 and stats["misses"] == 0
    assert stats["hit_rate"] == 0.0

    cache.put("a", np.ones(4))
    # 3 hits on the one resident key, 2 misses on absent keys -> 3/5 = 0.6.
    for _ in range(3):
        assert cache.get("a") is not None
    assert cache.get("missing-1") is None
    assert cache.get("missing-2") is None

    stats = cache.get_cache_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == pytest.approx(3 / 5)
    # A real hit rate is bounded by 1.0 regardless of how many keys are resident.
    assert 0.0 <= stats["hit_rate"] <= 1.0
