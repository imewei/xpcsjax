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
    # regardless of the q/phi/quality filter config details.
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


def test_aps_old_quality_filter_guards_allocation_before_accumulation(
    tmp_path, monkeypatch
) -> None:
    """2026-07-22 audit Fix 1: the APS-old quality-filtering branch must
    probe-then-guard the allocation budget BEFORE accumulating candidate
    matrices into a Python list — mirroring
    ``_guard_aps_u_intermediate_allocation``'s APS-U ordering — not only
    after, via ``_validate_loaded_arrays`` on the final stacked buffer.
    """
    pytest.importorskip("h5py")
    from xpcsjax.data import xpcs_loader as xl
    from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader

    hdf = tmp_path / "aps_old_quality.h5"
    _make_aps_old_hdf5(str(hdf), n_pairs=6, msize=8)

    config = {
        "analysis_mode": "static_isotropic",
        "experimental_data": {
            "data_folder_path": str(tmp_path),
            "data_file_name": "aps_old_quality.h5",
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 8,
            "scattering": {"wavevector_q": 0.03},
        },
        "data_filtering": {
            "enabled": True,
            "quality_filtering": {"enabled": True},
        },
    }
    loader = XPCSDataLoader(config_dict=config, configure_logging=False)
    # Every candidate survives metadata pre-filter -> the guard has real
    # candidates to bound before the accumulation loop runs.
    monkeypatch.setattr(
        loader, "_get_selected_indices", lambda dq, dphi, matrices=None: np.arange(len(dq))
    )
    # Tiny budget: even one 8x8 float64 matrix (512 bytes) trips it.
    monkeypatch.setattr(xl, "MAX_CORRELATION_ALLOC_BYTES", 100)

    with pytest.raises(XPCSDataFormatError, match="Refusing to allocate"):
        loader._load_aps_old_format(str(hdf))


def test_memory_map_manager_refcount_blocks_concurrent_eviction(tmp_path) -> None:
    """2026-07-22 audit Fix 2: MemoryMapManager must not evict (close) a file
    handle that is currently checked out via ``open_memory_mapped_hdf5``, even
    if it is the LRU-oldest candidate and ``max_open_files`` is exceeded. Once
    the checkout's ``with`` block exits and it becomes LRU-oldest again, it
    must be evictable.
    """
    h5py = pytest.importorskip("h5py")
    import threading

    from xpcsjax.data.performance_engine import MemoryMapManager

    file_a = tmp_path / "a.h5"
    file_b = tmp_path / "b.h5"
    for p in (file_a, file_b):
        with h5py.File(p, "w") as f:
            f.create_dataset("data", data=np.zeros(4))

    manager = MemoryMapManager(max_open_files=1)

    checked_out = threading.Event()
    release = threading.Event()

    def hold_a():
        with manager.open_memory_mapped_hdf5(str(file_a)):
            checked_out.set()
            release.wait(timeout=5)

    reader = threading.Thread(target=hold_a)
    reader.start()
    assert checked_out.wait(timeout=5), "reader thread never checked out file A"

    # Opening B while A is checked out must not evict A, even though A is the
    # only (hence LRU-oldest) open handle and max_open_files=1.
    with manager.open_memory_mapped_hdf5(str(file_b)):
        assert str(file_a) in manager._open_maps, "in-use handle A was evicted mid-read"

    release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()

    # A is no longer in use. Force it to look LRU-oldest and confirm cleanup
    # can now evict it.
    manager._last_access[str(file_a)] = 0.0
    manager._cleanup_old_mappings()
    assert str(file_a) not in manager._open_maps, "released handle A was never evicted"

    manager.close_all()
