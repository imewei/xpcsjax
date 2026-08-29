"""Regression tests for the 2026-08-05 data-module deep-RCA debug-audit fixes.

Each test fails against the pre-fix code and passes after the fix. Written in
response to the PR #35 code-review test-coverage gap: the audit fixed 12 real
bugs but shipped with zero committed regression tests (every fix was verified
manually with an ad-hoc repro script, not preserved as a test).
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# xpcs_loader._migrate_cache_template -- mixed $/{} syntax collision.
# ---------------------------------------------------------------------------
def test_migrate_cache_template_mixed_syntax_migrates_only_bare_braces() -> None:
    from xpcsjax.data.xpcs_loader import _migrate_cache_template

    out = _migrate_cache_template("cache_${wavevector_q}_{start_frame}_{end_frame}.npz")
    assert out == "cache_${wavevector_q}_${start_frame}_${end_frame}.npz"


def test_migrate_cache_template_all_old_style_migrates_both() -> None:
    from xpcsjax.data.xpcs_loader import _migrate_cache_template

    out = _migrate_cache_template("cache_{start_frame}_{end_frame}.npz")
    assert out == "cache_${start_frame}_${end_frame}.npz"


def test_migrate_cache_template_pure_dollar_syntax_unchanged() -> None:
    from xpcsjax.data.xpcs_loader import _migrate_cache_template

    template = "cache_${wavevector_q}_${start_frame}.npz"
    assert _migrate_cache_template(template) == template


def test_migrate_cache_template_format_spec_migrated() -> None:
    from xpcsjax.data.xpcs_loader import _migrate_cache_template

    out = _migrate_cache_template("cache_{wavevector_q:.4f}.npz")
    assert out == "cache_${wavevector_q}.npz"


# ---------------------------------------------------------------------------
# quality_controller.ValidationIssue fallback -- must expose the same fields
# as xpcsjax.data.validation.ValidationIssue, or a call site using
# parameter=/value= only crashes in the degraded-import environment.
# ---------------------------------------------------------------------------
def test_validation_issue_fallback_has_full_field_parity(monkeypatch) -> None:
    # Load an ISOLATED copy of the module under a throwaway name instead of
    # importlib.reload()-ing the real xpcsjax.data.quality_controller: reload
    # mutates the shared module object in place (new Enum/dataclass identities
    # for every class it defines), which silently breaks every other test
    # that already imported the old classes before this test runs.
    import dataclasses
    import importlib.util
    import sys

    import xpcsjax.data.quality_controller as real_qc

    monkeypatch.setitem(sys.modules, "xpcsjax.data.validation", None)

    spec = importlib.util.spec_from_file_location(
        "_test_only_quality_controller_fallback_probe", real_qc.__file__
    )
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    try:
        spec.loader.exec_module(probe)
        assert probe.HAS_VALIDATION is False

        fallback_fields = {f.name for f in dataclasses.fields(probe.ValidationIssue)}
        assert fallback_fields == {
            "severity",
            "category",
            "message",
            "parameter",
            "value",
            "recommendation",
        }

        # Every real ValidationIssue(...) call site in this file uses
        # keyword args including parameter= and value= -- confirm the
        # fallback actually accepts them without TypeError.
        issue = probe.ValidationIssue(
            severity="error",
            category="completeness",
            message="missing key",
            parameter="c2_exp",
            value=None,
            recommendation="check the loader",
        )
        assert issue.parameter == "c2_exp"
        assert issue.value is None
    finally:
        del sys.modules[spec.name]


# ---------------------------------------------------------------------------
# validators.validate_positive_value / validate_numeric_range -- NaN must not
# silently pass IEEE-754 relational comparisons.
# ---------------------------------------------------------------------------
def test_validate_positive_value_rejects_nan() -> None:
    from xpcsjax.data.validators import validate_positive_value

    errors = validate_positive_value(float("nan"), "dt")
    assert errors, "NaN silently passed the positive-value check"


def test_validate_numeric_range_rejects_nan_with_require_positive() -> None:
    from xpcsjax.data.validators import validate_numeric_range

    errors = validate_numeric_range(
        {"min": float("nan"), "max": 1.0}, "q_range", require_positive=True
    )
    assert errors, "NaN min silently passed a require_positive range check"


def test_validate_numeric_range_rejects_nan_when_wrapped_and_unbounded() -> None:
    # The allow_wrapped=True path skips the ordering check entirely, so a
    # NaN bound must still be caught by its own always-on finiteness check --
    # this is the one case test_validate_by_rules_phi_wrapped.py does NOT
    # cover, and it is not incidentally protected by value_bounds here.
    from xpcsjax.data.validators import validate_numeric_range

    errors = validate_numeric_range(
        {"min": 170.0, "max": float("nan")}, "phi_range", allow_wrapped=True
    )
    assert errors, "NaN max silently passed an unbounded wrapped-range check"


# ---------------------------------------------------------------------------
# filtering_utils.XPCSDataFilter -- NaN quality score must not bypass the
# `< quality_threshold` gate (NaN < threshold is False in Python).
# ---------------------------------------------------------------------------
def test_degenerate_matrix_nan_quality_score_is_dropped_not_kept() -> None:
    from xpcsjax.data.filtering_utils import FilteringResult, XPCSDataFilter

    f = XPCSDataFilter(config={"data_filtering": {"quality_threshold": 0.5}})
    good_matrix = np.full((8, 8), 1.3, dtype=np.float64)
    degenerate_matrix = np.empty((0, 0), dtype=np.float64)

    result = FilteringResult(
        selected_indices=None,
        total_available=2,
        total_selected=0,
        filters_applied=[],
        filter_statistics={},
        fallback_used=False,
        warnings=[],
        errors=[],
    )
    mask = f._apply_quality_filtering([good_matrix, degenerate_matrix], result)

    assert mask is not None
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False, "degenerate (NaN-score) matrix was kept, not dropped"


def test_calculate_matrix_quality_score_never_returns_nan() -> None:
    from xpcsjax.data.filtering_utils import XPCSDataFilter

    f = XPCSDataFilter()
    assert f._calculate_matrix_quality_score(np.empty((0, 0))) == 0.0
    assert f._calculate_matrix_quality_score(np.full((5, 5), np.nan)) == 0.0


# ---------------------------------------------------------------------------
# phi_filtering.PhiAngleFilter -- angles must be normalized to [-180, 180]
# before range comparison, matching angle_filtering.py's sibling behavior.
# ---------------------------------------------------------------------------
def test_phi_angle_filter_matches_raw_360_convention_angle() -> None:
    from xpcsjax.data.phi_filtering import PhiAngleFilter

    pf = PhiAngleFilter()
    # 355 degrees (raw [0, 360) convention) is physically -5 degrees, which
    # must match the default [-10, 10] target range exactly like -5.0 does.
    indices, _ = pf.filter_angles_for_optimization(
        [355.0, -5.0, 90.0], target_ranges=[(-10.0, 10.0)]
    )
    assert 0 in indices
    assert 1 in indices
    assert 2 not in indices


# ---------------------------------------------------------------------------
# angle_filtering.apply_angle_filtering_for_optimization -- an explicit None
# value must be treated the same as an absent key, not crash on np.asarray.
# ---------------------------------------------------------------------------
def test_apply_angle_filtering_explicit_none_values_do_not_crash() -> None:
    from xpcsjax.data.angle_filtering import apply_angle_filtering_for_optimization

    data = {"phi_angles_list": None, "c2_exp": None}
    result = apply_angle_filtering_for_optimization(data, config={})
    assert result is data


# ---------------------------------------------------------------------------
# config.validate_config_schema -- a None section is coerced and reported
# cleanly; a genuinely malformed (non-dict, non-None) section must still be
# flagged as an error, not silently absorbed into an empty section.
# ---------------------------------------------------------------------------
def test_validate_config_schema_none_section_reports_missing_params() -> None:
    from xpcsjax.data.config import validate_config_schema

    result = validate_config_schema({"experimental_data": None})
    assert not result.is_valid
    assert any("experimental_data" in e for e in result.errors)


def test_validate_config_schema_malformed_section_is_flagged_not_absorbed() -> None:
    from xpcsjax.data.config import validate_config_schema

    # A list where a mapping was required (e.g. a mis-indented YAML section)
    # must not be silently treated as "optional section, absent" -- that
    # would make the structural mistake invisible to the validator whose job
    # is to catch exactly this.
    result = validate_config_schema({"experimental_data": ["not", "a", "mapping"]})
    assert not result.is_valid
    assert any("experimental_data" in e and "mapping" in e.lower() for e in result.errors), (
        result.errors
    )


# ---------------------------------------------------------------------------
# optimization.DatasetOptimizer.create_chunked_iterator -- a phi array
# misaligned with data must raise, not silently truncate.
# ---------------------------------------------------------------------------
def test_create_chunked_iterator_rejects_misaligned_phi() -> None:
    from xpcsjax.data.optimization import DatasetOptimizer

    opt = DatasetOptimizer()
    data = np.arange(10, dtype=np.float64)
    sigma = np.ones(10, dtype=np.float64)
    t1 = np.zeros((2, 2))
    t2 = np.zeros((2, 2))
    phi_mismatched = np.array([0.0, 90.0, 180.0])  # length 3 != len(data) 10

    with pytest.raises(ValueError, match="phi length"):
        list(opt.create_chunked_iterator(data, sigma, t1, t2, phi_mismatched, chunk_size=4))


def test_create_chunked_iterator_slices_aligned_phi_correctly() -> None:
    from xpcsjax.data.optimization import DatasetOptimizer

    opt = DatasetOptimizer()
    data = np.arange(10, dtype=np.float64)
    sigma = np.ones(10, dtype=np.float64)
    t1 = np.zeros((2, 2))
    t2 = np.zeros((2, 2))
    phi_aligned = np.arange(10, dtype=np.float64) * 10.0

    chunks = list(opt.create_chunked_iterator(data, sigma, t1, t2, phi_aligned, chunk_size=4))
    assert len(chunks) == 3
    np.testing.assert_array_equal(np.asarray(chunks[0][4]), phi_aligned[0:4])
    np.testing.assert_array_equal(np.asarray(chunks[2][4]), phi_aligned[8:10])


# ---------------------------------------------------------------------------
# memory_manager.AdvancedMemoryManager -- the manager<->monitor bound-method
# callback cycle must not defer cleanup to a full GC pass.
# ---------------------------------------------------------------------------
def test_advanced_memory_manager_collected_without_gc_sweep() -> None:
    from xpcsjax.data.memory_manager import AdvancedMemoryManager

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        manager = AdvancedMemoryManager(config={"memory": {"enable_monitoring": False}})
        ref = weakref.ref(manager)
        del manager
        assert ref() is None, (
            "manager survived after its only reference was dropped -- "
            "a reference cycle is deferring cleanup to the cyclic GC"
        )
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


# ---------------------------------------------------------------------------
# performance_engine.MemoryMapManager.close_all -- a checked-out handle must
# not be closed out from under an active reader.
# ---------------------------------------------------------------------------
def test_close_all_skips_handle_still_checked_out() -> None:
    from unittest.mock import MagicMock

    from xpcsjax.data.performance_engine import MemoryMapManager

    manager = MemoryMapManager()
    handle = MagicMock()
    manager._open_maps["fake_path.h5"] = handle
    manager._in_use["fake_path.h5"] = 1  # simulate an active checkout

    manager.close_all()

    handle.close.assert_not_called()
    assert "fake_path.h5" in manager._open_maps


def test_close_all_closes_handle_not_in_use() -> None:
    from unittest.mock import MagicMock

    from xpcsjax.data.performance_engine import MemoryMapManager

    manager = MemoryMapManager()
    handle = MagicMock()
    manager._open_maps["fake_path.h5"] = handle

    manager.close_all()

    handle.close.assert_called_once()
    assert "fake_path.h5" not in manager._open_maps


# ---------------------------------------------------------------------------
# performance_engine.MultiLevelCache -- SSD usage-counter updates must be
# atomic under concurrent put() calls, or the tracked usage drifts below the
# real on-disk total and eviction silently stops firing.
# ---------------------------------------------------------------------------
def test_multi_level_cache_ssd_usage_counter_matches_disk_after_concurrent_puts(
    tmp_path, monkeypatch
) -> None:
    import threading

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from xpcsjax.data.performance_engine import MultiLevelCache

    cache = MultiLevelCache(memory_cache_mb=1.0, ssd_cache_mb=1000.0, hdd_cache_mb=1000.0)
    item = np.ones(2000, dtype=np.float64)  # small, fast to (de)serialize

    def _put(i: int) -> None:
        cache._put_ssd(f"key_{i}", item)

    threads = [threading.Thread(target=_put, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    actual_usage_mb = sum(
        f.stat().st_size for f in cache._ssd_cache_path.iterdir() if f.is_file()
    ) / (1024 * 1024)

    assert cache._ssd_usage_mb == pytest.approx(actual_usage_mb, rel=1e-6), (
        "tracked SSD usage drifted from the real on-disk total under concurrent put()"
    )
