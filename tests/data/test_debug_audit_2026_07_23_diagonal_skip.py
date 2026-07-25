"""Regression: the loader's mandatory diagonal correction must not silently
overwrite a diagonal correction preprocessing already applied.

Finding #2 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.data.preprocessing import PreprocessingPipeline


def _synthetic_data(n_mat: int = 2, n_t: int = 6) -> dict:
    rng = np.random.default_rng(0)
    c2 = np.ones((n_mat, n_t, n_t)) + 0.1 * rng.standard_normal((n_mat, n_t, n_t))
    # Make the diagonal deliberately wrong so a real correction changes it.
    for i in range(n_mat):
        np.fill_diagonal(c2[i], 5.0)
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01] * n_mat),
        "phi_angles_list": np.linspace(0.0, 90.0, n_mat),
    }


def test_correct_diagonal_stage_marks_data_as_corrected():
    config = {
        "preprocessing": {
            "enabled": True,
            "stages": {"correct_diagonal": {"method": "statistical"}},
        },
    }
    pipeline = PreprocessingPipeline(config)
    result = pipeline.process(_synthetic_data())
    assert result.success
    assert result.data.get("_diagonal_corrected") is True
    # The 'statistical' method must actually have changed the diagonal away
    # from the deliberately-wrong 5.0 seed value.
    assert not np.allclose(np.diagonal(result.data["c2_exp"][0]), 5.0)


def test_disabled_preprocessing_does_not_set_marker():
    # NOTE: PreprocessingPipeline.process() itself does not consult the
    # top-level "preprocessing.enabled" flag -- that gate lives one layer up,
    # in xpcs_loader.py's _apply_preprocessing_pipeline (it skips calling
    # PreprocessingPipeline entirely when disabled). Within
    # PreprocessingPipeline, per-stage gating is what actually controls
    # whether a stage -- and thus the marker -- runs, so disable the
    # correct_diagonal stage specifically to exercise the real mechanism.
    config = {
        "preprocessing": {"stages": {"correct_diagonal": {"enabled": False}}},
    }
    pipeline = PreprocessingPipeline(config)
    result = pipeline.process(_synthetic_data())
    assert "_diagonal_corrected" not in result.data


def test_loader_skips_mandatory_correction_when_already_corrected(monkeypatch):
    """End-to-end: apply_diagonal_correction_batch must NOT be called again
    when the preprocessing pipeline already set _diagonal_corrected."""
    import xpcsjax.data.xpcs_loader as xl

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    data = _synthetic_data()
    data["_diagonal_corrected"] = True
    result = xl._maybe_apply_mandatory_diagonal_correction(data)
    assert calls == [], "mandatory correction must be skipped when already corrected"
    assert result is data


def test_loader_applies_mandatory_correction_when_not_preprocessed(monkeypatch):
    import xpcsjax.data.xpcs_loader as xl

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    data = _synthetic_data()
    xl._maybe_apply_mandatory_diagonal_correction(data)
    assert len(calls) == 1, "mandatory correction must still run by default"


def test_loader_applies_configured_diagonal_correction_end_to_end(tmp_path, monkeypatch):
    """End-to-end (design spec Testing item 2): drive the REAL
    load_experimental_data with preprocessing enabled and a non-'basic'
    correct_diagonal.method, and confirm both that the mandatory post-load
    'basic' pass is skipped AND that the final c2_exp diagonal reflects the
    configured method's own correction -- not just the isolated helper's
    unit behavior. A wiring bug between _apply_preprocessing_pipeline and
    _maybe_apply_mandatory_diagonal_correction inside load_experimental_data
    itself would not be caught by the helper-level tests above; this is."""
    import xpcsjax.data.xpcs_loader as xl

    hdf_path = tmp_path / "fake.h5"
    hdf_path.write_bytes(b"")  # existence is all load_experimental_data checks
    # before handing off to _load_from_hdf, which is replaced below.

    def fake_load_from_hdf(self, path):
        return _synthetic_data()

    monkeypatch.setattr(xl.XPCSDataLoader, "_load_from_hdf", fake_load_from_hdf)

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    config = {
        "experimental_data": {
            "data_folder_path": str(tmp_path),
            "data_file_name": "fake.h5",
        },
        "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 6},
        "v2_features": {"cache_strategy": "none"},
        "preprocessing": {
            "enabled": True,
            "stages": {"correct_diagonal": {"method": "statistical"}},
        },
    }
    loader = xl.XPCSDataLoader(config_dict=config, configure_logging=False)
    result = loader.load_experimental_data()

    assert calls == [], (
        "the mandatory post-load 'basic' correction must be skipped end-to-end "
        "-- preprocessing's 'statistical' correction already ran"
    )
    c2 = np.asarray(result["c2_exp"])
    assert not np.allclose(np.diagonal(c2[0]), 5.0), (
        "preprocessing's configured 'statistical' correction must actually "
        "have changed the deliberately-wrong seeded diagonal, end-to-end"
    )
