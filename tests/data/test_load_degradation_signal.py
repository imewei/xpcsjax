"""DATA-1: degraded-fallback paths must leave a detectable signal.

The quality-gate audit (failure-hunter F3/F4) flagged that when angle filtering
or the preprocessing pipeline crashes, the loader silently substitutes the
optimizer's input (all angles / raw un-preprocessed data) with only a WARNING
and no signal on the result — a caller gating on exceptions cannot tell a
crash-fallback from an intended no-op.

The fix records every degraded fallback on ``loader.load_degradations`` (and
logs at ERROR, same severity as the failure), so the degradation is
programmatically detectable downstream.
"""

from __future__ import annotations

import logging

from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _bare_loader() -> XPCSDataLoader:
    """An instance with __init__ bypassed — we only exercise the helper."""
    inst = object.__new__(XPCSDataLoader)
    inst.load_degradations = []
    return inst


def test_record_degradation_appends_and_logs_error(caplog) -> None:
    loader = _bare_loader()
    with caplog.at_level(logging.ERROR, logger="xpcsjax.data.xpcs_loader"):
        loader._record_degradation("filtering crashed: boom")

    assert loader.load_degradations == ["filtering crashed: boom"]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_record_degradation_accumulates() -> None:
    loader = _bare_loader()
    loader._record_degradation("a")
    loader._record_degradation("b")
    assert loader.load_degradations == ["a", "b"]


def test_final_qc_gate_failure_sets_degraded(monkeypatch, tmp_path) -> None:
    """A dataset that fails the FINAL_DATA quality gate must set ``_degraded``
    even when nothing else (angle-filter/preprocessing fallbacks) went wrong
    (2026-08-05 pr-review-toolkit silent-failure-hunter finding, PR #36).

    Previously ``_degraded`` only reflected ``load_degradations`` and
    ``_preprocessing_degraded`` — a dataset that failed every quality gate
    (score 0, error-severity issues) but had no filtering/preprocessing
    crash would come back with ``_degraded=False``, silently telling a
    caller the data is trustworthy when it is not.
    """
    import numpy as np

    from xpcsjax.data.quality_controller import (
        QualityControlResult,
        QualityControlStage,
        QualityMetrics,
    )
    from xpcsjax.data.xpcs_loader import XPCSDataLoader

    class _StubQualityController:
        """Passes RAW_DATA/FILTERED_DATA, fails FINAL_DATA only."""

        def __init__(self) -> None:
            self.QualityControlStage = QualityControlStage
            self._prefilter_matrix_count: int | None = None

        def validate_data_stage(self, data, stage, previous_result=None):
            passed = stage != QualityControlStage.FINAL_DATA
            return QualityControlResult(
                stage=stage,
                passed=passed,
                metrics=QualityMetrics(overall_score=100.0 if passed else 0.0),
                issues=[],
            )

    def _synthetic_data() -> dict:
        n_t = 4
        return {
            "c2_exp": np.ones((1, n_t, n_t), dtype=np.float64) * 1.5,
            "t1": np.arange(n_t, dtype=np.float64),
            "t2": np.arange(n_t, dtype=np.float64),
            "wavevector_q_list": np.array([0.01], dtype=np.float64),
            "phi_angles_list": np.array([0.0], dtype=np.float64),
        }

    npz_path = tmp_path / "dummy.npz"
    npz_path.touch()  # never actually read: _load_from_cache is stubbed below

    loader = XPCSDataLoader(
        config_dict={
            "experimental_data": {
                "data_folder_path": str(tmp_path),
                "data_file_name": "dummy.npz",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 4},
            "quality_control": {"enabled": True},
        },
        generate_quality_reports=False,
    )
    monkeypatch.setattr(loader, "_load_from_cache", lambda _path: _synthetic_data())
    monkeypatch.setattr(loader, "_initialize_quality_control", _StubQualityController)

    data = loader.load_experimental_data()

    assert data["_degraded"] is True, (
        "a FINAL_DATA quality-gate failure must set _degraded=True even "
        "with no filtering/preprocessing fallback involved"
    )
    assert not loader.load_degradations, "no filtering/preprocessing fallback occurred"
