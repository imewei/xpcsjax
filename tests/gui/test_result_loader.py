"""Tests for the persisted-result summary loader (JAX-free)."""

import json

from xpcsjax.gui.result_loader import ResultSummary, load_result_summary


def _write_result(dir_path, **meta):
    payload = {
        "schema": "xpcsjax.nlsq.result/v1",
        "parameters": {"D0": {"value": 1234.5, "uncertainty": 1.0}, "alpha": {"value": 0.9, "uncertainty": None}},
        "metadata": {
            "success": True,
            "convergence_status": "converged",
            "chi_squared": 12.5,
            "reduced_chi_squared": 1.04,
            "quality_flag": "good",
            **meta,
        },
    }
    (dir_path / "nlsq_result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_result_summary_extracts_fields(tmp_path):
    _write_result(tmp_path)
    s = load_result_summary(tmp_path)
    assert isinstance(s, ResultSummary)
    assert s.success is True
    assert s.convergence_status == "converged"
    assert s.chi_squared == 12.5
    assert s.reduced_chi_squared == 1.04
    assert s.quality_flag == "good"
    assert s.parameters == {"D0": 1234.5, "alpha": 0.9}
    assert s.result_dir == tmp_path


def test_load_result_summary_missing_file_returns_none(tmp_path):
    assert load_result_summary(tmp_path) is None


def test_load_result_summary_corrupt_json_returns_none(tmp_path):
    (tmp_path / "nlsq_result.json").write_text("{not json", encoding="utf-8")
    assert load_result_summary(tmp_path) is None


def test_load_result_summary_extracts_diagnostics(tmp_path):
    _write_result(tmp_path, nlsq_diagnostics={"hierarchical_active": True})
    s = load_result_summary(tmp_path)
    assert s is not None
    assert s.diagnostics == {"hierarchical_active": True}


def test_load_result_summary_diagnostics_defaults_empty(tmp_path):
    _write_result(tmp_path)
    s = load_result_summary(tmp_path)
    assert s is not None
    assert s.diagnostics == {}
