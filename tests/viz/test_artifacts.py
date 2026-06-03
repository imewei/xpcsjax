"""Tests for NPZ + JSON artifact serialization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import _save_fit_artifacts, _write_npz_compressed


def _sample_arrays() -> dict[str, Any]:
    n_phi, n = 4, 32
    rng = np.random.default_rng(0)
    return {
        "c2_exp": rng.random((n_phi, n, n)) + 1.0,
        "c2_fitted": rng.random((n_phi, n, n)) + 1.0,
        "residuals": rng.normal(0, 0.05, (n_phi, n, n)),
        "phi_angles": np.array([0.0, 45.0, 90.0, 135.0]),
        "t1": np.arange(n) * 0.1,
        "t2": np.arange(n) * 0.1,
        "q": np.float64(0.0054),
        "params": np.array([100.0, -0.5, 0.0]),
        "contrast": np.float64(0.2),
        "offset": np.float64(1.0),
        "reduced_chi_squared": np.float64(0.906),
    }


def test_write_npz_lzma_roundtrip(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    out = tmp_path / "fit.npz"
    _write_npz_compressed(out, arrays, compression="lzma")
    assert out.exists()
    loaded = np.load(out, allow_pickle=False)
    for key, val in arrays.items():
        np.testing.assert_array_equal(loaded[key], val)


def test_write_npz_lzma_uses_lzma_zipmethod(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    out = tmp_path / "fit.npz"
    _write_npz_compressed(out, arrays, compression="lzma")
    with zipfile.ZipFile(out, "r") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_LZMA


def test_write_npz_deflate_compression(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    out = tmp_path / "fit.npz"
    _write_npz_compressed(out, arrays, compression="deflate")
    loaded = np.load(out, allow_pickle=False)
    np.testing.assert_array_equal(loaded["c2_exp"], arrays["c2_exp"])
    with zipfile.ZipFile(out, "r") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED


def test_write_npz_none_compression(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    out = tmp_path / "fit.npz"
    _write_npz_compressed(out, arrays, compression="none")
    loaded = np.load(out, allow_pickle=False)
    np.testing.assert_array_equal(loaded["c2_exp"], arrays["c2_exp"])


def test_write_npz_invalid_compression_raises() -> None:
    with pytest.raises(ValueError, match="compression"):
        _write_npz_compressed(Path("/tmp/x.npz"), {}, compression="brotli")  # type: ignore[arg-type]


def test_write_npz_atomic_no_partial_on_failure(tmp_path: Path) -> None:
    """If writing fails mid-stream, no stale .npz or .tmp should remain."""
    out = tmp_path / "fit.npz"
    bad_arrays = {"c2_exp": np.array([1, 2, 3]), "bad": object()}
    with pytest.raises((TypeError, ValueError, AttributeError)):
        _write_npz_compressed(out, bad_arrays, compression="lzma")
    assert not out.exists()
    assert not list(tmp_path.glob("*.tmp"))


def _sample_artifact_inputs(tmp_path: Path) -> dict[str, Any]:
    n_phi, n = 4, 32
    rng = np.random.default_rng(0)
    return dict(
        c2_exp=rng.random((n_phi, n, n)) + 1.0,
        c2_fitted=rng.random((n_phi, n, n)) + 1.0,
        residuals=rng.normal(0, 0.05, (n_phi, n, n)),
        phi_angles=np.array([0.0, 45.0, 90.0, 135.0]),
        t1=np.arange(n) * 0.1,
        t2=np.arange(n) * 0.1,
        q=0.0054,
        L=2_000_000.0,
        dt=0.1,
        params=np.array([100.0, -0.5, 0.0]),
        uncertainties=np.array([5.0, 0.05, 0.01]),
        parameter_names=["D0", "alpha", "D_offset"],
        contrast=0.2,
        offset=1.0,
        reduced_chi_squared=0.906,
        convergence_status="converged",
        iterations=42,
        execution_time=1.234,
        analysis_mode="static_isotropic",
        output_dir=tmp_path,
    )


def test_save_artifacts_writes_npz_and_json(tmp_path: Path) -> None:
    _save_fit_artifacts(**_sample_artifact_inputs(tmp_path))
    assert (tmp_path / "c2_fitted_data.npz").exists()
    assert (tmp_path / "simulation_config_fitted.json").exists()


def test_save_artifacts_npz_schema(tmp_path: Path) -> None:
    _save_fit_artifacts(**_sample_artifact_inputs(tmp_path))
    loaded = np.load(tmp_path / "c2_fitted_data.npz", allow_pickle=False)
    expected = {
        "c2_exp",
        "c2_fitted",
        "residuals",
        "phi_angles",
        "t1",
        "t2",
        "q",
        "params",
        "contrast",
        "offset",
        "reduced_chi_squared",
    }
    assert set(loaded.files) == expected


def test_save_artifacts_json_schema(tmp_path: Path) -> None:
    _save_fit_artifacts(**_sample_artifact_inputs(tmp_path))
    with open(tmp_path / "simulation_config_fitted.json") as f:
        meta = json.load(f)
    assert set(meta.keys()) == {"fit", "physics", "data"}
    p = meta["fit"]["parameters"]
    assert set(p.keys()) == {"values", "uncertainties", "names"}
    assert len(p["values"]) == len(p["uncertainties"]) == len(p["names"])
    assert meta["fit"]["reduced_chi_squared"] == pytest.approx(0.906)
    assert meta["fit"]["convergence_status"] == "converged"
    assert meta["fit"]["iterations"] == 42
    assert meta["physics"]["q_value_angstrom_inv"] == pytest.approx(0.0054)
    assert "q_list" not in meta["physics"]
    assert meta["physics"]["analysis_mode"] == "static_isotropic"
    assert meta["data"]["n_phi"] == 4
    assert meta["data"]["n_t1"] == 32 and meta["data"]["n_t2"] == 32


def test_save_artifacts_creates_output_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    _save_fit_artifacts(**_sample_artifact_inputs(nested))
    assert (nested / "c2_fitted_data.npz").exists()


def test_save_artifacts_lzma_fallback_on_memoryerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xpcsjax.viz import nlsq_plots as mod

    original = mod._write_npz_compressed
    call_log = []

    def flaky(path, arrays, *, compression):
        call_log.append(compression)
        if compression == "lzma":
            raise MemoryError("simulated LZMA OOM")
        return original(path, arrays, compression=compression)

    monkeypatch.setattr(mod, "_write_npz_compressed", flaky)
    _save_fit_artifacts(**_sample_artifact_inputs(tmp_path), compression="lzma")
    assert call_log == ["lzma", "deflate"]
    assert (tmp_path / "c2_fitted_data.npz").exists()
