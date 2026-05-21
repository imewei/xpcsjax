"""Tests for NPZ + JSON artifact serialization."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.viz.nlsq_plots import _write_npz_compressed


def _sample_arrays() -> dict[str, np.ndarray]:
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
    loaded = np.load(out)
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
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded["c2_exp"], arrays["c2_exp"])
    with zipfile.ZipFile(out, "r") as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_DEFLATED


def test_write_npz_none_compression(tmp_path: Path) -> None:
    arrays = _sample_arrays()
    out = tmp_path / "fit.npz"
    _write_npz_compressed(out, arrays, compression="none")
    loaded = np.load(out)
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
