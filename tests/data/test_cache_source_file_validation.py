"""Regression test for review finding A2 (2026-09-15).

The NPZ cache was keyed only on frame window + q-vector, not on the source
HDF5 file itself. Replacing the source file (or pointing ``data_file`` at a
same-named/sibling file with a matching frame window) would silently serve a
stale cache. ``_save_to_cache``/`_validate_cache_q_vector`` now record and
check ``source_file``/``source_size``/``source_mtime_ns``.
"""

from __future__ import annotations

import pytest

from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader


def _loader(tmp_path, data_file: str) -> XPCSDataLoader:
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"scattering": {"wavevector_q": 0.05}}
    loader.config = {}
    loader.exp_config = {
        "data_folder_path": str(tmp_path),
        "data_file_name": data_file,
    }
    return loader


def _base_metadata(loader: XPCSDataLoader) -> dict:
    stat = loader._source_hdf_stat()
    assert stat is not None
    name, size, mtime_ns = stat
    return {
        "config_wavevector_q": 0.05,
        "selective_q_caching": True,
        "source_file": name,
        "source_size": size,
        "source_mtime_ns": mtime_ns,
    }


def test_source_stat_none_without_existing_file(tmp_path):
    loader = _loader(tmp_path, "missing.h5")
    assert loader._source_hdf_stat() is None


def test_source_stat_matches_written_file(tmp_path):
    src = tmp_path / "run.h5"
    src.write_bytes(b"abc")
    loader = _loader(tmp_path, "run.h5")
    name, size, mtime_ns = loader._source_hdf_stat()
    assert name == "run.h5"
    assert size == 3
    assert mtime_ns == src.stat().st_mtime_ns


def test_replaced_source_file_is_rejected(tmp_path):
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)

    # Simulate the source file being replaced/re-reduced with new content.
    src.write_bytes(b"different content, different size")

    with pytest.raises(XPCSDataFormatError, match="source-file mismatch"):
        loader._validate_cache_q_vector(metadata)


def test_unchanged_source_file_is_accepted(tmp_path):
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)

    # No raise: source file untouched.
    loader._validate_cache_q_vector(metadata)


def test_missing_source_metadata_warns_not_raises(tmp_path, caplog: pytest.LogCaptureFixture):
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")

    with caplog.at_level("WARNING"):
        loader._validate_cache_q_vector({"config_wavevector_q": 0.05, "selective_q_caching": True})
    assert "predates source-file fingerprinting" in caplog.text
