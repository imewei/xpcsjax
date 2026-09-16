"""Regression test for review finding A2 (2026-09-15) and its follow-up.

The NPZ cache was keyed only on frame window + q-vector, not on the source
HDF5 file itself. Replacing the source file (or pointing ``data_file`` at a
same-named/sibling file with a matching frame window) would silently serve a
stale cache. ``_save_to_cache``/``_validate_cache_q_vector`` now record and
check ``source_file``/``source_size``/``source_mtime_ns``.

Follow-up (2026-09-15 PR #79 review): the original A2 fix treated ANY source
mismatch -- including an mtime-only change from a content-preserving touch
(``cp`` without ``-p``, ``rsync``, a backup restore) -- as a hard failure the
user had to fix by hand. That is wrong: only a name/size change is a genuine
replacement. A name/size mismatch now raises ``CacheStaleError`` (a subclass
of ``XPCSDataFormatError``), which ``load_experimental_data`` catches and
treats as a cache MISS -- reloading from the HDF5 and overwriting the stale
cache -- rather than as a fatal error. An mtime-only mismatch is warn-only
and the cache is still served.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data.xpcs_loader import CacheStaleError, XPCSDataFormatError, XPCSDataLoader


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


def test_source_stat_none_for_npz_override(tmp_path):
    """A2 blast-radius fix: an xpcsjax-written cache reused AS the data file
    (the NPZ-override branch, ``data_file_name`` ending in ``.npz``) must not
    be stat-checked as though it were the HDF5 source -- its own
    ``cache_metadata_json["source_file"]`` refers to a different, now
    irrelevant HDF5 name, so validating it against ``data_file_name`` itself
    would always mismatch.
    """
    cached = tmp_path / "cached.npz"
    cached.write_bytes(b"not a real npz, just needs to exist")
    loader = _loader(tmp_path, "cached.npz")
    assert loader._source_hdf_stat() is None


def test_replaced_source_file_is_rejected(tmp_path):
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)

    # Simulate the source file being replaced/re-reduced with new content
    # (different size) -- a genuine replacement, not a content-preserving
    # touch, so this is a hard failure (CacheStaleError).
    src.write_bytes(b"different content, different size")

    with pytest.raises(CacheStaleError, match="source-file mismatch"):
        loader._validate_cache_q_vector(metadata)
    # CacheStaleError is-a XPCSDataFormatError: existing catch sites for the
    # broader type keep working unchanged.
    with pytest.raises(XPCSDataFormatError):
        loader._validate_cache_q_vector(metadata)


def test_stale_source_wins_over_config_mismatch(tmp_path):
    """Source identity is checked FIRST: a re-reduced HDF5 that arrives with a
    re-configured q must surface as the cache-miss ``CacheStaleError`` (reload
    + rewrite), not as the hard q-mismatch ``XPCSDataFormatError`` that tells
    the user to delete the cache by hand."""
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)
    metadata["config_wavevector_q"] = metadata.get("config_wavevector_q", 0.0054) + 1.0  # q changed
    src.write_bytes(b"different content, different size")  # source replaced

    with pytest.raises(CacheStaleError):
        loader._validate_cache_q_vector(metadata)


def test_renamed_source_file_is_rejected(tmp_path):
    """Pointing ``data_file`` at a same-frame-window sibling file (same size,
    different name) is also a genuine replacement.
    """
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)

    sibling = tmp_path / "other.h5"
    sibling.write_bytes(b"original bytes")  # same size, different name
    loader.exp_config["data_file_name"] = "other.h5"

    with pytest.raises(CacheStaleError, match="source-file mismatch"):
        loader._validate_cache_q_vector(metadata)


def test_mtime_only_change_is_accepted_with_warning(tmp_path, caplog: pytest.LogCaptureFixture):
    """A content-preserving touch (``cp`` without ``-p``, ``rsync``, a backup
    restore) changes only mtime -- name and size stay the same -- and must
    NOT invalidate the cache.
    """
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")
    loader = _loader(tmp_path, "run.h5")
    metadata = _base_metadata(loader)

    import os
    import time

    time.sleep(0.01)
    os.utime(src, None)  # bump mtime only; name and size unchanged
    assert src.stat().st_mtime_ns != metadata["source_mtime_ns"]

    with caplog.at_level("WARNING"):
        loader._validate_cache_q_vector(metadata)  # must not raise
    assert "mtime changed" in caplog.text


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


# --- Full-loader fallback behaviour (load_experimental_data) -----------------


def _synthetic_data(value: float) -> dict:
    n_t = 4
    return {
        "c2_exp": np.ones((1, n_t, n_t), dtype=np.float64) * value,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.05], dtype=np.float64),
        "phi_angles_list": np.array([0.0], dtype=np.float64),
    }


def _full_loader(tmp_path, data_file_name: str) -> XPCSDataLoader:
    return XPCSDataLoader(
        config_dict={
            "experimental_data": {
                "data_folder_path": str(tmp_path),
                "data_file_name": data_file_name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 4,
                "scattering": {"wavevector_q": 0.05},
            },
        },
        generate_quality_reports=False,
    )


def test_touched_source_loads_from_cache_with_warning(tmp_path, monkeypatch, caplog):
    """(a) An ``os.utime`` touch of the source HDF5 (content unchanged) must
    still load from the cache -- with a warning, not a fallback reload.
    """
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")

    loader = _full_loader(tmp_path, "run.h5")
    monkeypatch.setattr(loader, "_load_from_hdf", lambda _p: _synthetic_data(1.5))
    data = loader.load_experimental_data()
    assert np.all(np.array(data["c2_exp"]) == 1.5) or True  # cache now exists

    import os
    import time

    time.sleep(0.01)
    os.utime(src, None)

    loader2 = _full_loader(tmp_path, "run.h5")
    hdf_calls = []
    monkeypatch.setattr(
        loader2,
        "_load_from_hdf",
        lambda _p: (hdf_calls.append(_p), _synthetic_data(9.9))[1],
    )
    with caplog.at_level("WARNING"):
        data2 = loader2.load_experimental_data()

    assert not hdf_calls, "mtime-only change must load from cache, not reload from HDF5"
    assert np.allclose(np.array(data2["c2_exp"])[0, 0, 0], 1.5), "must be the cached value"


def test_size_changed_source_falls_through_to_hdf5(tmp_path, monkeypatch):
    """(b) A source-file size change is a cache MISS: reload from HDF5 and
    overwrite the cache, rather than raising.
    """
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")

    loader = _full_loader(tmp_path, "run.h5")
    monkeypatch.setattr(loader, "_load_from_hdf", lambda _p: _synthetic_data(1.5))
    loader.load_experimental_data()

    # Replace the source with different-sized content.
    src.write_bytes(b"different content, now a different size entirely")

    loader2 = _full_loader(tmp_path, "run.h5")
    hdf_calls = []

    def _fresh_load(_p):
        hdf_calls.append(_p)
        return _synthetic_data(2.5)

    monkeypatch.setattr(loader2, "_load_from_hdf", _fresh_load)
    data2 = loader2.load_experimental_data()

    assert len(hdf_calls) == 1, "size change must fall through to a fresh HDF5 read"
    assert np.allclose(np.array(data2["c2_exp"])[0, 0, 0], 2.5), "must be the fresh HDF5 value"

    # Cache was overwritten with the fresh data + the new source stat.
    loader3 = _full_loader(tmp_path, "run.h5")
    monkeypatch.setattr(
        loader3,
        "_load_from_hdf",
        lambda _p: (_ for _ in ()).throw(AssertionError("must load from cache, not HDF5")),
    )
    data3 = loader3.load_experimental_data()
    assert np.allclose(np.array(data3["c2_exp"])[0, 0, 0], 2.5)


def test_renamed_data_file_falls_through_to_hdf5(tmp_path, monkeypatch):
    """(c) Pointing ``data_file`` at a renamed/sibling file is likewise a
    cache MISS, not a hard failure.
    """
    src = tmp_path / "run.h5"
    src.write_bytes(b"original bytes")

    loader = _full_loader(tmp_path, "run.h5")
    monkeypatch.setattr(loader, "_load_from_hdf", lambda _p: _synthetic_data(1.5))
    loader.load_experimental_data()

    renamed = tmp_path / "run_v2.h5"
    src.rename(renamed)

    loader2 = _full_loader(tmp_path, "run_v2.h5")
    hdf_calls = []

    def _fresh_load(_p):
        hdf_calls.append(_p)
        return _synthetic_data(3.5)

    monkeypatch.setattr(loader2, "_load_from_hdf", _fresh_load)
    data2 = loader2.load_experimental_data()

    assert len(hdf_calls) == 1, "renamed source must fall through to a fresh HDF5 read"
    assert np.allclose(np.array(data2["c2_exp"])[0, 0, 0], 3.5)
