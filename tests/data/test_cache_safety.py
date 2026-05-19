"""Trust-boundary regression tests for :meth:`XPCSDataLoader._load_from_cache`.

NPZ cache files live in config-controlled paths (``data_folder_path``,
``cache_file_path``, …), so the loader must treat them as untrusted input.
``allow_pickle=False`` blocks object deserialization, metadata is read from a
JSON-encoded scalar, and legacy object-serialized ``cache_metadata`` must be
refused — not transparently re-deserialized.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _bare_loader() -> XPCSDataLoader:
    """Bypass __init__ to avoid needing a YAML config on disk."""
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"scattering": {"wavevector_q": 0.0054}}
    loader.exp_config = {"cache_compression": False}
    return loader


def _good_payload() -> dict[str, np.ndarray]:
    return {
        "c2_exp": np.zeros((1, 4, 4), dtype=np.float64),
        "t1": np.arange(4, dtype=np.float64),
        "t2": np.arange(4, dtype=np.float64),
        "wavevector_q_list": np.array([0.0054], dtype=np.float64),
        "phi_angles_list": np.array([0.0], dtype=np.float64),
    }


def test_round_trip_with_json_metadata(tmp_path: Path):
    """Writer emits cache_metadata_json; reader loads it with allow_pickle=False."""
    loader = _bare_loader()
    cache_path = str(tmp_path / "good.npz")

    loader._save_to_cache(_good_payload(), cache_path)

    with np.load(cache_path, allow_pickle=False) as f:
        assert "cache_metadata_json" in f.files
        assert "cache_metadata" not in f.files
        metadata_text = str(np.asarray(f["cache_metadata_json"]).item())
        metadata = json.loads(metadata_text)
        assert metadata["config_wavevector_q"] == pytest.approx(0.0054)
        assert metadata["q_count"] == 1

    loaded = loader._load_from_cache(cache_path)
    assert loaded["c2_exp"].shape == (1, 4, 4)
    assert loaded["wavevector_q_list"].tolist() == [pytest.approx(0.0054)]


def test_legacy_object_metadata_is_refused(tmp_path: Path):
    """A pre-fix .npz with object-serialized cache_metadata must NOT load.

    Loading would require unsafe object deserialization from a
    config-controlled file. The loader surfaces a clear, actionable error
    instead.
    """
    cache_path = str(tmp_path / "legacy.npz")
    np.savez(
        cache_path,
        c2_exp=np.zeros((1, 4, 4), dtype=np.float64),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        wavevector_q_list=np.array([0.0054], dtype=np.float64),
        phi_angles_list=np.array([0.0], dtype=np.float64),
        cache_metadata=np.array(
            {"config_wavevector_q": 0.0054, "q_count": 1}, dtype=object
        ),
    )

    loader = _bare_loader()
    with pytest.raises(ValueError, match=r"legacy 'cache_metadata'"):
        loader._load_from_cache(cache_path)


def test_object_dtype_under_data_key_is_refused(tmp_path: Path):
    """A malicious .npz with an object-dtype data array must NOT load."""
    cache_path = str(tmp_path / "evil.npz")
    np.savez(
        cache_path,
        c2_exp=np.array([{"hack": "you"}], dtype=object),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        wavevector_q_list=np.array([0.0054], dtype=np.float64),
        phi_angles_list=np.array([0.0], dtype=np.float64),
        cache_metadata_json=np.asarray(
            json.dumps({"config_wavevector_q": 0.0054, "q_count": 1})
        ),
    )

    loader = _bare_loader()
    with pytest.raises(ValueError, match=r"object-dtype array under a data key"):
        loader._load_from_cache(cache_path)


def test_corrupt_metadata_json_is_rejected_clearly(tmp_path: Path):
    """A cache with non-JSON ``cache_metadata_json`` must error before validation."""
    cache_path = str(tmp_path / "corrupt_json.npz")
    np.savez(
        cache_path,
        c2_exp=np.zeros((1, 4, 4), dtype=np.float64),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        wavevector_q_list=np.array([0.0054], dtype=np.float64),
        phi_angles_list=np.array([0.0], dtype=np.float64),
        cache_metadata_json=np.asarray("not { valid json"),
    )

    loader = _bare_loader()
    with pytest.raises(ValueError, match=r"malformed cache_metadata_json"):
        loader._load_from_cache(cache_path)


def test_non_object_metadata_json_is_rejected(tmp_path: Path):
    """JSON that decodes to something other than a dict must be rejected."""
    cache_path = str(tmp_path / "wrong_type.npz")
    np.savez(
        cache_path,
        c2_exp=np.zeros((1, 4, 4), dtype=np.float64),
        t1=np.arange(4, dtype=np.float64),
        t2=np.arange(4, dtype=np.float64),
        wavevector_q_list=np.array([0.0054], dtype=np.float64),
        phi_angles_list=np.array([0.0], dtype=np.float64),
        cache_metadata_json=np.asarray(json.dumps(["not", "a", "dict"])),
    )

    loader = _bare_loader()
    with pytest.raises(ValueError, match=r"must encode a JSON object"):
        loader._load_from_cache(cache_path)
