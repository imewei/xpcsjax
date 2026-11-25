"""Red tests for Codex review round 1, shard data (claims data-B1..B18)."""

import gc

import h5py
import numpy as np
import pytest

from xpcsjax.data import load_xpcs_data
from xpcsjax.data.angle_filtering import apply_angle_filtering_for_optimization
from xpcsjax.data.memory_manager import AdvancedMemoryManager
from xpcsjax.data.phi_filtering import filter_phi_angles
from xpcsjax.data.quality_controller import DataQualityController, QualityControlStage
from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader

Q = 0.005


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # Loader side effects (text files, reports) must land in tmp_path.
    monkeypatch.chdir(tmp_path)


def _write_aps_u(path, halves, phis):
    """One-q APS-U file; ``halves[i]`` is the stored half matrix of phi bin i."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "xpcs/twotime/processed_bins", data=np.arange(1, len(halves) + 1, dtype=np.int64)
        )
        f.create_dataset("xpcs/qmap/dynamic_v_list_dim0", data=np.array([Q]))
        f.create_dataset("xpcs/qmap/dynamic_v_list_dim1", data=np.asarray(phis, dtype=float))
        grp = f.create_group("xpcs/twotime/correlation_map")
        for i, half in enumerate(halves):
            grp.create_dataset(f"c2_{i + 1:05d}", data=half)


def _bare_loader(config=None):
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {}
    loader.config = config or {}
    loader.load_degradations = []
    return loader


def _half(value, n_t=4, dtype=np.float64):
    return np.triu(np.full((n_t, n_t), value, dtype=dtype))


def _hdf_config(tmp_path, name, **extra):
    cfg = {
        "experimental_data": {"data_folder_path": str(tmp_path), "data_file_name": name},
        "analyzer_parameters": {
            "dt": 1.0,
            "start_frame": 1,
            "end_frame": 4,
            "scattering": {"wavevector_q": Q},
        },
        "quality_control": {"enabled": False},
    }
    cfg.update(extra)
    return cfg


def _qc_data():
    return {
        "c2_exp": np.ones((1, 4, 4), dtype=np.float64) * 1.5,
        "t1": np.arange(4, dtype=np.float64),
        "t2": np.arange(4, dtype=np.float64),
        "wavevector_q_list": np.array([0.01]),
        "phi_angles_list": np.array([0.0]),
    }


# data-B1 reclassified DESIGN by the lead (q documented "preserved unchanged"; slicing changes the fitted q); test removed.


def test_data_b2(tmp_path):
    path = tmp_path / "b2.h5"
    _write_aps_u(path, [_half(1.2), np.array([[1.5]])], [0.0, 90.0])
    with pytest.raises(XPCSDataFormatError):
        _bare_loader()._load_from_hdf(str(path))


def test_data_b3(tmp_path):
    path = tmp_path / "b3.h5"
    _write_aps_u(path, [_half(1.2, dtype=np.float32), _half(1.0000000001)], [0.0, 90.0])
    try:
        out = _bare_loader()._load_from_hdf(str(path))
    except XPCSDataFormatError:
        return  # rejecting a mixed-dtype file is an acceptable fix; silent narrowing is not
    assert np.asarray(out["c2_exp"]).shape == (2, 4, 4)  # control
    assert float(np.asarray(out["c2_exp"])[1, 0, 1]) == 1.0000000001


def test_data_b4(tmp_path):
    path = tmp_path / "b4.h5"
    _write_aps_u(path, [_half(2, dtype=np.int32)], [0.0])
    out = _bare_loader()._load_from_hdf(str(path))
    assert float(np.asarray(out["c2_exp"])[0, 0, 1]) == 2
    assert float(np.asarray(out["c2_exp"])[0, 0, 0]) == 2


def test_data_b5(tmp_path):
    path = tmp_path / "b5.h5"
    _write_aps_u(path, [_half(0.1), _half(1.2)], [0.0, 90.0])
    loader = _bare_loader({"data_filtering": {"enabled": True, "quality_threshold": 0.9}})
    out = loader._load_from_hdf(str(path))
    assert np.asarray(out["phi_angles_list"]).tolist() == [90.0]


def test_data_b6(tmp_path):
    path = tmp_path / "b6.h5"
    _write_aps_u(path, [_half(0.1), _half(1.2)], [0.0, 90.0])
    loader = _bare_loader(
        {
            "data_filtering": {
                "enabled": True,
                "quality_filtering": {"enabled": True},
                "quality_threshold": 0.9,
                "combine_criteria": "OR",
                "phi_range": {"min": -5, "max": 5},
            }
        }
    )
    out = loader._load_from_hdf(str(path))
    assert np.asarray(out["phi_angles_list"]).tolist() == [0.0, 90.0]


def test_data_b7(tmp_path):
    path = tmp_path / "b7.h5"
    _write_aps_u(path, [_half(1.2)], [0.0])
    cfg = _hdf_config(tmp_path, "b7.h5")
    cfg["experimental_data"]["cache_filename_template"] = "fixed.npz"
    cfg["analyzer_parameters"]["end_frame"] = 2
    first = load_xpcs_data(config_dict=cfg)
    assert first.c2.shape[-2:] == (2, 2)  # control: window applied
    assert (tmp_path / "fixed.npz").exists()  # control: fixed-name cache written
    cfg["analyzer_parameters"]["end_frame"] = 4
    second = load_xpcs_data(config_dict=cfg)
    assert second.c2.shape[-2:] == (4, 4)


def test_data_b8(tmp_path):
    path = tmp_path / "b8.h5"
    _write_aps_u(path, [_half(1.2)], [0.0])
    cfg = _hdf_config(
        tmp_path,
        "b8.h5",
        data_filtering={
            "enabled": True,
            "phi_range": {"min": 85, "max": 95},
            "fallback_on_empty": True,
        },
    )
    cold = load_xpcs_data(config_dict=cfg)
    assert cold["_degraded"] is True  # control: fallback recorded on the cold load
    assert list(tmp_path.glob("*.npz"))  # control: cache was written
    warm = load_xpcs_data(config_dict=cfg)
    assert np.asarray(warm["phi_angles_list"]).tolist() == [0.0]  # control: same fallback rows
    assert warm["_degraded"] is True


def test_data_b9(tmp_path):
    # Only the use_caching half is asserted; the cache_directory half is a DESIGN call.
    path = tmp_path / "b9.h5"
    _write_aps_u(path, [_half(1.2)], [0.0])
    cfg = _hdf_config(tmp_path, "b9.h5")
    load_xpcs_data(config_dict=cfg)
    cached = list(tmp_path.rglob("*.npz"))
    assert cached  # control: caching is on by default
    for p in cached:
        p.unlink()

    cfg["experimental_data"]["use_caching"] = False
    load_xpcs_data(config_dict=cfg)
    assert not list(tmp_path.rglob("*.npz"))


def test_data_b10(tmp_path):
    p = tmp_path / "source.npz"
    t = np.arange(6, dtype=np.float64)
    c2 = np.stack([1.0 + 0.3 * np.exp(-np.abs(t[:, None] - t[None, :]) / 2.0)] * 3)
    np.savez(
        p,
        c2_exp=c2,
        t1=t,
        t2=t,
        wavevector_q_list=np.full(3, 0.0237),
        phi_angles_list=np.linspace(0.0, 144.0, 3),
    )
    data = load_xpcs_data(
        {
            "experimental_data": {"file_path": str(p)},
            "analyzer_parameters": {"dt": 1.0, "start_frame": 1, "end_frame": 6},
        }
    )
    assert data.c2.shape == (3, 6, 6)


def test_data_b11(tmp_path):
    p = tmp_path / "mismatch.npz"
    np.savez(
        p,
        c2_exp=np.ones((1, 4, 4)),
        t1=np.arange(4.0),
        t2=np.arange(4.0),
        wavevector_q_list=np.array([0.01, 0.01]),
        phi_angles_list=np.array([0.0, 90.0]),
    )
    cfg = _hdf_config(tmp_path, "mismatch.npz")
    with pytest.raises(XPCSDataFormatError):
        load_xpcs_data(config_dict=cfg)


def test_data_b12():
    idx, _ = filter_phi_angles([0.0, 90.0, 180.0], target_ranges=[(0.0, 360.0)])
    assert list(idx) == [0, 1, 2]


def test_data_b13():
    import jax.numpy as jnp

    cfg = {"quality_control": {"enabled": True, "pass_threshold": 95}}
    d_np = _qc_data()
    r_np = DataQualityController(cfg).validate_data_stage(d_np, QualityControlStage.FINAL_DATA)
    assert r_np.passed  # control: the same data passes as NumPy

    d_jax = _qc_data()
    d_jax["c2_exp"] = jnp.asarray(d_jax["c2_exp"])
    r_jax = DataQualityController(cfg).validate_data_stage(d_jax, QualityControlStage.FINAL_DATA)
    assert r_jax.metrics.symmetry_score == r_np.metrics.symmetry_score
    assert r_jax.metrics.overall_score == r_np.metrics.overall_score
    assert r_jax.passed


def test_data_b16(tmp_path):
    path = tmp_path / "b16.h5"
    _write_aps_u(path, [_half(1.2)], [0.0])
    cfg = _hdf_config(
        tmp_path,
        "b16.h5",
        preprocessing={
            "enabled": True,
            "stages": {
                "reduce_noise": {
                    "enabled": True,
                    "method": "savgol",
                    "window_length": 3,
                    "polyorder": 4,
                }
            },
        },
    )
    assert load_xpcs_data(config_dict=cfg)["_degraded"] is True


def test_data_b18():
    before = gc.get_threshold()
    mgr = AdvancedMemoryManager({"memory": {"enable_monitoring": False, "gc_optimization": False}})
    try:
        host = (before[0] + 123, before[1], before[2])
        gc.set_threshold(*host)  # host application changes its own thresholds
        mgr.shutdown()
        assert gc.get_threshold() == host
    finally:
        gc.set_threshold(*before)
