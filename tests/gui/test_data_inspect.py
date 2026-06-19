import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from xpcsjax.gui.data_inspect import DatasetInfo, read_c2_preview, read_hdf5_metadata  # noqa: E402


def _make_h5(path):
    with h5py.File(path, "w") as f:
        f.attrs["detector"] = "test"
        f.create_dataset("c2", data=np.random.default_rng(0).random((2, 64, 64)))
        f.create_dataset("phi", data=np.array([0.0, 45.0]))


def test_metadata_lists_datasets_without_loading(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    infos = read_hdf5_metadata(p)
    names = {i.name.split("/")[-1]: i for i in infos}
    assert isinstance(infos[0], DatasetInfo)
    assert names["c2"].shape == (2, 64, 64)
    assert names["phi"].shape == (2,)


def test_c2_preview_slices_and_downsamples(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    img = read_c2_preview(p, "c2", phi_index=1, max_dim=32)
    assert img is not None and img.ndim == 2
    assert max(img.shape) <= 32


def test_c2_preview_missing_dataset_returns_none(tmp_path):
    p = tmp_path / "d.h5"
    _make_h5(p)
    assert read_c2_preview(p, "does_not_exist") is None


def test_c2_preview_missing_group_returns_none(tmp_path):
    # Spec §6: a known data_type whose C₂ group is absent yields "preview
    # unavailable" (None), never a guessed render.
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("c2", data=np.zeros((64, 64)))  # no exchange/C2T_all group
    assert read_c2_preview(p, "c2", data_type="aps_old") is None


def test_c2_preview_reconstructs_group_half_matrix(tmp_path):
    # Real formats store C₂ as a GROUP of per-angle 2-D half-matrices; the preview
    # reconstructs c2_half + c2_half.T (diag halved), mirroring the loader.
    p = tmp_path / "u.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("xpcs/twotime/correlation_map")
        for k in ("c2_00001", "c2_00002"):
            g.create_dataset(k, data=np.tril(np.ones((48, 48))))
    img = read_c2_preview(p, "unused", data_type="aps_u", phi_index=1, max_dim=32)
    assert img is not None and img.ndim == 2 and max(img.shape) <= 32
