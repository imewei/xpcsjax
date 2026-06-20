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
    # Use a NON-symmetric lower-triangular half (distinct values) so that symmetry
    # of the result is only achievable by a correct c2_half + c2_half.T reconstruction
    # (the raw half is asymmetric), not vacuously true.
    half = np.tril(np.arange(48 * 48, dtype=float).reshape(48, 48))
    assert not np.allclose(half, half.T)  # sanity: the stored half is asymmetric
    with h5py.File(p, "w") as f:
        g = f.create_group("xpcs/twotime/correlation_map")
        for k in ("c2_00001", "c2_00002"):
            g.create_dataset(k, data=half)
    img = read_c2_preview(p, "unused", data_type="aps_u", phi_index=1, max_dim=32)
    assert img is not None and img.ndim == 2 and max(img.shape) <= 32
    # The reconstructed two-time matrix must be symmetric (block-mean rasterization
    # with equal row/col strides preserves symmetry).
    np.testing.assert_allclose(img, img.T)


def test_c2_preview_nested_group_at_key_returns_none(tmp_path):
    # A malformed file where a C2 group key holds a nested GROUP (not a Dataset)
    # must yield None — not raise AttributeError on Group.ndim.
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("xpcs/twotime/correlation_map")  # aps_u layout group
        g.create_group("c2_00001")  # nested group where a 2-D dataset is expected
    assert read_c2_preview(p, "unused", data_type="aps_u", phi_index=0) is None


def test_c2_preview_unknown_type_group_path_returns_none(tmp_path):
    # The data_type=None best-effort branch reads f[dataset] directly; if that path
    # resolves to a GROUP rather than a Dataset, it must return None, not raise.
    p = tmp_path / "grp.h5"
    with h5py.File(p, "w") as f:
        f.create_group("some/group")
    assert read_c2_preview(p, "some/group") is None
