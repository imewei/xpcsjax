"""Regression test for review finding A1 (2026-09-15).

``_load_aps_u_format`` used to *skip* a correlation matrix on an
out-of-range ``bin_idx`` with only a ``logger.warning`` while leaving its
(q,phi) pair in place, silently shifting every later matrix by one slot
against ``filtered_dqlist``/``filtered_dphilist``. A file whose
``processed_bins`` mapping disagrees with the ``correlation_map`` group must
now fail loudly instead of mislabelling data.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader


def _write_aps_u_file(path, *, n_processed_bins: int, n_matrices: int) -> None:
    """Minimal APS-U HDF5 file: n_phi=1 so bin_idx == q_idx directly."""
    n_t = 4
    with h5py.File(path, "w") as f:
        # processed_bins are 1-based; with n_phi=1, bin_idx = processed_bin - 1
        # must land inside [0, n_processed_bins) for the (q,phi) grid mapping.
        f.create_dataset(
            "xpcs/twotime/processed_bins",
            data=np.arange(1, n_processed_bins + 1, dtype=np.int64),
        )
        f.create_dataset(
            "xpcs/qmap/dynamic_v_list_dim0",
            data=np.linspace(0.01, 0.02, n_processed_bins),
        )
        f.create_dataset("xpcs/qmap/dynamic_v_list_dim1", data=np.array([0.0]))
        grp = f.create_group("xpcs/twotime/correlation_map")
        for i in range(n_matrices):
            grp.create_dataset(f"c2_{i + 1:05d}", data=np.eye(n_t))


def _loader() -> XPCSDataLoader:
    return XPCSDataLoader.__new__(XPCSDataLoader)


def test_bin_skip_raises_instead_of_misaligning(tmp_path):
    """Fewer correlation matrices than processed_bins entries must raise."""
    path = tmp_path / "aps_u.h5"
    _write_aps_u_file(path, n_processed_bins=3, n_matrices=2)
    loader = _loader()
    with pytest.raises(XPCSDataFormatError, match="exceeds available matrices"):
        loader._load_aps_u_format(str(path))


def test_matched_bins_and_matrices_load_cleanly(tmp_path):
    """Sanity check: a consistent file still loads past the guarded loop."""
    path = tmp_path / "aps_u_ok.h5"
    _write_aps_u_file(path, n_processed_bins=2, n_matrices=2)
    loader = _loader()
    # Consistent counts must not raise inside the matrix-loading loop; the
    # method continues into selection code that needs a real config, so we
    # only assert it gets past the guarded block under test.
    with pytest.raises(AttributeError):
        # loader.config is unset (no __init__) - proves execution reached
        # past the fixed block into the (q,phi)-selection code, not the
        # XPCSDataFormatError this test targets.
        loader._load_aps_u_format(str(path))
