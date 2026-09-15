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
    """Bare loader with just enough state for the (q,phi)-selection code.

    D7 (2026-09-15) moved the optimal-q-vector selection (which reads
    ``self.analyzer_config``) and the data-filtering config lookup (which
    reads ``self.config``) ahead of the matrix-loading step, so both need to
    exist even to exercise the earlier out-of-range guard.
    """
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {}
    loader.config = {}
    return loader


def test_bin_skip_raises_instead_of_misaligning(tmp_path):
    """Fewer correlation matrices than processed_bins entries must raise.

    D7 (2026-09-15) made the loader selective: it only ever reads the
    matrices in the final (q,phi) selection, so the invalid/missing bin
    (index 2 of 3, q=0.02 -- the top of the linspace(0.01, 0.02, 3) range
    _write_aps_u_file lays out) must itself be the one selected, or the
    guard would never see it. Target that q directly instead of relying on
    the loader's unrelated hardcoded default.
    """
    path = tmp_path / "aps_u.h5"
    _write_aps_u_file(path, n_processed_bins=3, n_matrices=2)
    loader = _loader()
    loader.analyzer_config["scattering"] = {"wavevector_q": 0.02}
    with pytest.raises(XPCSDataFormatError, match="exceeds available matrices"):
        loader._load_aps_u_format(str(path))


def test_matched_bins_and_matrices_load_cleanly(tmp_path):
    """Sanity check: a consistent file loads all the way through cleanly."""
    path = tmp_path / "aps_u_ok.h5"
    _write_aps_u_file(path, n_processed_bins=2, n_matrices=2)
    loader = _loader()
    # Consistent counts must not raise inside the matrix-loading loop, and
    # every downstream default is sane enough for a full successful load —
    # proves execution reached (and passed) the fixed block under test.
    result = loader._load_aps_u_format(str(path))
    assert result["c2_exp"].shape[0] == 1
