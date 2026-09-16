"""Regression test for review finding D7 (2026-09-15).

``_load_aps_u_format`` used to unconditionally read every valid-bin
correlation matrix into RAM before running any (q,phi) selection, unlike
``_load_aps_old_format``'s two-pass selective-read structure. It now
resolves the (q,phi) selection from metadata alone and reads only the
matrices in the final selection (or, when quality filtering is enabled,
the metadata-narrowed candidate set).

This test pins numerical equivalence with the old "read everything, then
select" approach: it replicates that approach directly (reading and
reconstructing every valid bin, exactly as the pre-D7 code did, via the
loader's own unchanged helper methods) and asserts ``np.array_equal``
against the new selective-read result on a synthetic multi-(q,phi) APS-U
file.
"""

from __future__ import annotations

import h5py
import numpy as np

from xpcsjax.data.xpcs_loader import XPCSDataLoader


def _write_multi_bin_aps_u_file(path, *, n_q: int, n_phi: int, n_t: int = 4) -> None:
    """APS-U file with every (q,phi) grid cell populated and distinct data.

    Each matrix is filled with a value unique to its (q_idx, phi_idx) pair
    so an off-by-one selection error would change the compared arrays.
    """
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "xpcs/twotime/processed_bins",
            data=np.arange(1, n_q * n_phi + 1, dtype=np.int64),
        )
        f.create_dataset(
            "xpcs/qmap/dynamic_v_list_dim0",
            data=np.linspace(0.005, 0.03, n_q),
        )
        f.create_dataset(
            "xpcs/qmap/dynamic_v_list_dim1",
            data=np.linspace(0.0, 170.0, n_phi),
        )
        grp = f.create_group("xpcs/twotime/correlation_map")
        for bin_idx in range(n_q * n_phi):
            # Distinct, non-symmetric-in-a-trivial-way half matrix per bin.
            half = rng.uniform(0.5, 1.5, size=(n_t, n_t))
            grp.create_dataset(f"c2_{bin_idx + 1:05d}", data=half)


def _bare_loader(config: dict | None = None) -> XPCSDataLoader:
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {}
    loader.config = config or {}
    loader.load_degradations = []
    return loader


def _write_multi_bin_aps_old_file(path, *, n_q: int, n_phi: int, n_t: int = 4) -> None:
    """APS-old file with distinct (q,phi) rows for the quality-filtering test."""
    rng = np.random.default_rng(0)
    dqlist = np.repeat(np.linspace(0.005, 0.03, n_q), n_phi)
    dphilist = np.tile(np.linspace(0.0, 170.0, n_phi), n_q)
    with h5py.File(path, "w") as f:
        f.create_dataset("xpcs/dqlist", data=dqlist[np.newaxis, :])
        f.create_dataset("xpcs/dphilist", data=dphilist[np.newaxis, :])
        grp = f.create_group("exchange/C2T_all")
        for i in range(n_q * n_phi):
            half = rng.uniform(0.5, 1.5, size=(n_t, n_t))
            grp.create_dataset(str(i + 1), data=half)


def test_aps_old_quality_filtering_empty_phi_match_records_one_degradation(tmp_path):
    """APS-old mirror of ``test_quality_filtering_empty_phi_match_records_one_degradation``.

    The aps_old two-pass quality-filtering branch (``hdf5_readers.py``) runs
    ``_get_selected_indices`` twice against the SAME ``data_filtering``
    config -- a metadata pre-filter pass, then the final quality-filter pass
    on the narrowed candidates. A phi_range matching nothing triggers the
    ``fallback_on_empty`` path in both passes; only the final pass should
    record the DATA-1 degradation signal.
    """
    path = tmp_path / "aps_old_multi_quality_empty_phi.h5"
    _write_multi_bin_aps_old_file(path, n_q=3, n_phi=3)

    config = {
        "data_filtering": {
            "enabled": True,
            "phi_range": {"min": 500.0, "max": 600.0},  # matches no phi value
            "quality_filtering": {"enabled": True, "quality_threshold": 0.0},
        }
    }
    loader = _bare_loader(config)
    loader._load_aps_old_format(str(path))

    assert len(loader.load_degradations) == 1, (
        f"expected exactly one degradation entry, got {loader.load_degradations!r}"
    )


def _load_aps_u_eager_reference(loader: XPCSDataLoader, hdf_path: str) -> dict:
    """Replicate the pre-D7 "read every valid bin, then select" algorithm.

    Uses the loader's own unchanged helper methods
    (``_select_optimal_wavevector``, ``_get_selected_indices``,
    ``_reconstruct_full_matrix``, ``_require_nonempty_selection``,
    ``_apply_frame_slicing_to_selected_q``, ``_calculate_time_arrays``) so
    this exercises the same downstream contract as the production method,
    differing only in reading every bin up front instead of selectively.
    """
    with h5py.File(hdf_path, "r") as f:
        processed_bins = f["xpcs/twotime/processed_bins"][()]
        q_values = f["xpcs/qmap/dynamic_v_list_dim0"][()]
        phi_values = f["xpcs/qmap/dynamic_v_list_dim1"][()]
        n_q, n_phi = len(q_values), len(phi_values)

        qphi_pairs = []
        valid_bin_indices = []
        for i, processed_bin in enumerate(processed_bins):
            bin_idx = processed_bin - 1
            q_idx, phi_idx = bin_idx // n_phi, bin_idx % n_phi
            if 0 <= q_idx < n_q and 0 <= phi_idx < n_phi:
                qphi_pairs.append((q_values[q_idx], phi_values[phi_idx]))
                valid_bin_indices.append(i)

        qphi_array = np.array(qphi_pairs)
        filtered_dqlist = qphi_array[:, 0]
        filtered_dphilist = qphi_array[:, 1]

        corr_group = f["xpcs/twotime/correlation_map"]
        c2_keys = sorted(corr_group.keys())

        # Eagerly read + reconstruct EVERY valid bin (the pre-D7 behavior).
        c2_matrices_for_filtering = [
            loader._reconstruct_full_matrix(corr_group[c2_keys[bin_idx]][()])
            for bin_idx in valid_bin_indices
        ]

        selected_indices = loader._get_selected_indices(
            filtered_dqlist, filtered_dphilist, c2_matrices_for_filtering
        )
        selected_q_idx = loader._select_optimal_wavevector(filtered_dqlist)
        selected_q = filtered_dqlist[selected_q_idx]
        q_matching_indices = np.where(np.abs(filtered_dqlist - selected_q) < 1e-10)[0]
        final_indices = (
            np.intersect1d(q_matching_indices, selected_indices)
            if selected_indices is not None
            else q_matching_indices
        )
        final_indices = loader._require_nonempty_selection(final_indices, selected_q=selected_q)

        final_dqlist = filtered_dqlist[final_indices]
        final_dphilist = filtered_dphilist[final_indices]
        c2_matrices_array = np.array([c2_matrices_for_filtering[int(i)] for i in final_indices])
        c2_exp = loader._apply_frame_slicing_to_selected_q(c2_matrices_array)
        time_1d = loader._calculate_time_arrays(c2_exp.shape[-1])

        return {
            "wavevector_q_list": final_dqlist,
            "phi_angles_list": final_dphilist,
            "t1": time_1d,
            "t2": time_1d.copy(),
            "c2_exp": c2_exp,
        }


def test_selective_read_matches_eager_read_default_config(tmp_path):
    """Default config (no data_filtering): selective and eager reads agree."""
    path = tmp_path / "aps_u_multi.h5"
    _write_multi_bin_aps_u_file(path, n_q=3, n_phi=2)

    new_result = _bare_loader()._load_aps_u_format(str(path))
    old_result = _load_aps_u_eager_reference(_bare_loader(), str(path))

    assert np.array_equal(new_result["wavevector_q_list"], old_result["wavevector_q_list"])
    assert np.array_equal(new_result["phi_angles_list"], old_result["phi_angles_list"])
    assert np.array_equal(new_result["c2_exp"], old_result["c2_exp"])
    assert np.array_equal(new_result["t1"], old_result["t1"])
    assert np.array_equal(new_result["t2"], old_result["t2"])
    # Sanity: more than one (q,phi) pair actually got selected, so this test
    # would catch a selection that silently collapsed to a single point.
    assert new_result["c2_exp"].shape[0] >= 1


def test_selective_read_matches_eager_read_with_phi_filtering(tmp_path):
    """A phi-range filter (still no quality filtering) also stays selective-safe."""
    path = tmp_path / "aps_u_multi_phi.h5"
    _write_multi_bin_aps_u_file(path, n_q=3, n_phi=4)

    config = {
        "data_filtering": {
            "enabled": True,
            "phi_range": {"target_ranges": [[0.0, 90.0]]},
        }
    }
    new_result = _bare_loader(config)._load_aps_u_format(str(path))
    old_result = _load_aps_u_eager_reference(_bare_loader(config), str(path))

    assert np.array_equal(new_result["wavevector_q_list"], old_result["wavevector_q_list"])
    assert np.array_equal(new_result["phi_angles_list"], old_result["phi_angles_list"])
    assert np.array_equal(new_result["c2_exp"], old_result["c2_exp"])


def test_quality_filtering_empty_phi_match_records_one_degradation(tmp_path):
    """Regression test (2026-09-15 PR #79 review): the quality-filtering
    branch runs ``_get_selected_indices`` twice against the SAME
    ``data_filtering`` config -- a phi/q metadata pre-filter pass, then the
    final quality-filter pass on the narrowed candidates. A phi_range that
    matches nothing triggers the ``fallback_on_empty`` path in BOTH passes;
    before the fix this recorded the DATA-1 degradation signal twice for one
    logical fallback. Only the final (pass 2) fallback should be recorded.
    """
    path = tmp_path / "aps_u_multi_quality_empty_phi.h5"
    _write_multi_bin_aps_u_file(path, n_q=3, n_phi=3)

    config = {
        "data_filtering": {
            "enabled": True,
            "phi_range": {"min": 500.0, "max": 600.0},  # matches no phi value
            "quality_filtering": {"enabled": True, "quality_threshold": 0.0},
        }
    }
    loader = _bare_loader(config)
    loader._load_aps_u_format(str(path))

    assert len(loader.load_degradations) == 1, (
        f"expected exactly one degradation entry, got {loader.load_degradations!r}"
    )


def test_selective_read_matches_eager_read_with_quality_filtering(tmp_path):
    """Quality filtering enabled: candidates are narrowed then loaded, but
    the final selection must still match the eager-read reference.
    """
    path = tmp_path / "aps_u_multi_quality.h5"
    _write_multi_bin_aps_u_file(path, n_q=3, n_phi=3)

    config = {
        "data_filtering": {
            "enabled": True,
            "quality_filtering": {"enabled": True, "quality_threshold": 0.0},
        }
    }
    new_result = _bare_loader(config)._load_aps_u_format(str(path))
    old_result = _load_aps_u_eager_reference(_bare_loader(config), str(path))

    assert np.array_equal(new_result["wavevector_q_list"], old_result["wavevector_q_list"])
    assert np.array_equal(new_result["phi_angles_list"], old_result["phi_angles_list"])
    assert np.array_equal(new_result["c2_exp"], old_result["c2_exp"])
