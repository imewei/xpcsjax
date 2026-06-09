"""Round-trip load test using a self-contained synthetic NPZ cache.

Historically this pointed at a maintainer-local Simon-dataset cache and SKIPped
everywhere else. It now SYNTHESIZES a valid 1-D-time-axis NPZ cache in ``tmp_path``
and drives the real ``XPCSDataLoader._load_from_cache`` path, so it runs on every
machine (CI included) with zero external data — exercising more of the loader than
the old placeholder ever did off the maintainer box.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.data import load_xpcs_data


def _write_synthetic_npz_cache(path, *, n_phi: int = 3, n_t: int = 6, q: float = 0.0237):
    """Write a minimal, valid 1-D-time-axis NPZ cache the loader can read directly.

    Schema mirrors ``XPCSDataLoader._load_from_cache``: ``c2_exp`` ``(n_phi, N, N)``,
    1-D monotonic ``t1``/``t2``, plus ``wavevector_q_list`` / ``phi_angles_list``.
    No ``cache_metadata_json`` is written, so the q-vector cross-check is bypassed
    (it is optional). Saved UNCOMPRESSED so ``np.load(mmap_mode="r")`` works.
    """
    t = np.arange(n_t, dtype=np.float64)  # 1-D, strictly monotonic
    phi = np.linspace(0.0, 144.0, n_phi, dtype=np.float64)
    # Symmetric, finite, square per-angle correlation surface (g2-like decay).
    dtau = np.abs(t[:, None] - t[None, :])
    c2 = np.stack(
        [1.0 + (0.30 + 0.02 * i) * np.exp(-dtau / 2.0) for i in range(n_phi)]
    ).astype(np.float64)
    np.savez(
        path,
        c2_exp=c2,
        t1=t,
        t2=t,
        wavevector_q_list=np.full(n_phi, q, dtype=np.float64),
        phi_angles_list=phi,
    )
    return n_phi, n_t


def test_load_synthetic_npz_cache_roundtrip(tmp_path):
    """Load a synthetic NPZ cache end-to-end and assert the XPCS data invariants."""
    n_phi, n_t = _write_synthetic_npz_cache(tmp_path / "synthetic_cache.npz")

    config_dict = {
        "experimental_data": {
            "data_folder_path": str(tmp_path),
            "data_file_name": "synthetic_cache.npz",
        },
        "analyzer_parameters": {
            "dt": 0.5,
            "start_frame": 1,
            "end_frame": n_t,
            "scattering": {"wavevector_q": 0.0237},
        },
    }

    data = load_xpcs_data(config_dict=config_dict)

    # Sanity invariants for any homodyne XPCS file:
    assert "c2_exp" in data
    assert "phi_angles_list" in data
    assert "t1" in data and "t2" in data
    c2 = np.asarray(data["c2_exp"])
    # c2 shape: (n_phi, N, N) — 3-D float64. Some callers add a leading q-dimension
    # producing (n_q, n_phi, N, N), so accept both.
    assert c2.ndim >= 3
    assert c2.dtype == np.float64
    # Last two axes are square (N × N correlation matrix), matching what we wrote.
    N = c2.shape[-1]
    assert N == c2.shape[-2] == n_t
    assert np.all(np.isfinite(c2))
    assert np.asarray(data["phi_angles_list"]).shape[-1] == n_phi
    # Time arrays are monotonic.
    assert np.all(np.diff(np.asarray(data["t1"])) > 0)
    assert np.all(np.diff(np.asarray(data["t2"])) > 0)
