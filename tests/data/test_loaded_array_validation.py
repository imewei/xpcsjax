"""DATA-2 / threat-03: hard-fail validation of loaded correlation arrays.

The quality-gate audit flagged two boundary gaps:

* The project's own I/O rule ("validate shape/dtype/NaN/monotonicity") was
  opt-in, so a corrupt HDF5 with NaN/inf values silently drove a numerically
  wrong fit (DATA-2).
* The ``.npz`` cache path bypassed the frame-count/allocation guard entirely,
  so a crafted cache could OOM the process (threat-03 / F4).

``_validate_loaded_arrays`` is the unconditional boundary check that closes
both. It hard-fails (raises ``XPCSDataFormatError``) per the chosen contract.

Monotonicity is asserted on the *time* axes (t1/t2 = [0, dt, 2dt, ...], which
are monotonic by construction), NOT on ``wavevector_q_list`` — in XPCS the
q-list holds one entry per (q, phi) pair and is legitimately non-monotonic.
"""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data.xpcs_loader import (
    XPCSDataFormatError,
    _validate_loaded_arrays,
)


def _good_data(n_mat: int = 2, n_t: int = 4) -> dict:
    return {
        "c2_exp": np.ones((n_mat, n_t, n_t), dtype=np.float64),
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01, 0.01, 0.02]),  # non-monotonic OK
        "phi_angles_list": np.array([0.0, 45.0, 90.0]),
    }


def test_accepts_valid_data():
    assert _validate_loaded_arrays(_good_data(), source="ok.h5") is None


def test_accepts_non_monotonic_q_list():
    # q repeats across phi pairs — must NOT be rejected.
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.05, 0.01, 0.05, 0.02])
    assert _validate_loaded_arrays(data, source="ok.h5") is None


def test_rejects_nan_in_c2_exp():
    data = _good_data()
    data["c2_exp"][0, 0, 0] = np.nan
    with pytest.raises(XPCSDataFormatError, match="NaN/inf"):
        _validate_loaded_arrays(data, source="evil.h5")


def test_rejects_inf_in_q_list():
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.01, np.inf, 0.02])
    with pytest.raises(XPCSDataFormatError, match="NaN/inf"):
        _validate_loaded_arrays(data, source="evil.h5")


def test_rejects_non_monotonic_time_axis():
    data = _good_data()
    data["t1"] = np.array([0.0, 2.0, 1.0, 3.0])  # decreasing step
    with pytest.raises(XPCSDataFormatError, match="monotonic"):
        _validate_loaded_arrays(data, source="evil.h5")


def test_rejects_non_square_cached_correlation_buffer():
    # A 3-D c2_exp whose trailing axes aren't square is malformed (threat-03).
    data = _good_data()
    data["c2_exp"] = np.ones((2, 4, 3), dtype=np.float64)
    with pytest.raises(XPCSDataFormatError, match="square"):
        _validate_loaded_arrays(data, source="evil.h5")


def test_frame_guard_is_invoked_for_3d_buffer(monkeypatch):
    # Lower the cap so a small array trips it — proves the frame guard runs.
    import xpcsjax.data.xpcs_loader as loader

    monkeypatch.setattr(loader, "MAX_CORRELATION_FRAMES", 3)
    data = _good_data(n_mat=2, n_t=4)  # last axis 4 > cap 3
    with pytest.raises(XPCSDataFormatError):
        _validate_loaded_arrays(data, source="evil.h5")
