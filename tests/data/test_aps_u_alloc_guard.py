"""Quality-gate finding #5: the APS-U loader builds an unbounded intermediate
list of reconstructed matrices BEFORE its post-selection allocation guard runs.
A crafted APS-U file with many large bins could exhaust RAM during that
accumulation. ``_guard_aps_u_intermediate_allocation`` probes the first valid
matrix (h5py metadata only) and applies the square/frame/budget guards up front,
mirroring the APS-old probe-then-guard ordering.

These tests pin the helper with a lightweight h5py-like fake (only ``.shape`` /
``.dtype`` are touched on the probe — no full read), so no real HDF5 is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data import xpcs_loader as xl
from xpcsjax.data.xpcs_loader import XPCSDataFormatError


class _FakeDS:
    """Minimal h5py-dataset stand-in: exposes shape + dtype, no data read."""

    def __init__(self, shape, dtype=np.float64):
        self.shape = shape
        self.dtype = np.dtype(dtype)


def test_guard_rejects_oversized_frame_count():
    corr = {"c2_00001": _FakeDS((xl.MAX_CORRELATION_FRAMES + 1, xl.MAX_CORRELATION_FRAMES + 1))}
    with pytest.raises(XPCSDataFormatError, match="frame count"):
        xl._guard_aps_u_intermediate_allocation(corr, ["c2_00001"], [0], source="test")


def test_guard_rejects_budget_exceeded(monkeypatch):
    # Many in-range bins of a legal-but-not-tiny n_t, against a tiny budget.
    monkeypatch.setattr(xl, "MAX_CORRELATION_ALLOC_BYTES", 1024)
    corr = {f"c2_{i:05d}": _FakeDS((64, 64)) for i in range(1, 11)}
    keys = sorted(corr)
    with pytest.raises(XPCSDataFormatError, match="Refusing to allocate"):
        xl._guard_aps_u_intermediate_allocation(corr, keys, list(range(10)), source="test")


def test_guard_rejects_non_square():
    corr = {"c2_00001": _FakeDS((10, 20))}
    with pytest.raises(XPCSDataFormatError, match="square"):
        xl._guard_aps_u_intermediate_allocation(corr, ["c2_00001"], [0], source="test")


def test_guard_passes_for_legitimate_small_input():
    corr = {f"c2_{i:05d}": _FakeDS((32, 32)) for i in range(1, 4)}
    keys = sorted(corr)
    # Must not raise.
    xl._guard_aps_u_intermediate_allocation(corr, keys, [0, 1, 2], source="test")


def test_guard_noop_when_no_valid_bins_in_range():
    corr = {"c2_00001": _FakeDS((32, 32))}
    # bin index out of range -> nothing to probe -> no raise.
    xl._guard_aps_u_intermediate_allocation(corr, ["c2_00001"], [5], source="test")
