"""Guard against unbounded allocation from a crafted/corrupt correlation file.

Quality-gate finding: the HDF5 loaders read the time-axis dimension straight
from a dataset shape and allocate ``(n_sel, n_t, n_t)`` with no upper bound, so
a file declaring a huge ``n_t`` triggers a multi-hundred-GB allocation (OOM/DoS)
before any validation runs. ``_check_frame_count`` is the cheap sanity check
that belongs at that I/O boundary.
"""

from __future__ import annotations

import pytest

from xpcsjax.data.xpcs_loader import (
    MAX_CORRELATION_ALLOC_BYTES,
    MAX_CORRELATION_FRAMES,
    XPCSDataFormatError,
    _check_allocation_budget,
    _check_frame_count,
    _check_square_matrix,
)


def test_rejects_absurd_frame_count():
    with pytest.raises(XPCSDataFormatError, match="frame"):
        _check_frame_count(MAX_CORRELATION_FRAMES + 1, source="evil.h5")


def test_rejects_nonpositive_frame_count():
    with pytest.raises(XPCSDataFormatError):
        _check_frame_count(0, source="evil.h5")


def test_accepts_realistic_frame_count():
    # A normal experiment size passes silently (returns None).
    assert _check_frame_count(500, source="ok.h5") is None


# ---------------------------------------------------------------------------
# SEC-2: total-byte allocation budget. ``_check_frame_count`` bounds only the
# time axis; the product n_matrices * n_t * n_t * itemsize is what drives the
# real allocation, so a legal n_t with a huge matrix count still OOMs.
# ---------------------------------------------------------------------------


def test_rejects_huge_matrix_count_even_with_legal_frame_count():
    # n_t is well under MAX_CORRELATION_FRAMES, but the product is petabytes.
    with pytest.raises(XPCSDataFormatError, match="GiB"):
        _check_allocation_budget(n_matrices=10_000, n_t=10_000, itemsize=8, source="evil.h5")


def test_rejects_negative_matrix_count():
    with pytest.raises(XPCSDataFormatError):
        _check_allocation_budget(n_matrices=-1, n_t=100, itemsize=8, source="evil.h5")


def test_accepts_realistic_allocation():
    # 500 matrices x 1000 x 1000 x 8 bytes = 4 GB, well under the ceiling.
    assert _check_allocation_budget(n_matrices=500, n_t=1000, itemsize=8, source="ok.h5") is None


def test_budget_ceiling_is_a_sane_positive_value():
    # The ceiling must be a positive byte count (regression guard against a
    # zero/None default that would reject every allocation).
    assert MAX_CORRELATION_ALLOC_BYTES > 0


# ---------------------------------------------------------------------------
# SEC-2: square-matrix shape. The allocation assumes (n_t, n_t) from shape[0]
# only; a non-square or non-2D stored half-matrix must be rejected.
# ---------------------------------------------------------------------------


def test_accepts_square_matrix_shape():
    assert _check_square_matrix((128, 128), source="ok.h5") is None


def test_rejects_non_square_matrix_shape():
    with pytest.raises(XPCSDataFormatError, match="square"):
        _check_square_matrix((128, 64), source="evil.h5")


def test_rejects_non_2d_matrix_shape():
    with pytest.raises(XPCSDataFormatError):
        _check_square_matrix((128,), source="evil.h5")
