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
    MAX_CORRELATION_FRAMES,
    XPCSDataFormatError,
    _check_frame_count,
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
