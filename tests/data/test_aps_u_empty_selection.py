"""Regression test for APS-U empty (q,phi) selection (audit C9).

When phi filtering intersected with q-vector selection yields no indices, the
loader silently fell back to (q,phi) pair index 0 — returning arbitrary-q data.
An empty selection must instead raise, mirroring the APS-old branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader


def _loader() -> XPCSDataLoader:
    return XPCSDataLoader.__new__(XPCSDataLoader)


def test_empty_selection_raises():
    loader = _loader()
    with pytest.raises(XPCSDataFormatError):
        loader._require_nonempty_selection(np.array([], dtype=int), selected_q=0.05)


def test_nonempty_selection_returned_unchanged():
    loader = _loader()
    idx = np.array([2, 5, 7], dtype=int)
    out = loader._require_nonempty_selection(idx, selected_q=0.05)
    np.testing.assert_array_equal(out, idx)
