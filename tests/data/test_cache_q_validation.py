"""Regression test for cache q-vector validation (audit C1).

A q-keyed selective cache must NOT be silently reused for a different
configured wavevector_q. The old `_validate_cache_q_vector` only logged a
warning on mismatch, so a different q received another q's correlation data.
It must now reject the cache (raise), mirroring the existing legacy-cache
refusal.
"""

from __future__ import annotations

import pytest

from xpcsjax.data.xpcs_loader import XPCSDataFormatError, XPCSDataLoader


def _loader_with_q(q: float) -> XPCSDataLoader:
    # _validate_cache_q_vector reads only self.analyzer_config — bypass the
    # heavy constructor.
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"scattering": {"wavevector_q": q}}
    return loader


def test_q_mismatch_is_rejected():
    loader = _loader_with_q(0.05)
    with pytest.raises(XPCSDataFormatError):
        loader._validate_cache_q_vector(
            {"config_wavevector_q": 0.01, "selective_q_caching": True}
        )


def test_matching_q_is_accepted():
    loader = _loader_with_q(0.05)
    # Same q within tolerance: no raise.
    loader._validate_cache_q_vector(
        {"config_wavevector_q": 0.05, "selective_q_caching": True}
    )
