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
    # _validate_cache_q_vector reads self.analyzer_config and (for the
    # filter-config fingerprint check) self.config — bypass the heavy
    # constructor but still stub both attributes.
    loader = XPCSDataLoader.__new__(XPCSDataLoader)
    loader.analyzer_config = {"scattering": {"wavevector_q": q}}
    loader.config = {}
    return loader


def test_q_mismatch_is_rejected():
    loader = _loader_with_q(0.05)
    with pytest.raises(XPCSDataFormatError):
        loader._validate_cache_q_vector({"config_wavevector_q": 0.01, "selective_q_caching": True})


def test_matching_q_is_accepted():
    loader = _loader_with_q(0.05)
    # Same q within tolerance: no raise.
    loader._validate_cache_q_vector({"config_wavevector_q": 0.05, "selective_q_caching": True})


def test_dt_mismatch_is_rejected():
    """A cache built for a different dt must not silently reuse its t1/t2 axes."""
    loader = _loader_with_q(0.05)
    loader.analyzer_config["dt"] = 0.1
    with pytest.raises(XPCSDataFormatError):
        loader._validate_cache_q_vector(
            {"config_wavevector_q": 0.05, "selective_q_caching": True, "dt": 0.5}
        )


def test_matching_dt_is_accepted():
    loader = _loader_with_q(0.05)
    loader.analyzer_config["dt"] = 0.1
    # Same dt within tolerance: no raise.
    loader._validate_cache_q_vector(
        {"config_wavevector_q": 0.05, "selective_q_caching": True, "dt": 0.1}
    )


def test_missing_dt_metadata_warns_not_raises():
    """Pre-existing caches without dt fingerprinting must still load (warn-only)."""
    loader = _loader_with_q(0.05)
    loader.analyzer_config["dt"] = 0.1
    loader._validate_cache_q_vector({"config_wavevector_q": 0.05, "selective_q_caching": True})
