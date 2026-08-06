"""Regression test for cache q-vector validation (audit C1).

A q-keyed selective cache must NOT be silently reused for a different
configured wavevector_q. The old `_validate_cache_q_vector` only logged a
warning on mismatch, so a different q received another q's correlation data.
It must now reject the cache (raise), mirroring the existing legacy-cache
refusal.
"""

from __future__ import annotations

import logging

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


def test_missing_dt_metadata_warns_not_raises(caplog: pytest.LogCaptureFixture):
    """Pre-existing caches without dt fingerprinting must still load (warn-only)."""
    loader = _loader_with_q(0.05)
    loader.analyzer_config["dt"] = 0.1
    with caplog.at_level(logging.WARNING):
        loader._validate_cache_q_vector({"config_wavevector_q": 0.05, "selective_q_caching": True})
    assert "predates dt fingerprinting" in caplog.text


def _angle_hash(loader: XPCSDataLoader) -> str:
    from xpcsjax.data.xpcs_loader import _hash_filter_config

    return _hash_filter_config(
        loader.config.get("optimization_config", {}).get("angle_filtering", {})
    )


def test_angle_filtering_mismatch_is_rejected():
    """optimization_config.angle_filtering shapes the cached (q, phi) selection.

    It lives outside the data_filtering subtree that filter_config_hash covers,
    so a target_ranges edit must not silently reuse the old angle set's cache.
    """
    loader = _loader_with_q(0.05)
    loader.config = {
        "optimization_config": {
            "angle_filtering": {
                "enabled": True,
                "target_ranges": [{"min_angle": 80, "max_angle": 100}],
            }
        }
    }
    stale = _loader_with_q(0.05)
    stale.config = {
        "optimization_config": {
            "angle_filtering": {
                "enabled": True,
                "target_ranges": [{"min_angle": -10, "max_angle": 10}],
            }
        }
    }
    with pytest.raises(XPCSDataFormatError, match="angle-filtering mismatch"):
        loader._validate_cache_q_vector(
            {
                "config_wavevector_q": 0.05,
                "selective_q_caching": True,
                "angle_filtering_hash": _angle_hash(stale),
            }
        )


def test_matching_angle_filtering_is_accepted():
    loader = _loader_with_q(0.05)
    loader.config = {
        "optimization_config": {
            "angle_filtering": {"enabled": True, "fallback_to_all_angles": False}
        }
    }
    loader._validate_cache_q_vector(
        {
            "config_wavevector_q": 0.05,
            "selective_q_caching": True,
            "angle_filtering_hash": _angle_hash(loader),
        }
    )


def test_missing_angle_filtering_metadata_warns_not_raises(caplog: pytest.LogCaptureFixture):
    """Pre-existing caches without the new fingerprint must still load (warn-only)."""
    loader = _loader_with_q(0.05)
    with caplog.at_level(logging.WARNING):
        loader._validate_cache_q_vector({"config_wavevector_q": 0.05, "selective_q_caching": True})
    assert "predates angle-filtering fingerprinting" in caplog.text
