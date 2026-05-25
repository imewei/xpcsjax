"""Tests for heterodyne memory routing and adapter helper logic.

* heterodyne_memory: memory detection, peak estimate, env/clamp/fallback
  threshold resolution, and the STANDARD/LARGE/STREAMING decision tree.
* heterodyne_adapter (pure/helper surface): optimizer-kwarg mapping, the
  CurveFit model cache (miss/hit/clear/stats), post-fit convergence
  assessment, and the NLSQWrapper fallback-tier ordering. The full optimizer
  fit() paths are out of scope here (they run real NLSQ).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq import heterodyne_adapter as ad
from xpcsjax.optimization.nlsq import heterodyne_memory as hm
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig

# ===========================================================================
# heterodyne_memory
# ===========================================================================


def test_detect_total_system_memory_gb() -> None:
    total = hm.detect_total_system_memory()
    assert total is not None and total > 0  # GB, psutil available


def test_estimate_peak_memory_gb() -> None:
    peak = hm.estimate_peak_memory_gb(1_000_000, 53)
    expected = 1_000_000 * 53 * 8 * 6.5 / (1024**3)
    assert peak == pytest.approx(expected)


def test_get_memory_threshold_default() -> None:
    total = hm.detect_total_system_memory()
    assert total is not None
    threshold = hm._get_memory_threshold(0.75)
    assert threshold == pytest.approx(total * 0.75)


def test_get_memory_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hm.MEMORY_FRACTION_ENV_VAR, "0.5")
    total = hm.detect_total_system_memory()
    assert total is not None
    assert hm._get_memory_threshold(0.75) == pytest.approx(total * 0.5)


def test_get_memory_threshold_invalid_env_keeps_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hm.MEMORY_FRACTION_ENV_VAR, "garbage")
    total = hm.detect_total_system_memory()
    assert total is not None
    # Invalid env -> logged, keeps the passed fraction (0.6).
    assert hm._get_memory_threshold(0.6) == pytest.approx(total * 0.6)


def test_get_memory_threshold_clamps() -> None:
    total = hm.detect_total_system_memory()
    assert total is not None
    assert hm._get_memory_threshold(0.99) == pytest.approx(total * 0.9)  # clamped high
    assert hm._get_memory_threshold(0.01) == pytest.approx(total * 0.1)  # clamped low


def test_get_memory_threshold_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "detect_total_system_memory", lambda: None)
    assert hm._get_memory_threshold(0.75) == hm.FALLBACK_THRESHOLD_GB


def _patch_threshold(monkeypatch: pytest.MonkeyPatch, gb: float) -> None:
    monkeypatch.setattr(hm, "_get_memory_threshold", lambda f: gb)


def test_select_strategy_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    d = hm.select_nlsq_strategy(1000, 10)
    assert d.strategy is hm.NLSQStrategy.STANDARD
    assert "within threshold" in d.reason


def test_select_strategy_large(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    d = hm.select_nlsq_strategy(10_000_000, 53)  # peak >> 1 GB, index < 1 GB
    assert d.strategy is hm.NLSQStrategy.LARGE


def test_select_strategy_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    d = hm.select_nlsq_strategy(200_000_000, 53)  # index array > 1 GB
    assert d.strategy is hm.NLSQStrategy.STREAMING


def test_select_strategy_zero_params(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_threshold(monkeypatch, 1.0)
    d = hm.select_nlsq_strategy(1000, 0)
    assert d.peak_memory_gb == 0.0
    assert d.strategy is hm.NLSQStrategy.STANDARD


# ===========================================================================
# heterodyne_adapter — optimizer kwargs
# ===========================================================================


def test_optimizer_kwargs_omits_none_max_nfev() -> None:
    kw = ad._optimizer_kwargs(NLSQConfig(), "trf")
    assert kw["method"] == "trf"
    assert kw["loss"] == NLSQConfig().loss
    assert "max_nfev" not in kw  # None -> omitted so nlsq keeps its default


def test_optimizer_kwargs_includes_explicit_max_nfev() -> None:
    kw = ad._optimizer_kwargs(NLSQConfig(max_nfev=250), "lm")
    assert kw["method"] == "lm"
    assert kw["max_nfev"] == 250


# ===========================================================================
# heterodyne_adapter — model cache
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_cache():
    ad.clear_model_cache()
    yield
    ad.clear_model_cache()


def test_cache_miss_then_hit() -> None:
    fitter1, hit1 = ad.get_or_create_fitter(n_data=100, n_params=5)
    assert hit1 is False  # first call is a miss
    fitter2, hit2 = ad.get_or_create_fitter(n_data=100, n_params=5)
    assert hit2 is True  # identical key -> hit
    assert fitter1 is fitter2  # same cached instance
    stats = ad.get_cache_stats()
    assert stats == {"hits": 1, "misses": 1, "size": 1}


def test_cache_distinct_keys() -> None:
    ad.get_or_create_fitter(n_data=100, n_params=5)
    ad.get_or_create_fitter(n_data=200, n_params=5)  # different n_data -> new entry
    ad.get_or_create_fitter(n_data=100, n_params=5, scaling_mode="individual")
    assert ad.get_cache_stats()["size"] == 3


def test_clear_model_cache_resets() -> None:
    ad.get_or_create_fitter(n_data=100, n_params=5)
    ad.clear_model_cache()
    assert ad.get_cache_stats() == {"hits": 0, "misses": 0, "size": 0}


def test_cache_key_equality() -> None:
    k1 = ad.ModelCacheKey(100, 5, None, "auto")
    k2 = ad.ModelCacheKey(100, 5, None, "auto")
    k3 = ad.ModelCacheKey(100, 5, (0.0, 45.0), "auto")
    assert k1 == k2
    assert k1 != k3


# ===========================================================================
# heterodyne_adapter — convergence assessment
# ===========================================================================


def test_assess_convergence_success() -> None:
    success, msg, reason = ad._assess_convergence(
        np.array([1.0, 2.0]), np.array([0.5, 0.5]), reduced_chi2=1.2
    )
    assert success is True
    assert reason == "tolerance"


def test_assess_convergence_nonfinite() -> None:
    success, _msg, reason = ad._assess_convergence(
        np.array([np.nan, 2.0]), np.array([0.5, 0.5]), reduced_chi2=1.0
    )
    assert success is False
    assert reason == "failed"


def test_assess_convergence_poor_fit() -> None:
    success, _msg, reason = ad._assess_convergence(
        np.array([1.0, 2.0]), np.array([0.5, 0.5]), reduced_chi2=1e7
    )
    assert success is False
    assert reason == "poor_fit"


def test_assess_convergence_no_progress() -> None:
    params = np.array([1.0, 2.0])
    success, _msg, reason = ad._assess_convergence(params, params.copy(), reduced_chi2=1.0)
    assert success is False
    assert reason == "no_progress"


# ===========================================================================
# heterodyne_adapter — wrapper / adapter metadata + tier list
# ===========================================================================


def test_nlsq_wrapper_metadata_and_retry_clamp() -> None:
    w = ad.NLSQWrapper(["a", "b"], max_retries=0)
    assert w.name == "nlsq.NLSQWrapper"
    assert w.supports_bounds() is True
    assert w.supports_jacobian() is True
    assert w._max_retries == 1  # clamped up from 0


def test_nlsq_adapter_metadata() -> None:
    a = ad.NLSQAdapter(["a", "b", "c"])
    assert a.name == "nlsq.CurveFit"
    assert a.supports_bounds() is True
    assert a.supports_jacobian() is True


def test_build_tier_list_from_each_start() -> None:
    w = ad.NLSQWrapper(["a"])
    S = hm.NLSQStrategy
    assert w._build_tier_list(S.STREAMING) == [S.STREAMING, S.LARGE, S.STANDARD]
    assert w._build_tier_list(S.LARGE) == [S.LARGE, S.STANDARD]
    assert w._build_tier_list(S.STANDARD) == [S.STANDARD]


def test_build_tier_list_drops_large_when_disabled() -> None:
    w = ad.NLSQWrapper(["a"], enable_large_dataset=False)
    tiers = w._build_tier_list(hm.NLSQStrategy.STREAMING)
    assert hm.NLSQStrategy.LARGE not in tiers
    assert tiers == [hm.NLSQStrategy.STREAMING, hm.NLSQStrategy.STANDARD]
