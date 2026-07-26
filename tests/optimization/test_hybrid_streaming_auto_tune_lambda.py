"""Regression: ``regularization.auto_tune_lambda`` must reach the L3
``AdaptiveRegularizationConfig`` actually built INSIDE the streaming fit path
(``fit_with_stratified_hybrid_streaming``), not just the config-parsing layer.

``test_anti_degeneracy_layers.py::test_regularization_auto_tune_lambda_wired_through_controller``
only proves the field survives ``AntiDegeneracyConfig.from_dict`` and reaches
the shared ``AntiDegeneracyController`` -- it never drives an actual streaming
fit call. This test reuses the ``_CapturingOptimizer`` harness from
``test_hybrid_streaming_constant_quantile_fallback.py`` (which already proves
that harness reaches ``fit_with_stratified_hybrid_streaming``'s internals) to
capture the real ``AdaptiveRegularizationConfig(...)`` call site inside the
streaming function and pin the configured value.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

from .test_hybrid_streaming_constant_quantile_fallback import (
    _PHYSICAL_NAMES,
    _CapturingOptimizer,
    _laminar_dataset,
    _NullLogger,
)


def test_streaming_auto_tune_lambda_reaches_adaptive_regularization_config(monkeypatch):
    """``auto_tune_lambda=False`` configured via ``anti_degeneracy_config``
    must reach the ``AdaptiveRegularizationConfig`` built inside
    ``fit_with_stratified_hybrid_streaming`` -- the streaming-path L3 site
    that ``test_regularization_auto_tune_lambda_wired_through_controller``
    (config-parsing layer) never exercises.
    """
    stratified_data, n_phi = _laminar_dataset()

    initial_params = np.concatenate(
        [
            np.full(n_phi, 0.3),
            np.full(n_phi, 1.0),
            np.array([1000.0, 0.9, 5.0, 0.5, 0.0, 0.0, 45.0]),
        ]
    )
    lower = np.concatenate(
        [
            np.zeros(n_phi),
            np.full(n_phi, 0.5),
            np.array([1.0, 0.1, 0.0, 0.0, -1.0, -1.0, 0.0]),
        ]
    )
    upper = np.concatenate(
        [
            np.ones(n_phi),
            np.full(n_phi, 1.5),
            np.array([1e5, 2.0, 100.0, 100.0, 1.0, 1.0, 360.0]),
        ]
    )
    bounds = (lower, upper)

    def _raise_quantile(*args, **kwargs):
        raise RuntimeError("quantile estimation forced failure")

    captured: dict = {}
    real_adaptive_regularization_config = hs.AdaptiveRegularizationConfig

    def _spy_config(**kwargs):
        captured["kwargs"] = kwargs
        return real_adaptive_regularization_config(**kwargs)

    _CapturingOptimizer.last_p0 = None
    _CapturingOptimizer.last_bounds = None
    monkeypatch.setattr(hs, "_compute_quantile_per_angle_scaling", _raise_quantile)
    monkeypatch.setattr(hs, "HYBRID_STREAMING_AVAILABLE", True)
    monkeypatch.setattr(hs, "HybridStreamingConfig", lambda **kw: kw)
    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _CapturingOptimizer)
    monkeypatch.setattr(hs, "AdaptiveRegularizationConfig", _spy_config)

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=_PHYSICAL_NAMES,
        initial_params=initial_params,
        bounds=bounds,
        logger=_NullLogger(),
        hybrid_config={},
        anti_degeneracy_config={
            "per_angle_mode": "constant",
            "regularization": {"auto_tune_lambda": False},
            "hierarchical": {"enable": False},
            "gradient_monitoring": {"enable": False},
        },
    )

    assert "kwargs" in captured, "AdaptiveRegularizationConfig must have been constructed"
    assert captured["kwargs"]["auto_tune_lambda"] is False, (
        "regularization.auto_tune_lambda=False must reach the streaming path's "
        "AdaptiveRegularizationConfig; got "
        f"{captured['kwargs']['auto_tune_lambda']!r}. Regression: this streaming "
        "call site could silently ignore the configured value while the "
        "config-parsing-layer test (test_anti_degeneracy_layers.py) stays green."
    )
