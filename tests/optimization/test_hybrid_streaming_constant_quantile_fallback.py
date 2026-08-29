"""Regression: explicit ``per_angle_mode="constant"`` must not silently corrupt
the parameter layout when quantile-based scaling estimation raises.

Root cause (P1, basin-risk): the except-path fallback for the 'constant' mode's
quantile-estimation failure set ``use_fixed_scaling=False`` but had no matching
parameter-vector reduction — only ``use_averaged_scaling`` had one.
``model_fn_pointwise``'s unpacking for ``per_angle_mode_actual == "constant"``
ALWAYS takes the 2-scalar ``[contrast, offset, physical...]`` branch (gated on
``use_constant``, not on whether quantile estimation actually succeeded), so a
failed estimation left ``fit_initial_params`` at the untransformed full length
(``2*n_phi + n_physical``) while the model expects only ``2 + n_physical`` —
per-angle contrast/offset values leak into physics parameter slots (D0, alpha, ...).

See ``xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py`` around the
quantile-estimation except block, the ``elif use_constant:`` fallback added to
mirror the ``use_averaged_scaling`` forward transform, and the matching inverse
transform (``elif use_constant:`` after ``elif use_averaged_scaling:`` in the
INVERSE TRANSFORMATION section) which restores ``popt`` to the full per-angle
layout the function's docstring promises.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

from ._hybrid_streaming_fixtures import (
    _PHYSICAL_NAMES,
    _CapturingOptimizer,
    _laminar_dataset,
    _NullLogger,
)


def test_explicit_constant_mode_quantile_failure_uses_reduced_layout(monkeypatch):
    """When quantile-based fixed-scaling estimation raises for explicit
    ``per_angle_mode="constant"``, the optimizer must be called with the
    2 + n_physical vector ``model_fn_pointwise`` expects -- not the corrupted
    full-length (2*n_phi + n_physical) vector."""
    stratified_data, n_phi = _laminar_dataset()
    n_physical = len(_PHYSICAL_NAMES)

    initial_params = np.concatenate(
        [
            np.full(n_phi, 0.3),  # per-angle contrast
            np.full(n_phi, 1.0),  # per-angle offset
            np.array([1000.0, 0.9, 5.0, 0.5, 0.0, 0.0, 45.0]),  # physical
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

    def _raise_quantile(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("quantile estimation forced failure")

    _CapturingOptimizer.last_p0 = None
    _CapturingOptimizer.last_bounds = None
    monkeypatch.setattr(hs, "_compute_quantile_per_angle_scaling", _raise_quantile)
    monkeypatch.setattr(hs, "HYBRID_STREAMING_AVAILABLE", True)
    monkeypatch.setattr(hs, "HybridStreamingConfig", lambda **kw: kw)
    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _CapturingOptimizer)

    popt, pcov, info = hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=_PHYSICAL_NAMES,
        initial_params=initial_params,
        bounds=bounds,
        logger=_NullLogger(),
        hybrid_config={},
        anti_degeneracy_config={
            "per_angle_mode": "constant",
            "hierarchical": {"enable": False},
            "regularization": {"enable": False},
            "gradient_monitoring": {"enable": False},
        },
    )

    # The optimizer must see the reduced [contrast, offset, physical...] vector
    # -- matching model_fn_pointwise's `elif use_constant:` branch -- not the
    # full per-angle vector with per-angle values leaking into physics slots.
    assert _CapturingOptimizer.last_p0 is not None
    assert len(_CapturingOptimizer.last_p0) == 2 + n_physical
    assert _CapturingOptimizer.last_bounds is not None
    assert len(_CapturingOptimizer.last_bounds[0]) == 2 + n_physical

    # Docstring contract: popt is expanded back to the full per-angle layout
    # for the caller (residual_fn / diagnostics expect this length).
    assert len(popt) == 2 * n_phi + n_physical
    assert np.all(np.isfinite(popt))
    assert pcov.shape == (2 * n_phi + n_physical, 2 * n_phi + n_physical)


def test_constant_mode_quantile_failure_resolves_l3_l4_as_averaged(monkeypatch):
    """The SAME quantile-failure fallback as above must also resolve L3
    (adaptive regularization group indices) and L4 (gradient-collapse
    monitor watched indices) as if per_angle_mode were "averaged" (the real
    2-param [contrast, offset, physical...] vector this fallback builds),
    not literal "constant" (which ParameterIndexMapper.canonical resolves to
    n_optimized=0 -- a frozen, zero-width scaling head). Using "constant"
    there would point the L4 monitor's physical_indices at the scaling head
    instead of physics, and collapse L3's group_indices to None (silently
    skipping regularization for a scaling head that does exist).
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

    def _raise_quantile(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("quantile estimation forced failure")

    _CapturingOptimizer.last_p0 = None
    _CapturingOptimizer.last_bounds = None
    monkeypatch.setattr(hs, "_compute_quantile_per_angle_scaling", _raise_quantile)
    monkeypatch.setattr(hs, "HYBRID_STREAMING_AVAILABLE", True)
    monkeypatch.setattr(hs, "HybridStreamingConfig", lambda **kw: kw)
    monkeypatch.setattr(hs, "AdaptiveHybridStreamingOptimizer", _CapturingOptimizer)

    from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

    modes_seen: list[str] = []
    _real_canonical = ParameterIndexMapper.canonical

    def _capturing_canonical(mode: str, n_phi: int, n_physics: int):
        modes_seen.append(mode)
        return _real_canonical(mode, n_phi=n_phi, n_physics=n_physics)

    monkeypatch.setattr(ParameterIndexMapper, "canonical", staticmethod(_capturing_canonical))

    popt, pcov, info = hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=_PHYSICAL_NAMES,
        initial_params=initial_params,
        bounds=bounds,
        logger=_NullLogger(),
        hybrid_config={},
        anti_degeneracy_config={
            "per_angle_mode": "constant",
            "hierarchical": {"enable": False},
            "regularization": {"enable": True},
            "gradient_monitoring": {"enable": True},
        },
    )

    assert modes_seen, "ParameterIndexMapper.canonical was never called"
    assert all(m == "averaged" for m in modes_seen), (
        f"L3/L4 must resolve the constant-mode quantile-failure fallback as "
        f"'averaged' (real 2-param vector), got: {modes_seen}"
    )
    assert np.all(np.isfinite(popt))
