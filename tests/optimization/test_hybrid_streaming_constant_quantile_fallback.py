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

_PHYSICAL_NAMES = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]


class _FakeStratifiedData:
    def __init__(
        self,
        phi_flat: np.ndarray,
        t1_flat: np.ndarray,
        t2_flat: np.ndarray,
        g2_flat: np.ndarray,
    ) -> None:
        self.phi_flat = phi_flat
        self.t1_flat = t1_flat
        self.t2_flat = t2_flat
        self.g2_flat = g2_flat
        self.q = 0.0237
        self.L = 2_000_000.0
        self.dt = 0.1


class _CapturingOptimizer:
    """Stand-in for NLSQ's ``AdaptiveHybridStreamingOptimizer``: captures the
    ``p0``/``bounds`` it was given and echoes ``p0`` back as ``x`` (no real
    solve), so the test observes exactly what shape the driver constructs."""

    last_p0: np.ndarray | None = None
    last_bounds: Any = None

    def __init__(self, config: Any) -> None:
        self.config = config

    def fit(
        self,
        *,
        data_source: Any,
        func: Any,
        p0: np.ndarray,
        bounds: Any,
        sigma: Any = None,
        verbose: int = 1,
    ) -> dict:
        _CapturingOptimizer.last_p0 = np.asarray(p0, dtype=float)
        _CapturingOptimizer.last_bounds = bounds
        n = len(p0)
        return {
            "x": np.asarray(p0, dtype=float),
            "pcov": np.eye(n),
            "success": True,
            "streaming_diagnostics": {},
        }


class _NullLogger:
    def info(self, *a: Any, **k: Any) -> None:
        pass

    def warning(self, *a: Any, **k: Any) -> None:
        pass

    def error(self, *a: Any, **k: Any) -> None:
        pass

    def debug(self, *a: Any, **k: Any) -> None:
        pass


def _laminar_dataset(n_phi: int = 4, n_t: int = 5) -> tuple[_FakeStratifiedData, int]:
    phi_unique = np.linspace(0.0, 90.0, n_phi)
    t_unique = np.linspace(0.1, 0.1 * n_t, n_t)
    phi_g, t1_g, t2_g = np.meshgrid(phi_unique, t_unique, t_unique, indexing="ij")
    mask = t1_g != t2_g  # off-diagonal points only (diagonal is filtered internally)
    phi_flat = phi_g[mask]
    t1_flat = t1_g[mask]
    t2_flat = t2_g[mask]
    g2_flat = np.full(phi_flat.shape, 1.2)
    return _FakeStratifiedData(phi_flat, t1_flat, t2_flat, g2_flat), n_phi


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
