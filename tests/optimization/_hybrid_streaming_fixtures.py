"""Shared fakes/fixtures for laminar-flow hybrid-streaming regression tests.

Extracted from ``test_hybrid_streaming_constant_quantile_fallback.py`` so that
sibling test modules (e.g. ``test_hybrid_streaming_auto_tune_lambda.py``) can
reuse the same fake ``AdaptiveHybridStreamingOptimizer``/dataset without
importing across test modules and coupling their collection order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
    solve), so the test observes exactly what shape the driver constructs.

    ``last_p0``/``last_bounds`` are reset by each test before use -- every
    consumer writes them immediately before reading, so the shared class
    state is not read-before-write across tests.
    """

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


def _constant_mode_initial_and_bounds(
    n_phi: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Full per-angle [contrast, offset, *physical] initial values + bounds
    for the constant-mode quantile-failure fallback tests."""
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
    return initial_params, (lower, upper)


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
