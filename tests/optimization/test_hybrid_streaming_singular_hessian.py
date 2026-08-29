"""Regression: a singular Hessian in the laminar L2 hierarchical covariance
path must fall back to the identity placeholder, NOT a pseudo-inverse.

Mirrors tests/optimization/test_heterodyne_hybrid_streaming.py::
test_streaming_l2_singular_hessian_falls_back_to_placeholder, but for the
laminar/homodyne strategies/hybrid_streaming.py module, which received the
matching fix (pinv -> identity placeholder + covariance_is_placeholder=True)
but had no test of its own for this exact code path.

np.linalg.pinv's Moore-Penrose null-space treatment reports the unidentified
direction's variance as EXACTLY 0.0 (infinite precision) rather than the
statistically correct answer (unbounded uncertainty, since a singular
Hessian means that direction is genuinely unconstrained by the fit).
"""

from __future__ import annotations

import logging

import numpy as np

from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

from ._hybrid_streaming_fixtures import _PHYSICAL_NAMES, _laminar_dataset


def test_streaming_l2_singular_hessian_falls_back_to_placeholder(monkeypatch, caplog):
    stratified_data, n_phi = _laminar_dataset()

    # "individual" per-angle scaling: full [contrast_0..n, offset_0..n, *physical].
    initial_params = np.concatenate(
        [
            np.full(n_phi, 0.3),
            np.full(n_phi, 1.0),
            np.array([1000.0, 0.9, 5.0, 0.5, 0.0, 0.0, 45.0]),
        ]
    )
    lower = np.concatenate(
        [np.zeros(n_phi), np.full(n_phi, 0.5), np.array([1.0, 0.1, 0.0, 0.0, -1.0, -1.0, 0.0])]
    )
    upper = np.concatenate(
        [
            np.ones(n_phi),
            np.full(n_phi, 1.5),
            np.array([1e5, 2.0, 100.0, 100.0, 1.0, 1.0, 360.0]),
        ]
    )
    bounds = (lower, upper)

    def _zero_hessian(fn):
        def _inner(p):
            n = p.shape[0]
            import jax.numpy as jnp

            return jnp.zeros((n, n))

        return _inner

    monkeypatch.setattr(hs.jax, "hessian", _zero_hessian)

    with caplog.at_level("ERROR", logger="xpcsjax.optimization.nlsq.strategies.hybrid_streaming"):
        popt, pcov, info = hs.fit_with_stratified_hybrid_streaming(
            stratified_data=stratified_data,
            per_angle_scaling=True,
            physical_param_names=_PHYSICAL_NAMES,
            initial_params=initial_params,
            bounds=bounds,
            logger=logging.getLogger("test_singular_hessian"),
            hybrid_config={
                "warmup_iterations": 2,
                "max_warmup_iterations": 3,
                "gauss_newton_max_iterations": 2,
                "verbose": 0,
            },
            anti_degeneracy_config={
                "per_angle_mode": "individual",
                "hierarchical": {"enable": True, "max_outer_iterations": 1},
                "regularization": {"enable": False},
                "gradient_monitoring": {"enable": False},
            },
        )

    n = popt.shape[0]
    assert pcov.shape == (n, n)
    assert np.array_equal(pcov, np.eye(n))
    assert info["hybrid_streaming_diagnostics"]["covariance_is_placeholder"] is True
    assert any(
        "singular" in r.getMessage().lower() and r.levelname == "ERROR" for r in caplog.records
    ), "expected an ERROR log naming the singular-Hessian failure specifically"
