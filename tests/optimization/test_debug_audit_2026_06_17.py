"""Regression tests for the 2026-06-17 debug-audit fixes (optimization/core)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# jax_backend.gradient_chi2 / hessian_chi2 — must not raise on the static
# q/L/dt args (finding #5). Pre-fix the bare outer jit forwarded them as
# tracers into the static-arg inner jit, raising "Non-hashable static
# arguments".
# ---------------------------------------------------------------------------
def _chi2_args() -> tuple:
    params = jnp.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0], dtype=jnp.float64)
    t = jnp.linspace(1e-3, 1e-2, 5, dtype=jnp.float64)
    t1, t2 = jnp.meshgrid(t, t, indexing="ij")
    data = jnp.ones_like(t1)
    sigma = jnp.ones_like(t1)
    phi = jnp.array([0.0], dtype=jnp.float64)
    # q (6), L (7), dt (10) are static.
    return (params, data, sigma, t1, t2, phi, 0.01, 1e6, 0.5, 1.0, 1e-3)


def test_gradient_chi2_does_not_raise_on_static_args() -> None:
    from xpcsjax.core.jax_backend import gradient_chi2

    grad = gradient_chi2(*_chi2_args())
    grad = np.asarray(grad)
    assert grad.shape[0] == 7
    assert np.all(np.isfinite(grad))


def test_hessian_chi2_does_not_raise_on_static_args() -> None:
    from xpcsjax.core.jax_backend import hessian_chi2

    hess = np.asarray(hessian_chi2(*_chi2_args()))
    assert hess.shape == (7, 7)
    assert np.all(np.isfinite(hess))


# ---------------------------------------------------------------------------
# AdaptiveRegularizer must receive the TRUE angle count (n_phi), not the
# per-group count, so its `auto` mode threshold (n_phi > 5) sees the real
# number of angles even in averaged scaling (finding #9 / uncertain #2).
# ---------------------------------------------------------------------------
def test_controller_passes_true_n_phi_to_regularizer() -> None:
    from xpcsjax.optimization.nlsq.adaptive_regularization import (
        AdaptiveRegularizationConfig,
        AdaptiveRegularizer,
    )
    from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

    # Averaged scaling (use_constant layout): n_per_group == 1 but the true
    # angle count is 7.
    mapper = ParameterIndexMapper(n_phi=7, n_physical=7, use_constant=True)
    assert mapper.n_per_group == 1  # the value the bug used to pass

    reg_config = AdaptiveRegularizationConfig(
        enable=True,
        mode="auto",
        group_indices=mapper.get_group_indices(),
    )
    # Mirror the controller call site: the true angle count, not n_per_group.
    regularizer = AdaptiveRegularizer(reg_config, mapper.n_phi)
    assert regularizer.n_phi == 7


# ---------------------------------------------------------------------------
# log_heterodyne_completion must read physics from the layout indicated by the
# explicit scaling_first marker, NOT guess from the "averaged" mode token —
# the legacy producer is physics-first, the engine route scaling-first (#1).
# ---------------------------------------------------------------------------
def _completion_physics(caplog) -> dict:
    out = {}
    for rec in caplog.records:
        msg = rec.getMessage().strip()
        for name in ("A", "B", "C"):
            if msg.startswith(f"{name}:"):
                out[name] = float(msg.split(":")[1].split("+/-")[0])
    return out


def _het_result(parameters: list, diag: dict):
    import types

    return types.SimpleNamespace(
        nlsq_diagnostics=diag,
        parameters=np.asarray(parameters, dtype=np.float64),
        uncertainties=None,
        success=True,
        iterations=1,
        execution_time=0.0,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        quality_flag="good",
    )


def test_averaged_scaling_first_marker_reads_physics_from_tail(caplog) -> None:
    import logging

    from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_completion

    # engine-route averaged: scaling-first [c, o | physics(3)]
    params = [0.5, 1.0, 10.0, 11.0, 12.0]
    diag = {"per_angle_mode": "averaged", "scaling_first": True}
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
        log_heterodyne_completion(_het_result(params, diag), ["A", "B", "C"], n_physics=3, n_phi=1)
    assert _completion_physics(caplog) == {"A": 10.0, "B": 11.0, "C": 12.0}


def test_averaged_physics_first_marker_reads_physics_from_head(caplog) -> None:
    import logging

    from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_completion

    # legacy averaged: physics-first [physics(3) | c, o]
    params = [10.0, 11.0, 12.0, 0.5, 1.0]
    diag = {"per_angle_mode": "averaged", "scaling_first": False}
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
        log_heterodyne_completion(_het_result(params, diag), ["A", "B", "C"], n_physics=3, n_phi=1)
    assert _completion_physics(caplog) == {"A": 10.0, "B": 11.0, "C": 12.0}
