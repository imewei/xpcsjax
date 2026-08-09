"""Coverage for Layer-3 adaptive regularization (audit finding #15).

These exercise the previously-uncovered ``compute_regularization`` branches:
disabled short-circuit, the NaN-guard (audit fix #12), out-of-range and
too-small group indices, and the auto / absolute mode paths. They are pure,
deterministic unit tests — no optimizer execution.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)


def _make(
    *,
    group_indices: list[tuple[int, int]] | None = None,
    mode: Literal["absolute", "relative", "auto"] = "relative",
    n_phi: int = 3,
    enable: bool = True,
    n_params: int | None = None,
) -> AdaptiveRegularizer:
    cfg = AdaptiveRegularizationConfig(enable=enable, mode=mode, group_indices=group_indices)
    return AdaptiveRegularizer(cfg, n_phi=n_phi, n_params=n_params)


def test_disabled_returns_zero() -> None:
    reg = _make(enable=False)
    assert reg.compute_regularization(np.ones(8), mse=0.04, n_points=1000) == 0.0


def test_nonfinite_params_return_inf() -> None:
    """H-4: NaN/inf params (a diverged step) must force trust-region rejection.

    The penalty must never be NaN (which would poison the loss ambiguously) and
    must not be 0.0 either (which would silently drop the stabilizing term at the
    moment it is most needed). Returning +inf makes the augmented loss
    unambiguously bad so the step is rejected.
    """
    reg = _make(group_indices=[(0, 3), (3, 6)])
    params = np.array([1.0, np.nan, 2.0, 0.5, 0.6, np.inf])
    out = reg.compute_regularization(params, mse=0.04, n_points=1000)
    assert out == np.inf
    assert not np.isnan(out)


def test_out_of_range_group_is_skipped_not_crash() -> None:
    reg = _make(group_indices=[(0, 3), (3, 100)])  # second group runs past params
    params = np.array([0.3, 0.4, 0.5, 1.0, 1.1, 1.2])
    out = reg.compute_regularization(params, mse=0.04, n_points=1000)
    assert np.isfinite(out) and out >= 0.0


def test_singleton_group_is_skipped() -> None:
    reg = _make(group_indices=[(0, 1)])  # n_group < 2 -> no variance to regularize
    params = np.array([0.3, 0.4])
    out = reg.compute_regularization(params, mse=0.04, n_points=1000)
    assert np.isfinite(out) and out >= 0.0


def test_auto_and_absolute_modes_both_finite() -> None:
    params = np.array([0.2, 0.5, 0.8, 1.0, 1.1, 1.2])
    out_auto = _make(group_indices=[(0, 3), (3, 6)], mode="auto", n_phi=6).compute_regularization(
        params, mse=0.04, n_points=1000
    )
    out_abs = _make(
        group_indices=[(0, 3), (3, 6)], mode="absolute", n_phi=6
    ).compute_regularization(params, mse=0.04, n_points=1000)
    assert np.isfinite(out_auto) and out_auto >= 0.0
    assert np.isfinite(out_abs) and out_abs >= 0.0


def test_jax_path_finite_gradient_at_zero_mean_group() -> None:
    """compute_regularization_jax's CV safe-divide must not poison jax.grad.

    A group whose mean is exactly 0 (e.g. a symmetric offset/velocity term
    straddling zero) is the degenerate case the CV penalty exists to catch.
    Regression guard for the both-branches-evaluated-under-JIT gradient
    contamination bug (0/0 -> Inf -> nan in the untaken jnp.where branch).
    """
    reg = _make(group_indices=[(0, 3)], mode="relative", n_phi=6)
    params = jnp.array([-1.0, 0.0, 1.0])

    def loss(p: jnp.ndarray) -> jnp.ndarray:
        return reg.compute_regularization_jax(p, mse=jnp.array(0.04), n_points=1000)

    value = loss(params)
    grad = jax.grad(loss)(params)
    assert jnp.isfinite(value)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_jax_path_matches_numpy_fallback_at_zero_mean_group() -> None:
    """At mean~=0, compute_regularization_jax's CV must fall back to std (not 0),
    matching compute_regularization's documented "Fallback to absolute std"
    contract for the same input. Regression guard for a fix that sanitized the
    CV numerator (not just the denominator) and silently zeroed the penalty at
    the exact degenerate case it exists to catch.
    """
    group_indices = [(0, 3)]
    params_np = np.array([-1.0, 0.0, 1.0])
    numpy_reg = _make(group_indices=group_indices, mode="relative", n_phi=6)
    numpy_out = numpy_reg.compute_regularization(params_np, mse=0.04, n_points=1000)

    jax_reg = _make(group_indices=group_indices, mode="relative", n_phi=6)
    jax_out = jax_reg.compute_regularization_jax(
        jnp.array(params_np), mse=jnp.array(0.04), n_points=1000
    )
    assert float(jax_out) > 0.0
    assert np.isclose(float(jax_out), numpy_out, rtol=1e-10)
