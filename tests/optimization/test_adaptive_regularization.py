"""Coverage for Layer-3 adaptive regularization (audit finding #15).

These exercise the previously-uncovered ``compute_regularization`` branches:
disabled short-circuit, the NaN-guard (audit fix #12), out-of-range and
too-small group indices, and the auto / absolute mode paths. They are pure,
deterministic unit tests — no optimizer execution.
"""

from __future__ import annotations

from typing import Literal

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
