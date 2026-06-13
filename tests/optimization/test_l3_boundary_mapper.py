"""Phase 1+2: shared L3 boundary reads the canonical mapper, and AdaptiveRegularizer
handles constant mode (n_optimized==0) WITHOUT regularizing physics params as scaling."""
from __future__ import annotations

from xpcsjax.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizer,
    AdaptiveRegularizationConfig,
)


def test_adaptive_regularizer_constant_mode_has_empty_groups():
    # constant: optimizer vector is PHYSICS-ONLY (n_params == n_physical, < 2*n_phi).
    # The old n_params < 2*n_phi fallback would wrongly emit [(0,1),(1,2)] and
    # regularize the first two PHYSICS params as scaling. Constant must yield NO groups.
    cfg = AdaptiveRegularizationConfig(group_indices=None)
    reg = AdaptiveRegularizer(cfg, n_phi=23, n_params=7, n_optimized=0)
    assert reg.group_indices == []


def test_adaptive_regularizer_averaged_groups():
    cfg = AdaptiveRegularizationConfig(group_indices=None)
    reg = AdaptiveRegularizer(cfg, n_phi=23, n_params=7 + 2, n_optimized=2)
    assert reg.group_indices == [(0, 1), (1, 2)]


def test_adaptive_regularizer_individual_groups():
    cfg = AdaptiveRegularizationConfig(group_indices=None)
    reg = AdaptiveRegularizer(cfg, n_phi=4, n_params=7 + 8, n_optimized=8)
    assert reg.group_indices == [(0, 4), (4, 8)]
