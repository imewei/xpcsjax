"""Tests that heterodyne L2-L4 defense layers activate under config flags.

L5 (shear-sensitivity weighting) is intentionally homodyne-only — see
docs/theory/heterodyne_anti_degeneracy.rst (added in Task D4) for the
physics rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.results import OptimizationResult


def test_l2_hierarchical_activates_for_heterodyne() -> None:
    """`config.enable_hierarchical=True` runs the two-stage solve for heterodyne."""
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")

    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="fourier",
        enable_hierarchical=True,
        max_nfev=30,
    )
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    assert isinstance(result, OptimizationResult)
    diag = result.nlsq_diagnostics or {}
    assert diag.get("hierarchical_stages") == 2, (
        "L2 hierarchical must record two-stage execution in diagnostics; "
        f"got hierarchical_stages={diag.get('hierarchical_stages')!r}"
    )


def test_l3_adaptive_regularization_activates_for_heterodyne() -> None:
    """`config.regularization_mode='adaptive'` wraps the residual."""
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")

    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="fourier",
        regularization_mode="adaptive",
        max_nfev=30,
    )
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    diag = result.nlsq_diagnostics or {}
    assert diag.get("regularization_active") is True, (
        "L3 adaptive regularization must record regularization_active=True"
    )
    assert "regularization_lambda_applied" in diag, (
        "L3 must record the applied lambda in diagnostics"
    )


def test_l4_gradient_monitor_activates_for_heterodyne() -> None:
    """`config.enable_gradient_monitoring=True` installs the collapse callback."""
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")

    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="fourier",
        enable_gradient_monitoring=True,
        gradient_ratio_threshold=100.0,
        gradient_consecutive_triggers=3,
        max_nfev=30,
    )
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    diag = result.nlsq_diagnostics or {}
    assert "gradient_monitor" in diag, (
        "L4 gradient monitor must record its block in nlsq_diagnostics"
    )
    monitor = diag["gradient_monitor"]
    assert isinstance(monitor, dict)
    assert "collapse_detected" in monitor
    assert "max_gradient_ratio" in monitor
    assert "trigger_count" in monitor


def test_l5_shear_weighting_is_homodyne_only() -> None:
    """Heterodyne explicitly records L5 as not applicable.

    Locks in the structural decision documented in
    docs/theory/heterodyne_anti_degeneracy.rst: heterodyne's two-component
    model lacks homodyne's φ=0 shear-sensitivity peak, so L5 has no physics
    justification here. The `shear_weighting='not_applicable_heterodyne'`
    marker is populated by _build_heterodyne_diagnostics (added in C2).
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")

    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(per_angle_mode="fourier", max_nfev=30)
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    diag = result.nlsq_diagnostics or {}
    assert diag.get("shear_weighting") == "not_applicable_heterodyne", (
        f"expected L5 marker 'not_applicable_heterodyne', got {diag.get('shear_weighting')!r}"
    )
