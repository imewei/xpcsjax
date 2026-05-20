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
