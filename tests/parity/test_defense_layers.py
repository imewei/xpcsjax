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


def test_l2_hierarchical_two_stage_actually_runs() -> None:
    """`enable_hierarchical=True` runs constant-mode physics-only THEN joint refine.

    Strengthens :func:`test_l2_hierarchical_activates_for_heterodyne` by
    asserting behavioural evidence — both stages must record a chi^2 in
    diagnostics, and the joint stage-2 refine must not be worse than the
    physics-only stage-1 result (modulo a tiny floating-point slack).
    """
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
        fourier_order=2,
        enable_hierarchical=True,
        max_nfev=60,  # generous so both stages have budget
    )
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    diag = result.nlsq_diagnostics or {}

    assert diag.get("hierarchical_stages") == 2
    # Evidence that two real stages ran, not just a diagnostic stub:
    assert "hierarchical_stage1_chi2" in diag, "stage 1 chi2 must be recorded"
    assert "hierarchical_stage2_chi2" in diag, "stage 2 chi2 must be recorded"
    # Stage 2 should not be worse than stage 1 (joint refine should
    # monotonically improve fit modulo a small floating-point slack).
    assert diag["hierarchical_stage2_chi2"] <= diag["hierarchical_stage1_chi2"] * 1.01, (
        f"stage 2 chi2 ({diag['hierarchical_stage2_chi2']:.4f}) should not exceed "
        f"stage 1 chi2 ({diag['hierarchical_stage1_chi2']:.4f}) — joint refine should improve"
    )
    assert diag.get("hierarchical_scope") == "full_two_stage"


def test_l2_hierarchical_two_stage_averaged_mode() -> None:
    """L2 two-stage hierarchical also runs in averaged mode.

    Note: averaged mode's stage 1 uses *per-angle quantile* scaling
    (`(n_phi,)` independent contrast/offset values), while stage 2
    constrains scaling to a single averaged `(contrast, offset)` pair —
    so the two chi^2 values are computed against subtly different
    objectives and need not satisfy `stage2 <= stage1` (a tighter
    scaling DoF in stage 1 can fit residual noise that stage 2's single
    pair cannot). We assert both stages recorded their chi^2 and the
    scope is the full two-stage label — not a strict monotonicity claim.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="averaged",
        enable_hierarchical=True,
        max_nfev=60,
    )
    n_phi = 4  # below fourier_auto_threshold so averaged is a stable pick
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    diag = result.nlsq_diagnostics or {}

    assert diag.get("hierarchical_stages") == 2
    assert "hierarchical_stage1_chi2" in diag
    assert "hierarchical_stage2_chi2" in diag
    # Both stages must report finite, non-negative chi^2 — the
    # full-two-stage scope marker is what proves both stages ran.
    assert float(diag["hierarchical_stage1_chi2"]) >= 0.0
    assert float(diag["hierarchical_stage2_chi2"]) >= 0.0
    assert np.isfinite(float(diag["hierarchical_stage1_chi2"]))
    assert np.isfinite(float(diag["hierarchical_stage2_chi2"]))
    assert diag.get("hierarchical_scope") == "full_two_stage"


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


def test_l3_adaptive_regularization_actually_penalizes() -> None:
    """`config.regularization_mode='adaptive'` actually adds penalty rows to the residual.

    Verifies behaviorally that regularization is in the solver loop, not just
    diagnostic-only. The wired AdaptiveRegularizer appends `n_groups` penalty
    rows (one per parameter group — contrast group, offset group) whose
    sum-of-squares equals ``lambda * sum_g(CV_g^2)``.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from tests.optimization.test_heterodyne_return_shape import (  # type: ignore[import-untyped]
        _build_minimal_heterodyne_model_for_fourier,
        _build_synthetic_c2_stack_for_fourier,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    n_phi = 6
    c2 = _build_synthetic_c2_stack_for_fourier(n_phi=n_phi, n_t=16, model=model)
    phi = np.linspace(0, 150, n_phi)

    # Baseline: no regularization
    config_baseline = NLSQConfig(
        per_angle_mode="fourier",
        fourier_order=2,
        regularization_mode="none",
        max_nfev=60,
    )
    baseline = fit_nlsq_multi_phi(model, c2, phi, config_baseline, weights=None)

    # Regularized — re-build model since fit mutates it
    model_reg = _build_minimal_heterodyne_model_for_fourier()
    config_reg = NLSQConfig(
        per_angle_mode="fourier",
        fourier_order=2,
        regularization_mode="adaptive",
        group_variance_lambda=0.01,
        max_nfev=60,
    )
    reg = fit_nlsq_multi_phi(model_reg, c2, phi, config_reg, weights=None)

    diag = reg.nlsq_diagnostics or {}
    assert diag.get("regularization_active") is True
    # Behavioral evidence — penalty rows in solver:
    assert "regularization_penalty_count" in diag, (
        "L3 must record how many penalty rows were appended"
    )
    assert diag["regularization_penalty_count"] > 0, (
        f"penalty count must be positive, got {diag['regularization_penalty_count']}"
    )
    # SSR conservation still holds for the *data* residual (excluding penalty rows):
    assert "regularization_data_residual_ssr" in diag, (
        "L3 must report the data-only SSR (excluding penalty contribution)"
    )
    # Data-only SSR must match chi_squared (the SSR conservation invariant: the
    # OptimizationResult chi_squared is the DATA-only SSR, not the total).
    data_ssr = float(diag["regularization_data_residual_ssr"])
    assert np.isclose(data_ssr, reg.chi_squared, rtol=1e-9), (
        f"data-only SSR ({data_ssr}) must equal chi_squared ({reg.chi_squared}) — "
        "SSR conservation requires chi_squared to exclude the penalty contribution"
    )
    # The data-only SSR with regularization should not be wildly better than baseline.
    # (Regularization trades data fit for parameter smoothness; with small lambda
    # the trade-off should be modest.)
    assert reg.chi_squared >= baseline.chi_squared * 0.5, (
        "regularized fit should not be dramatically better (would indicate a bug)"
    )
    # The scope must now flag full integration, not the MVP placeholder.
    assert diag.get("regularization_scope") == "full_residual_augmentation", (
        f"L3 scope must be 'full_residual_augmentation', got {diag.get('regularization_scope')!r}"
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
