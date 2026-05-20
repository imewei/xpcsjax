"""Tests for heterodyne post-hoc per-angle view helpers
(reconstruct_per_angle_scaling, per_angle_chi2).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.results import OptimizationResult


def test_reconstruct_per_angle_scaling_fourier_mode() -> None:
    """Helper reconstructs per-angle contrast from Fourier coefficients."""
    from xpcsjax.optimization.nlsq.heterodyne_views import (
        reconstruct_per_angle_scaling,
    )

    # Fake result: 3 physics + 5 contrast Fourier (K=2) + 5 offset Fourier
    # Coefficients: contrast_coeffs = [0.4, 0, 0, 0, 0]  (constant 0.4)
    #               offset_coeffs   = [1.0, 0, 0, 0, 0]  (constant 1.0)
    params = np.array([0.5, 0.1, 0.01, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "fourier",
            "fourier_basis_dim": 5,
            "scaling_source": "fitted",
        },
    )

    phi = np.array([0.0, 45.0, 90.0])
    layout = {"n_physics": 3, "fourier_order": 2}
    out = reconstruct_per_angle_scaling(
        result=result, phi_angles=phi, mode="fourier", layout=layout
    )

    assert set(out.keys()) == {"contrast", "offset"}
    np.testing.assert_allclose(out["contrast"], [0.4, 0.4, 0.4], atol=1e-12)
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0], atol=1e-12)


def test_reconstruct_per_angle_scaling_fourier_matches_canonical_basis() -> None:
    """Helper's Fourier basis layout matches FourierReparameterizer's.

    Exercises every harmonic with non-zero coefficients at K=2 and compares
    the helper output against B @ coeffs, where B is the canonical basis
    matrix built by `FourierReparameterizer._compute_basis_matrix`. This is
    the load-bearing test for the interleaved [c0, c1, s1, c2, s2] layout.
    """
    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    K = 2
    phi_deg = np.array([0.0, 30.0, 60.0, 90.0, 135.0])
    n_physics = 3
    basis_dim = 2 * K + 1  # 5

    # Non-zero coefficients to exercise every harmonic.
    contrast_coeffs = np.array([0.40, 0.05, 0.03, -0.02, 0.01])  # [c0, c1, s1, c2, s2]
    offset_coeffs = np.array([1.00, 0.10, -0.05, 0.02, -0.03])

    params = np.concatenate([
        np.array([0.5, 0.1, 0.01]),  # physics
        contrast_coeffs,
        offset_coeffs,
    ])

    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "fourier",
            "fourier_basis_dim": basis_dim,
            "scaling_source": "fitted",
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=phi_deg,
        mode="fourier",
        layout={"n_physics": n_physics, "fourier_order": K},
    )

    # Build the canonical basis matrix the same way `_fit_joint_multi_phi`
    # would. FourierReparameterizer takes phi in *radians*; the helper takes
    # phi in *degrees* and deg2rads internally — so we feed the radian form
    # to the reparameterizer for an apples-to-apples comparison.
    config = FourierReparamConfig(mode="fourier", fourier_order=K)
    reparam = FourierReparameterizer(
        phi_angles=np.deg2rad(phi_deg), config=config
    )
    assert reparam.use_fourier, "expected Fourier mode to be active"
    B = reparam.get_basis_matrix()
    assert B is not None
    expected_contrast = B @ contrast_coeffs
    expected_offset = B @ offset_coeffs

    np.testing.assert_allclose(out["contrast"], expected_contrast, atol=1e-12)
    np.testing.assert_allclose(out["offset"], expected_offset, atol=1e-12)


def test_reconstruct_per_angle_scaling_individual_mode() -> None:
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    n_physics = 3
    params = np.concatenate([
        np.array([0.5, 0.1, 0.01]),       # physics
        np.array([0.4, 0.42, 0.38]),       # contrast per angle
        np.array([1.0, 1.0, 1.0]),         # offset per angle
    ])
    result = OptimizationResult(
        parameters=params,
        uncertainties=np.zeros_like(params),
        covariance=np.eye(len(params)),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={"per_angle_mode": "individual", "scaling_source": "fitted"},
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="individual",
        layout={"n_physics": n_physics},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0])


def test_reconstruct_per_angle_scaling_constant_mode() -> None:
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    result = OptimizationResult(
        parameters=np.array([0.5, 0.1, 0.01]),  # physics only
        uncertainties=np.zeros(3),
        covariance=np.eye(3),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "constant",
            "scaling_source": "quantile_fixed",
            "contrast_per_angle_fixed": np.array([0.4, 0.42, 0.38]),
            "offset_per_angle_fixed": np.array([1.0, 1.0, 1.0]),
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="constant",
        layout={"n_physics": 3},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])
    np.testing.assert_allclose(out["offset"], [1.0, 1.0, 1.0])


def test_reconstruct_per_angle_scaling_auto_mode_resolves_via_diagnostics() -> None:
    """`mode='auto'` reads the actual dispatched mode from diagnostics and recurses."""
    from xpcsjax.optimization.nlsq.heterodyne_views import reconstruct_per_angle_scaling

    # Build a result that came out of `auto` dispatch but landed in `constant`.
    result = OptimizationResult(
        parameters=np.array([0.5, 0.1, 0.01]),
        uncertainties=np.zeros(3),
        covariance=np.eye(3),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={
            "per_angle_mode": "constant",  # the resolved mode
            "scaling_source": "quantile_fixed",
            "contrast_per_angle_fixed": np.array([0.4, 0.42, 0.38]),
            "offset_per_angle_fixed": np.array([1.0, 1.0, 1.0]),
        },
    )

    out = reconstruct_per_angle_scaling(
        result=result,
        phi_angles=np.array([0.0, 45.0, 90.0]),
        mode="auto",
        layout={"n_physics": 3},
    )

    np.testing.assert_allclose(out["contrast"], [0.4, 0.42, 0.38])


def test_per_angle_chi2_reads_from_diagnostics() -> None:
    """`per_angle_chi2()` retrieves the array from nlsq_diagnostics."""
    from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

    expected = np.array([1.0, 2.0, 3.0])
    result = OptimizationResult(
        parameters=np.array([0.5]),
        uncertainties=np.zeros(1),
        covariance=np.eye(1),
        chi_squared=6.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics={"chi2_per_angle": expected},
    )
    np.testing.assert_array_equal(per_angle_chi2(result), expected)


def test_per_angle_chi2_raises_when_missing() -> None:
    """`per_angle_chi2()` raises ValueError if the key isn't present."""
    from xpcsjax.optimization.nlsq.heterodyne_views import per_angle_chi2

    result = OptimizationResult(
        parameters=np.array([0.5]),
        uncertainties=np.zeros(1),
        covariance=np.eye(1),
        chi_squared=0.0,
        reduced_chi_squared=0.0,
        convergence_status="converged",
        iterations=0,
        execution_time=0.0,
        device_info={},
        recovery_actions=[],
        quality_flag="good",
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=None,
    )
    with pytest.raises(ValueError, match="chi2_per_angle"):
        per_angle_chi2(result)


# ---------------------------------------------------------------------------
# C2: integration test — Fourier-mode joint fit returns one OptimizationResult
# ---------------------------------------------------------------------------
#
# Self-contained heterodyne config sufficient for HeterodyneModel.from_config.
# Pattern mirrors test_heterodyne_constant_mode.py's B2 helpers — tiny problem
# size, registry-default physics, no external fixtures. ``n_phi=6`` is the
# minimum to keep ``auto`` dispatch in the Fourier window (>= fourier_auto_
# threshold of 6), but the explicit ``per_angle_mode="fourier"`` setting makes
# the dispatch deterministic regardless of threshold defaults.
_C2_N_TIMES = 16
_C2_DT = 1.0
_C2_Q = 0.0054
_C2_PHI_ANGLES = np.linspace(0.0, 150.0, 6, dtype=np.float64)
_C2_NOISE_SIGMA = 5e-4


def _c2_config_dict() -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _C2_DT,
            "start_frame": 1,
            "end_frame": _C2_N_TIMES,
            "scattering": {"wavevector_q": _C2_Q},
        },
        "scaling": {
            "n_angles": len(_C2_PHI_ANGLES),
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 30,
                "enable_cmaes": False,
            },
        },
    }


def _build_minimal_heterodyne_model_for_fourier():
    """Build a minimal HeterodyneModel via the same config path the smoke tests use.

    Pattern mirrors ``_build_minimal_heterodyne_model`` in
    ``test_heterodyne_constant_mode.py`` — duplicated here (rather than imported)
    to keep this test file self-contained; the constant-mode helper uses a
    different ``n_phi`` and time window.
    """
    import tempfile
    from pathlib import Path

    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "c2_fourier.yaml"
        cfg_path.write_text(yaml.safe_dump(_c2_config_dict()))
        cfg = ConfigManager(str(cfg_path))
        assert cfg.config is not None, "ConfigManager.config must not be None"
        return HeterodyneModel.from_config(cfg.config)


def _build_synthetic_c2_stack_for_fourier(
    n_phi: int, n_t: int, model
) -> np.ndarray:
    """Forward-evaluate the model at each phi to build a (n_phi, N, N) stack."""
    assert model.n_times == n_t, (
        f"model.n_times={model.n_times} does not match requested n_t={n_t}"
    )
    rng = np.random.default_rng(seed=20260520)
    c2_stack = np.empty((n_phi, n_t, n_t), dtype=np.float64)
    for i, phi in enumerate(_C2_PHI_ANGLES[:n_phi]):
        c2 = np.asarray(model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        c2_stack[i] = c2 + rng.normal(0.0, _C2_NOISE_SIGMA, size=c2.shape)
    return c2_stack


def test_fourier_mode_returns_single_optimization_result() -> None:
    """``per_angle_mode='fourier'`` returns one OptimizationResult.

    The optimizer parameter vector is
    ``[physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]``
    where each Fourier block has ``2K+1`` coefficients (K = fourier_order).
    Per-angle chi^2 lands in ``nlsq_diagnostics['chi2_per_angle']``, and
    SSR conservation (``chi2_per_angle.sum() == chi_squared``) must hold
    — same invariant as B2's constant-mode result.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    K = 2
    config = NLSQConfig(
        per_angle_mode="fourier", fourier_order=K, max_nfev=30
    )
    n_phi = len(_C2_PHI_ANGLES)
    c2 = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )
    phi = _C2_PHI_ANGLES

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)

    assert isinstance(result, OptimizationResult), (
        f"expected OptimizationResult, got {type(result).__name__}"
    )

    # Parameter vector layout: physics_varying + 2*(2K+1) Fourier coeffs
    # (contrast block + offset block).
    expected_dim = model.param_manager.n_varying + 2 * (2 * K + 1)
    assert result.parameters.shape == (expected_dim,), (
        f"Fourier mode parameter vector should be physics + 2*(2K+1) coeffs; "
        f"got {result.parameters.shape}"
    )

    assert result.nlsq_diagnostics is not None
    diag = result.nlsq_diagnostics
    assert diag["per_angle_mode"] == "fourier"
    # fourier_basis_dim is the per-block coefficient count (2K+1), matching
    # B2's per_angle_mode='constant' convention where it is None and the
    # post-hoc heterodyne_views helpers consume this key.
    assert diag["fourier_basis_dim"] == 2 * K + 1
    assert diag["scaling_source"] == "fitted"
    assert diag["shear_weighting"] == "not_applicable_heterodyne"
    assert "chi2_per_angle" in diag
    assert diag["chi2_per_angle"].shape == (n_phi,)

    # SSR conservation (locked in by B2 — same convention applies here).
    np.testing.assert_allclose(
        diag["chi2_per_angle"].sum(),
        result.chi_squared,
        rtol=1e-6,
        err_msg="chi2_per_angle.sum() must equal chi_squared (SSR conservation)",
    )


# ---------------------------------------------------------------------------
# C3: integration test — averaged-mode joint fit returns one OptimizationResult
# ---------------------------------------------------------------------------
#
# Reuses the C2 fixture builders. The averaged path is taken when
# ``per_angle_mode='auto'`` and ``constant_threshold <= n_phi < fourier_threshold``.
# Optimizer parameter vector is ``[physics_varying | avg_contrast | avg_offset]``
# (2 scaling parameters, not 2*(2K+1) Fourier coefficients).


def test_averaged_path_returns_single_optimization_result() -> None:
    """`per_angle_mode='auto'` with constant_threshold <= n_phi < fourier_threshold returns OptimizationResult."""
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="auto",
        constant_scaling_threshold=3,
        fourier_auto_threshold=6,
        max_nfev=30,
    )
    n_phi = 4  # in the averaged window (3 <= n_phi < 6)
    c2 = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )
    phi = _C2_PHI_ANGLES[:n_phi]

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)

    assert isinstance(result, OptimizationResult), (
        f"expected OptimizationResult, got {type(result).__name__}"
    )
    n_physics = model.param_manager.n_varying
    # Averaged path: physics + 2 scaling parameters
    assert result.parameters.shape == (n_physics + 2,), (
        f"averaged mode adds 2 scaling params; got {result.parameters.shape}"
    )
    diag = result.nlsq_diagnostics
    assert diag is not None
    assert diag["per_angle_mode"] == "averaged"
    assert diag["scaling_source"] == "averaged_then_fitted"
    assert diag["fourier_basis_dim"] is None
    assert diag["shear_weighting"] == "not_applicable_heterodyne"
    assert "chi2_per_angle" in diag
    assert diag["chi2_per_angle"].shape == (n_phi,)
    # Extras specific to averaged mode
    assert "averaged_contrast" in diag
    assert "averaged_offset" in diag
    # SSR conservation (B2's regression lock)
    np.testing.assert_allclose(
        diag["chi2_per_angle"].sum(),
        result.chi_squared,
        rtol=1e-6,
        err_msg="chi2_per_angle.sum() must equal chi_squared (SSR conservation)",
    )


# ---------------------------------------------------------------------------
# C4: stub paths — CMA-ES escape and per-angle multistart — return
# OptimizationResult (not list[NLSQResult] / NLSQResult)
# ---------------------------------------------------------------------------
#
# Both ``_fit_joint_cmaes_multi_phi`` and ``_fit_multistart`` are currently
# Phase-6 stubs that raised ``NotImplementedError``.  The C4 conversion gives
# them minimal viable bodies that delegate to the standard joint Fourier
# fit, so the dispatcher's ``return _fit_*(...)`` arrows can be typed as
# returning ``OptimizationResult`` (and so the final ``fit_nlsq_multi_phi``
# annotation in C5 can be tightened too).
#
# Note that ``_fit_multistart`` is *per-angle* (signature parallels
# ``_fit_local`` / ``_fit_cmaes``), while ``_fit_joint_cmaes_multi_phi`` is
# multi-phi.  The tests reflect that asymmetry.


def test_cmaes_path_returns_single_optimization_result() -> None:
    """CMA-ES escape stub (multi-phi) must return one OptimizationResult."""
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _fit_joint_cmaes_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(
        per_angle_mode="fourier", fourier_order=2, enable_cmaes=True, max_nfev=30
    )
    n_phi = len(_C2_PHI_ANGLES)
    c2 = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )
    phi = _C2_PHI_ANGLES

    result = _fit_joint_cmaes_multi_phi(
        _model=model,
        _c2_data=c2,
        _phi_angles=phi,
        _config=config,
        _weights=None,
    )
    assert isinstance(result, OptimizationResult)


def test_multistart_path_returns_single_optimization_result() -> None:
    """Per-angle multistart stub must return a single OptimizationResult.

    ``_fit_multistart`` is the per-angle global-search entry (signature
    mirrors ``_fit_local`` / ``_fit_cmaes`` — takes a scalar ``phi_angle``,
    not the multi-phi ``phi_angles`` array).  The C4 conversion still
    returns one ``OptimizationResult`` per call.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _fit_multistart

    model = _build_minimal_heterodyne_model_for_fourier()
    config = NLSQConfig(per_angle_mode="fourier", fourier_order=2, max_nfev=30)
    n_phi = len(_C2_PHI_ANGLES)
    c2_stack = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )

    # Per-angle entry: pick a single c2 matrix and a scalar phi.
    result = _fit_multistart(
        _model=model,
        _c2_data=c2_stack[0],
        _phi_angle=float(_C2_PHI_ANGLES[0]),
        _config=config,
        _weights=None,
        _use_nlsq_library=True,
    )
    assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# C5: top-level contract — fit_nlsq_multi_phi returns OptimizationResult
# across all dispatched modes plus the three auto-routing windows.
# ---------------------------------------------------------------------------
#
# Each parametrization exercises a single dispatch branch:
#
# * ``"constant"``   → :func:`_fit_joint_constant_multi_phi`
# * ``"individual"`` → sequential per-angle path (still returns
#   ``list[NLSQResult]``; tracked as Phase-6 follow-up C5b — xfailed below)
# * ``"fourier"``    → :func:`_fit_joint_multi_phi`
# * ``"auto"``       → routed by ``n_phi`` to constant / averaged / fourier
#
# The fourier auto window uses ``n_phi=6`` (the fixture's full angle set,
# matching ``fourier_auto_threshold``); using ``n_phi=8`` would exceed the
# fixture's 6-angle ``_C2_PHI_ANGLES`` array.


@pytest.mark.parametrize(
    "mode,n_phi,fourier_order",
    [
        ("constant", 2, None),
        pytest.param(
            "individual",
            4,
            None,
            marks=pytest.mark.xfail(
                reason=(
                    "individual mode falls through to the sequential per-angle "
                    "warm-start chain which still returns list[NLSQResult]. "
                    "Aggregating that into a single OptimizationResult is "
                    "tracked as Phase-6 follow-up C5b."
                ),
                strict=True,
            ),
        ),
        ("fourier", 6, 2),
        ("auto", 2, None),  # n_phi < constant_scaling_threshold (3) → constant
        ("auto", 4, None),  # constant_threshold <= n_phi < fourier_threshold → averaged
        ("auto", 6, None),  # n_phi >= fourier_auto_threshold (6) → fourier
    ],
)
def test_fit_nlsq_multi_phi_top_level_returns_optimization_result(
    mode: str, n_phi: int, fourier_order: int | None
) -> None:
    """The public entry point returns one OptimizationResult in all modes.

    SSR conservation (``chi2_per_angle.sum() == chi_squared``) — the
    invariant locked in by B2's constant-mode result and reasserted by
    C2's Fourier and C3's averaged tests — is checked uniformly across
    every dispatched branch here.
    """
    pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi

    model = _build_minimal_heterodyne_model_for_fourier()
    kwargs: dict[str, Any] = {"per_angle_mode": mode, "max_nfev": 30}
    if fourier_order is not None:
        kwargs["fourier_order"] = fourier_order
    config = NLSQConfig(**kwargs)
    c2 = _build_synthetic_c2_stack_for_fourier(
        n_phi=n_phi, n_t=_C2_N_TIMES, model=model
    )
    phi = _C2_PHI_ANGLES[:n_phi]

    result = fit_nlsq_multi_phi(model, c2, phi, config, weights=None)
    assert isinstance(result, OptimizationResult)
    assert result.nlsq_diagnostics is not None
    diag = result.nlsq_diagnostics
    assert "chi2_per_angle" in diag
    assert diag["chi2_per_angle"].shape == (n_phi,)
    # SSR conservation across all modes
    np.testing.assert_allclose(
        diag["chi2_per_angle"].sum(), result.chi_squared, rtol=1e-6
    )
