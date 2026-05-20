"""Tests for true `constant` mode in heterodyne (quantile-frozen scaling)."""
from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.core.heterodyne_scaling_utils import (
    estimate_per_angle_scaling_from_quantile,
)


def _make_synthetic_c2(n_phi: int = 3, n_t: int = 32, seed: int = 0) -> dict:
    """Build a tiny synthetic two-time correlation stack for unit tests."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_t)
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    # Toy heterodyne forward: c2 = offset + contrast * exp(-D * |t1-t2|)
    true_contrast = np.array([0.45, 0.42, 0.40][:n_phi])
    true_offset = np.array([1.00, 1.00, 1.00][:n_phi])
    # D = 10 over t in [0, 1] gives max-lag decay = exp(-10) ~ 5e-5, so the
    # large-lag corner c2 ~ offset. Lower D leaves residual signal in the
    # corners and the dual-region offset estimate cannot reach the true value.
    D = 10.0
    decay = np.exp(-D * np.abs(t1 - t2))
    c2 = np.stack(
        [true_offset[i] + true_contrast[i] * decay for i in range(n_phi)], axis=0
    )
    c2 += 0.005 * rng.standard_normal(c2.shape)
    return {
        "c2": c2,
        "t1": np.broadcast_to(t1, c2.shape).copy(),
        "t2": np.broadcast_to(t2, c2.shape).copy(),
        "phi_indices": np.repeat(np.arange(n_phi), n_t * n_t).reshape(c2.shape),
        "true_contrast": true_contrast,
        "true_offset": true_offset,
    }


def test_quantile_estimator_recovers_synthetic_contrast() -> None:
    """Dual-region quantile estimator recovers per-angle contrast and offset.

    Small-lag high-quantile gives the ceiling (offset + contrast); large-lag
    low-quantile gives the floor (offset after decay). Their difference
    recovers contrast.
    """
    data = _make_synthetic_c2(n_phi=3)
    contrast_hat, offset_hat = estimate_per_angle_scaling_from_quantile(
        c2_data=data["c2"],
        t1=data["t1"],
        t2=data["t2"],
        phi_indices=data["phi_indices"],
        n_phi=3,
        quantile=0.95,
    )

    assert contrast_hat.shape == (3,)
    assert offset_hat.shape == (3,)
    # Tolerance reflects synthetic noise level (sigma=0.005, contrast~0.4).
    np.testing.assert_allclose(contrast_hat, data["true_contrast"], rtol=0.10)
    np.testing.assert_allclose(offset_hat, data["true_offset"], rtol=0.02)


def test_quantile_estimator_raises_on_empty_phi_cell() -> None:
    """A phi index with no samples in `phi_indices` is a malformed input."""
    data = _make_synthetic_c2(n_phi=2)
    # phi_indices claims n_phi=2 but the array assigns all samples to index 0
    bad_phi_indices = np.zeros_like(data["phi_indices"])
    with pytest.raises(
        ValueError, match=r"no samples for phi index 1|only.*finite samples"
    ):
        estimate_per_angle_scaling_from_quantile(
            c2_data=data["c2"],
            t1=data["t1"],
            t2=data["t2"],
            phi_indices=bad_phi_indices,
            n_phi=2,
        )


def test_quantile_estimator_diagonal_only_fails_to_recover_offset() -> None:
    """Locks in the dual-region rationale: diagonal-only input cannot recover offset.

    If a future contributor optimizes the wrapper to use only diagonal samples,
    this test will fail and force the regression review.
    """
    # Build a fixture where t1 == t2 everywhere — i.e., only diagonal samples.
    # The dual-region estimator's large-lag region has no data, so it should
    # either raise (preferred) or return a wildly wrong offset.
    n_phi, n_t = 2, 256
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n_t)
    # All samples are at t1 = t2 = same point → no off-diagonal coverage
    c2_diag = np.stack(
        [0.45 + 1.0 + 0.005 * rng.standard_normal(n_t) for _ in range(n_phi)]
    )
    t1 = np.broadcast_to(t, c2_diag.shape).copy()
    t2 = t1.copy()
    phi_indices = np.repeat(np.arange(n_phi), n_t).reshape(c2_diag.shape)

    # Expect either ValueError (insufficient samples in large-lag region after
    # the guard fires) or a wildly-wrong offset that fails the rtol=0.02 check.
    try:
        contrast_hat, offset_hat = estimate_per_angle_scaling_from_quantile(
            c2_data=c2_diag,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
            n_phi=n_phi,
        )
    except ValueError:
        # Acceptable outcome — guard caught the malformed input.
        return
    # If no exception, the offset estimate must NOT pass the tight tolerance:
    del contrast_hat  # not asserted here; the lock-in is on offset
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(offset_hat, [1.0, 1.0], rtol=0.02)


# ---------------------------------------------------------------------------
# B2: integration test — the true `constant` mode fit
# ---------------------------------------------------------------------------
#
# Self-contained heterodyne config sufficient for HeterodyneModel.from_config.
# Pattern mirrors tests/heterodyne/test_two_component_smoke.py — tiny problem
# size, registry-default physics, no external fixtures. The `scaling.mode` is
# left at "constant" because it controls PerAngleScaling.from_config; this is
# distinct from NLSQConfig.per_angle_mode which is set by the test directly.
_B2_N_TIMES = 24
_B2_DT = 1.0
_B2_Q = 0.0054
_B2_PHI_ANGLES = np.array([0.0, 30.0, 60.0], dtype=np.float64)
_B2_NOISE_SIGMA = 5e-4


def _b2_config_dict() -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _B2_DT,
            "start_frame": 1,
            "end_frame": _B2_N_TIMES,
            "scattering": {"wavevector_q": _B2_Q},
        },
        "scaling": {
            "n_angles": len(_B2_PHI_ANGLES),
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


def _build_minimal_heterodyne_model():
    """Build a minimal HeterodyneModel via the same config path the smoke tests use.

    Returns a ``HeterodyneModel`` instance constructed via
    ``HeterodyneModel.from_config(cfg.config)`` where ``cfg`` is a
    ``ConfigManager`` loaded from a temporary YAML file. The proven
    integration-test construction pattern; see
    ``tests/heterodyne/test_two_component_smoke.py``.

    Uses ``tempfile`` rather than the pytest ``tmp_path`` fixture so callers
    can be plain functions — the reviewer wanted the second (slow) test to
    call this helper without taking a fixture arg.
    """
    import tempfile
    from pathlib import Path

    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "b2_constant.yaml"
        cfg_path.write_text(yaml.safe_dump(_b2_config_dict()))
        cfg = ConfigManager(str(cfg_path))
        # Pyright: cfg.config narrowing — the manager promises a populated
        # dict once a path is loaded, but the type is ``dict | None``.
        assert cfg.config is not None, "ConfigManager.config must not be None"
        return HeterodyneModel.from_config(cfg.config)


def _build_synthetic_c2_stack(n_phi: int, n_t: int, model) -> np.ndarray:  # noqa: ARG001
    """Forward-evaluate the model at each phi to build a (n_phi, N, N) stack.

    ``n_t`` is consumed implicitly via ``model.n_times``; the parameter is
    accepted for caller readability (matches the reviewer's snippet) and
    asserted-equal so a mismatch surfaces immediately.
    """
    assert model.n_times == n_t, (
        f"model.n_times={model.n_times} does not match requested n_t={n_t}"
    )
    rng = np.random.default_rng(seed=20260520)
    c2_stack = np.empty((n_phi, n_t, n_t), dtype=np.float64)
    for i, phi in enumerate(_B2_PHI_ANGLES[:n_phi]):
        c2 = np.asarray(model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        c2_stack[i] = c2 + rng.normal(0.0, _B2_NOISE_SIGMA, size=c2.shape)
    return c2_stack


def _expected_synthetic_scaling(model, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (contrast, offset) arrays the synthetic builder used.

    ``_build_synthetic_c2_stack`` forward-evaluates ``model.compute_correlation``,
    which reads contrast/offset from ``model.scaling``. So the ground-truth
    scaling for the recovery assertion is whatever the freshly-built model
    carries in ``model.scaling.contrast / offset`` (truncated to ``n_phi``).
    """
    contrast = np.asarray(model.scaling.contrast[:n_phi], dtype=np.float64)
    offset = np.asarray(model.scaling.offset[:n_phi], dtype=np.float64)
    return contrast, offset


def test_constant_mode_fit_optimizes_only_physics() -> None:
    """``_fit_joint_constant_multi_phi`` returns OptimizationResult with
    ``parameters.shape == (n_physics_varying,)`` — scaling is frozen, not in
    the optimizer vector.

    Fast structural test (~7s). Convergence quality is asserted separately in
    ``test_constant_mode_recovers_synthetic_truth``.
    """
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _fit_joint_constant_multi_phi,
    )
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    model = _build_minimal_heterodyne_model()
    n_phi = len(_B2_PHI_ANGLES)
    c2_data = _build_synthetic_c2_stack(n_phi=n_phi, n_t=_B2_N_TIMES, model=model)
    config = NLSQConfig(per_angle_mode="constant", max_nfev=30)
    # max_nfev=30 keeps this test fast (~7s); we only verify the
    # OptimizationResult shape/diagnostic contract here. Convergence quality
    # is tested separately in test_constant_mode_recovers_synthetic_truth.

    result = _fit_joint_constant_multi_phi(
        model=model,
        c2_data=c2_data,
        phi_angles=_B2_PHI_ANGLES,
        config=config,
        weights=None,
    )

    assert isinstance(result, OptimizationResult)
    n_physics = model.param_manager.n_varying
    assert result.parameters.shape == (n_physics,), (
        f"constant mode must optimize only physics; got {result.parameters.shape}"
    )
    assert result.nlsq_diagnostics is not None
    diag = result.nlsq_diagnostics
    assert diag["scaling_source"] == "quantile_fixed"
    assert diag["per_angle_mode"] == "constant"
    assert diag["fourier_basis_dim"] is None
    assert diag["shear_weighting"] == "not_applicable_heterodyne"
    assert "contrast_per_angle_fixed" in diag
    assert "offset_per_angle_fixed" in diag
    assert diag["contrast_per_angle_fixed"].shape == (n_phi,)
    assert diag["offset_per_angle_fixed"].shape == (n_phi,)
    assert "chi2_per_angle" in diag
    assert diag["chi2_per_angle"].shape == (n_phi,)

    # SSR conservation: per-angle chi^2 must sum to the global chi_squared.
    # Locks in the SSR convention in heterodyne_constant_mode.py — chi_squared
    # is computed from the raw final-residual SSR (not 2*final_cost) so the
    # invariant holds for both linear and robust-loss configurations.
    np.testing.assert_allclose(
        diag["chi2_per_angle"].sum(),
        result.chi_squared,
        rtol=1e-6,
        err_msg="chi2_per_angle.sum() must equal chi_squared (SSR conservation)",
    )


def test_constant_mode_recovers_synthetic_truth() -> None:
    """End-to-end: converging constant-mode fit recovers synthetic ground truth.

    This is the slow test (max_nfev=200, ~15-30s). It complements the fast
    structural test and catches bugs where shape is correct but values are
    wrong — specifically, drift in either the quantile estimator or the
    physics-only residual closure.
    """
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _fit_joint_constant_multi_phi,
    )
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    model = _build_minimal_heterodyne_model()
    n_phi = len(_B2_PHI_ANGLES)
    c2_data = _build_synthetic_c2_stack(n_phi=n_phi, n_t=_B2_N_TIMES, model=model)
    config = NLSQConfig(per_angle_mode="constant", max_nfev=200)

    result = _fit_joint_constant_multi_phi(
        model=model,
        c2_data=c2_data,
        phi_angles=_B2_PHI_ANGLES,
        config=config,
        weights=None,
    )

    assert isinstance(result, OptimizationResult)
    # If the fitter converged, chi_squared should be small relative to the
    # synthetic noise floor. NOISE_SIGMA=5e-4 over n_phi*n_t*(n_t-1) ~ 1656
    # residuals gives an expected SSR of order n_residuals*sigma^2 ~ 4e-4.
    # 1.0 is a very generous ceiling — catches "stuck at initial point" without
    # over-pinning convergence quality.
    assert result.chi_squared < 1.0, (
        f"converged chi_squared should be small, got {result.chi_squared}"
    )
    # Frozen scaling should be close to the model's true scaling values
    # (within the quantile-estimator tolerance — wider on contrast than offset).
    diag = result.nlsq_diagnostics
    assert diag is not None
    true_contrast, true_offset = _expected_synthetic_scaling(model, n_phi)
    np.testing.assert_allclose(
        diag["contrast_per_angle_fixed"],
        true_contrast,
        rtol=0.15,
        err_msg="quantile-estimated contrast deviates from synthetic ground truth",
    )
    np.testing.assert_allclose(
        diag["offset_per_angle_fixed"],
        true_offset,
        rtol=0.05,
        err_msg="quantile-estimated offset deviates from synthetic ground truth",
    )
