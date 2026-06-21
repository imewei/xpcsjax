"""CMA-ES auto-triggers at scale_ratio >= 1000 (homodyne default).

XPCS multi-scale problems span >3 orders of magnitude (e.g., D0 ~ 1e4 vs
gamma_dot ~ 1e-3 → ratio ~ 1e7). This is the documented escape hatch; we
verify it directly so a regression localizes to the trigger function rather
than only surfacing via characterization."""

import inspect

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapper


@pytest.fixture()
def wrapper():
    return CMAESWrapper()


def test_high_scale_ratio_triggers_cmaes(wrapper):
    """Realistic XPCS multi-scale bounds (D0 ~ 1e4 vs gamma_dot ~ 1e-3) must trigger.

    scale_ratio here measures spread of parameter widths across the parameter
    vector — not range within a single parameter. Three params spanning 9
    orders of magnitude reliably clear the 1000 threshold."""
    lower = np.array([1.0e2, 1.0e-4, -0.5])
    upper = np.array([5.0e4, 1.0, 0.5])
    assert wrapper.should_use_cmaes((lower, upper)), (
        f"multi-scale XPCS bounds must enable CMA-ES "
        f"(scale_ratio={wrapper.compute_scale_ratio((lower, upper))})"
    )


def test_low_scale_ratio_does_not_trigger(wrapper):
    """Tightly-clustered parameter widths must NOT enable CMA-ES.

    All three parameters with width ≈ 1 — scale_ratio = 1, below 1000 threshold."""
    lower = np.array([1.0, 2.0, 3.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert not wrapper.should_use_cmaes((lower, upper)), (
        f"unimodal-scale bounds must not enable CMA-ES "
        f"(scale_ratio={wrapper.compute_scale_ratio((lower, upper))})"
    )


def test_default_threshold_is_1000():
    """The documented default scale_threshold is 1000.0."""
    sig = inspect.signature(CMAESWrapper.should_use_cmaes)
    threshold_param = next(
        (
            p
            for name, p in sig.parameters.items()
            if "threshold" in name.lower() or "scale_thr" in name.lower()
        ),
        None,
    )
    assert threshold_param is not None, (
        "CMAESWrapper.should_use_cmaes has no threshold parameter — "
        "homodyne's documented API has scale_threshold=1000.0 by default."
    )
    assert threshold_param.default == pytest.approx(1000.0), (
        f"default scale_threshold drifted from documented 1000.0 to {threshold_param.default}"
    )


def test_compute_scale_ratio_increases_with_spread(wrapper):
    """compute_scale_ratio reports parameter-width spread; wider spread → higher ratio.

    Two parameters of width 1 → ratio = 1. Add a parameter of width 1e6 →
    ratio explodes. Verifies the spread metric responds monotonically."""
    tight_lower = np.array([1.0, 2.0])
    tight_upper = np.array([2.0, 3.0])
    tight_ratio = wrapper.compute_scale_ratio((tight_lower, tight_upper))

    wide_lower = np.array([1.0, 1.0e-4])
    wide_upper = np.array([2.0, 1.0e3])
    wide_ratio = wrapper.compute_scale_ratio((wide_lower, wide_upper))

    assert wide_ratio > tight_ratio * 100, (
        f"compute_scale_ratio should respond to width spread: "
        f"tight={tight_ratio}, wide={wide_ratio}"
    )


# --- Phase 6: laminar CMA-ES surviving per-angle modes ---


def _laminar_cmaes_config(per_angle_mode: str):
    from xpcsjax.config import ConfigManager

    return ConfigManager(
        config_override={
            "analysis_mode": "laminar_flow",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 10,
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 10},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2_000_000.0},
            },
            "optimization": {
                "method": "nlsq",
                "nlsq": {
                    "max_iterations": 20,
                    "cmaes": {"enable": True, "auto_select": False},
                    "multi_start": {"enable": False},
                    "anti_degeneracy": {
                        "enable": True,
                        "per_angle_mode": per_angle_mode,
                        "constant_scaling_threshold": 3,
                    },
                },
                "stratification": {"enabled": False},
            },
        }
    )


def _tiny_laminar_data(n_phi=5, n_t=10):
    from xpcsjax.core.homodyne_model import HomodyneModel

    phi = np.linspace(0.0, 144.0, n_phi)
    t = np.linspace(0.0, float(n_t - 1), n_t)
    truth = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    cfg = _laminar_cmaes_config("individual")
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(truth, phi, contrast=0.3, offset=1.0), dtype=np.float64)
    return {
        "phi_angles_list": phi,
        "wavevector_q_list": np.array([0.0237]),
        "t1": t,
        "t2": t,
        "c2_exp": c2,
    }


@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_cmaes_laminar_result_param_count(mode):
    """All resolved per-angle modes expand back to the dense per-angle
    2*n_phi + n_physical layout.

    ``constant`` (fixed-scaling) reduces the CMA-ES optimizer vector to
    physics-only (n_physical); the post-solve expansion must reconstruct the
    dense layout from the frozen per-angle scaling rather than crashing in
    ``expand_per_angle_parameters`` (which would otherwise be swallowed into a
    failed result).
    """
    from xpcsjax.optimization.nlsq.core import fit_nlsq_cmaes

    res = fit_nlsq_cmaes(_tiny_laminar_data(), _laminar_cmaes_config(mode))
    assert len(res.parameters) == 2 * 5 + 7


@pytest.mark.parametrize("mode", ["individual", "auto", "constant"])
def test_cmaes_laminar_does_not_silently_fail(mode):
    """A CMA-ES solve that actually ran must not be discarded into a failed
    result. ``constant`` mode previously raised ValueError in the post-solve
    expansion (wrong param-count contract), got caught, and returned
    ``_cmaes_failed_result`` (inf chi-squared) despite a converged solve."""
    from xpcsjax.optimization.nlsq.core import fit_nlsq_cmaes

    res = fit_nlsq_cmaes(_tiny_laminar_data(), _laminar_cmaes_config(mode))
    assert res.convergence_status == "converged", (
        f"mode={mode}: converged solve was discarded into a failed result "
        f"(status={res.convergence_status}, chi2={res.chi_squared})"
    )
    assert np.isfinite(res.chi_squared), f"mode={mode}: chi_squared={res.chi_squared}"


def test_cmaes_fixed_constant_survives_none_covariance(monkeypatch):
    """A converged fixed-constant CMA-ES solve whose covariance is ``None`` must
    not be discarded into a failed result (codex adversarial review).

    When CMA-ES runs without L-M refinement, ``CMAESResult.covariance`` is
    ``None`` (it comes from ``result.get("pcov", None)``). Fixed-constant mode
    expands ``final_params`` to the dense ``2*n_phi + n_physical`` layout, but
    the ``None``-covariance placeholders (``np.zeros`` / ``np.eye``) were sized
    by the EFFECTIVE constrained DOF (``n_physical``) rather than the expanded
    length. ``OptimizationResult.__post_init__`` then raised a shape ValueError
    that the surrounding ``except ValueError`` silently converted into
    ``_cmaes_failed_result`` — losing an otherwise-successful global search.
    """
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESResult, CMAESWrapper
    from xpcsjax.optimization.nlsq.core import fit_nlsq_cmaes

    n_phi, n_physical = 5, 7

    def _fake_fit(self, **_kwargs):
        # Fixed-constant optimizer vector is physics-only; no refinement ran,
        # so pcov is absent -> covariance None.
        return CMAESResult(
            parameters=np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0]),
            covariance=None,
            chi_squared=1.0,
            success=True,
            diagnostics={},
            nlsq_refined=False,
        )

    monkeypatch.setattr(CMAESWrapper, "fit", _fake_fit)

    cfg = _laminar_cmaes_config("constant")
    # Disable the NLSQ warm-start so the (covariance=None) CMA-ES result is the
    # one finalized; otherwise a warm-start covariance would mask the defect.
    cfg.config["optimization"]["nlsq"]["cmaes"]["nlsq_warmstart"] = False

    res = fit_nlsq_cmaes(_tiny_laminar_data(n_phi=n_phi), cfg)

    assert res.convergence_status == "converged", (
        f"converged fixed-constant solve with None covariance was discarded "
        f"(status={res.convergence_status}, chi2={res.chi_squared})"
    )
    assert np.isfinite(res.chi_squared)
    # Dense per-angle layout, with placeholders sized to MATCH it.
    assert len(res.parameters) == 2 * n_phi + n_physical
    assert len(res.uncertainties) == len(res.parameters)
    assert np.asarray(res.covariance).shape == (len(res.parameters), len(res.parameters))
