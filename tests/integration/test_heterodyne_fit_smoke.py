"""Integration smoke tests: real heterodyne NLSQ fits on tiny synthetic data.

Unlike the unit suites (which mock the optimizer boundary), these run an actual
NLSQ optimization end-to-end on a small two-component dataset. They exercise the
fit-orchestration paths that unit tests cannot reach: the single-angle entry
(`fit_nlsq_jax` -> `_fit_local` -> NLSQ adapter) and the multi-angle dispatch
(`fit_nlsq_multi_phi` -> individual / auto->fourier joint fits).

Kept fast via tiny grids and `max_nfev=30` — these assert the result *contract*
(type, shape, finiteness, diagnostics), not convergence quality (covered by the
dedicated parity/constant-mode suites).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xpcsjax.core.heterodyne_model_stateful")

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig  # noqa: E402
from xpcsjax.optimization.nlsq.heterodyne_core import (  # noqa: E402
    fit_nlsq_jax,
    fit_nlsq_multi_phi,
)
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult  # noqa: E402
from xpcsjax.optimization.nlsq.results import OptimizationResult  # noqa: E402

_N_TIMES = 16
_DT = 1.0
_Q = 0.0054
_PHI_ANGLES = np.linspace(0.0, 150.0, 6, dtype=np.float64)
_NOISE = 5e-4


def _config_dict() -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _DT,
            "start_frame": 1,
            "end_frame": _N_TIMES,
            "scattering": {"wavevector_q": _Q},
        },
        "scaling": {
            "n_angles": len(_PHI_ANGLES),
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


def _build_model():
    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(_config_dict()))
        cfg = ConfigManager(str(cfg_path))
        assert cfg.config is not None
        return HeterodyneModel.from_config(cfg.config)


def _synthetic_stack(model, n_phi: int) -> np.ndarray:
    rng = np.random.default_rng(seed=20260524)
    stack = np.empty((n_phi, _N_TIMES, _N_TIMES), dtype=np.float64)
    for i, phi in enumerate(_PHI_ANGLES[:n_phi]):
        c2 = np.asarray(model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        stack[i] = c2 + rng.normal(0.0, _NOISE, size=c2.shape)
    return stack


# ---------------------------------------------------------------------------
# single-angle entry: fit_nlsq_jax -> _fit_local -> adapter
# ---------------------------------------------------------------------------


def test_single_angle_fit_local_runs() -> None:
    model = _build_model()
    c2 = _synthetic_stack(model, n_phi=1)[0]  # (N, N)
    config = NLSQConfig(max_nfev=30, enable_cmaes=False, multistart=False)

    result = fit_nlsq_jax(model, c2, phi_angle=float(_PHI_ANGLES[0]), config=config)

    assert isinstance(result, NLSQResult)
    assert np.all(np.isfinite(result.parameters))
    assert result.parameters.shape[0] >= model.param_manager.n_varying
    assert result.reduced_chi_squared is None or np.isfinite(result.reduced_chi_squared)


# ---------------------------------------------------------------------------
# multi-angle dispatch: fit_nlsq_multi_phi
# ---------------------------------------------------------------------------


def test_individual_mode_joint_fit() -> None:
    model = _build_model()
    n_phi = 3
    c2 = _synthetic_stack(model, n_phi=n_phi)
    config = NLSQConfig(per_angle_mode="individual", max_nfev=30, enable_cmaes=False)

    result = fit_nlsq_multi_phi(model, c2, _PHI_ANGLES[:n_phi], config, weights=None)

    assert isinstance(result, OptimizationResult)
    # individual mode: physics + 2*n_phi (contrast + offset per angle)
    assert result.parameters.shape == (model.param_manager.n_varying + 2 * n_phi,)
    assert np.all(np.isfinite(result.parameters))
    diag = result.nlsq_diagnostics
    assert diag is not None and diag["per_angle_mode"] == "individual"


def test_auto_mode_resolves_to_averaged_for_many_angles() -> None:
    # Unified auto rule: n_phi >= 3 -> averaged, regardless of how large.
    # Even at n_phi=6 (the OLD fourier_auto_threshold) auto stays "averaged";
    # fourier is selected only when the user requests it explicitly.
    model = _build_model()
    n_phi = 6
    c2 = _synthetic_stack(model, n_phi=n_phi)
    config = NLSQConfig(per_angle_mode="auto", max_nfev=30, enable_cmaes=False)

    result = fit_nlsq_multi_phi(model, c2, _PHI_ANGLES[:n_phi], config, weights=None)

    assert isinstance(result, OptimizationResult)
    assert np.all(np.isfinite(result.parameters))
    diag = result.nlsq_diagnostics
    assert diag is not None and diag["per_angle_mode"] == "averaged"


def test_cmaes_path_runs() -> None:
    # enable_cmaes routes the single-angle entry through the CMA-ES branch
    # (_fit_cmaes). Tiny budget keeps it fast; we assert the result contract.
    model = _build_model()
    c2 = _synthetic_stack(model, n_phi=1)[0]
    config = NLSQConfig(
        enable_cmaes=True,
        cmaes_max_iterations=5,
        cmaes_population_size=8,
        cmaes_restart_strategy="none",
        cmaes_max_restarts=0,
        multistart=False,
        max_nfev=30,
    )
    result = fit_nlsq_jax(model, c2, phi_angle=float(_PHI_ANGLES[0]), config=config)
    assert isinstance(result, NLSQResult)
    assert np.all(np.isfinite(result.parameters))


def test_multistart_path_runs() -> None:
    # multistart routes the single-angle entry through the multi-start branch
    # (_fit_multistart). Two starts is enough to exercise the orchestration.
    model = _build_model()
    c2 = _synthetic_stack(model, n_phi=1)[0]
    config = NLSQConfig(
        multistart=True, multistart_n=2, enable_cmaes=False, max_nfev=30
    )
    result = fit_nlsq_jax(model, c2, phi_angle=float(_PHI_ANGLES[0]), config=config)
    assert isinstance(result, NLSQResult)
    assert np.all(np.isfinite(result.parameters))


def test_multi_phi_ssr_conservation() -> None:
    # SSR conservation: per-angle chi2 sums to total chi_squared (a cross-cutting
    # invariant of the joint-fit result assembly).
    model = _build_model()
    n_phi = 3
    c2 = _synthetic_stack(model, n_phi=n_phi)
    config = NLSQConfig(per_angle_mode="individual", max_nfev=30, enable_cmaes=False)
    result = fit_nlsq_multi_phi(model, c2, _PHI_ANGLES[:n_phi], config, weights=None)
    diag = result.nlsq_diagnostics
    assert diag is not None
    assert diag["chi2_per_angle"].shape == (n_phi,)
    np.testing.assert_allclose(
        diag["chi2_per_angle"].sum(), result.chi_squared, rtol=1e-6
    )
