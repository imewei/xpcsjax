"""Heterodyne + CMA-ES end-to-end smoke test.

Closes the /double-check Phase 5 gap: ``_fit_cmaes`` in
``xpcsjax.optimization.nlsq.heterodyne_core`` was calling
``fit_with_cmaes(objective_fn=..., residual_fn=..., n_data=..., ...)`` with
keyword arguments that the real ``cmaes_wrapper.fit_with_cmaes`` signature
``(model_func, xdata, ydata, p0, bounds, sigma, config)`` never accepted.
Mypy flagged ~25 errors in that function; the smoke suite never reached the
branch (no heterodyne config in ``tests/`` enabled CMA-ES), so the bug was
silently latent. Enabling ``cmaes.enable: true`` on a real two-component
heterodyne config in v0.1 would have crashed with a ``TypeError``.

This test runs the real search on a tiny synthetic one-angle dataset through
the joint escape and asserts:

* The search really ran: the escape is best-effort and falls back to the plain
  joint fit on any exception, which leaves no ``global_escape`` tag.
* The result is a joint ``individual`` result with finite parameters.

Signature drift in the per-angle ``_fit_cmaes`` call (reached only by calling
it directly) is guarded by the tests that call it.

The fit quality envelope is intentionally loose — this is the architectural
sibling of ``test_two_component_smoke``, not a precision test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

# Tiny problem size — keeps the fit under ~30 seconds while still exercising
# the full CMA-ES warm-start → global-search → Phase-3 comparison chain.
# Single phi angle: one angle runs the joint CMA-ES escape of its per-angle
# mode (``individual`` below the auto threshold), the same as several angles.
_N_TIMES = 16
_DT = 1.0
_Q = 0.0054
_PHI_ANGLES = np.array([0.0], dtype=np.float64)
_NOISE_SIGMA = 5e-3


def _cmaes_smoke_config_dict() -> dict:
    """Self-contained heterodyne config with CMA-ES enabled and tight budget.

    The ``cmaes_*`` fields here are the **heterodyne** field names
    (``cmaes_max_iterations`` not ``cmaes_max_generations``,
    ``cmaes_tolx`` not ``cmaes_tol_x``) — the test would silently no-op the
    settings if anyone refactored heterodyne_config.py to match homodyne's
    naming without updating the wrapper.
    """
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
                "enable_cmaes": True,
                "cmaes": {
                    "enable": True,
                    # Heterodyne field-name dialect — keep these in sync with
                    # xpcsjax.optimization.nlsq.heterodyne_config.NLSQConfig.
                    "max_iterations": 25,
                    "population_size": 8,
                    "tolx": 1e-4,
                    "tolfun": 1e-4,
                    "restart_strategy": "none",
                    "max_restarts": 0,
                    # Short-circuit the auto-skip so the CMA-ES Phase 2 actually
                    # runs even if NLSQ warm-start lands a decent chi².
                    "warmstart_auto_skip": False,
                },
            },
        },
    }


def _cmaes_available() -> bool:
    """Skip-gate for hosts without evosax (CPU-only or barebones installs)."""
    try:
        from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAES_AVAILABLE

        return bool(CMAES_AVAILABLE)
    except ImportError:
        return False


@pytest.mark.skipif(
    not _cmaes_available(),
    reason="CMA-ES backend (evosax) not installed; per-angle CMA-ES path not testable",
)
def test_heterodyne_one_angle_cmaes_runs_the_joint_search(tmp_path: Path) -> None:
    """End-to-end: one angle with CMA-ES runs the joint search to completion."""
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    cfg_path = tmp_path / "cmaes_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(_cmaes_smoke_config_dict()))
    cfg = ConfigManager(str(cfg_path))
    assert cfg.config is not None, "ConfigManager.config typed Optional; runtime non-None"

    # Synthesise self-consistent c2 at the model's initial parameters.
    truth_model = HeterodyneModel.from_config(cfg.config)
    rng = np.random.default_rng(seed=20260519)
    c2_stack = np.empty((len(_PHI_ANGLES), _N_TIMES, _N_TIMES), dtype=np.float64)
    for i, phi in enumerate(_PHI_ANGLES):
        c2 = np.asarray(truth_model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        c2_stack[i] = c2 + rng.normal(0.0, _NOISE_SIGMA, size=c2.shape)

    results = fit_nlsq({"c2": c2_stack, "phi": _PHI_ANGLES}, cfg)

    assert isinstance(results, OptimizationResult), (
        f"expected OptimizationResult, got {type(results)}"
    )
    diag = results.nlsq_diagnostics or {}
    assert diag.get("per_angle_mode") == "individual"
    assert diag.get("global_escape") in {"cmaes", "cmaes_warmstart_kept"}, (
        f"the joint CMA-ES search did not run; got {diag.get('global_escape')!r}"
    )
    params = np.asarray(results.parameters, dtype=np.float64)
    assert diag["parameter_names"][:2] == ["contrast_0", "offset_0"]
    assert np.all(np.isfinite(params)), f"NaN/Inf in fitted parameters: {params}"
    assert np.isfinite(results.chi_squared), (
        f"CMA-ES path must return finite chi_squared; got {results.chi_squared!r}"
    )
    assert results.chi_squared < 5.0, (
        f"CMA-ES chi_squared must remain bounded on the smoke fixture; "
        f"got {results.chi_squared:.2e} — investigate a stochastic-search "
        f"regression or fixture corruption."
    )
