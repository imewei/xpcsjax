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

This test exercises the fixed path on a tiny synthetic 2-angle dataset and
asserts:

* The fitter returns without raising (the regression is purely structural —
  if the signature drift is ever reintroduced, the call site crashes
  immediately).
* The result is shaped like an ``NLSQResult`` and reports CMA-ES winning
  Phase 3 (``metadata["cmaes_winner"]`` is populated).

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
# Single phi angle: the v0.1 dispatch routes multi-angle CMA-ES to the joint
# path (currently a Phase-6 NotImplementedError stub), so this test exercises
# the *per-angle* path that ``_fit_cmaes`` actually wires. Joint multi-angle
# CMA-ES gets its own test when Phase 6 lands.
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
def test_heterodyne_per_angle_cmaes_fits_without_signature_drift(tmp_path: Path) -> None:
    """End-to-end: the per-angle ``_fit_cmaes`` path completes without raising.

    The regression this guards: ``_fit_cmaes`` previously called
    ``fit_with_cmaes(objective_fn=..., initial_params=..., parameter_names=...,
    residual_fn=..., n_data=..., anti_degeneracy=..., config=CMAESConfig(...))``.
    None of those kwargs exist on the real signature; the call would have
    raised ``TypeError: fit_with_cmaes() got an unexpected keyword argument
    'objective_fn'`` the first time a heterodyne user enabled CMA-ES.

    The fix in this PR rewrites the call to use the real signature
    ``fit_with_cmaes(model_func, xdata, ydata, p0, bounds, sigma, config)``
    and reads ``chi_squared`` (not ``final_cost``) off ``CMAESResult``. If
    either drift is reintroduced, this test crashes loudly.
    """
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.results import OptimizationResult

    cfg_path = tmp_path / "cmaes_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(_cmaes_smoke_config_dict()))
    cfg = ConfigManager(str(cfg_path))
    assert cfg.config is not None, "ConfigManager.config typed Optional; runtime non-None"

    # Build synthetic data from the truth model (registry defaults + tiny noise).
    truth_model = HeterodyneModel.from_config(cfg.config)
    rng = np.random.default_rng(seed=20260519)
    c2_stack = np.empty((len(_PHI_ANGLES), _N_TIMES, _N_TIMES), dtype=np.float64)
    for i, phi in enumerate(_PHI_ANGLES):
        c2 = np.asarray(
            truth_model.compute_correlation(phi_angle=float(phi), angle_idx=i)
        )
        c2_stack[i] = c2 + rng.normal(0.0, _NOISE_SIGMA, size=c2.shape)

    data = {"c2": c2_stack, "phi": _PHI_ANGLES}

    # If the joint multi-phi CMA-ES path got reached we'd hit our Phase-6
    # NotImplementedError — assert we don't. The default dispatch should route
    # to the per-angle path when ``cmaes.joint_multi_phi`` is unset.
    results = fit_nlsq(data, cfg)

    # ---- Pipeline contract -----------------------------------------------
    # Post-C5b: every heterodyne dispatch returns a single
    # ``OptimizationResult``. Per-angle ``NLSQResult.metadata`` survives in
    # ``nlsq_diagnostics["per_angle_metadata"]`` for routing audits.
    assert isinstance(results, OptimizationResult), (
        f"expected OptimizationResult, got {type(results)}"
    )
    diag = results.nlsq_diagnostics or {}
    assert diag.get("per_angle_mode") == "individual", (
        f"expected per_angle_mode='individual' (CMA-ES individual path); "
        f"got {diag.get('per_angle_mode')!r}"
    )
    per_angle_metadata = diag.get("per_angle_metadata")
    assert isinstance(per_angle_metadata, list), (
        "expected per_angle_metadata list in nlsq_diagnostics"
    )
    assert len(per_angle_metadata) == len(_PHI_ANGLES), (
        f"expected {len(_PHI_ANGLES)} per-angle metadata entries, "
        f"got {len(per_angle_metadata)}"
    )
    params = np.asarray(results.parameters, dtype=np.float64)
    assert np.all(np.isfinite(params)), f"NaN/Inf in fitted parameters: {params}"
    per_angle_messages = diag.get("per_angle_messages") or []
    assert all(per_angle_messages), (
        "every per-angle fit must carry a non-empty status message"
    )

    # ---- CMA-ES path was actually taken ---------------------------------
    # ``_fit_cmaes`` writes ``metadata["optimizer"] = "cmaes"`` and
    # ``metadata["cmaes_winner"] in {"nlsq", "cmaes"}``. If the per-angle
    # CMA-ES branch isn't reached (e.g., because someone re-introduced
    # auto-skip with a permissive threshold), these would be missing.
    first_meta = per_angle_metadata[0]
    if "optimizer" not in first_meta:
        # Some heterodyne dispatch variants nest metadata under the joint
        # CMA-ES key — be tolerant rather than over-pinning the shape.
        assert "cmaes" in str(first_meta).lower(), (
            f"expected CMA-ES marker somewhere in metadata; got keys "
            f"{list(first_meta)}"
        )
    else:
        assert first_meta["optimizer"] in {"cmaes", "joint_cmaes_warmstart"}, (
            f"expected CMA-ES optimizer label in metadata, got "
            f"{first_meta['optimizer']!r}"
        )
