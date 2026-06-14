"""Phase 5 dual-gate (b): default auto->averaged is no-worse-SSR vs explicit individual.

Directionality (spec Risk 2): averaged is MORE constrained (2 scaling DOF vs 2*n_phi),
so on the same data SSR can only stay equal or DEGRADE. "No worse" = the degradation
stays within the parity threshold; it is the INTENDED default change, not a regression.
"""
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

# Reuse the live homodyne config registry + availability gate from the A/B suite.
from tests.characterization.test_homodyne_equivalence import CONFIGS

_NO_WORSE_REL = 1e-3  # ~1e-3 no-worse contract (CLAUDE.md two_component engine-unification)
_GATE_OPT_IN = os.environ.get("XPCSJAX_RUN_AB_PARITY") == "1"
_SKIP = (
    "C020/Simon no-worse oracle needs the registered datasets on disk + "
    "XPCSJAX_RUN_AB_PARITY=1; skips on CI / fresh clones."
)


def _fit(mode, n_phi, n_t=10, seed=11):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.homodyne_model import HomodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    ad = {"enable": True, "per_angle_mode": mode, "constant_scaling_threshold": 3}
    cfg = ConfigManager(config_override={
        "analysis_mode": "laminar_flow",
        "analyzer_parameters": {
            "dt": 0.1, "start_frame": 1, "end_frame": n_t,
            "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": n_t},
            "scattering": {"wavevector_q": 0.0237},
            "geometry": {"stator_rotor_gap": 2000000},
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0",
                                "beta", "gamma_dot_t_offset", "phi0"],
            "values": true.tolist(),
        },
        "optimization": {
            "method": "nlsq",
            "nlsq": {
                "analysis_mode": "laminar_flow", "max_iterations": 80, "loss": "linear",
                "cmaes": {"enable": False, "auto_select": False},
                "multi_start": {"enable": False},
                "anti_degeneracy": ad,
            },
            "stratification": {"enabled": False},
        },
    })
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0))
    c2 = c2 + np.random.default_rng(seed).normal(0.0, 5e-4, size=c2.shape)
    data = {"phi_angles_list": phi, "c2_exp": c2, "t1": t, "t2": t,
            "wavevector_q_list": np.array([0.0237])}
    return fit_nlsq(data, cfg)


def test_synthetic_default_averaged_no_worse_than_individual():
    """CI-safe: averaged SSR must not exceed individual SSR by more than the threshold."""
    res_ind = _fit("individual", n_phi=4)
    res_avg = _fit("auto", n_phi=4)  # -> averaged
    ssr_ind = float(res_ind.chi_squared)
    ssr_avg = float(res_avg.chi_squared)
    # averaged is more-constrained: degrade-or-equal, within ~1e-3 relative band.
    assert ssr_avg <= ssr_ind * (1.0 + _NO_WORSE_REL) + 1e-12, (
        f"averaged SSR {ssr_avg} worse than individual {ssr_ind} beyond no-worse band"
    )
    # sanity: averaged actually ran averaged
    assert dict(res_avg.nlsq_diagnostics or {}).get("per_angle_mode") == "averaged"


@pytest.mark.skipif(not _GATE_OPT_IN, reason=_SKIP)
@pytest.mark.parametrize(
    "label", sorted(k for k, v in CONFIGS.items() if Path(v).exists())
)
def test_c020_simon_default_no_worse(label):
    """Availability-gated: run the real homodyne config under explicit individual vs
    default auto(->averaged); averaged SSR must be no-worse within the band.

    NOTE (Finding 8): only LAMINAR configs exercise the auto->averaged default change.
    Static modes (e.g. `static_simon`) are gated OUT of Phase 5 (the laminar-only
    `analysis_mode == LAMINAR_FLOW` branch keeps individual), so an `auto` fit there
    still resolves individual and SSR is trivially EQUAL — a vacuous pass. Skip static
    labels explicitly so the oracle is honestly scoped to `laminar_c020`; Simon becomes a
    real averaged case only when static unification lands (deferred, spec §9)."""
    if importlib.util.find_spec("homodyne") is None:
        pytest.skip("upstream homodyne not importable")
    from xpcsjax.config import ConfigManager
    from xpcsjax.data import load_xpcs_data
    from xpcsjax.optimization.nlsq import fit_nlsq

    config_path = CONFIGS[label]
    # Skip static configs: auto->averaged does not change a static fit (no flow direction;
    # static keeps individual), so the no-worse comparison is vacuous (SSR equal).
    # NOTE: ConfigManager's path argument is `config_file` (the plan's `config_path=`
    # kwarg is stale; the sibling A/B-parity suite passes the path positionally too).
    _probe = ConfigManager(config_file=config_path)
    if str(_probe.config.get("analysis_mode", "")).startswith("static"):
        pytest.skip(f"{label}: static mode is deferred (auto->averaged is a no-op); not a Phase-5 oracle")
    # individual baseline
    cfg_i = ConfigManager(config_file=config_path)
    cfg_i.config.setdefault("optimization", {}).setdefault("nlsq", {}).setdefault(
        "anti_degeneracy", {}
    ).update({"enable": True, "per_angle_mode": "individual"})
    # load_xpcs_data takes the path (the data load does not depend on per_angle_mode,
    # which only affects the fit); the mutated cfg drives fit_nlsq.
    # The registered laminar config (C020) is a ~23M-point dataset routed to the
    # heavy stratified-LS tier; keep only the scalar SSR and release the data +
    # result between the two fits so two in-process 23M fits don't overcommit RAM
    # (the documented OOM hazard — never run this under pytest-xdist `-n auto`).
    data_i = load_xpcs_data(config_path)
    res_i = fit_nlsq(data_i, cfg_i)
    ssr_i = float(res_i.chi_squared)
    del data_i, res_i, cfg_i
    gc.collect()

    # default auto baseline
    cfg_a = ConfigManager(config_file=config_path)
    cfg_a.config.setdefault("optimization", {}).setdefault("nlsq", {}).setdefault(
        "anti_degeneracy", {}
    ).update({"enable": True, "per_angle_mode": "auto"})
    data_a = load_xpcs_data(config_path)
    res_a = fit_nlsq(data_a, cfg_a)
    ssr_a = float(res_a.chi_squared)
    assert ssr_a <= ssr_i * (1.0 + _NO_WORSE_REL) + 1e-9, (
        f"{label}: auto/averaged SSR {ssr_a} worse than individual {ssr_i} "
        f"beyond no-worse band"
    )
