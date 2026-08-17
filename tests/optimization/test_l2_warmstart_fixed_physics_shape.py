"""Regression test for the C045 crash (2026-08-17): the L2 Stage-1 warm-start
delegates to ``_fit_joint_constant_multi_phi``, whose ``OptimizationResult.parameters``
is always the FULL 14-physics vector (``expand_reduced_result``). When any
physics parameter is fixed (``initial_parameters.active_parameters`` narrower
than all 14 names, e.g. the reference-transport triplet ``D0_ref``/``alpha_ref``/
``D_offset_ref`` held fixed), ``physics_lower``/``physics_upper`` are
varying-only (11 here) and ``np.clip(stage1_physics, physics_lower,
physics_upper)`` raised ``ValueError: operands could not be broadcast
together with shapes (14,) (11,) (11,)`` in both call sites
(``_build_joint_problem`` for individual mode, ``_fit_joint_averaged_multi_phi``
for averaged mode). Fixed by reducing ``stage1_result.parameters`` back to the
varying subset via ``param_manager.extract_varying`` before clipping.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import yaml

from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import (
    _build_joint_problem,
    _fit_joint_averaged_multi_phi,
)

_DT = 1.0
_Q = 0.0054
_NOISE = 5e-4

# Mirrors the real C045 config: reference-transport (D0_ref/alpha_ref/D_offset_ref)
# fixed, only the sample/velocity/fraction/angle physics + scaling vary.
_PHYSICS_NAMES = [
    "D0_sample",
    "alpha_sample",
    "D_offset_sample",
    "v0",
    "beta",
    "v_offset",
    "f0",
    "f1",
    "f2",
    "f3",
    "phi0",
]
_PHYSICS_VALUES = [1000.0, 0.9, 1.0, 5.0, 0.5, 0.01, 0.5, 0.1, 0.01, 0.001, 0.0]
_FIXED_REF_ACTIVE_PARAMS = [*_PHYSICS_NAMES, "contrast", "offset"]


def _make_model_with_fixed_ref_params(n_phi: int, n_t: int = 16):
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    config_dict = {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _DT,
            "start_frame": 1,
            "end_frame": n_t,
            "scattering": {"wavevector_q": _Q},
        },
        "scaling": {
            "n_angles": n_phi,
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "initial_parameters": {
            "parameter_names": _PHYSICS_NAMES,
            "values": _PHYSICS_VALUES,
            "active_parameters": _FIXED_REF_ACTIVE_PARAMS,
        },
        "optimization": {
            "nlsq": {"analysis_mode": "two_component", "max_iterations": 30, "enable_cmaes": False}
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "fixture.yaml"
        cfg_path.write_text(yaml.safe_dump(config_dict))
        cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)

    phi = np.linspace(0.0, 150.0, n_phi, dtype=np.float64)
    rng = np.random.default_rng(seed=20260524)
    c2 = np.empty((n_phi, n_t, n_t), dtype=np.float64)
    for i, phi_angle in enumerate(phi):
        corr = np.asarray(model.compute_correlation(phi_angle=float(phi_angle), angle_idx=i))
        c2[i] = corr + rng.normal(0.0, _NOISE, size=corr.shape)

    return model, c2, phi


def test_build_joint_problem_individual_mode_with_fixed_physics_params():
    # n_phi=2 < constant_scaling_threshold (3) => auto resolves to individual,
    # the exact path that crashed in the C045 log.
    model, c2, phi = _make_model_with_fixed_ref_params(n_phi=2)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "auto",
            "enable_hierarchical": True,
        }
    )
    n_varying = model.param_manager.n_varying
    assert n_varying == 11  # 13 active_parameters minus the 2 scaling names

    prob = _build_joint_problem(model, c2, phi, cfg, weights=None)

    # x0 = [scaling_head | physics]; individual mode scaling head = 2 * n_phi.
    assert prob.x0.shape == (2 * 2 + n_varying,)
    assert prob.lb.shape == prob.x0.shape
    assert prob.ub.shape == prob.x0.shape


def test_averaged_path_with_fixed_physics_params():
    model, c2, phi = _make_model_with_fixed_ref_params(n_phi=3)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_hierarchical": True,
        }
    )
    n_varying = model.param_manager.n_varying
    assert n_varying == 11

    result = _fit_joint_averaged_multi_phi(model, c2, phi, cfg, weights=None)

    # Result parameters: full 14-physics layout + 2 averaged scaling (contrast, offset).
    assert result.parameters.shape == (16,)
