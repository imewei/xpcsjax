"""Regression test for log_heterodyne_completion param layout (audit C4).

log_heterodyne_completion read physics from the TAIL (scaling-first), but the
averaged joint path and the sequential individual aggregate return PHYSICS-FIRST
vectors, so the wrong elements were logged for those two paths.

Pure logging — assertions inspect captured log records only.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_completion

_PHYS = ["A", "B", "C"]


def _result(parameters, diag):
    return SimpleNamespace(
        nlsq_diagnostics=diag,
        parameters=np.asarray(parameters, dtype=np.float64),
        uncertainties=None,
        success=True,
        iterations=1,
        execution_time=0.0,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        quality_flag="good",
    )


def _logged_physics(caplog):
    out = {}
    for rec in caplog.records:
        msg = rec.getMessage().strip()
        for name in _PHYS:
            if msg.startswith(f"{name}:"):
                out[name] = float(msg.split(":")[1].split("+/-")[0])
    return out


def test_averaged_reads_physics_from_head(caplog):
    # physics-first: [physics(3) | contrast, offset]
    params = [10.0, 11.0, 12.0, 0.5, 1.0]
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
        log_heterodyne_completion(
            _result(params, {"per_angle_mode": "averaged"}), _PHYS, n_physics=3, n_phi=1
        )
    assert _logged_physics(caplog) == {"A": 10.0, "B": 11.0, "C": 12.0}


def test_sequential_individual_reads_physics_from_head(caplog):
    # physics-first sequential aggregate: [physics(3) | per-angle scaling tail]
    params = [10.0, 11.0, 12.0, 0.5, 0.6, 1.0, 1.1]
    diag = {"per_angle_mode": "individual", "covariance_structure": "block_diagonal_sequential"}
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
        log_heterodyne_completion(_result(params, diag), _PHYS, n_physics=3, n_phi=2)
    assert _logged_physics(caplog) == {"A": 10.0, "B": 11.0, "C": 12.0}


def test_scaling_first_joint_reads_physics_from_tail(caplog):
    # scaling-first (individual joint / streaming / engine): [scaling | physics(3)]
    params = [0.5, 0.6, 1.0, 1.1, 10.0, 11.0, 12.0]
    diag = {"per_angle_mode": "individual"}  # no sequential marker
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_core"):
        log_heterodyne_completion(_result(params, diag), _PHYS, n_physics=3, n_phi=2)
    assert _logged_physics(caplog) == {"A": 10.0, "B": 11.0, "C": 12.0}
