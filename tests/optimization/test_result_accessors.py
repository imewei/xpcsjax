"""Typed accessors on OptimizationResult (quality-gate type-design fixes).

F8: ``parameters`` is a flat array whose physics-first ``[physics | scaling]``
layout was implicit — the root of the recurring viz mis-slicing bug. Named
``physics_parameters`` / ``scaling_parameters`` accessors make the split
explicit (given ``n_physics``).

F3: ``nlsq_diagnostics`` is an untyped dict, so ``["global_escape"]`` typos
silently return ``None``. A typed ``global_escape`` accessor centralises the
lookup.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.results import OptimizationResult


def _result(**overrides):
    base = dict(
        parameters=np.array([1.0, 2.0, 3.0, 0.5, 0.1]),  # 3 physics + 2 scaling
        uncertainties=np.zeros(5),
        covariance=np.eye(5),
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        convergence_status="converged",
        iterations=3,
        execution_time=0.01,
        device_info={"device": "cpu"},
    )
    base.update(overrides)
    return OptimizationResult(**base)


def test_physics_and_scaling_split_when_n_physics_known():
    res = _result(n_physics=3)
    assert np.array_equal(res.physics_parameters, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(res.scaling_parameters, np.array([0.5, 0.1]))


def test_accessors_raise_clearly_when_n_physics_unknown():
    res = _result()  # n_physics defaults to None
    with pytest.raises(ValueError, match="n_physics"):
        _ = res.physics_parameters


def test_global_escape_typed_read():
    assert _result(nlsq_diagnostics={"global_escape": "cmaes"}).global_escape == "cmaes"
    # Missing key or no diagnostics → None (no silent KeyError on a typo path).
    assert _result(nlsq_diagnostics={"other": 1}).global_escape is None
    assert _result(nlsq_diagnostics=None).global_escape is None
