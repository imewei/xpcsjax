"""Phase 5 — optimizer vector + bounds lengths per resolved mode."""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper


def _vlen(mode: str, n_phi: int, n_physics: int = 7) -> int:
    return ParameterIndexMapper.canonical(
        mode=mode, n_phi=n_phi, n_physics=n_physics
    ).vector_length


def test_vector_length_per_mode():
    assert _vlen("individual", 4) == 7 + 2 * 4   # 15
    assert _vlen("averaged", 4) == 7 + 2          # 9
    assert _vlen("constant", 4) == 7              # 7


def test_fit_vector_length_observed_individual():
    res = _run("individual", n_phi=4)
    # individual: 7 physics + 2*4 scaling
    assert len(np.asarray(res.parameters)) == 7 + 2 * 4


def test_fit_vector_length_observed_averaged_expands_to_dense():
    # Result builder expands averaged back to dense per-angle (expand_back contract),
    # so result.parameters is the DENSE 7 + 2*n_phi layout even though the optimizer
    # solved 7 + 2 params. The diagnostics report the optimizer length.
    res = _run("auto", n_phi=4)  # -> averaged
    diag = dict(res.nlsq_diagnostics or {})
    assert diag.get("per_angle_mode") == "averaged"
    assert int(diag.get("n_optimized")) == 2
    # dense result layout (expand_back) for viz/results parity:
    assert len(np.asarray(res.parameters)) == 7 + 2 * 4


def _run(mode, n_phi):
    from tests.optimization.test_phase5_standard_resolver import _fit
    return _fit(mode, n_phi)
