"""Phase 5 — constant/averaged/individual laminar fits run end-to-end without JIT
crashes and return the canonical dense vector lengths."""

from __future__ import annotations

import numpy as np
import pytest

from tests.optimization.test_phase5_standard_resolver import _laminar_cfg
from xpcsjax.core.homodyne_model import HomodyneModel
from xpcsjax.optimization.nlsq import fit_nlsq

_PHYS = 7


def _run(mode, n_phi, n_t=12, seed=3):
    phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
    t = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
    true = np.array([1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0])
    cfg = _laminar_cfg(mode, n_t)
    model = HomodyneModel(cfg.config)
    c2 = np.asarray(model.compute_c2(true, phi, contrast=0.3, offset=1.0))
    c2 = c2 + np.random.default_rng(seed).normal(0.0, 5e-4, size=c2.shape)
    data = {
        "phi_angles_list": phi,
        "c2_exp": c2,
        "t1": t,
        "t2": t,
        "wavevector_q_list": np.array([0.0237]),
    }
    return fit_nlsq(data, cfg)


@pytest.mark.parametrize(
    "mode,n_phi,expect_optimized",
    [
        ("individual", 4, 2 * 4),
        ("auto", 4, 2),  # -> averaged
        ("constant", 4, 0),
        ("individual", 2, 2 * 2),
        ("auto", 2, 2 * 2),  # n_phi<3 -> individual
    ],
)
def test_laminar_modes_no_crash_correct_lengths(mode, n_phi, expect_optimized):
    res = _run(mode, n_phi)
    params = np.asarray(res.parameters, dtype=np.float64)
    # Result is ALWAYS dense scaling-first: 2*n_phi + n_physics.
    assert params.shape[0] == 2 * n_phi + _PHYS, (
        f"{mode} n_phi={n_phi}: dense result length {params.shape[0]} != {2 * n_phi + _PHYS}"
    )
    assert np.all(np.isfinite(params))
    diag = dict(res.nlsq_diagnostics or {})
    assert int(diag.get("n_optimized")) == expect_optimized
    # objective is finite (the JIT actually ran)
    assert np.isfinite(float(res.chi_squared))
