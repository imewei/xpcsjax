"""Characterization test for the joint global-escape result contract.

CLAUDE.md documents that a *kept* global-escape result carries NaN covariance /
uncertainties and ``iterations == 0`` (no covariance solve on the kept vector).
The escape test suite asserted keep-better and the tag, but not this contract —
so a stray covariance solve on the escape vector would go uncaught. This pins
it, via the typed ``OptimizationResult.global_escape`` accessor.
"""

from __future__ import annotations

import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi


def test_kept_cmaes_escape_carries_nan_covariance_and_zero_iterations():
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": "averaged",
            "enable_cmaes": True,
        }
    )
    res = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)

    tag = res.global_escape  # typed accessor (F3)
    assert tag is not None and tag.startswith("cmaes")

    if tag == "cmaes":
        # Escape WON → contract applies: no covariance solve on the kept vector.
        assert res.iterations == 0
        assert np.all(np.isnan(np.asarray(res.covariance)))
        assert np.all(np.isnan(np.asarray(res.uncertainties)))
    else:
        # Warm-start kept (tag == "cmaes_warmstart_kept"): the plain joint fit's
        # finite covariance is retained — the escape contract does not apply.
        assert np.isfinite(np.asarray(res.covariance)).any()
