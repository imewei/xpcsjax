"""The joint global escape must honour the resolved per-angle mode.

Quality-gate gap: escape tests only exercised ``averaged``/``constant`` modes.
CLAUDE.md's consistency invariant says enabling CMA-ES must NOT change which
scaling layout is used (it must not silently switch to Fourier). This pins that
for ``individual`` and ``fourier``: the escape fit keeps the same parameter
layout as the plain fit, and is keep-better.
"""

from __future__ import annotations

import pytest

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi


def _fit(model, c2, phi, mode, *, cmaes):
    cfg = NLSQConfig.from_dict(
        {
            "analysis_mode": "two_component",
            "per_angle_mode": mode,
            "enable_cmaes": cmaes,
        }
    )
    return fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)


@pytest.mark.parametrize("mode", ["individual", "fourier"])
def test_escape_preserves_layout_and_is_keep_better(mode):
    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    plain = _fit(model, c2, phi, mode, cmaes=False)

    model2, c2_2, phi2 = make_synthetic_two_component(n_phi=3, n_t=20)
    escaped = _fit(model2, c2_2, phi2, mode, cmaes=True)

    # Escape must not switch the scaling layout (no silent Fourier tail).
    assert len(escaped.parameters) == len(plain.parameters)
    # Keep-better: the escape never returns a worse fit than the plain path.
    assert escaped.chi_squared <= plain.chi_squared * (1 + 1e-6)
