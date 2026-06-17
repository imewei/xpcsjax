"""Regression test for CombinedModel.compute_chi_squared dt drift (audit C11).

The backend `compute_chi_squared` gained a required `dt` argument; the
`CombinedModel.compute_chi_squared` wrapper was never updated to accept and
forward it, so any call raised TypeError.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.models import CombinedModel


def test_compute_chi_squared_accepts_and_forwards_dt():
    model = CombinedModel()
    params = model.get_default_parameters()

    t1 = jnp.array([0.0])
    t2 = jnp.array([1.0])
    phi = jnp.array([0.0])
    q, L, contrast, offset, dt = 0.01, 1e6, 0.8, 1.0, 0.1

    data = model.compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt)
    sigma = jnp.ones_like(data)

    chi2 = model.compute_chi_squared(
        params, data, sigma, t1, t2, phi, q, L, contrast, offset, dt
    )
    chi2 = float(chi2)
    # data == theory at the same params -> chi^2 is ~0 and finite.
    assert np.isfinite(chi2)
    assert chi2 == 0.0
