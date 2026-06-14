"""Phase 5 — quantile scaling accepts a raw (non-stratified, <100k) grid object."""
from __future__ import annotations

import numpy as np

from xpcsjax.optimization.nlsq.parameter_utils import compute_quantile_per_angle_scaling


class _RawGrid:
    """The object the <100k unstratified path returns: grid arrays, NO phi_flat."""

    def __init__(self, n_phi=4, n_t=40):
        self.phi = np.linspace(0.0, 90.0, n_phi, dtype=np.float64)
        self.t1 = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
        self.t2 = np.linspace(0.0, float(n_t - 1), n_t, dtype=np.float64)
        t1m, t2m = np.meshgrid(self.t1, self.t2, indexing="ij")
        dtau = np.abs(t1m - t2m)
        # C2 = contrast * exp(-2*dtau/10) + offset, per angle.
        g2 = np.empty((n_phi, n_t, n_t), dtype=np.float64)
        for i in range(n_phi):
            g2[i] = 0.3 * np.exp(-2.0 * dtau / 10.0) + 1.0
        self.g2 = g2
        self.q = 0.0237
        self.L = 2_000_000.0
        self.dt = 0.1


def test_quantile_on_raw_grid_no_attribute_error():
    grid = _RawGrid(n_phi=4)
    c, o = compute_quantile_per_angle_scaling(grid)
    assert c.shape == (4,)
    assert o.shape == (4,)
    # offset ~ 1.0 (floor), contrast ~ 0.3 (ceiling - floor); generous tolerance
    np.testing.assert_allclose(o, np.full(4, 1.0), atol=0.15)
    assert np.all(c > 0.0)
