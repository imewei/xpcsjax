"""Regression test for NLSQWrapper._prepare_sigma_data's stratified phi indexing.

Rounding only the query (data.phi_flat) while leaving the search table
(data.phi) unrounded shifts every searchsorted index by +1 whenever the
rounded query lands fractionally above the table's raw floating-point
value -- silently assigning each point another angle's sigma row instead
of its own. _prepare_sigma_data doesn't reference ``self``, so it can be
called directly on a minimal stand-in data object without constructing a
full NLSQWrapper.
"""

from types import SimpleNamespace

import numpy as np

from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper


def test_stratified_sigma_phi_index_matches_own_angle():
    phi_deg = np.array([10.0, 45.0, 80.0])
    phi_unique = np.deg2rad(phi_deg)
    t1_unique = np.array([0.0, 0.1])
    t2_unique = np.array([0.0, 0.1])

    # sigma_3d[phi_idx, t1_idx, t2_idx]: give each phi angle a distinct,
    # easily-identifiable sigma value so a wrong phi_idx is obvious.
    sigma_3d = np.zeros((3, 2, 2))
    sigma_3d[0] = 1.0
    sigma_3d[1] = 2.0
    sigma_3d[2] = 3.0

    # One flat point per angle, independently computed (mirrors the
    # interleaved-copy pattern that introduces tiny float differences
    # between the "unique" table and the "flat" per-point arrays).
    phi_flat = np.array([np.deg2rad(10.0), np.deg2rad(45.0), np.deg2rad(80.0)])
    t1_flat = np.array([0.0, 0.0, 0.0])
    t2_flat = np.array([0.0, 0.0, 0.0])

    data = SimpleNamespace(
        sigma=sigma_3d,
        phi_flat=phi_flat,
        t1_flat=t1_flat,
        t2_flat=t2_flat,
        phi=phi_unique,
        t1=t1_unique,
        t2=t2_unique,
    )

    result = NLSQWrapper._prepare_sigma_data(None, data, n_data=3)  # noqa: SLF001

    assert result is not None
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
