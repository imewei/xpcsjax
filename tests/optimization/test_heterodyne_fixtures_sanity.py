"""Pin the precondition the shared heterodyne fixture's covariance tests rely on.

``finalize_covariance`` reports the all-NaN placeholder whenever any variance
is exactly zero, and nlsq's SVD-truncated pcov has an exactly-zero variance
whenever the Jacobian has an exactly-zero column. The registry default
``f1 = 0`` produced one (``d/d f2`` of ``f0*exp(f1*(t - f2)) + f3`` vanishes),
which only surfaced on platforms whose solver converges right at ``x0`` (CI run
35005275832, macOS). This solver-free check fails on every platform if someone
reseeds the fixture back onto that point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
from xpcsjax.optimization.nlsq.heterodyne_core import compute_multi_angle_residuals

N_PHI, N_T = 3, 20


def test_fixture_jacobian_has_no_zero_column_at_x0():
    model, c2, phi = make_synthetic_two_component(n_phi=N_PHI, n_t=N_T)
    pm = model.param_manager
    fixed = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    vidx = jnp.asarray(pm.varying_indices, dtype=jnp.int32)
    c2j = jnp.asarray(c2)
    ones = jnp.ones_like(c2j)
    phij = jnp.asarray(phi)

    def resid(x):  # [contrast, offset | physics]
        full = fixed.at[vidx].set(x[2:])
        return compute_multi_angle_residuals(
            full,
            model.t,
            model.q,
            model.dt,
            phij,
            c2j,
            ones,
            jnp.full((N_PHI,), x[0]),
            jnp.full((N_PHI,), x[1]),
        )

    x0 = np.concatenate([[0.3, 1.0], np.asarray(pm.get_initial_values())])
    J = np.asarray(jax.jacfwd(resid)(jnp.asarray(x0)))
    col_norm = np.linalg.norm(J, axis=0)
    names = ["contrast", "offset", *pm.varying_names]
    assert np.all(np.isfinite(J))
    zero = [n for n, c in zip(names, col_norm, strict=True) if c == 0.0]
    assert not zero, f"exactly-zero Jacobian columns at x0: {zero}"
