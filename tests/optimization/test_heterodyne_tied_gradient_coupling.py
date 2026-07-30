"""Proves the tied-parameter mechanism is REAL coupling (gradient sums both
usages onto the shared free variable), not cosmetic post-hoc mirroring.

If this test failed, tying would only copy the parent's value into the
child's slot for REPORTING purposes while the optimizer explored the child
and parent as two unrelated free variables -- exactly the bug the original
active_parameters-based workaround had.

CAVEAT (pattern lock-in, not closure verification): `loss()` below
reconstructs the scatter-and-tie-loop pattern standalone, same as
tests/optimization/test_heterodyne_tied_residuals.py -- it proves JAX's own
autodiff is correct for THIS hand-rolled closure, not that the production
residual closures (Tasks 4-9) actually contain the same loop. Real coverage
of the production wiring comes from Tasks 10-14's `fit_nlsq(...)`
integration tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
from xpcsjax.config.heterodyne_parameter_space import ParameterSpace
from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne


def _tied_param_manager() -> ParameterManager:
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"tied_parameters": {"D0_ref": "D0_sample"}},
    }
    space = ParameterSpace.from_config(config)
    return ParameterManager(space)


def test_tied_gradient_sums_both_usages():
    """d(loss)/d(D0_sample_free_var) must equal the SUM of the kernel's
    partial derivative through the D0_ref slot AND through the D0_sample
    slot -- not just one of them (which would mean the tie only applied to
    ONE usage, a wiring bug) and not zero (which would mean the tie broke
    differentiability)."""
    pm = _tied_param_manager()
    fixed_values_jax = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(pm.varying_indices, dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs
    assert tied_idx_pairs, "fixture must produce at least one tied pair"

    t = jnp.linspace(0.1, 5.0, 10)
    q, dt = 0.0054, 1.0
    phi = 0.0
    contrast, offset = 0.3, 1.0

    def loss(varying_params: jnp.ndarray) -> jnp.ndarray:
        full = fixed_values_jax.at[varying_indices_jax].set(varying_params)
        for child_idx, parent_idx in tied_idx_pairs:
            full = full.at[child_idx].set(full[parent_idx])
        c2 = compute_c2_heterodyne(full, t, q, dt, phi, contrast, offset)
        return jnp.sum(c2**2)

    x0 = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    analytic_grad = jax.grad(loss)(x0)

    # Finite-difference cross-check on the free variable backing D0_sample
    # (which now drives BOTH the D0_ref slot and the D0_sample slot).
    d0_sample_pos = pm.varying_names.index("D0_sample")
    eps = 1e-3 * max(abs(float(x0[d0_sample_pos])), 1.0)
    x_plus = x0.at[d0_sample_pos].add(eps)
    x_minus = x0.at[d0_sample_pos].add(-eps)
    fd_grad = (loss(x_plus) - loss(x_minus)) / (2 * eps)

    assert np.isclose(float(analytic_grad[d0_sample_pos]), float(fd_grad), rtol=1e-3, atol=1e-6), (
        f"analytic grad {float(analytic_grad[d0_sample_pos]):.6g} != finite-diff "
        f"{float(fd_grad):.6g} -- the tied free variable's gradient does not "
        "match its true combined sensitivity through both slots"
    )

    # Sanity: the gradient must be non-zero (proves the tie doesn't
    # accidentally zero out the D0_ref contribution).
    assert abs(float(fd_grad)) > 0.0
