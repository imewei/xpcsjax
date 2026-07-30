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
residual closures (Tasks 4-9) actually contain the same loop. A closure with
only ONE physics usage of the tied slot, or a cosmetic-mirror-only
implementation that patches the child in post-hoc after an untied solve,
would pass this same-closure check trivially -- it cannot tell real
during-solve coupling apart from report-time-only mirroring.

The real end-to-end discriminator is
`test_tied_fit_ssr_matches_recompute_from_reported_parameters` below: it runs
an actual tied fit and checks that recomputing SSR from the REPORTED
parameters agrees with the solver's own `chi_squared`. Real coverage of the
production wiring also comes from Tasks 10-14's `fit_nlsq(...)` integration
tests.
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
    """Within THIS hand-rolled closure, d(loss)/d(D0_sample_free_var) must
    equal the SUM of the kernel's partial derivative through the D0_ref slot
    AND through the D0_sample slot -- not just one of them, and not zero
    (which would mean the tie broke differentiability).

    This proves JAX's autodiff is internally consistent for a closure that
    already has the tie-mirror loop wired in by hand. It does NOT prove the
    production residual closures actually contain that same loop -- see the
    module docstring's CAVEAT and
    `test_tied_fit_ssr_matches_recompute_from_reported_parameters` below for
    the real end-to-end discriminator."""
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


def test_tied_fit_ssr_matches_recompute_from_reported_parameters(tmp_path):
    """End-to-end discriminator: real during-solve tying vs report-time-only
    mirroring, using an actual fit rather than a hand-rolled closure.

    Runs a real ``constant``-mode tied fit (scaling is frozen pre-solve, so
    the diagnostics' ``contrast_per_angle_fixed``/``offset_per_angle_fixed``
    unambiguously give the scaling the solver actually used -- no
    scaling_first/physics_first split to resolve first), then independently
    recomputes SSR by feeding the model's own production residual kernel
    (``compute_multi_angle_residuals``, the same function
    ``heterodyne_constant_mode.py`` uses internally) the REPORTED tied
    physics parameters and the original synthetic data.

    If the tie-mirror step were missing from the production residual closure
    (the bug this whole feature guards against) the optimizer would have
    explored the tied child as its own untied, unmirrored fixed slot during
    the solve, and a post-hoc ``expand_varying_to_full``-style mirror would then
    force child=parent for REPORTING only. In that world the vector behind
    the solver's own ``chi_squared`` (child at its untied value) would not be
    the same vector as the REPORTED, mirrored parameters -- so recomputing
    SSR from the reported vector would disagree with ``result.chi_squared``.
    Under real, during-solve tying (what production actually does) both are
    the same vector, so they must agree.
    """
    import jax.numpy as jnp
    import yaml

    from tests.optimization.test_heterodyne_tied_result_assembly import (
        _build_synthetic_c2,
        _run_tied_fit,
        _tied_config_dict,
    )
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    result = _run_tied_fit(tmp_path, phi_angles, "constant")
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag, "fixture must produce at least one tied pair"

    # Rebuild the exact same synthetic data the fit was run against
    # (deterministic seed in `_build_synthetic_c2`; same config -> same model
    # -> same true physics -> same c2 stack).
    cfg_path = tmp_path / "tied_recheck.yaml"
    cfg_path.write_text(yaml.safe_dump(_tied_config_dict(phi_angles, "constant")))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)

    reported_physics = np.asarray(result.parameters, dtype=np.float64)
    assert reported_physics.size == 14, "constant mode reports the full 14-physics vector"

    contrast_fixed = np.asarray(diag["contrast_per_angle_fixed"], dtype=np.float64)
    offset_fixed = np.asarray(diag["offset_per_angle_fixed"], dtype=np.float64)

    residual = compute_multi_angle_residuals(
        jnp.asarray(reported_physics, dtype=jnp.float64),
        jnp.asarray(model.t, dtype=jnp.float64),
        model.q,
        model.dt,
        jnp.asarray(phi_angles, dtype=jnp.float64),
        jnp.asarray(c2, dtype=jnp.float64),
        jnp.ones_like(jnp.asarray(c2, dtype=jnp.float64)),
        jnp.asarray(contrast_fixed, dtype=jnp.float64),
        jnp.asarray(offset_fixed, dtype=jnp.float64),
    )
    recomputed_ssr = float(np.sum(np.asarray(residual) ** 2))

    assert np.isclose(recomputed_ssr, result.chi_squared, rtol=1e-6), (
        f"recomputed SSR {recomputed_ssr:.6e} from the REPORTED tied "
        f"parameters disagrees with result.chi_squared {result.chi_squared:.6e} "
        "-- this would happen if the tie were applied only at reporting time "
        "(post-hoc mirroring) rather than enforced during the solve"
    )
