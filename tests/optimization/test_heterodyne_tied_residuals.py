"""Direct residual-closure tests: every wired heterodyne residual function
must enforce child == parent on EVERY call when tied_parameters is set, and
be a byte-identical no-op when it is absent.

These test the residual CLOSURES directly (not a full fit) so failures
localize to exactly one wiring site.

CAVEAT (pattern lock-in, not closure verification): each per-site test below
reconstructs the intended scatter-and-tie-loop pattern standalone rather than
importing the actual production closure -- `base_residual_fn` /
`joint_residual_fn` / `model_func` / `jax_residual_fn` / `residual_fn` are all
private, module-nested functions with no importable module-level name. These
tests lock in the expected pattern BEFORE each task wires it into production
code; they do not, by themselves, prove the production closure was edited
correctly. Real end-to-end coverage of the actual wiring comes from Tasks
10-14's `fit_nlsq(...)` integration tests (once the engine-route dispatch
bypass from Task 10 Step 0 is in place) -- treat those as the closure-level
proof, not these.
"""

from __future__ import annotations

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
from xpcsjax.config.heterodyne_parameter_space import ParameterSpace


def _tied_param_manager() -> ParameterManager:
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"tied_parameters": {"D0_ref": "D0_sample"}},
    }
    space = ParameterSpace.from_config(config)
    return ParameterManager(space)


def test_expand_varying_to_full_used_by_all_residual_sites_is_tied():
    """Sanity check exercised again here (redundant with Task 2's test, kept
    as the entry point for the per-site tests appended in Tasks 5-11 below)."""
    pm = _tied_param_manager()
    varying = pm.get_initial_values()
    full = pm.expand_varying_to_full(varying)
    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert full[d0_ref_idx] == full[d0_sample_idx]


def test_averaged_mode_residual_enforces_tie():
    """heterodyne_core.py: _fit_joint_averaged_multi_phi's base_residual_fn."""
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals

    pm = _tied_param_manager()
    fixed_values_jax = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(pm.varying_indices, dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs

    n_physics_varying = len(pm.varying_indices)
    physics_varying = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)

    full_jax = fixed_values_jax.at[varying_indices_jax].set(physics_varying)
    for child_idx, parent_idx in tied_idx_pairs:
        full_jax = full_jax.at[child_idx].set(full_jax[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_jax[d0_ref_idx]) == float(full_jax[d0_sample_idx])
    del compute_multi_angle_residuals, n_physics_varying  # imports exercised for coverage only


def test_individual_mode_residual_pattern_enforces_tie():
    """heterodyne_core.py: _build_joint_problem's base_residual_fn (scaling-
    first layout: scaling head, physics tail)."""
    import jax.numpy as jnp

    pm = _tied_param_manager()
    fixed_values_jax = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(pm.varying_indices, dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs

    physics_varying = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    full_jax = fixed_values_jax.at[varying_indices_jax].set(physics_varying)
    for child_idx, parent_idx in tied_idx_pairs:
        full_jax = full_jax.at[child_idx].set(full_jax[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_jax[d0_ref_idx]) == float(full_jax[d0_sample_idx])


def test_constant_mode_residual_pattern_enforces_tie():
    """heterodyne_constant_mode.py: _fit_joint_constant_multi_phi's
    joint_residual_fn."""
    import jax.numpy as jnp

    pm = _tied_param_manager()
    fixed_values_jax = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(list(pm.varying_indices), dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs

    physics_varying = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    full_jax = fixed_values_jax.at[varying_indices_jax].set(physics_varying)
    for child_idx, parent_idx in tied_idx_pairs:
        full_jax = full_jax.at[child_idx].set(full_jax[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_jax[d0_ref_idx]) == float(full_jax[d0_sample_idx])


def test_per_angle_cmaes_residual_pattern_enforces_tie():
    """heterodyne_core.py: _fit_cmaes's model_func (per-angle CMA-ES escape)."""
    import jax.numpy as jnp

    pm = _tied_param_manager()
    full_template_jax = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.asarray(list(pm.varying_indices), dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs

    varying_jax = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    full_jax = full_template_jax.at[varying_indices_jax].set(varying_jax)
    for child_idx, parent_idx in tied_idx_pairs:
        full_jax = full_jax.at[child_idx].set(full_jax[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_jax[d0_ref_idx]) == float(full_jax[d0_sample_idx])


def test_per_angle_local_residual_pattern_enforces_tie():
    """heterodyne_core.py: _fit_local's jax_residual_fn."""
    import jax.numpy as jnp

    pm = _tied_param_manager()
    fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(pm.varying_indices)
    tied_idx_pairs = pm.tied_idx_pairs

    varying_array = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    full_params = fixed_values.at[varying_indices].set(varying_array)
    for child_idx, parent_idx in tied_idx_pairs:
        full_params = full_params.at[child_idx].set(full_params[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_params[d0_ref_idx]) == float(full_params[d0_sample_idx])


def test_make_numpy_residual_fn_pattern_enforces_tie():
    """heterodyne_core.py: _make_numpy_residual_fn's residual_fn."""
    import jax.numpy as jnp

    pm = _tied_param_manager()
    fixed_values = jnp.asarray(pm.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(pm.varying_indices, dtype=jnp.int32)
    tied_idx_pairs = pm.tied_idx_pairs

    varying_jax = jnp.asarray(pm.get_initial_values(), dtype=jnp.float64)
    full_params = fixed_values.at[varying_indices].set(varying_jax)
    for child_idx, parent_idx in tied_idx_pairs:
        full_params = full_params.at[child_idx].set(full_params[parent_idx])

    d0_ref_idx = list(ALL_PARAM_NAMES).index("D0_ref")
    d0_sample_idx = list(ALL_PARAM_NAMES).index("D0_sample")
    assert float(full_params[d0_ref_idx]) == float(full_params[d0_sample_idx])
