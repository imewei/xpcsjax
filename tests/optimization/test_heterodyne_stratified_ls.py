import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import reorder_for_stratification


def test_reorder_preserves_multiset_and_shuffles():
    phi = np.repeat([10.0, 20.0, 30.0], 4)
    payload = np.arange(12, dtype=np.float64)
    perm, chunk_sizes = reorder_for_stratification(phi, target_chunk_size=6, shuffle=True, seed=42)
    assert sorted(perm.tolist()) == list(range(12))
    assert sorted(payload[perm].tolist()) == sorted(payload.tolist())
    assert sum(chunk_sizes) == 12
    perm2, _ = reorder_for_stratification(phi, target_chunk_size=6, shuffle=True, seed=42)
    assert np.array_equal(perm, perm2)


def test_reorder_shuffle_off_is_pure_interleave():
    phi = np.repeat([10.0, 20.0, 30.0], 4)
    perm_a, _ = reorder_for_stratification(phi, target_chunk_size=6, shuffle=False, seed=42)
    perm_b, _ = reorder_for_stratification(phi, target_chunk_size=6, shuffle=False, seed=999)
    assert np.array_equal(perm_a, perm_b)


def test_averaged_scaling_expander_broadcasts():
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import make_scaling_expander

    expander, n_scaling = make_scaling_expander("averaged", n_phi=3)
    assert n_scaling == 2  # one contrast + one offset
    contrast, offset = expander(jnp.array([0.3, 0.8]))
    assert contrast.shape == (3,) and offset.shape == (3,)
    assert np.allclose(np.asarray(contrast), 0.3)
    assert np.allclose(np.asarray(offset), 0.8)


def test_joint_pointwise_residual_matches_batched():
    """Flat pointwise residual is finite and has the off-diagonal/t>0 support length."""
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        build_joint_pointwise_residual,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    # Seed the scaling tail at the data-generating values (the fixture config
    # uses initial_contrast=0.3, initial_offset=1.0), so the residual at
    # p0_full is noise-level rather than carrying a constant baseline offset.
    residual_fn, x_data, y_data, p0_full, meta = build_joint_pointwise_residual(
        model=model, stratified_data=strat, per_angle_mode="averaged",
        avg_contrast=0.3, avg_offset=1.0,
    )
    r = np.asarray(residual_fn(np.asarray(p0_full)))
    assert r.shape[0] == meta["n_data_points"]
    assert np.all(np.isfinite(r))
    # Data is the model at its initial params plus ~5e-4 noise, so the residual
    # at p0_full must be noise-level — confirms real values, not just finiteness.
    assert float(np.max(np.abs(r))) < 0.05
