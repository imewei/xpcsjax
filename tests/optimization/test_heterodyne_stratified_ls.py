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


def test_stratified_ls_matches_joint_fit_shuffle_off():
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        build_joint_pointwise_residual,
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})

    joint = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    strat = fit_heterodyne_stratified_least_squares(
        model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False,
    )

    # --- Parity proof: identical objective ---------------------------------
    # The two paths solve the SAME least-squares objective; the only difference
    # is the point ORDER (angle-major batched vs interleaved-stratified) which
    # steers the trust-region solve into a slightly different basin of the
    # near-degenerate two_component landscape (the documented C044 degeneracy:
    # parameters diverge while SSR is nearly identical). Prove objective
    # equality directly by scoring BOTH fitted parameter vectors against a
    # single shared residual function — each must reproduce its own reported
    # chi_squared exactly.
    # For this synthetic fixture the data-generating contrast/offset (0.3, 1.0)
    # match the driver's quantile estimates, so this shared residual is
    # equivalent to the one the driver builds — a fair common objective.
    st = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    shared_resid, _x, _y, _p0, _meta = build_joint_pointwise_residual(
        model=model, stratified_data=st, per_angle_mode="averaged",
        avg_contrast=0.3, avg_offset=1.0,
    )
    ssr_joint = float(np.sum(np.asarray(shared_resid(np.asarray(joint.parameters)))**2))
    ssr_strat = float(np.sum(np.asarray(shared_resid(np.asarray(strat.parameters)))**2))
    assert np.isclose(ssr_joint, joint.chi_squared, rtol=1e-9)
    assert np.isclose(ssr_strat, strat.chi_squared, rtol=1e-9)

    # Both land at near-optimal SSR; the residual convergence spread on this
    # degenerate objective is ~0.2% (robust to tightening solver tolerances to
    # 1e-12 — see Task 4 investigation), so the chi_squared agreement tolerance
    # reflects that empirically-measured spread, not solver slop.
    assert np.isclose(strat.chi_squared, joint.chi_squared, rtol=5e-3)

    # SSR conservation: per-angle chi^2 decomposition sums to the total.
    diag = strat.nlsq_diagnostics
    assert np.isclose(float(np.sum(diag["chi2_per_angle"])), strat.chi_squared, rtol=1e-6)
