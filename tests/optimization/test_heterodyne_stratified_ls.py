import logging

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import reorder_for_stratification


def test_stratified_ls_emits_laminar_parity_banners(caplog):
    """The stratified-LS path (the >=1M solver the C044 two_component run took)
    historically logged NOTHING between the adapter call and completion, leaving
    a multi-minute silent gap. It must now narrate the laminar_flow log surface
    end to end (path activation -> mode -> quantiles -> gradient sanity ->
    fit start -> results -> complete)."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})

    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_logging"):
        fit_heterodyne_stratified_least_squares(
            model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False
        )
    text = caplog.text
    for expected in (
        "STRATIFIED LEAST-SQUARES PATH ACTIVATED",
        "Physical parameters for two_component",
        "Quantile-based per-angle estimation complete",
        "Contrast: mean=",
        "Offset: mean=",
        "ANTI-DEGENERACY: Effective per-angle mode 'averaged'",
        "GRADIENT SANITY CHECK",
        "Gradient sanity check passed",
        "Starting NLSQ least_squares() optimization",
        "OPTIMIZATION RESULTS",
        "STRATIFIED LEAST-SQUARES COMPLETE",
    ):
        assert expected in text, f"missing laminar-parity banner: {expected!r}"


def test_stratified_ls_gradient_sanity_perturbs_physics_first_param(caplog):
    """Heterodyne's joint vector is PHYSICS-FIRST ([physics | scaling]), so the
    gradient sanity check must perturb param[0] (the first physical parameter) --
    NOT the scaling-first index (2*n_phi) the homodyne/laminar path uses. This
    pins the layout-correct port; a verbatim copy of laminar's index would
    perturb a scaling coefficient instead and silently weaken the check."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_logging"):
        fit_heterodyne_stratified_least_squares(
            model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False
        )
    assert "perturbation of param[0]" in caplog.text


def test_completion_emits_honest_anti_degeneracy_defense(caplog):
    """The shared completion chokepoint must emit an anti-degeneracy DEFENSE
    summary reading REAL per-path diagnostics. The stratified-LS path runs a
    plain joint solve, so it must HONESTLY report L2/L3 inactive (not fabricate
    'Enabled: True' the way laminar's controller-driven path does)."""
    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import log_heterodyne_completion
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    result = fit_heterodyne_stratified_least_squares(
        model=model, c2=c2, phi=phi, config=cfg, weights=None, shuffle=False
    )

    with caplog.at_level(logging.INFO, logger="xpcsjax.optimization.nlsq.heterodyne_logging"):
        log_heterodyne_completion(
            result,
            list(model.param_manager.varying_names),
            int(model.param_manager.n_varying),
            len(phi),
        )
    text = caplog.text
    assert "ANTI-DEGENERACY DEFENSE" in text
    assert "L2 hierarchical_active: False" in text
    assert "L3 regularization_active: False" in text
    # Heterodyne has no shear term -> structural L5 sentinel, NOT laminar's
    # "Enabled: True" shear banner.
    assert "not_applicable_heterodyne" in text


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


def test_preshuffle_preserves_chunk_angle_balance():
    """Seed-42 shuffle is a PRE-shuffle that preserves per-chunk angle balance.

    A correct pre-shuffle re-derives stratification from the relabeled angles,
    so each chunk keeps its balanced angle multiset; only WHICH concrete points
    fill each angle's slots changes. A post-stratification global shuffle would
    scramble that per-chunk composition (the bug this guards against).
    """
    phi = np.repeat([10.0, 20.0, 30.0], 8)  # 3 angles, 8 pts each
    perm_off, sizes_off = reorder_for_stratification(phi, target_chunk_size=6, shuffle=False)
    perm_on, sizes_on = reorder_for_stratification(phi, target_chunk_size=6, shuffle=True, seed=42)

    # Same chunk boundaries regardless of shuffle.
    assert sizes_on == sizes_off

    # Per-chunk angle multiset preserved -> stratified balance intact.
    bounds = np.cumsum([0, *sizes_off])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        assert sorted(phi[perm_on[a:b]].tolist()) == sorted(phi[perm_off[a:b]].tolist())

    # The pre-shuffle still changed the concrete ordering.
    assert not np.array_equal(perm_on, perm_off)


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
        model=model,
        stratified_data=strat,
        per_angle_mode="averaged",
        init_scaling=np.array([0.3, 1.0]),
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
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=False,
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
        model=model,
        stratified_data=st,
        per_angle_mode="averaged",
        init_scaling=np.array([0.3, 1.0]),
    )
    ssr_joint = float(np.sum(np.asarray(shared_resid(np.asarray(joint.parameters))) ** 2))
    ssr_strat = float(np.sum(np.asarray(shared_resid(np.asarray(strat.parameters))) ** 2))
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


def test_individual_scaling_expander_splits_blocks():
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import make_scaling_expander

    expander, n_scaling = make_scaling_expander("individual", n_phi=3)
    assert n_scaling == 6
    c, o = expander(jnp.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert np.allclose(np.asarray(c), [0.1, 0.2, 0.3])
    assert np.allclose(np.asarray(o), [0.7, 0.8, 0.9])


def test_fourier_scaling_expander_shapes():
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import make_scaling_expander

    n_phi = 7
    K = 2
    phi_rad = np.deg2rad(np.linspace(0.0, 150.0, n_phi))
    fr = FourierReparameterizer(
        phi_rad,
        FourierReparamConfig(mode="fourier", fourier_order=K, auto_threshold=6),
    )
    # Confirm fourier mode is actually active (not auto-downgraded).
    assert fr.use_fourier
    assert fr.n_coeffs_per_param == 2 * K + 1

    expander, n_scaling = make_scaling_expander("fourier", n_phi=n_phi, fourier=fr)
    assert n_scaling == 2 * (2 * K + 1)

    # DC (constant) coeff of contrast is index 0; DC of offset is index n_coeffs_per_param.
    coeffs = jnp.zeros(n_scaling).at[0].set(0.3).at[2 * K + 1].set(0.8)
    c, o = expander(coeffs)
    assert c.shape == (n_phi,) and o.shape == (n_phi,)
    assert np.all(np.isfinite(np.asarray(c))) and np.all(np.isfinite(np.asarray(o)))
    # Only the DC term is nonzero -> constant contrast/offset across angles.
    assert np.allclose(np.asarray(c), 0.3)
    assert np.allclose(np.asarray(o), 0.8)


def test_fourier_scaling_expander_requires_reparameterizer():
    import pytest

    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import make_scaling_expander

    with pytest.raises(ValueError, match="fourier"):
        make_scaling_expander("fourier", n_phi=7, fourier=None)


def test_stratified_ls_individual_mode():
    """Individual mode is SCOPED OUT of stratified-LS (Fix 1).

    The existing heterodyne ``individual`` mode is sequential per-angle; the
    stratified driver would treat it as one joint solve (a different objective).
    Policy: only ``averaged``/``fourier`` use stratified-LS. The driver keeps a
    defensive ``NotImplementedError`` so it can never silently mis-handle
    individual even if called directly.
    """
    import pytest

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "individual"})
    # Pre-assert the config resolves to individual on this fixture.
    assert _resolve_effective_mode(cfg, len(phi)) == "individual"

    with pytest.raises(NotImplementedError):
        fit_heterodyne_stratified_least_squares(
            model=model,
            c2=c2,
            phi=phi,
            config=cfg,
            weights=None,
            shuffle=False,
        )


def test_stratified_ls_fourier_parity():
    """Fourier-mode stratified-LS matches the in-memory joint fourier fit objective.

    Mirrors ``test_stratified_ls_matches_joint_fit_shuffle_off`` but for FOURIER
    mode (n_phi=7 so fourier resolves). Both fitted vectors are cross-evaluated
    against a single shared fourier residual and must reproduce their own
    reported chi_squared exactly; the stratified chi2 must be within rtol 5e-2
    of the joint fit (documenting the near-degenerate spread).
    """
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.fourier_reparam import (
        FourierReparamConfig,
        FourierReparameterizer,
    )
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import (
        _resolve_effective_mode,
        fit_nlsq_multi_phi,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        build_joint_pointwise_residual,
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=7, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "fourier"})
    # Pre-assert the config resolves to fourier (fourier is never auto-selected).
    assert _resolve_effective_mode(cfg, len(phi)) == "fourier"

    joint = fit_nlsq_multi_phi(model, c2, phi, cfg, weights=None)
    strat = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=False,
    )

    # Shared fourier residual scoring both fitted vectors against one objective.
    st = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    from xpcsjax.optimization.nlsq.parameter_utils import (
        compute_quantile_per_angle_scaling,
    )

    contrast_pa, offset_pa = compute_quantile_per_angle_scaling(st)
    fr = FourierReparameterizer(
        np.deg2rad(np.asarray(phi).astype(np.float64)),
        FourierReparamConfig(
            mode="fourier",
            fourier_order=cfg.fourier_order,
            auto_threshold=cfg.fourier_auto_threshold,
        ),
    )
    init_scaling = np.asarray(
        fr.per_angle_to_fourier(
            np.asarray(contrast_pa, np.float64), np.asarray(offset_pa, np.float64)
        ),
        dtype=np.float64,
    )
    shared_resid, _x, _y, _p0, _meta = build_joint_pointwise_residual(
        model=model,
        stratified_data=st,
        per_angle_mode="fourier",
        init_scaling=init_scaling,
        fourier=fr,
    )
    ssr_joint = float(np.sum(np.asarray(shared_resid(np.asarray(joint.parameters))) ** 2))
    ssr_strat = float(np.sum(np.asarray(shared_resid(np.asarray(strat.parameters))) ** 2))
    assert np.isfinite(ssr_joint) and np.isfinite(ssr_strat)
    assert np.isclose(ssr_joint, joint.chi_squared, rtol=1e-9)
    assert np.isclose(ssr_strat, strat.chi_squared, rtol=1e-9)

    # Near-optimal SSR on the degenerate fourier objective.
    assert np.isclose(strat.chi_squared, joint.chi_squared, rtol=5e-2)

    # SSR conservation: per-angle chi^2 decomposition sums to the total.
    diag = strat.nlsq_diagnostics
    assert np.isclose(float(np.sum(diag["chi2_per_angle"])), strat.chi_squared, rtol=1e-6)


def test_stratified_ls_fourier_mode():
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import _resolve_effective_mode
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    # n_phi=7 with explicit fourier request so the mode resolves to "fourier"
    # (fourier is never auto-selected — _resolve_effective_mode passes the
    # explicit request through unchanged).
    model, c2, phi = make_synthetic_two_component(n_phi=7, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "fourier"})
    # Guard: confirm the config actually resolves to fourier before asserting.
    assert _resolve_effective_mode(cfg, len(phi)) == "fourier"

    res = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=False,
    )
    assert res.nlsq_diagnostics["per_angle_mode"] == "fourier"
    d = res.nlsq_diagnostics
    assert np.isclose(float(np.sum(d["chi2_per_angle"])), res.chi_squared, rtol=1e-6)
    assert np.all(np.isfinite(res.parameters))


def test_stratified_ls_attaches_diagnostics():

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    res = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi,
        config=cfg,
        weights=None,
        shuffle=True,
    )
    assert res.stratification_diagnostics is not None
    diag = res.stratification_diagnostics
    # n_chunks is always >= 1 for any non-empty dataset
    assert diag.n_chunks >= 1
    # chunk_sizes must be a non-empty list summing to the number of filtered points
    assert isinstance(diag.chunk_sizes, list)
    assert len(diag.chunk_sizes) == diag.n_chunks
    # use_index_based reflects the stratified-LS path (always True here)
    assert diag.use_index_based is True
    # execution_time_ms is non-negative
    assert diag.execution_time_ms >= 0.0


def test_stratified_ls_parameter_names_match_full_vector():
    """Fix 4: diagnostics parameter_names must align with the FULL popt length.

    The stratified popt includes the scaling tail (physics + scaling), so the
    diagnostics ``parameter_names`` must be the full joint name list, not the
    physics-only ``varying_names``. Checked for both averaged and fourier.
    """

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    # averaged: tail = [contrast, offset]
    model_a, c2_a, phi_a = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg_a = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})
    res_a = fit_heterodyne_stratified_least_squares(
        model=model_a, c2=c2_a, phi=phi_a, config=cfg_a, weights=None, shuffle=False
    )
    names_a = res_a.nlsq_diagnostics["parameter_names"]
    assert len(names_a) == len(res_a.parameters)

    # fourier: tail = fourier coefficient names
    model_f, c2_f, phi_f = make_synthetic_two_component(n_phi=7, n_t=20)
    cfg_f = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "fourier"})
    res_f = fit_heterodyne_stratified_least_squares(
        model=model_f, c2=c2_f, phi=phi_f, config=cfg_f, weights=None, shuffle=False
    )
    names_f = res_f.nlsq_diagnostics["parameter_names"]
    assert len(names_f) == len(res_f.parameters)


def test_stratified_ls_shuffle_on_deterministic_and_comparable():
    import numpy as np

    from tests.optimization._heterodyne_fixtures import make_synthetic_two_component
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    model, c2, phi = make_synthetic_two_component(n_phi=3, n_t=20)
    cfg = NLSQConfig.from_dict({"analysis_mode": "two_component", "per_angle_mode": "averaged"})

    def _fit(shuffle):
        return fit_heterodyne_stratified_least_squares(
            model=model,
            c2=c2,
            phi=phi,
            config=cfg,
            weights=None,
            shuffle=shuffle,
        )

    # 1) Determinism: shuffle=True (seed 42) is reproducible run-to-run.
    a = _fit(True)
    b = _fit(True)
    assert np.allclose(a.parameters, b.parameters, rtol=1e-8, atol=1e-10)
    assert np.isclose(a.chi_squared, b.chi_squared, rtol=1e-10)

    # 2) Comparable to shuffle-off: same objective scale, not bit-equal
    #    (the seed-42 reorder may land in a nearby basin — documented C044
    #    degeneracy — so assert SSR is comparable, not identical).
    off = _fit(False)
    assert a.chi_squared <= off.chi_squared * 2.0 + 1e-12
    assert off.chi_squared <= a.chi_squared * 2.0 + 1e-12
