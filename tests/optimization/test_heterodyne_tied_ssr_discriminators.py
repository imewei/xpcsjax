"""SSR-recompute discriminator tests for the tied-parameters wired closures
NOT already covered by an SSR-recompute check.

``tests/optimization/test_heterodyne_tied_result_assembly.py`` and
``test_heterodyne_tied_gradient_coupling.py`` only prove real during-solve
tie coupling (as opposed to cosmetic post-hoc mirroring by
``expand_reduced_result``/``expand_varying_to_full``) for TWO of the eight
production closures wired into this PR: the ``constant``-mode joint fit and
the single-angle ``_fit_local``/NLSQAdapter path. Every other wired closure
only has a SHAPE check (``result.parameters[child] == result.parameters[parent]``),
which passes identically whether the tie was enforced during the solve or
mirrored only at report time -- ``ParameterManager.expand_reduced_result``
unconditionally mirrors the tied child onto its parent when assembling the
reported ``OptimizationResult``, regardless of what the optimizer actually
searched.

This module adds the same SSR-recompute discriminator technique (run a real
tied fit; independently recompute SSR by feeding the model's own forward
computation the REPORTED tied parameters; assert it agrees with the
solver's own reported objective) for:

1. Averaged mode joint fit (``_fit_joint_averaged_multi_phi``).
2. Individual mode joint fit (``_build_joint_problem``'s closure -- reused
   verbatim by BOTH the joint CMA-ES escape and joint multistart escape via
   ``prob.joint_residual_fn``, so this test also validates those two paths
   transitively; not separately re-tested here since they'd construct the
   identical ``joint_residual_fn`` closure).
3. Stratified-LS (``fit_heterodyne_stratified_least_squares``), called
   directly, bypassing the dispatcher's >=1M size gate.
4. Hybrid-streaming (``fit_with_stratified_hybrid_streaming_heterodyne``),
   forced via the memory-tier monkeypatch technique.
6. The NLSQWrapper-fallback closure (``_make_numpy_residual_fn``, used when
   ``use_nlsq_library=False``).

Item 5 (the per-angle CMA-ES escape, ``_fit_cmaes``) is NOT included here --
see ``test_per_angle_cmaes_escape_ssr_matches_recompute`` docstring below for
why the technique does not discriminate for that specific closure (this was
verified empirically, not assumed).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import yaml

from tests.optimization.test_heterodyne_tied_result_assembly import (
    _build_synthetic_c2,
    _run_tied_fit,
    _tied_config_dict,
)
from xpcsjax.config import ConfigManager
from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals
from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel


def _recompute_joint_ssr(model, phi_angles: np.ndarray, c2: np.ndarray, result) -> float:
    """Recompute SSR from a joint-fit ``OptimizationResult``'s OWN REPORTED
    parameters, using the production multi-angle residual kernel.

    Handles the canonical scaling-first layout
    (``[scaling_head | physics(14)]``) shared by every joint-fit producer:
    ``n_scaling == 2`` (averaged, one contrast/offset pair broadcast across
    angles) or ``n_scaling == 2 * n_phi`` (individual, per-angle scaling).
    """
    n_phi = len(phi_angles)
    params = np.asarray(result.parameters, dtype=np.float64)
    assert result.n_physics == 14, "joint-fit producers report the full 14-physics tail"
    physics = result.physics_parameters
    n_scaling = params.size - 14
    scaling_head = params[:n_scaling]
    if n_scaling == 2:
        contrast_per_angle = np.full(n_phi, scaling_head[0], dtype=np.float64)
        offset_per_angle = np.full(n_phi, scaling_head[1], dtype=np.float64)
    elif n_scaling == 2 * n_phi:
        contrast_per_angle = scaling_head[:n_phi]
        offset_per_angle = scaling_head[n_phi : 2 * n_phi]
    else:
        raise AssertionError(f"unexpected scaling-head width {n_scaling} for n_phi={n_phi}")

    c2_jax = jnp.asarray(c2, dtype=jnp.float64)
    residual = compute_multi_angle_residuals(
        jnp.asarray(physics, dtype=jnp.float64),
        jnp.asarray(model.t, dtype=jnp.float64),
        model.q,
        model.dt,
        jnp.asarray(phi_angles, dtype=jnp.float64),
        c2_jax,
        jnp.ones_like(c2_jax),
        jnp.asarray(contrast_per_angle, dtype=jnp.float64),
        jnp.asarray(offset_per_angle, dtype=jnp.float64),
    )
    return float(np.sum(np.asarray(residual) ** 2))


def _tied_model_and_c2(tmp_path, phi_angles: np.ndarray, per_angle_mode: str):
    """Rebuild the exact same deterministic model/data a ``_run_tied_fit``
    call was run against (same config -> same model -> same synthetic c2,
    per the seeded ``_build_synthetic_c2``)."""
    cfg_path = tmp_path / "tied_recheck.yaml"
    cfg_path.write_text(yaml.safe_dump(_tied_config_dict(phi_angles, per_angle_mode)))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)
    return model, c2


def _build_perturbed_tied_model_and_c2(tmp_path, phi_angles: np.ndarray, per_angle_mode: str):
    """Like ``_tied_model_and_c2``, but the synthetic "true" data is generated
    from physics perturbed FAR from the config's initial/registry-default
    values (mirroring ``test_engine_heterodyne_fit_parity.py``'s
    ``_make_well_posed_case`` / ``_TRUE_PERTURB`` technique), not from the
    model's own defaults.

    This matters empirically for the SSR-recompute discriminator technique:
    ``_build_synthetic_c2`` (the shared helper) generates data from the
    model's OWN default physics, so a fit barely needs to move its
    parameters to converge. A tied child accidentally frozen at its
    config-time-synced INITIAL value (the exact bug this helper is designed
    to catch) is then numerically almost indistinguishable from the
    correctly-tracked value, since the parent barely moved from that same
    initial value either -- silently defeating the discriminator (verified
    empirically: ablating the stratified-LS tie loop against the default-value
    fixture moved chi_squared by ~1e-4 relative, an easy false pass at any
    reasonable tolerance). Perturbing the true D0_ref/D0_sample (and
    alpha_ref/alpha_sample) pair by a large, deliberate offset forces real
    parameter movement, so a frozen-vs-tracked mismatch is no longer masked
    by a near-degenerate optimization landscape (verified: the same ablation
    against this perturbed fixture moved chi_squared by >700% relative).
    """
    from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES

    cfg_path = tmp_path / "tied_perturbed.yaml"
    cfg_path.write_text(yaml.safe_dump(_tied_config_dict(phi_angles, per_angle_mode)))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)

    names = list(ALL_PARAM_NAMES)
    true_full = np.asarray(model.param_manager.get_full_values(), dtype=np.float64).copy()
    true_full[names.index("D0_sample")] *= 3.0
    true_full[names.index("D0_ref")] = true_full[names.index("D0_sample")]
    true_full[names.index("alpha_sample")] += 0.3
    true_full[names.index("alpha_ref")] = true_full[names.index("alpha_sample")]

    rng = np.random.default_rng(seed=20260729)
    n_t = 40
    c2_stack = np.empty((len(phi_angles), n_t, n_t), dtype=np.float64)
    for i, phi in enumerate(phi_angles):
        c2 = np.asarray(
            model.compute_correlation(
                phi_angle=float(phi), params=true_full, contrast=0.3, offset=1.0, angle_idx=i
            )
        )
        c2_stack[i] = c2 + rng.normal(0.0, 1e-3, size=c2.shape)
    return model, c2_stack


# ===========================================================================
# 1. Averaged mode joint fit (_fit_joint_averaged_multi_phi)
# ===========================================================================
def test_averaged_mode_tied_fit_ssr_matches_recompute(tmp_path):
    """Real during-solve coupling discriminator for the averaged-mode joint
    residual closure (``heterodyne_core.py`` ~line 1313). A shape-only check
    (D0_ref == D0_sample in the reported vector) cannot tell real coupling
    apart from a post-hoc-mirror-only implementation; recomputing SSR from
    the REPORTED tied parameters and comparing against the solver's own
    ``chi_squared`` can."""
    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)  # n_phi=3 -> averaged
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    diag = result.nlsq_diagnostics or {}
    assert diag.get("per_angle_mode") == "averaged", "fixture must resolve to averaged mode"
    assert "tied_parameters" in diag, "fixture must produce at least one tied pair"

    model, c2 = _tied_model_and_c2(tmp_path, phi_angles, "auto")
    recomputed_ssr = _recompute_joint_ssr(model, phi_angles, c2, result)

    assert np.isclose(recomputed_ssr, result.chi_squared, rtol=1e-4, atol=1e-8), (
        f"recomputed SSR {recomputed_ssr:.6e} from the REPORTED tied parameters "
        f"disagrees with result.chi_squared {result.chi_squared:.6e} -- signature "
        "of a dropped tie-enforcement loop in _fit_joint_averaged_multi_phi's "
        "residual closure"
    )


# ===========================================================================
# 2. Individual mode joint fit (_build_joint_problem's closure -- also
#    covers the joint CMA-ES / multistart escapes transitively, since both
#    reuse prob.joint_residual_fn verbatim rather than building their own).
# ===========================================================================
def test_individual_mode_tied_fit_ssr_matches_recompute(tmp_path):
    """Real during-solve coupling discriminator for the individual-mode
    joint residual closure (``_build_joint_problem``, ``heterodyne_core.py``
    ~line 3102). This closure's ``joint_residual_fn`` is reused verbatim by
    BOTH the joint CMA-ES escape (``_fit_joint_cmaes_multi_phi``) and the
    joint multistart escape (``_fit_joint_multistart``) -- see the CLAUDE.md
    "Heterodyne joint global escapes" section -- so this single test
    transitively validates all three paths' shared closure."""
    phi_angles = np.array([0.0, 90.0], dtype=np.float64)  # n_phi=2 -> individual
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    diag = result.nlsq_diagnostics or {}
    assert diag.get("per_angle_mode") == "individual", "fixture must resolve to individual mode"
    assert "tied_parameters" in diag, "fixture must produce at least one tied pair"

    model, c2 = _tied_model_and_c2(tmp_path, phi_angles, "auto")
    recomputed_ssr = _recompute_joint_ssr(model, phi_angles, c2, result)

    assert np.isclose(recomputed_ssr, result.chi_squared, rtol=1e-4, atol=1e-8), (
        f"recomputed SSR {recomputed_ssr:.6e} from the REPORTED tied parameters "
        f"disagrees with result.chi_squared {result.chi_squared:.6e} -- signature "
        "of a dropped tie-enforcement loop in _build_joint_problem's residual "
        "closure (shared verbatim by the joint CMA-ES/multistart escapes)"
    )


# ===========================================================================
# 3. Stratified-LS (fit_heterodyne_stratified_least_squares), called
#    directly, bypassing the dispatcher's >=1M-point size gate.
# ===========================================================================
def test_stratified_ls_tied_fit_ssr_matches_recompute(tmp_path):
    """Real during-solve coupling discriminator for the stratified-LS (>=1M)
    residual closure (``heterodyne_stratified_ls.py`` ~line 362). Mirrors the
    call pattern of ``test_stratified_ls_tied_fit_reports_full_physics``.

    Uses the PERTURBED fixture (see ``_build_perturbed_tied_model_and_c2``):
    the default-value fixture barely moves the tied parent from its initial
    (== frozen-child) value, which empirically defeats this discriminator
    (verified: ablating the tie loop against the default-value fixture only
    moved chi_squared by ~1e-4 relative -- an easy false pass)."""
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    model, c2 = _build_perturbed_tied_model_and_c2(tmp_path, phi_angles, "auto")

    nlsq_dict = dict(_tied_config_dict(phi_angles, "auto")["optimization"]["nlsq"])
    nlsq_dict.setdefault("analysis_mode", "two_component")
    nlsq_cfg = NLSQConfig.from_dict(nlsq_dict)

    result = fit_heterodyne_stratified_least_squares(
        model=model,
        c2=c2,
        phi=phi_angles,
        config=nlsq_cfg,
        weights=None,
        target_chunk_size=10,
        check_memory_safety=False,
    )
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag, "fixture must produce at least one tied pair"

    recomputed_ssr = _recompute_joint_ssr(model, phi_angles, c2, result)

    assert np.isclose(recomputed_ssr, result.chi_squared, rtol=1e-3, atol=1e-6), (
        f"recomputed SSR {recomputed_ssr:.6e} from the REPORTED tied parameters "
        f"disagrees with result.chi_squared {result.chi_squared:.6e} -- signature "
        "of a dropped tie-enforcement loop in the stratified-LS residual closure"
    )


# ===========================================================================
# 4. Hybrid-streaming (fit_with_stratified_hybrid_streaming_heterodyne),
#    forced via the memory-tier monkeypatch technique used by the existing
#    hybrid-streaming end-to-end test.
# ===========================================================================
def test_hybrid_streaming_tied_fit_ssr_matches_recompute(tmp_path, monkeypatch):
    """Real during-solve coupling discriminator for the hybrid-streaming
    ``model_fn`` residual closure
    (``strategies/heterodyne_hybrid_streaming.py`` ~line 300). Forces the
    STREAMING/LARGE dispatch on a tiny synthetic fixture the same way
    ``test_hybrid_streaming_tied_fit_reports_full_physics`` does (a call-through
    spy proves the streaming path actually ran).

    Uses the PERTURBED fixture (see ``_build_perturbed_tied_model_and_c2``) for
    the same reason as the stratified-LS test above: the default-value
    fixture barely moves the tied parent, which empirically defeats this
    style of discriminator."""
    import xpcsjax.optimization.nlsq.heterodyne_memory as heterodyne_memory
    import xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming as hs_mod
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.heterodyne_memory import NLSQStrategy, StrategyDecision

    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    model, c2 = _build_perturbed_tied_model_and_c2(tmp_path, phi_angles, "auto")
    config = _tied_config_dict(phi_angles, "auto")
    config["optimization"]["nlsq"]["hybrid_streaming"] = {"enable": True}

    forced_decision = StrategyDecision(
        strategy=NLSQStrategy.LARGE,
        threshold_gb=0.0,
        peak_memory_gb=999.0,
        reason="forced-for-test",
    )
    monkeypatch.setattr(
        heterodyne_memory, "select_nlsq_strategy", lambda *args, **kwargs: forced_decision
    )

    real_fit = hs_mod.fit_with_stratified_hybrid_streaming_heterodyne
    called = {"hit": False}

    def _spy(*args, **kwargs):
        called["hit"] = True
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(hs_mod, "fit_with_stratified_hybrid_streaming_heterodyne", _spy)

    cfg_path = tmp_path / "tied_streaming.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    cfg = ConfigManager(str(cfg_path))
    result = fit_nlsq({"c2": c2, "phi": phi_angles}, cfg)

    assert called["hit"], "the memory-tier override failed to force the streaming dispatch"
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag, "fixture must produce at least one tied pair"

    recomputed_ssr = _recompute_joint_ssr(model, phi_angles, c2, result)

    assert np.isclose(recomputed_ssr, result.chi_squared, rtol=5e-2, atol=1e-6), (
        f"recomputed SSR {recomputed_ssr:.6e} from the REPORTED tied parameters "
        f"disagrees with result.chi_squared {result.chi_squared:.6e} -- signature "
        "of a dropped tie-enforcement loop in the hybrid-streaming model_fn closure"
    )


# ===========================================================================
# 6. The NLSQWrapper-fallback closure (_make_numpy_residual_fn, used when
#    use_nlsq_library=False).
# ===========================================================================
def test_numpy_wrapper_fallback_tied_fit_ssr_matches_recompute():
    """Real during-solve coupling discriminator for the numpy/SciPy-wrapper
    fallback residual closure (``_make_numpy_residual_fn``,
    ``heterodyne_core.py`` ~line 4623), used when ``use_nlsq_library=False``.
    Distinct from ``jax_residual_fn`` (the NLSQAdapter/JAX path already
    covered by ``test_fit_nlsq_jax_single_angle_tied_fit_enforces_tie_in_residual``
    in test_heterodyne_tied_result_assembly.py) -- a different closure, with
    its own separate tie-mirror loop, so it needs its own discriminator."""
    from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
    from xpcsjax.core.heterodyne_jax_backend import compute_residuals
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_jax

    _D0_REF_IDX = list(ALL_PARAM_NAMES).index("D0_ref")
    _D0_SAMPLE_IDX = list(ALL_PARAM_NAMES).index("D0_sample")

    phi_angles = np.array([0.0], dtype=np.float64)
    config_dict = _tied_config_dict(phi_angles, "constant")
    model = HeterodyneModel.from_config(config_dict)
    c2 = _build_synthetic_c2(model, phi_angles)[0]

    result = fit_nlsq_jax(
        model,
        c2,
        phi_angle=float(phi_angles[0]),
        config=None,
        weights=None,
        use_nlsq_library=False,  # force the numpy_residual_fn / NLSQWrapper path
        _skip_global_selection=True,  # force the _fit_local path (no CMA-ES)
        angle_idx=0,
    )
    assert result.success

    full_tied = model.param_manager.expand_varying_to_full(np.asarray(result.parameters))
    assert full_tied[_D0_REF_IDX] == full_tied[_D0_SAMPLE_IDX]

    contrast_val, offset_val = model.scaling.get_for_angle(0)
    recomputed = compute_residuals(
        jnp.asarray(full_tied, dtype=jnp.float64),
        model.t,
        model.q,
        model.dt,
        float(phi_angles[0]),
        jnp.asarray(c2, dtype=jnp.float64),
        None,
        contrast_val,
        offset_val,
    )
    recomputed_ssr = float(np.sum(np.asarray(recomputed) ** 2))
    # build_result_from_nlsq (heterodyne_result_builder.py) sets
    # ``final_cost = 0.5 * SSR``, the same scipy-style convention as the
    # NLSQAdapter/JAX path -- matches ``reported_ssr`` at line 494 below.
    reported_ssr = 2.0 * float(result.final_cost)
    assert np.isclose(recomputed_ssr, reported_ssr, rtol=1e-4, atol=1e-8), (
        f"recomputed SSR at the tied point ({recomputed_ssr}) diverges from "
        f"the optimizer's reported objective ({reported_ssr}) -- the signature "
        "of a dropped tie-enforcement loop in _make_numpy_residual_fn's "
        "residual closure (the NLSQWrapper fallback path would have tracked a "
        "stale, un-tied D0_ref instead of the converged D0_sample)."
    )


# ===========================================================================
# 5. Per-angle CMA-ES escape (_fit_cmaes) -- NOT a valid SSR-recompute
#    discriminator for this specific closure. Documented, not silently
#    skipped.
# ===========================================================================
def test_per_angle_cmaes_escape_reports_tied_parameters_and_a_self_consistent_cost():
    """``_fit_cmaes`` (``heterodyne_core.py`` ~line 3946) DOES have its own
    tie-mirror loop inside ``model_func`` (the CMA-ES objective closure).
    However, empirically verified: the standard "recompute SSR from the
    REPORTED parameters and compare against the solver's own reported cost"
    technique used by every other test in this module does NOT discriminate
    a broken ``model_func`` tie loop for THIS closure specifically.

    Why: Phase 3 of ``_fit_cmaes`` (its NLSQ-vs-CMA-ES cost comparison, and
    thus the ``final_cost`` ultimately reported) recomputes BOTH candidates'
    cost via ``param_manager.expand_varying_to_full(result.parameters)`` ->
    ``compute_residuals`` -- i.e. production ITSELF applies the exact same
    post-hoc tie-mirror-then-evaluate recipe an external discriminator test
    would use to build its own comparison value. So ``result.final_cost`` is
    *already* a function of "expand the reported parameters and re-evaluate",
    not of whatever ``model_func`` actually explored during the CMA-ES
    search -- an ablated (tie-loop-removed) ``model_func`` still converges to
    SOME point (with a stale, un-tied D0_ref explored during the search),
    but Phase 3 then re-evaluates that same converged point through the
    ALWAYS-tying-aware ``expand_varying_to_full`` before computing the cost
    it reports. A discriminator that recomputes from ``result.parameters``
    the same way is therefore trivially self-consistent regardless of
    whether ``model_func``'s internal loop is live or removed.

    This was verified empirically (not assumed): the ablation described in
    the review brief -- remove the ``model_func`` tie-mirror loop, rerun --
    left this style of test passing unchanged, because both branches of the
    comparison already go through the same post-hoc expansion. A real
    discriminator for this specific closure would need to intercept
    ``model_func``'s calls during the live CMA-ES search (e.g. a call-count/
    call-value spy on ``fit_with_cmaes``) rather than comparing post-hoc
    recomputed costs -- out of scope for this pass; tracked as a known gap,
    not silently ignored.

    This test instead pins the two properties that ARE meaningfully checked
    end-to-end for this path: (a) the escape reaches a tied result at all
    (shape/marker check, like the existing per-mode tests), and (b) the
    reported cost is internally self-consistent with a separately-derived
    recompute from the reported (tied) parameters -- which is a real,
    if weaker, contract: it catches a broken *result-assembly* mirror (the
    Component 5 bug class this whole PR targets), even though it cannot
    catch a broken *model_func* mirror specifically.
    """
    from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
    from xpcsjax.core.heterodyne_jax_backend import compute_residuals
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_jax

    _D0_REF_IDX = list(ALL_PARAM_NAMES).index("D0_ref")
    _D0_SAMPLE_IDX = list(ALL_PARAM_NAMES).index("D0_sample")

    phi_angles = np.array([0.0], dtype=np.float64)
    config_dict = _tied_config_dict(phi_angles, "constant")
    model = HeterodyneModel.from_config(config_dict)
    c2 = _build_synthetic_c2(model, phi_angles)[0]

    cmaes_cfg = NLSQConfig(
        method="trf",
        loss="linear",
        max_nfev=200,
        enable_cmaes=True,
        multistart=False,
        cmaes_warmstart_auto_skip=False,  # force real Phase-2 CMA-ES search
        cmaes_max_iterations=15,
        cmaes_population_size=8,
    )

    result = fit_nlsq_jax(
        model,
        c2,
        phi_angle=float(phi_angles[0]),
        config=cmaes_cfg,
        weights=None,
        use_nlsq_library=True,
        angle_idx=0,
    )
    assert result.success
    assert result.metadata.get("optimizer") == "cmaes"

    full_tied = model.param_manager.expand_varying_to_full(np.asarray(result.parameters))
    assert full_tied[_D0_REF_IDX] == full_tied[_D0_SAMPLE_IDX]

    contrast_val, offset_val = model.scaling.get_for_angle(0)
    recomputed = compute_residuals(
        jnp.asarray(full_tied, dtype=jnp.float64),
        model.t,
        model.q,
        model.dt,
        float(phi_angles[0]),
        jnp.asarray(c2, dtype=jnp.float64),
        None,
        contrast_val,
        offset_val,
    )
    recomputed_ssr = float(np.sum(np.asarray(recomputed) ** 2))
    reported_ssr = 2.0 * float(result.final_cost)
    assert np.isclose(recomputed_ssr, reported_ssr, rtol=1e-4, atol=1e-8)
