"""End-to-end result-assembly tests: a tied config's OptimizationResult must
report the full 14-physics block (D0_ref == D0_sample exactly) with mirrored
covariance/uncertainty, on every wired in-memory joint-fit path.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES

_D0_REF_IDX = list(ALL_PARAM_NAMES).index("D0_ref")
_D0_SAMPLE_IDX = list(ALL_PARAM_NAMES).index("D0_sample")

_N_TIMES = 40
_DT = 1.0
_Q = 0.0054
_NOISE_SIGMA = 1e-3


def _tied_config_dict(phi_angles: np.ndarray, per_angle_mode: str) -> dict:
    return {
        "analysis_mode": "two_component",
        "analyzer_parameters": {
            "dt": _DT,
            "start_frame": 1,
            "end_frame": _N_TIMES,
            "scattering": {"wavevector_q": _Q},
        },
        "scaling": {
            "n_angles": len(phi_angles),
            "mode": "constant",
            "initial_contrast": 0.3,
            "initial_offset": 1.0,
        },
        "initial_parameters": {
            "tied_parameters": {
                "D0_ref": "D0_sample",
                "alpha_ref": "alpha_sample",
                "D_offset_ref": "D_offset_sample",
            },
        },
        "optimization": {
            "nlsq": {
                "analysis_mode": "two_component",
                "max_iterations": 50,
                "enable_cmaes": False,
                "anti_degeneracy": {"per_angle_mode": per_angle_mode},
            },
        },
    }


def _build_synthetic_c2(model, phi_angles):
    rng = np.random.default_rng(seed=20260729)
    c2_stack = np.empty((len(phi_angles), _N_TIMES, _N_TIMES), dtype=np.float64)
    for i, phi in enumerate(phi_angles):
        c2 = np.asarray(model.compute_correlation(phi_angle=float(phi), angle_idx=i))
        c2_stack[i] = c2 + rng.normal(0.0, _NOISE_SIGMA, size=c2.shape)
    return c2_stack


def _run_tied_fit(tmp_path, phi_angles, per_angle_mode):
    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq

    cfg_path = tmp_path / "tied.yaml"
    cfg_path.write_text(yaml.safe_dump(_tied_config_dict(phi_angles, per_angle_mode)))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)
    data = {"c2": c2, "phi": phi_angles}
    return fit_nlsq(data, cfg)


def _assert_tied_result_shape(result):
    diag = result.nlsq_diagnostics or {}
    assert "tied_parameters" in diag, "nlsq_diagnostics must record the tied_parameters map"
    params = np.asarray(result.parameters, dtype=np.float64)
    assert params.size >= 14, f"expected physics block full-length 14, got total {params.size}"
    # locate the physics block: it's always 14 contiguous entries at the
    # position recorded by n_physics (or, for constant mode, the whole vector)
    if result.n_physics is not None:
        physics = result.physics_parameters
    else:
        physics = params
    assert physics.size == 14
    assert physics[_D0_REF_IDX] == physics[_D0_SAMPLE_IDX]
    cov = np.asarray(result.covariance, dtype=np.float64)
    unc = np.asarray(result.uncertainties, dtype=np.float64)
    # find the tied pair's position within the full vector for the cov/unc check
    if result.n_physics is not None:
        offset = params.size - 14
    else:
        offset = 0
    ref_pos, sample_pos = offset + _D0_REF_IDX, offset + _D0_SAMPLE_IDX
    if np.isfinite(unc[sample_pos]):
        assert unc[ref_pos] == unc[sample_pos]
        assert cov[ref_pos, ref_pos] == cov[sample_pos, sample_pos]

    # ``nlsq_diagnostics["parameter_names"]`` must zip 1:1 (same LENGTH and
    # same ORDER, not just length) against `result.parameters` -- this is the
    # exact invariant codex/agy audit findings #1/#2/#8 (2026-07-29) found
    # broken: the label list stayed at the REDUCED, pre-expansion length/
    # order after `result.parameters` was expanded to the full-14-physics
    # layout. A regression here would have left every OTHER assertion in
    # this helper green (they only check parameter VALUES), so this is the
    # one that would have caught the shipped bug.
    names = diag.get("parameter_names")
    assert names is not None, "nlsq_diagnostics must record parameter_names"
    assert len(names) == params.size, (
        f"parameter_names length ({len(names)}) must match result.parameters length ({params.size})"
    )
    assert list(names[offset : offset + 14]) == list(ALL_PARAM_NAMES), (
        "parameter_names' physics block must be the full ALL_PARAM_NAMES, "
        "in order, at the same position as result.parameters' physics block"
    )


def test_averaged_mode_tied_fit_reports_full_physics(tmp_path):
    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)  # n_phi=3 -> averaged
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    _assert_tied_result_shape(result)


def test_individual_mode_tied_fit_reports_full_physics(tmp_path):
    phi_angles = np.array([0.0, 90.0], dtype=np.float64)  # n_phi=2 -> individual
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    _assert_tied_result_shape(result)


def test_constant_mode_tied_fit_reports_full_physics(tmp_path):
    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    result = _run_tied_fit(tmp_path, phi_angles, "constant")
    _assert_tied_result_shape(result)


def test_single_angle_tied_fit_reports_full_physics(tmp_path):
    """Single-angle (``len(phi_angles) <= 1``) top-level tied fits fall
    through ``fit_nlsq_multi_phi``'s joint-dispatch guard (``config is not
    None and len(phi_angles) > 1``) straight to the sequential per-angle
    warm-start chain, aggregated by ``_aggregate_individual_results`` --
    a real, non-hypothetical production path (PR #27 review item 2), and
    NOT the same call path as
    ``test_fit_nlsq_jax_single_angle_tied_fit_enforces_tie_in_residual``
    above (that test calls ``fit_nlsq_jax`` directly, bypassing the
    aggregate entirely). Before this fix,
    ``_aggregate_individual_results`` reported the REDUCED
    ``[physics_varying | contrast | offset]`` vector (13 physics here, one
    tied child completely absent) instead of the full-14-physics,
    scaling-first contract every other producer emits."""
    phi_angles = np.array([0.0], dtype=np.float64)
    result = _run_tied_fit(tmp_path, phi_angles, "auto")
    assert result.n_physics == 14
    _assert_tied_result_shape(result)


def test_fit_nlsq_jax_single_angle_tied_fit_enforces_tie_in_residual():
    """``fit_nlsq_jax`` -> ``_fit_local`` (single-angle, per-angle local
    optimization) is a DISTINCT production entry point from the joint-fit
    dispatch every other test in this module exercises, and is not exercised
    by ``test_heterodyne_tied_residuals.py`` (which only hand-reconstructs
    the tie loop, never calling the real closures). Regression coverage for
    codex/agy audit finding #4, 2026-07-29.

    This proves the RESIDUAL closure (not just post-hoc
    ``expand_varying_to_full``) enforced child==parent throughout the solve:
    if ``_fit_local``'s ``jax_residual_fn`` tie loop were dropped, the
    optimizer would have searched with ``D0_ref`` FROZEN at its stale initial
    value while ``D0_sample`` moved -- so re-evaluating the SAME production
    residual kernel at the post-hoc-tied converged point would then diverge
    from the optimizer's own reported objective (``final_cost``), since that
    optimizer never actually evaluated the model at the tied point.
    """
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_residuals
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_jax

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
        use_nlsq_library=True,
        _skip_global_selection=True,  # force the _fit_local path
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
    reported_ssr = 2.0 * float(result.final_cost)
    assert np.isclose(recomputed_ssr, reported_ssr, rtol=1e-4, atol=1e-8), (
        f"recomputed SSR at the tied point ({recomputed_ssr}) diverges from "
        f"the optimizer's reported objective ({reported_ssr}) -- the "
        "signature of a dropped tie-enforcement loop in _fit_local's "
        "residual closure (the solve would have tracked a stale, un-tied "
        "D0_ref instead of the converged D0_sample)."
    )


def test_build_hybrid_streaming_result_expands_fixed_physics_param():
    """Regression test for the pre-existing bug: a fixed (non-tied) physics
    param must not shrink the reported parameters below 14 physics + scaling."""
    from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
    from xpcsjax.config.heterodyne_parameter_space import ParameterSpace
    from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
        build_hybrid_streaming_result,
    )

    space = ParameterSpace.from_config(
        {
            "analysis_mode": "two_component",
        }
    )
    pm = ParameterManager(space)
    pm.set_vary("D0_ref", False)
    n_varying = len(pm.varying_indices)
    n_phi = 2
    n_scaling = 2 * n_phi  # individual layout
    popt = np.concatenate([np.full(n_scaling, 0.5), pm.get_initial_values()])
    pcov = np.eye(n_varying + n_scaling, dtype=np.float64)

    class _FakeModel:
        param_manager = pm

    result = build_hybrid_streaming_result(
        model=_FakeModel(),
        popt=popt,
        pcov=pcov,
        info={"nit": 1, "success": True},
        phi_angles=np.array([0.0, 90.0]),
        per_angle_mode="individual",
    )
    assert result.parameters.size == 14 + n_scaling, (
        f"expected 14 + {n_scaling} = {14 + n_scaling}, got {result.parameters.size}"
    )
    # codex/agy audit finding #2: parameter_names must match this expanded
    # length/order, not the reduced pre-expansion list.
    names = result.nlsq_diagnostics["parameter_names"]
    assert len(names) == result.parameters.size
    assert list(names[-14:]) == list(ALL_PARAM_NAMES)


def test_build_hybrid_streaming_result_mirrors_tied_child():
    from xpcsjax.config.heterodyne_parameter_manager import ParameterManager
    from xpcsjax.config.heterodyne_parameter_space import ParameterSpace
    from xpcsjax.optimization.nlsq.heterodyne_result_builder import (
        build_hybrid_streaming_result,
    )

    space = ParameterSpace.from_config(
        {
            "analysis_mode": "two_component",
            "initial_parameters": {"tied_parameters": {"D0_ref": "D0_sample"}},
        }
    )
    pm = ParameterManager(space)
    n_varying = len(pm.varying_indices)
    n_phi = 2
    n_scaling = 2 * n_phi
    popt = np.concatenate([np.full(n_scaling, 0.5), pm.get_initial_values()])
    pcov = np.eye(n_varying + n_scaling, dtype=np.float64)

    class _FakeModel:
        param_manager = pm

    result = build_hybrid_streaming_result(
        model=_FakeModel(),
        popt=popt,
        pcov=pcov,
        info={"nit": 1, "success": True},
        phi_angles=np.array([0.0, 90.0]),
        per_angle_mode="individual",
    )
    physics = result.parameters[-14:]
    assert physics[_D0_REF_IDX] == physics[_D0_SAMPLE_IDX]
    # codex/agy audit finding #2: parameter_names must match this expanded
    # length/order, not the reduced pre-expansion list.
    names = result.nlsq_diagnostics["parameter_names"]
    assert len(names) == result.parameters.size
    assert list(names[-14:]) == list(ALL_PARAM_NAMES)


def test_stratified_ls_tied_fit_reports_full_physics(tmp_path):
    """Call fit_heterodyne_stratified_least_squares directly with a tied
    ParameterManager and a small synthetic dataset, bypassing the
    dispatcher's >=1M-point size gate entirely.

    The brief's suggested call (``fit_heterodyne_stratified_least_squares(
    model=model, c2_data=c2, phi_angles=phi_angles)``) was NOT re-verified
    during planning. Grepping the real signature
    (heterodyne_stratified_ls.py:550) shows it is keyword-only
    ``model, c2, phi, config, weights, target_chunk_size=..., shuffle=...,
    use_index_based=..., check_memory_safety=..., anti_degeneracy_dict=...``
    -- the parameter names are ``c2``/``phi``, not ``c2_data``/``phi_angles``,
    and ``config`` must be an ``NLSQConfig`` instance (built the same way the
    dispatcher in ``xpcsjax/optimization/nlsq/__init__.py`` builds it: unwrap
    ``optimization.nlsq`` then ``NLSQConfig.from_dict``), not a raw dict.
    """
    import yaml

    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
    from xpcsjax.optimization.nlsq.heterodyne_stratified_ls import (
        fit_heterodyne_stratified_least_squares,
    )

    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
    config = _tied_config_dict(phi_angles, "auto")
    cfg_path = tmp_path / "tied_stratified.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    cfg = ConfigManager(str(cfg_path))
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)

    nlsq_dict = dict(config["optimization"]["nlsq"])
    nlsq_dict.setdefault("analysis_mode", config["analysis_mode"])
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
    _assert_tied_result_shape(result)


def test_hybrid_streaming_tied_fit_reports_full_physics(tmp_path, monkeypatch):
    """Force the hybrid-streaming dispatch on a small synthetic dataset so
    Task 9's wired residual + Task 13's build_hybrid_streaming_result fix
    are exercised together end-to-end.

    ``hybrid_streaming.enable: true`` alone is NOT sufficient: the dispatcher
    (``_fit_nlsq_heterodyne`` in ``xpcsjax/optimization/nlsq/__init__.py``)
    additionally gates on ``select_nlsq_strategy(...)`` returning
    ``NLSQStrategy.LARGE``/``STREAMING`` (``heterodyne_memory.py``), which for
    this tiny synthetic fixture (n_phi=3, N=40) would naturally resolve to
    ``STANDARD`` and silently fall through to the plain joint fit. Both
    ``select_nlsq_strategy`` and
    ``fit_with_stratified_hybrid_streaming_heterodyne`` are imported with a
    LOCAL ``from ... import ...`` inside the dispatch branch (not at module
    load time), so monkeypatching the module-level attributes here is picked
    up by that late-bound import. A call-through spy on
    ``fit_with_stratified_hybrid_streaming_heterodyne`` proves the streaming
    path actually ran rather than trusting a green result blindly.
    """
    import yaml

    import xpcsjax.optimization.nlsq.heterodyne_memory as heterodyne_memory
    import xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming as hs_mod
    from xpcsjax.config import ConfigManager
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel
    from xpcsjax.optimization.nlsq import fit_nlsq
    from xpcsjax.optimization.nlsq.heterodyne_memory import NLSQStrategy, StrategyDecision

    phi_angles = np.array([0.0, 45.0, 90.0], dtype=np.float64)
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
    model = HeterodyneModel.from_config(cfg.config)
    c2 = _build_synthetic_c2(model, phi_angles)
    result = fit_nlsq({"c2": c2, "phi": phi_angles}, cfg)

    assert called["hit"], (
        "fit_with_stratified_hybrid_streaming_heterodyne was never invoked -- "
        "the memory-tier override failed to force the STREAMING/LARGE "
        "dispatch, so this test would otherwise have silently passed against "
        "the standard joint-fit path instead of the hybrid-streaming path"
    )
    _assert_tied_result_shape(result)
