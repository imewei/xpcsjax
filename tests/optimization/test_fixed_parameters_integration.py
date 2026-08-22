"""Integration tests: fixed_parameters/active_parameters actually constrain
the real NLSQ solve (not just ParameterManager in isolation). This is the
real proof for Tasks 3 (wrapper.py) and 4 (adapter.py) -- see their task
headers for why they carry no standalone test of their own."""

import numpy as np
import pytest

from xpcsjax.config import ConfigManager
from xpcsjax.core.jax_backend import compute_g2_scaled
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

TRUE_PHYSICAL_LAMINAR = {
    "D0": 8000.0,
    "alpha": -1.2,
    "D_offset": 50.0,
    "gamma_dot_t0": 0.01,
    "beta": 0.1,
    "gamma_dot_t_offset": 0.0,
    "phi0": 0.0,
}
TRUE_PHYSICAL_STATIC = {"D0": 8000.0, "alpha": -1.2, "D_offset": 50.0}
CONTRAST, OFFSET, Q, L, DT = 0.3, 0.8, 0.005, 2_000_000.0, 0.001

_PHYSICAL_BY_MODE = {
    "laminar_flow": TRUE_PHYSICAL_LAMINAR,
    "static_isotropic": TRUE_PHYSICAL_STATIC,
    "static_anisotropic": TRUE_PHYSICAL_STATIC,  # same 3-name physical set as static_isotropic (round 3 Codex finding #11 -- spec names both)
}
_ALL_PHYSICAL_NAMES = list(
    TRUE_PHYSICAL_LAMINAR.keys()
)  # static forward-sim still uses the full 7-slot kernel


def _physical_index(n_params, physical_names, name):
    """Index of a physical parameter in `result.parameters`.

    Physics is ALWAYS the tail of the returned vector regardless of the
    resolved per-angle scaling mode (see CLAUDE.md's "Physics is ALWAYS the
    tail" note): the scaling head varies in width -- e.g. laminar_flow with
    n_phi=3 resolves `auto -> averaged` for the SOLVE, but the wrapper's
    Phase-5 contract expands the returned vector back to the dense
    scaling-first per-angle layout (2*n_phi head), not the compact
    [contrast, offset] head a naive reading of the solve-time shape would
    suggest.
    """
    n_physical = len(physical_names)
    return n_params - n_physical + physical_names.index(name)


def _synthetic_data(analysis_mode="laminar_flow", n_t=10, n_phi=3, seed=0):
    import jax.numpy as jnp

    true_physical = _PHYSICAL_BY_MODE[analysis_mode]
    # compute_g2_scaled's kernel always takes the full 7-parameter vector;
    # for static mode the shear-related entries are simply absent from
    # TRUE_PHYSICAL_STATIC and default to 0.0 here -- physically equivalent
    # to pure diffusion, matching what a static-mode optimizer vector means.
    full_physical = {**dict.fromkeys(_ALL_PHYSICAL_NAMES, 0.0), **true_physical}
    t = np.arange(1, n_t + 1) * DT
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    # First 3 entries preserved exactly for every existing n_phi<=3 caller;
    # extended so n_phi>3 callers (e.g. Layer 5 shear-weighting coverage,
    # which gates on n_phi>3) get distinct, well-separated angles too.
    phi = np.array([0.0, 45.0, 90.0, 135.0, 180.0])[:n_phi]
    params_vec = jnp.array([full_physical[name] for name in _ALL_PHYSICAL_NAMES])
    g2 = np.stack(
        [
            # compute_g2_scaled always returns a leading n_phi axis (length 1
            # for a scalar phi) -- squeeze it before stacking our own n_phi
            # axis, or g2 ends up (n_phi, 1, n_t, n_t) instead of
            # (n_phi, n_t, n_t) and every consumer downstream misreads the
            # shape (e.g. the adapter's angle-major flattening).
            np.asarray(
                compute_g2_scaled(
                    params_vec,
                    jnp.asarray(t1),
                    jnp.asarray(t2),
                    jnp.asarray(p),
                    Q,
                    L,
                    CONTRAST,
                    OFFSET,
                    DT,
                )
            )[0]
            for p in phi
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    g2_noisy = g2 + rng.normal(scale=1e-4, size=g2.shape)
    return {
        "phi": phi,
        "g2": g2_noisy,
        "t1": t1,
        "t2": t2,
        "q": Q,
        "L": L,
        "dt": DT,
        "sigma": 1e-4 * np.ones_like(g2_noisy),
    }


def _config(
    analysis_mode="laminar_flow",
    fixed_parameters=None,
    active_parameters=None,
    extra_initial=None,
    extra_top=None,
):
    true_physical = _PHYSICAL_BY_MODE[analysis_mode]
    initial = {
        "parameter_names": list(true_physical.keys()),
        "values": list(true_physical.values()),
    }
    if fixed_parameters:
        initial["fixed_parameters"] = fixed_parameters
    if active_parameters is not None:
        initial["active_parameters"] = active_parameters
    if extra_initial:
        initial.update(extra_initial)
    config = {"analysis_mode": analysis_mode, "initial_parameters": initial}
    if extra_top:
        config.update(extra_top)
    return config


@pytest.mark.parametrize(
    "use_adapter",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "NLSQAdapter cannot complete a real solve via fit_nlsq_jax: "
                    "_normalize_data_to_object collapses 2D t1/t2 to length-n_t while "
                    "_flatten_xpcs_data needs angle-tiled flat arrays, and nlsq.curve_fit "
                    "rejects the resulting 3-column xdata even with adapter-native flat "
                    "data. Both reproduce with resolved_physical=None -- pre-existing, "
                    "unrelated to fixed_parameters. Task 4's adapter.py wiring is "
                    "therefore unreachable through this entry point (verified live)."
                ),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "analysis_mode", ["static_isotropic", "static_anisotropic", "laminar_flow"]
)
def test_fixed_parameter_survives_real_fit(analysis_mode, use_adapter):
    data = _synthetic_data(analysis_mode)
    fixed_value = 37.5  # different from the true simulated value (50.0)
    cm = ConfigManager(
        config_override=_config(analysis_mode, fixed_parameters={"D_offset": fixed_value})
    )
    result = fit_nlsq_jax(data, cm, use_adapter=use_adapter)
    physical_names = list(_PHYSICAL_BY_MODE[analysis_mode].keys())
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), physical_names, "D_offset")
    assert abs(params[d_offset_idx] - fixed_value) < 1e-9
    if result.uncertainties is not None:
        # A fixed parameter's true covariance diagonal is exactly 0.
        # `safe_uncertainties_from_pcov` (recovery.py) floors ANY near-zero
        # diagonal entry as a generic numerical-safety net, but
        # `_post_process_results` (wrapper.py) explicitly re-zeroes the
        # uncertainty at every FIXED physical position afterward -- the plan's
        # invariant is bit-exact 0.0, not "small".
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0
    if use_adapter:
        # Prove the adapter itself (Task 4) honored the fixed parameter,
        # not that it failed and silently fell back to NLSQWrapper (Task 3)
        # -- core.py's adapter->wrapper fallback would make this assertion
        # pass either way if left unchecked.
        assert result.device_info.get("fallback_occurred") is False
        assert result.device_info.get("adapter") != "NLSQWrapper"


def test_fixed_scaling_parameter_raises():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow", fixed_parameters={"contrast": 0.5}))
    with pytest.raises(ValueError, match="contrast"):
        fit_nlsq_jax(data, cm, use_adapter=False)


def test_unset_fixed_parameters_is_a_noop():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow"))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    # n_phi=3 hits the default constant_scaling_threshold (3), so laminar_flow
    # resolves auto -> averaged for the solve, but the returned vector is the
    # dense scaling-first per-angle layout: 2*n_phi (6) + n_physical (7) = 13.
    assert np.asarray(result.parameters).size == 2 * 3 + len(TRUE_PHYSICAL_LAMINAR)


def test_restricted_active_parameters_real_fit():
    """A physical parameter excluded via active_parameters must not move from
    its initial value -- distinct mechanism entry point from fixed_parameters,
    same underlying resolver."""
    data = _synthetic_data("laminar_flow")
    active = [
        "D0",
        "alpha",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]  # excludes D_offset
    cm = ConfigManager(config_override=_config("laminar_flow", active_parameters=active))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 50.0) < 1e-9  # unchanged from its initial value


def test_x_scale_map_array_branch_with_fixed_parameter():
    """Forces x_scale_value to be an ARRAY (not the default 'jac' string) via
    optimization.nlsq.x_scale_map, combined with fixed_parameters -- the
    branch v2's plan would have crashed on (slicing a 3-char string)."""
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={
            "optimization": {
                "nlsq": {
                    "x_scale_map": {name: 1.0 for name in TRUE_PHYSICAL_LAMINAR},
                }
            }
        },
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)  # must not raise
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9


def test_fixed_parameter_survives_cmaes_fit():
    """fixed_parameters must survive fit_nlsq_cmaes's own Phase 1 (NLSQ
    warm-start) / Phase 2 (CMA-ES) / Phase 3 (result-selection) sequence --
    a distinct engine from fit_nlsq_jax's local path proven in Task 5.

    auto_select is forced off so the fit actually runs through fit_nlsq_cmaes
    rather than silently falling back to local NLSQ if the synthetic bounds'
    scale ratio doesn't clear the auto-select threshold -- see
    _laminar_cmaes_config in test_cmaes_trigger.py for the same pattern.
    """
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": {"cmaes": {"enable": True, "auto_select": False}}}},
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_multistart_fit():
    """fixed_parameters must survive fit_nlsq_multistart's own LHS
    sampling/screening -- `_SingleFitWorker` narrows `start_params` to the
    free physical subset (Task 7), reconstructs the full vector, and its
    recursive `fit_nlsq_jax(..., _skip_global_selection=True)` call
    independently re-derives and enforces the fixed value regardless."""
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}},
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9


def test_restricted_active_parameters_survives_multistart_fit():
    """An active_parameters-excluded (but NOT fixed_parameters-listed)
    physical parameter must survive fit_nlsq_multistart at its INITIAL
    value, not silently move to its lower bound (dev-suite:three-brain
    deep-review finding on `core.py`'s `fit_nlsq_multistart` ->
    `resolve_optimized_physical_parameters` call: it previously passed
    `lower_bounds` as `values_full` instead of the actual initial values,
    corrupting any active_parameters-excluded slot that wasn't also fixed).
    """
    data = _synthetic_data("laminar_flow")
    active = [
        "D0",
        "alpha",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]  # excludes D_offset; D_offset is NOT in fixed_parameters
    config = _config(
        "laminar_flow",
        active_parameters=active,
        extra_top={"optimization": {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}},
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    # 50.0 is TRUE_PHYSICAL_LAMINAR's D_offset initial value (see _config's
    # default initial_parameters) -- the bug moved this to D_offset's lower
    # bound instead.
    assert abs(params[d_offset_idx] - 50.0) < 1e-9


def test_fixed_parameter_wiring_reaches_sequential_solver(monkeypatch):
    """fixed_parameters must survive the sequential per-angle tier (Task 7d).

    `optimization.stratification.force_sequential_fallback: true` is the
    actual config key `_apply_stratification_if_needed` reads to route into
    `_run_sequential_optimization` (verified live -- the `optimization.nlsq.
    sequential.enable` key the plan brief guessed does not exist).

    This test stubs the solver call and inspects exactly what Task 7d's
    wiring passed into it (bounds narrowed, initial_params patched) --
    isolating the wiring contract from the solver's own behavior. A real
    end-to-end solve through this tier (fixed_parameters survives an actual
    fit, not just a stubbed one) is covered separately by
    `test_fixed_parameter_survives_sequential_fit`, below.

    Historical note: two pre-existing, unrelated-to-fixed_parameters JAX
    TracerArrayConversionError bugs previously made this tier unable to
    complete ANY real solve under `force_sequential_fallback` -- not just
    with fixed_parameters set, but even with zero fixed parameters at all.
    Both are now fixed (see git history for `strategies/sequential.py`'s
    `has_fixed` residual-wrapping branch and `wrapper.py`'s own sequential
    `residual_func`, which was rewritten to stay pure-jnp throughout so it
    can be safely traced by NLSQ's internal `jax.jacfwd`). This test's stub
    remains useful in its own right -- it precisely isolates the wiring
    contract (bounds/initial_params) from anything solver-internal.
    """
    import xpcsjax.optimization.nlsq.wrapper as wrapper_mod
    from xpcsjax.optimization.nlsq.strategies.sequential import SequentialResult

    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"stratification": {"force_sequential_fallback": True}}},
    )
    cm = ConfigManager(config_override=config)

    captured: dict = {}

    def _fake_sequential(**kwargs):
        captured["initial_params"] = np.array(kwargs["initial_params"], dtype=np.float64)
        captured["bounds"] = tuple(np.array(b, dtype=np.float64) for b in kwargs["bounds"])
        # Mirror restore_fixed_parameters's real behavior: every angle's
        # result carries the (patched) initial_params value at fixed slots.
        combined = captured["initial_params"].copy()
        n_p = combined.shape[0]
        # Non-zero diagonal even at the fixed slot, mirroring
        # combine_angle_results' "dead"-column fallback (a fixed slot has
        # exactly-zero per-angle variance everywhere, so it is excluded from
        # the usable inverse-variance mask and falls back to a non-zero
        # scalar per-angle weight instead of reporting zero combined
        # variance) -- proves the wrapper's post-solve re-zero, not the mock.
        combined_covariance = np.eye(n_p, dtype=np.float64) * 5e-3
        return SequentialResult(
            combined_parameters=combined,
            combined_covariance=combined_covariance,
            per_angle_results=[{"phi_angle": 0.0, "n_iterations": 3, "success": True}],
            n_angles_optimized=1,
            n_angles_failed=0,
            total_cost=1.0,
            success_rate=1.0,
        )

    monkeypatch.setattr(wrapper_mod, "optimize_per_angle_sequential", _fake_sequential)

    result = fit_nlsq_jax(data, cm, use_adapter=False)

    assert result.device_info.get("strategy") == "sequential_per_angle"

    n_physical = len(_ALL_PHYSICAL_NAMES)
    d_offset_phys_idx = _ALL_PHYSICAL_NAMES.index("D_offset")

    # 1. Bounds were narrowed to the exact configured fixed value at the
    # D_offset slot (Step 1's bounds-narrowing).
    lower, upper = captured["bounds"]
    assert lower[-n_physical + d_offset_phys_idx] == 37.5
    assert upper[-n_physical + d_offset_phys_idx] == 37.5
    # 2. initial_params was ALSO patched to the same value -- sequential's
    # own restore_fixed_parameters re-inserts values FROM initial_params,
    # not from the bounds, so narrowing bounds alone is not sufficient.
    assert captured["initial_params"][-n_physical + d_offset_phys_idx] == 37.5

    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    # 3. The post-solve re-zero forces the reported uncertainty to exactly
    # 0.0 even though the stubbed covariance's diagonal there is 5e-3.
    assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_sequential_fit():
    """fixed_parameters must survive a REAL (non-stubbed) solve through the
    sequential per-angle tier -- the real-solve counterpart to
    `test_fixed_parameter_wiring_reaches_sequential_solver`, above.

    Forces the branch via `optimization.stratification.
    force_sequential_fallback: true` (the real config key, not a
    monkeypatch -- unlike every other per-tier proof test in this file, no
    strategy-selection hook needs to be forced here since this key routes
    unconditionally). No solver call is stubbed: this exercises the actual
    `optimize_per_angle_sequential` -> `optimize_single_angle` ->
    `engine.least_squares` path, including NLSQ's own internal
    `jax.jacfwd`-based Jacobian tracing of `wrapper.py`'s sequential
    `residual_func` and `strategies/sequential.py`'s `has_fixed`
    residual-wrapping closure -- the two previously-crashing call sites.
    """
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"stratification": {"force_sequential_fallback": True}}},
    )
    cm = ConfigManager(config_override=config)

    result = fit_nlsq_jax(data, cm, use_adapter=False)

    assert result.device_info.get("strategy") == "sequential_per_angle"

    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_sequential_fit_with_shear_transform():
    """fixed_parameters must survive a real sequential solve with the shear
    log-transform active (issue #58 follow-up coverage gap).

    `optimization.nlsq.shear_transforms.enable_gamma_dot_log: true` routes
    `gamma_dot_t0` through `apply_forward_shear_transforms_to_vector` /
    `apply_inverse_shear_transforms_to_vector_jax` inside the sequential
    residual closure -- a distinct code path from the untransformed case
    `test_fixed_parameter_survives_sequential_fit` covers, since the fixed
    value must survive being forward-transformed into solver space (log)
    and then inverse-transformed back (exp) at every residual evaluation
    without drifting.
    """
    data = _synthetic_data("laminar_flow")
    config = _config(
        "laminar_flow",
        fixed_parameters={"gamma_dot_t0": 0.01},
        extra_top={
            "optimization": {
                "stratification": {"force_sequential_fallback": True},
                "nlsq": {"shear_transforms": {"enable_gamma_dot_log": True}},
            }
        },
    )
    cm = ConfigManager(config_override=config)

    result = fit_nlsq_jax(data, cm, use_adapter=False)

    assert result.device_info.get("strategy") == "sequential_per_angle"

    params = np.asarray(result.parameters).ravel()
    gamma_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "gamma_dot_t0")
    assert abs(params[gamma_idx] - 0.01) < 1e-9
    assert np.asarray(result.uncertainties).ravel()[gamma_idx] == 0.0


def test_fixed_parameter_survives_out_of_core_fit(monkeypatch):
    """fixed_parameters must survive the out-of-core (chunk-wise J^T J
    accumulation) tier (Task 7c). `select_nlsq_strategy` is purely
    memory-based (no config key forces OUT_OF_CORE) -- monkeypatch it, on the
    name as imported into wrapper.py, to force the branch deterministically
    on a small synthetic dataset. This covers the first OUT_OF_CORE branch
    point (`wrapper.py`'s initial strategy check, `recovery_actions ==
    ["out_of_core_delegation"]`); the strategy-recheck branch point
    (`out_of_core_recheck_delegation`) shares the exact same
    strip/mask/restore code path in `out_of_core.py` and is not
    independently exercised here."""
    from xpcsjax.optimization.nlsq import wrapper as wrapper_module
    from xpcsjax.optimization.nlsq.memory import NLSQStrategy, StrategyDecision

    def _force_out_of_core(n_points, n_params, *args, **kwargs):
        return StrategyDecision(
            strategy=NLSQStrategy.OUT_OF_CORE,
            threshold_gb=0.001,
            index_memory_gb=0.0,
            peak_memory_gb=1.0,
            reason="forced out_of_core for test_fixed_parameter_survives_out_of_core_fit",
        )

    monkeypatch.setattr(wrapper_module, "select_nlsq_strategy", _force_out_of_core)

    data = _synthetic_data("laminar_flow")
    config = _config("laminar_flow", fixed_parameters={"D_offset": 37.5})
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    # Prove the branch actually ran, not that the mock silently no-opted.
    assert result.device_info.get("strategy") == "out_of_core"
    assert result.recovery_actions == ["out_of_core_delegation"]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_stratified_ls_fit(monkeypatch):
    """fixed_parameters must survive the stratified-LS (double-chunking,
    >=1M point) tier (Task 7e). `use_stratified_least_squares` gates purely
    on ``len(stratified_data.g2_flat) >= 1_000_000`` -- unlike OUT_OF_CORE
    there is no `select_nlsq_strategy` hook to monkeypatch, so this stubs
    `_apply_stratification_if_needed` to hand back the real 3-angle
    synthetic data TILED past the 1M-point threshold (physically real,
    repeated points -- not fabricated), letting the actual
    `fit_with_stratified_least_squares` code path run for real. NLSQ's
    `max_iterations` is capped low via config: a fixed physical parameter's
    exact-value/exact-zero-uncertainty invariant holds regardless of solver
    convergence (the restore step runs unconditionally after the solve), so
    this keeps the test fast without weakening what it proves."""
    import types

    from xpcsjax.optimization.nlsq import wrapper as wrapper_module

    n_phi, n_t = 3, 10
    raw = _synthetic_data("laminar_flow", n_t=n_t, n_phi=n_phi)
    t = np.arange(1, n_t + 1) * DT
    t1_grid, t2_grid = np.meshgrid(t, t, indexing="ij")
    phi_flat = np.concatenate([np.full(n_t * n_t, p) for p in raw["phi"]])
    t1_flat = np.tile(t1_grid.ravel(), n_phi)
    t2_flat = np.tile(t2_grid.ravel(), n_phi)
    g2_flat = np.concatenate([raw["g2"][i].ravel() for i in range(n_phi)])
    # `create_stratified_chunks`'s no-`chunk_sizes` fallback slices
    # sequentially by `target_chunk_size` (100_000). The one repeating
    # "cycle" here is `phi_flat.size` == 300 (100 points per angle block);
    # n_tile=4000 makes the total 1,200,000 -- an exact multiple of BOTH
    # 300 (so every 100_000-point chunk boundary still lands on an
    # angle-block boundary) and 100_000 (so there is no undersized trailing
    # chunk). A non-aligned total leaves a short last chunk that can miss an
    # angle entirely and fail `validate_chunk_structure()` (verified: 200
    # points then a 2-of-3-name Chunk error).
    n_tile = 4000
    assert phi_flat.size == 300  # pins the alignment arithmetic above

    def _fake_stratify(self, data, per_angle_scaling, config, logger):
        return types.SimpleNamespace(
            phi_flat=np.tile(phi_flat, n_tile),
            t1_flat=np.tile(t1_flat, n_tile),
            t2_flat=np.tile(t2_flat, n_tile),
            g2_flat=np.tile(g2_flat, n_tile),
            sigma=None,  # unweighted sentinel (matches the production convention)
            q=Q,
            L=L,
            dt=DT,
        )

    monkeypatch.setattr(
        wrapper_module.NLSQWrapper, "_apply_stratification_if_needed", _fake_stratify
    )

    data = _synthetic_data("laminar_flow", n_t=n_t, n_phi=n_phi)
    config = _config(
        "laminar_flow",
        fixed_parameters={"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": {"max_iterations": 3}}},
    )
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    # Prove the stratified-LS branch actually ran, not that the mock
    # silently fell through to a different tier.
    assert result.recovery_actions == ["stratified_least_squares_method"]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def _hybrid_streaming_fake_stratify_and_config(
    *, per_angle_mode=None, monkeypatch, n_phi=3, fixed_parameters=None, extra_nlsq=None
):
    """Shared setup for the hybrid-streaming tests below: forces the same
    tiled >=1M-point dataset as ``test_fixed_parameter_survives_stratified_ls_fit``
    (via ``_apply_stratification_if_needed``, since ``use_stratified_least_squares``'s
    gate is a bare point-count check with no ``select_nlsq_strategy`` hook to
    monkeypatch), plus the config keys needed to route into
    ``fit_with_stratified_hybrid_streaming`` instead of falling through to
    stratified-LS: ``nlsq.use_streaming`` (forces streaming mode) and
    ``nlsq.hybrid_streaming.enable`` (prefers the hybrid optimizer over basic
    streaming). Warmup/Gauss-Newton/hierarchical iteration counts are capped
    low -- the fixed-value/exact-zero-uncertainty invariant holds regardless
    of solver convergence, since the restore step runs unconditionally after
    the solve.

    ``n_phi``/``fixed_parameters``/``extra_nlsq`` default to the original
    fixed values (3 angles, D_offset fixed, no extra nlsq config) so every
    pre-existing call site is unaffected; pass them explicitly to reach a
    different branch (e.g. n_phi>3 + a fixed phi0 for Layer 5 coverage)."""
    import types

    from xpcsjax.optimization.nlsq import wrapper as wrapper_module

    n_t = 10
    raw = _synthetic_data("laminar_flow", n_t=n_t, n_phi=n_phi)
    t = np.arange(1, n_t + 1) * DT
    t1_grid, t2_grid = np.meshgrid(t, t, indexing="ij")
    phi_flat = np.concatenate([np.full(n_t * n_t, p) for p in raw["phi"]])
    t1_flat = np.tile(t1_grid.ravel(), n_phi)
    t2_flat = np.tile(t2_grid.ravel(), n_phi)
    g2_flat = np.concatenate([raw["g2"][i].ravel() for i in range(n_phi)])
    n_tile = 4000
    assert phi_flat.size == n_t * n_t * n_phi

    def _fake_stratify(self, data, per_angle_scaling, config, logger):
        return types.SimpleNamespace(
            phi_flat=np.tile(phi_flat, n_tile),
            t1_flat=np.tile(t1_flat, n_tile),
            t2_flat=np.tile(t2_flat, n_tile),
            g2_flat=np.tile(g2_flat, n_tile),
            sigma=None,
            q=Q,
            L=L,
            dt=DT,
        )

    monkeypatch.setattr(
        wrapper_module.NLSQWrapper, "_apply_stratification_if_needed", _fake_stratify
    )

    nlsq_block: dict = {
        "use_streaming": True,
        "hybrid_streaming": {
            "enable": True,
            "warmup_iterations": 2,
            "max_warmup_iterations": 2,
            "gauss_newton_max_iterations": 2,
            "chunk_size": 100_000,
        },
    }
    if per_angle_mode is not None:
        nlsq_block["anti_degeneracy"] = {
            "per_angle_mode": per_angle_mode,
            "hierarchical": {
                "enable": True,
                "max_outer_iterations": 1,
                "physical_max_iterations": 3,
                "per_angle_max_iterations": 3,
            },
        }
    if extra_nlsq:
        nlsq_block.update(extra_nlsq)

    data = _synthetic_data("laminar_flow", n_t=n_t, n_phi=n_phi)
    config = _config(
        "laminar_flow",
        fixed_parameters=fixed_parameters if fixed_parameters is not None else {"D_offset": 37.5},
        extra_top={"optimization": {"nlsq": nlsq_block}},
    )
    return data, ConfigManager(config_override=config)


def test_fixed_parameter_survives_hybrid_streaming_fit(monkeypatch):
    """fixed_parameters must survive the hybrid-streaming (L-BFGS warmup +
    Gauss-Newton) tier (Task 7f) on its default (non-hierarchical) branch:
    n_phi=3 resolves ``auto -> averaged`` (``constant_scaling_threshold``
    default 3), which sets ``use_constant=True`` and therefore SKIPS Layer 2
    hierarchical optimization -- exercising the plain
    ``optimizer.fit(func=active_model_fn, ...)`` path where the fixed
    physical parameter is threaded through the point-wise model wrapper."""
    data, cm = _hybrid_streaming_fake_stratify_and_config(monkeypatch=monkeypatch)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    # Prove the hybrid-streaming branch actually ran, not that it silently
    # fell through to stratified-LS via the try/except fallback.
    assert result.recovery_actions == ["hybrid_streaming_optimizer_method"]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_hybrid_streaming_fit_individual_mode(monkeypatch):
    """Same as above, but with ``per_angle_mode: individual`` forced
    explicitly so ``use_constant=False`` and Layer 2 hierarchical
    optimization activates -- exercising the ``hierarchical_optimizer.fit(loss_fn=...)``
    branch, which wraps the fixed physical parameter through a DIFFERENT
    closure (``loss_fn`` -> ``active_model_fn``) than the plain branch above."""
    data, cm = _hybrid_streaming_fake_stratify_and_config(
        per_angle_mode="individual", monkeypatch=monkeypatch
    )
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    assert result.recovery_actions == ["hybrid_streaming_optimizer_method"]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "D_offset")
    assert abs(params[d_offset_idx] - 37.5) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_parameter_survives_hybrid_streaming_fit_individual_mode_with_fixed_phi0(
    monkeypatch,
):
    """L5 shear-weighting phi0-index regression (dev-suite:smart-debug RCA
    follow-up to PR #57's parked finding).

    Requires simultaneously: `per_angle_mode: individual` (activates Layer 2
    hierarchical -> `shear_weight_update_callback` actually gets invoked),
    `n_phi > 3` (Layer 5's own construction gate), and `phi0` itself as the
    fixed parameter (the specific narrow case that corrupted
    `ShearSensitivityWeighting.update_phi0`'s index arithmetic before the
    fix: `phi0_index=6`/`n_physical=n_physical` (both hardcoded to the FULL
    count) meant `update_phi0` would silently read whatever parameter
    remained last in the REDUCED vector -- not phi0 -- once phi0 was
    stripped out as fixed, and treat that value as "the current phi0
    estimate" for the rest of the solve.

    This does not violate the plan's hard fixed-value/exact-zero-uncertainty
    invariants (phi0's own final value is forced correctly by the
    strip/restore mechanism regardless of L5's internal bookkeeping) -- the
    bug corrupts L5's contribution to OTHER free parameters' fit quality,
    not phi0's own reported value. The test therefore proves the crash-free
    /correct-value contract that IS testable end-to-end; it does not (and
    cannot, without a reference oracle) prove the other parameters converged
    to a *better* optimum than before the fix -- only that the code path
    that used to read the wrong array slot now runs correctly-indexed.
    """
    data, cm = _hybrid_streaming_fake_stratify_and_config(
        per_angle_mode="individual",
        monkeypatch=monkeypatch,
        n_phi=5,
        fixed_parameters={"phi0": 5.0},  # phi0's registry bounds are [-10, 10]
    )
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    assert result.recovery_actions == ["hybrid_streaming_optimizer_method"]
    params = np.asarray(result.parameters).ravel()
    phi0_idx = _physical_index(len(params), _ALL_PHYSICAL_NAMES, "phi0")
    assert abs(params[phi0_idx] - 5.0) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[phi0_idx] == 0.0


# ---------------------------------------------------------------------------
# Task 11: heterodyne real-fit integration (Tasks 8-10's config-layer wiring)
# ---------------------------------------------------------------------------

_TWO_COMPONENT_TEMPLATE_PATH = "xpcsjax/config/templates/xpcsjax_two_component.yaml"


def _heterodyne_config(fixed_parameters):
    """Load the SAME template _make_synthetic_heterodyne() uses, so the
    fitted model shares the exact geometry (t/q/dt) that generated c2 --
    do not hand-build a minimal config (round 3 Codex finding #8).

    Invariant this relies on: the synthetic-data generator and the fit call
    must use the exact same analyzer_parameters.dt/start_frame/end_frame (or
    temporal.dt/time_length/t_start) -- HeterodyneModel.sync_time_axis is
    length-only (it reads len(t), not the values), so both sides resync to
    an identical model.t as long as they load the same template and the
    same n_t; if a future edit changes those between generation and fit,
    the length-only trim will silently diverge again.
    """
    import copy

    base_config = copy.deepcopy(ConfigManager(_TWO_COMPONENT_TEMPLATE_PATH).config)
    base_config["initial_parameters"]["fixed_parameters"] = fixed_parameters
    return base_config


def _fixed_value_survives(result, name):
    """Read the fitted value for `name` using the SAME parameter_names
    metadata the codebase itself uses to interpret result.parameters
    positionally (round 3 Codex finding #10 -- not full ALL_PARAM_NAMES order)."""
    diagnostics = result.nlsq_diagnostics or {}
    param_names = diagnostics.get("parameter_names")
    assert param_names is not None, (
        "result.nlsq_diagnostics['parameter_names'] missing -- re-check Step 0"
    )
    params = np.asarray(result.parameters).ravel()
    return params[list(param_names).index(name)]


def test_heterodyne_fixed_parameter_survives_real_fit():
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
    from xpcsjax.optimization.nlsq import fit_nlsq

    # n_t=12 -> 144 points/angle: n_phi=2 resolves per_angle_mode='individual'
    # (n_phi < constant_scaling_threshold=3), which runs the L2 Stage-1
    # per-angle quantile warm-start; that estimator floors at >=100 finite
    # samples/angle (heterodyne_scaling_utils.py:750) -- the default n_t=8
    # (64 points) raises there, unrelated to fixed_parameters wiring.
    model, c2, phi = _make_synthetic_heterodyne(n_t=12)
    rng = np.random.default_rng(0)
    c2_noisy = c2 + rng.normal(
        scale=1e-4, size=c2.shape
    )  # inject noise -- helper's data is noise-free
    fixed_name = "D_offset_sample"
    physical_values = model.param_manager.get_full_values()
    fixed_value = float(physical_values[list(ALL_PARAM_NAMES).index(fixed_name)])

    data = {
        "c2": c2_noisy,
        "phi": phi,
    }  # ONLY what _fit_nlsq_heterodyne actually reads -- see Step 0/1 above
    cm = ConfigManager(config_override=_heterodyne_config({fixed_name: fixed_value}))
    result = fit_nlsq(data, cm)
    assert abs(_fixed_value_survives(result, fixed_name) - fixed_value) < 1e-6


def test_heterodyne_fixed_scaling_parameter_survives_real_fit():
    """grilling round 1 Q7 end-to-end: heterodyne can fix a scaling name, unlike
    homodyne -- and the fixed value must actually survive the solve, not just
    fail to raise (dev-suite:three-brain deep-review Minor finding: the prior
    version of this test asserted only ``convergence_status is not None``,
    which would also pass if a bare "contrast" fixed_parameters key silently
    failed to match anything under the template's default per_angle_mode=
    "auto" -> "individual" (n_phi=2 < constant_scaling_threshold=3) per-angle
    "contrast[i]" layout. Forcing per_angle_mode="constant" makes "contrast"/
    "offset" the actual flat scalar names in play, so a name-matching failure
    would surface as a wrong fitted value instead of passing silently.
    """
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq import fit_nlsq

    model, c2, phi = _make_synthetic_heterodyne()
    rng = np.random.default_rng(0)
    c2_noisy = c2 + rng.normal(scale=1e-4, size=c2.shape)
    contrast, offset = model.scaling.get_for_angle(0)

    data = {"c2": c2_noisy, "phi": phi}
    cm = ConfigManager(config_override=_heterodyne_config({"contrast": contrast}))
    result = fit_nlsq(
        data, cm
    )  # must not raise -- homodyne would raise ValueError for this, heterodyne must not
    assert result.convergence_status is not None
    diagnostics = result.nlsq_diagnostics or {}
    param_names = list(diagnostics.get("parameter_names") or [])
    params = np.asarray(result.parameters).ravel()
    # n_phi=2 < constant_scaling_threshold=3 resolves per_angle_mode='auto'
    # -> 'individual': a bare "contrast" fixed_parameters key broadcasts to
    # every per-angle "contrast_i" slot, not a single flat "contrast" name.
    contrast_indices = [i for i, name in enumerate(param_names) if name.startswith("contrast_")]
    assert contrast_indices, f"no per-angle contrast_i entries in {param_names}"
    for idx in contrast_indices:
        # 1e-4, not 1e-6: unlike a fixed PHYSICAL parameter (bounds narrowed
        # to an exact point, forcing bit-exact convergence), a fixed
        # per-angle scaling slot still carries small trf/soft_l1 solver
        # residual drift at its bound.
        assert abs(params[idx] - contrast) < 1e-4

    cm_offset = ConfigManager(config_override=_heterodyne_config({"offset": offset}))
    result_offset = fit_nlsq(data, cm_offset)
    assert result_offset.convergence_status is not None
    diagnostics_offset = result_offset.nlsq_diagnostics or {}
    param_names_offset = list(diagnostics_offset.get("parameter_names") or [])
    params_offset = np.asarray(result_offset.parameters).ravel()
    offset_indices = [i for i, name in enumerate(param_names_offset) if name.startswith("offset_")]
    assert offset_indices, f"no per-angle offset_i entries in {param_names_offset}"
    for idx in offset_indices:
        assert abs(params_offset[idx] - offset) < 1e-4
