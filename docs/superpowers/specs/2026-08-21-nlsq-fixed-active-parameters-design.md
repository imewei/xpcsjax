# NLSQ `fixed_parameters` / `active_parameters` correctness fix

Status: approved (design), pending implementation plan
Date: 2026-08-21
Scope: `static_anisotropic`, `static_isotropic`, `laminar_flow` (homodyne), `two_component` (heterodyne)

## Problem

Deep-RCA (this session) found `initial_parameters.fixed_parameters` is a
silent no-op in **all three analysis-mode families**, and
`initial_parameters.active_parameters` is a silent no-op in
**static/laminar_flow** (homodyne) only. Both are documented, user-facing
config keys in every mode's YAML template
(`xpcsjax/config/templates/xpcsjax_{static_isotropic,static_anisotropic,laminar_flow,two_component}.yaml`),
with worked examples ("Common simplification: `fixed_parameters: {D_offset:
0.0}`"). Neither has any effect on the actual NLSQ solve.

### Root cause — homodyne (static / laminar_flow)

`optimization/nlsq/core.py` has three independent, duplicated x0/bounds
construction sites — `fit_nlsq_jax` (main entry), `fit_nlsq_cmaes`,
`fit_nlsq_multistart` — each calling a hardcoded `_get_param_names(analysis_mode)`
(returns the full 5/9-parameter list for the mode) instead of anything
active/fixed-aware. Initial values come from `_load_initial_params_from_config`,
which reads `config["initial_parameters"]["parameter_names"]/["values"]`
**directly**, bypassing `ConfigManager.get_initial_parameters()` (the one
method that correctly filters `fixed_parameters`) entirely.
`ParameterManager.get_fixed_parameters()` / `get_optimizable_parameters()` /
`is_parameter_active()` — the correct API — have **zero callers** anywhere
under `xpcsjax/optimization/`. Every downstream strategy tier consumes
whatever x0/bounds one of the three `core.py` entry points hands to
`NLSQAdapter`/`NLSQWrapper`, so the bug is universal across tiers, with one
exception: `strategies/sequential.py` already does this correctly via
`strip_fixed_parameters`/`restore_fixed_parameters`.

Empirically confirmed (this session):
```
fixed_parameters: {D_offset: 0.0}, laminar_flow
-> get_optimizable_parameters() correctly excludes D_offset (unused)
-> x0 dict built by the real fit path still includes D_offset
-> solver bounds for D_offset: {min: -100000.0, max: 100000.0}  (fully free)
```

### Root cause — heterodyne (two_component)

Structurally different. `active_parameters` and `tied_parameters` **are**
correctly wired: `heterodyne_parameter_space.py::_apply_initial_parameters` /
`_apply_tied_parameters` set `space.vary`/`space.tied`, and every strategy
tier (CMA-ES, multistart, hybrid-streaming, stratified-LS, engine-route,
sequential) shares one `ParameterManager` instance and reads
`varying_names`/`expand_varying_to_full`. But `fixed_parameters` is never
read anywhere in `heterodyne_parameter_space.py` or
`heterodyne_parameter_manager.py` — `get_fixed_parameters()` there is a
pure `vary==False` *read-out*, not a config *consumer*.

Empirically confirmed (this session):
```
fixed_parameters: {v_beta: 0.0, v_offset: 0.0}
-> varying_names still includes 'v_offset' (should be excluded; is not)
```

### Why untested

Every existing `fixed_parameters` test
(`tests/config/test_debug_audit_2026_06_17.py`,
`test_debug_audit_2026_07_22_config.py`, `tests/test_debug_audit_2026_06_18.py`)
exercises `ParameterManager`/`ConfigManager` in isolation. None run an actual
`fit_nlsq`/`fit_nlsq_jax`/heterodyne fit and assert the fixed parameter
didn't move.

## Spike: does "desugar to equal (min==max) bounds" work?

The originally proposed fix — turn `fixed_parameters: {name: value}` into
`parameter_space.bounds: [{name, min: value, max: value}]` and let it flow
through unchanged — was **spiked and falsified** before this plan was
written. The upstream `nlsq` library's `least_squares` validator hard-rejects
degenerate bounds:

```
nlsq/core/least_squares.py::_validate_least_squares_inputs
ValueError: Each lower bound must be strictly less than each upper bound.
```

Reproduced against **both** `NLSQAdapter` (`use_adapter=True`) and
`NLSQWrapper` (`use_adapter=False`, the default) on a synthetic
`laminar_flow` fit: both raised, the wrapper's retry/recovery logic
exhausted 3 attempts, and the fit returned `status=failed, chi2=inf`.
Passing equal bounds through as-is would make `fixed_parameters` **worse**
than the current no-op — currently it's silently ignored (wrong but the fit
runs); with literal equal bounds it would make every such fit crash.

This is not a new finding — `strategies/sequential.py::strip_fixed_parameters`
already documents the identical constraint verbatim: *"The TRF solver used
by sequential optimization requires strict lower < upper for every
parameter. Fixed parameters (equality constraints encoded as lower == upper)
must be stripped before the call and their known values re-inserted into the
result."* That function does real vector-dimensionality reduction — drop the
fixed dimensions before the solver call, reinsert after — and is already
proven and tested. It is currently wired into only the `sequential` strategy
tier.

**Conclusion: fix by dimensionality reduction (strip/restore), not by
degenerate bounds.** This is what the rest of this spec designs.

## Scope decisions (confirmed with user)

1. **All strategy tiers** must honor `fixed_parameters`/`active_parameters`,
   not just the primary in-memory path — CMA-ES, multistart,
   hybrid-streaming, stratified-LS, out-of-core, sequential.
2. **Bundle both fixes** — `fixed_parameters` (all 3 modes) and
   `active_parameters` (static/laminar_flow only) — in one plan, since they
   share the identical root cause and fix mechanism in `core.py`.

## Design

### Architecture

Two independent fixes:

**Homodyne**: extract the proven `strip_fixed_parameters`/`restore_fixed_parameters`
pattern from `strategies/sequential.py` into a shared location, add a
resolver that computes the *optimized* (free) parameter subset from
`ParameterManager.get_optimizable_parameters()` intersected with
`active_parameters` when explicitly set, and route all three `core.py` entry
points through it. Because every downstream strategy tier is a pure
consumer of the x0/bounds those three entry points construct, fixing the
three sites fixes every tier at once (sequential.py already does this and
becomes a thin caller of the relocated shared functions — dedup only, no
behavior change).

**Heterodyne**: `fixed_parameters` only needs to set `space.vary[name] =
False`, exactly mirroring the existing `active_parameters` code path in
`heterodyne_parameter_space.py::_apply_initial_parameters`. Every tier
already reads `varying_names`/`expand_varying_to_full` off the same
`ParameterManager` instance, so this one function-level change propagates
everywhere automatically.

### Components

1. **`xpcsjax/optimization/nlsq/parameter_utils.py`** (existing file,
   extend): relocate `strip_fixed_parameters`/`restore_fixed_parameters`
   here from `strategies/sequential.py` (re-exported from there for
   back-compat). Add:

   ```python
   @dataclass
   class ResolvedParameterSet:
       param_names: list[str]        # full mode list, unchanged order
       x0_full: np.ndarray
       lower_full: np.ndarray
       upper_full: np.ndarray
       free_mask: np.ndarray         # True where optimized
       x0_free: np.ndarray
       lower_free: np.ndarray
       upper_free: np.ndarray

   def resolve_optimized_parameter_set(
       param_manager: ParameterManager,
       analysis_mode: AnalysisMode,
       x0_full: np.ndarray,
       lower_full: np.ndarray,
       upper_full: np.ndarray,
   ) -> ResolvedParameterSet: ...
   ```

   `free_mask` is derived from
   `param_manager.get_optimizable_parameters()` (already correctly computes
   active-minus-fixed) mapped onto `param_names` position order. When
   `active_parameters`/`fixed_parameters` are both unset (template default),
   `free_mask` is all-`True` and every `_free` array equals its `_full`
   counterpart — byte-identical to the current arrays, so this is a provable
   no-op for every currently-passing config.

2. **`core.py`**: all three entry points (`fit_nlsq_jax`, `fit_nlsq_cmaes`,
   `fit_nlsq_multistart`) call `resolve_optimized_parameter_set` instead of
   using `_get_param_names(analysis_mode)` bounds directly. They pass the
   `_free` x0/bounds to `NLSQAdapter`/`NLSQWrapper`/CMA-ES/multistart. The
   model/residual callable each already builds
   (`adapter.py::_build_model_function`'s `model_func`; `wrapper.py`'s
   internal equivalent) is wrapped in a `restore_fixed_parameters` closure —
   identical in shape to `sequential.py`'s existing
   `def residual_func(params, *args, **kwargs): full = restore_fixed_parameters(...)`
   — so the physics kernel (`compute_g2_scaled` etc.) always receives the
   full-length positional vector it expects. The returned `OptimizationResult`
   has fixed values restored into `.parameters` and the covariance/uncertainty
   rows for fixed positions zero-padded (matches `sequential.py`'s existing,
   documented convention: fixed parameters report `uncertainty == 0.0` — "never
   estimated, not because perfectly known" — not `NaN`).

3. **`heterodyne_parameter_space.py::_apply_initial_parameters`**: add a
   `fixed_parameters` block, same shape as the adjacent `active_parameters`
   block:
   ```python
   fixed_raw = initial.get("fixed_parameters")
   if fixed_raw is not None and isinstance(fixed_raw, dict):
       for name in fixed_raw:
           canonical = ...  # same name-mapping chain as active_parameters
           space.vary[canonical] = False
   ```
   Conflict rules, mirroring the existing tied-vs-active precedence already
   in `_apply_tied_parameters`:
   - A name in both `active_parameters` and `fixed_parameters` → fixed wins
     (a stronger constraint), warning logged. Apply `_apply_initial_parameters`'s
     `fixed_parameters` block **after** its `active_parameters` block so this
     falls out naturally from ordering.
   - A tied child listed in `fixed_parameters` → `ValueError` at config-load
     time in `_apply_tied_parameters` (children are already forced
     non-varying via the tie; being separately "fixed" is a contradiction in
     terms the existing tie-validation should reject the same way it already
     rejects a tied child appearing in `active_parameters`).

4. **`strategies/sequential.py`**: `strip_fixed_parameters`/
   `restore_fixed_parameters` become thin re-exports from
   `parameter_utils.py`. No behavior change — deduplication only.

### Data flow (homodyne, `fixed_parameters` set)

```
config.initial_parameters.fixed_parameters
  -> ParameterManager.get_optimizable_parameters()          (already correct)
  -> resolve_optimized_parameter_set(...) builds free_mask
  -> core.py entry point masks x0/bounds to the free subset
  -> solver (NLSQAdapter / NLSQWrapper / CMA-ES / multistart)
       sees only strictly lower < upper free dimensions      (satisfies the
                                                                spike's hard
                                                                constraint)
  -> model closure re-expands to the full vector on every residual eval
  -> OptimizationResult: fixed values restored, uncertainty=0.0 at those
     positions
```

### Error handling

- `fixed_parameters` naming an unknown or scaling parameter: existing
  `ParameterManager` warnings already cover this (unchanged).
- `fixed_parameters` reducing the free set to zero (every physics parameter
  fixed): raise `ValueError` before calling the solver — there is nothing
  left to optimize. A "compute residual once, no fit" degenerate path is out
  of scope for this plan.
- Heterodyne fixed-child-of-a-tie conflict: `ValueError` at config-load
  time, in the same validation pass and message style as the existing
  tied-parameters checks.

### Testing

- New integration test per mode (the exact coverage gap the RCA found): a
  real `fit_nlsq_jax` (static, laminar_flow) / heterodyne `fit_nlsq_multi_phi`
  call on synthetic data with `fixed_parameters` set. Assert the fixed
  parameter's fitted value equals the configured value exactly and its
  reported uncertainty is `0.0`.
- One test per strategy tier confirming a fixed parameter survives: CMA-ES,
  multistart, hybrid-streaming, stratified-LS, sequential (regression — must
  still pass unchanged), out-of-core.
- `active_parameters` regression test, static/laminar_flow: a config with a
  restricted `active_parameters` list; assert excluded physics parameters
  never move from their initial value.
- Full `tests/parity/_golden/` (`rtol=1e-10`) and
  `test_phase5_default_no_worse.py` re-run — must be untouched, since
  `fixed_parameters: null` / `active_parameters: null` (the template
  default) is provably the identical code path as today (`free_mask`
  all-`True`).
- `make test-optimization`, `make test-heterodyne`,
  `XPCSJAX_RUN_CHARACTERIZATION=1 XPCSJAX_RUN_ENGINE_PARITY=1 make test-full-local`
  (forced locally per root `CLAUDE.md`, since this touches the engine seam).

### Out of scope

- Fixing the separate CLI-override bug found during the RCA
  (`cli/config_handling.py`'s `--initial-*` override re-introducing a fixed
  parameter into `parameter_names`/`values` because it checks `active_names`
  without excluding `fixed_parameters`) — related but independent; not
  blocking this plan, filed as a follow-up.
- `xpcsjax/core/fitting.py::ParameterSpace` (the hardcoded-bounds fallback
  class, `HAS_PARAMETER_MANAGER=False` path only) and
  `xpcsjax/config/parameter_space.py::ParameterSpace` (the orphaned,
  no-production-caller class that already respects `active_parameters`
  correctly) — dead-code cleanup, not required for this fix, noted as a
  follow-up in the RCA.
- Correcting the misleading docstring/comment in
  `config/manager.py::_calculate_midpoint_defaults` ("already excludes
  fixed parameters" — false) — trivial, can ride along with this PR or be
  filed separately.
