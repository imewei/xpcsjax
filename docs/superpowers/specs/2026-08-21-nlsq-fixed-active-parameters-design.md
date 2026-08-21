# NLSQ `fixed_parameters` / `active_parameters` correctness fix

Status: revised after Codex review + grilling session, pending implementation plan
Date: 2026-08-21 (design approved); revised 2026-08-21 (Codex review); revised 2026-08-21 (grilling session)
Scope: `static_anisotropic`, `static_isotropic`, `laminar_flow` (homodyne), `two_component` (heterodyne)

## Revision note (Codex review)

The first version of this spec was reviewed via `codex exec` against the
actual current source (not just re-read by the author). 15 claims were
checked; 9 came back `CONFIRMED`, 1 `REFUTED` (a call-site detail), and 5
`NEEDS-FIX` — three of them structural, not cosmetic. The **Design** section
below is rewritten to incorporate every confirmed finding. Two changes are
significant enough to flag explicitly:

1. **The reduction boundary moves.** The original design planned to mask
   x0/bounds once in `core.py` and hand a shorter vector to
   `NLSQAdapter`/`NLSQWrapper`. Codex found both `adapter.py` and
   `wrapper.py` build their solver-facing model function around a
   **per-angle-expanded** vector (`[contrast_0..N, offset_0..N,
   *physical]`), constructed *after* the model function itself and
   *between* it and the actual `curve_fit()` call — a compact 5/9-length
   masked vector from `core.py` would never reach the solver as-is. The
   strip/restore step has to live inside `adapter.py`/`wrapper.py`,
   applied to the trailing physical-parameter slice of whichever vector
   (compact or expanded) each file actually sends to its own solver call.
   `core.py`'s job shrinks to: resolve *which* physical parameters are free
   vs. fixed and their values, and pass that descriptor down — not to mask
   arrays itself.
2. **Scope narrows to physical parameters only — for homodyne.** `ParameterManager.get_optimizable_parameters()`
   is physics-only by contract (excludes contrast/offset); `core.py`'s
   compact vectors always start with `[contrast, offset, ...]`. A naive
   name-membership mask over the full vector would silently freeze both
   scaling parameters on every single fit. Rather than teach the resolver a
   separate scaling-parameter policy (which then has to reason about
   per-angle-expanded contrast/offset — ambiguous, and no template example
   ever fixes a scaling parameter this way), `fixed_parameters` /
   `active_parameters` are scoped to **physical parameters only** for
   homodyne (static/laminar_flow). This is a controlled scope cut, not a
   compromise: every real `fixed_parameters` example across the three
   homodyne templates targets a physical parameter (`D_offset`, `beta`,
   `gamma_dot_t_offset`); contrast/offset already has its own dedicated
   control surface there (`per_angle_scaling` initial values + the
   `constant`/individual per-angle-mode machinery). **The grilling session
   below (§Q7) found this does *not* transfer to heterodyne**, whose own
   `active_parameters` already legitimately targets its top-level
   `contrast`/`offset` — see the Grilling revision note.

The rest of this section records what Codex confirmed, refuted, or flagged;
skip to **Grilling revision note** for the follow-up interrogation, or to
**Design** for the fully corrected plan.

| # | Claim | Verdict | Disposition |
|---|---|---|---|
| 1 | All 3 `core.py` entry points call `_get_param_names` for bounds | REFUTED | Corrected below — `fit_nlsq_multistart`/`fit_nlsq_cmaes` call `get_parameter_bounds()` with no name filter (`core.py:1539`, `:1771`); only `fit_nlsq_jax` calls `_get_param_names` directly (`core.py:453`). Net effect (full vector, fixed-unaware) is unchanged; call-site description corrected. |
| 2 | `_load_initial_params_from_config` bypasses `ConfigManager.get_initial_parameters()` | CONFIRMED | No change. |
| 3 | Mask can come straight from `get_optimizable_parameters()` | NEEDS-FIX | Physics-only API vs. scaling-prefixed vectors — see scope narrowing above. |
| 4 | — (new finding) `active_parameters: []` isn't treated as "fix everything" in homodyne (`parameter_manager.py:515`, truthiness check, not `is not None`) | NEEDS-FIX | In scope now — fixed alongside `active_parameters`, see Components §5. |
| 5 | `strip_fixed_parameters`/`restore_fixed_parameters` exist with described signatures | CONFIRMED | No change. |
| 6 | "Sequential already does this correctly" (end-to-end) | NEEDS-FIX | Overclaim — sequential only strips *already-degenerate* bounds; it doesn't consume `fixed_parameters` itself. Also: sequential's all-fixed case returns zero-length covariance rather than raising (`sequential.py:550-564`) — this plan adopts that convention instead of a new `ValueError`, see Error handling. |
| 7 | `adapter.py`/`wrapper.py` each have a model-building seam | CONFIRMED | `adapter.py::_build_model_function` (`:861`); `wrapper.py::_create_residual_function` (`:3834`). |
| 8 | Model closure can be wrapped once in `core.py` | NEEDS-FIX | See "reduction boundary moves" above. Verified directly: `wrapper.py` builds `residual_fn` at Step 6 (`:1809`) *before* expanding compact→per-angle at Step 6.6 (`:1821`); `adapter.py`'s `model_func` (`:968`) checks `n_params >= n_physical + 2*n_phi`, so `adapter.fit()` (`:1256`) must expand `x0` between `_build_model_function` (`:1342`) and `curve_fit()` (`:1393`) too. |
| 9 | Masking `fit_nlsq_multistart`'s vectors is a drop-in change | NEEDS-FIX | `_SingleFitWorker` (`core.py:1399-1409`) zips sampled starts against the *full* `_get_param_names` list and recurses into `fit_nlsq_jax` itself. Since that recursive call re-applies fixed_parameters internally once `fit_nlsq_jax` is fixed, correctness isn't blocked — but the worker's own LHS-sampling scope and zip need updating to the free physical subset, see Components §2. |
| 10 | Heterodyne `active_parameters` "correctly wired" | CONFIRMED, caveat | `_apply_initial_parameters` only runs its active/fixed logic when *both* flat `parameter_names` and `values` are present (`heterodyne_parameter_space.py:504-513`) — true of every template (they're always sibling keys), but worth loosening; see Components §4. |
| 11 | Heterodyne fix only needs `space.vary[name] = False` | NEEDS-FIX | `expand_varying_to_full` fills fixed positions from `space.values` (`heterodyne_parameter_manager.py:259`), not from a separate override — the fix must also write `space.values[canonical] = value`. |
| 12 | Tied-child-fixed conflict rule "already mirrors" the tied-vs-active rule | NEEDS-FIX | Overclaim — `_apply_tied_parameters` only checks *active*-vs-tied conflicts (`heterodyne_parameter_space.py:635-648`) and only rejects a tied child when its *parent* is non-varying (`:670-675`). A fixed-child check is genuinely new code, same file, same validation pass, same style. |
| 13 | Zero callers of `get_optimizable_parameters()`/`get_fixed_parameters()`/`is_parameter_active()` under `xpcsjax/optimization/` | CONFIRMED | No change. |
| 14 | `parameter_utils.py` exists, no naming collision | CONFIRMED | No change. |
| 15 | Covariance zero-padding convention matches spec | CONFIRMED | No change. |

## Grilling revision note

A `/grilling` session (3 rounds, 9 questions) interrogated the Codex-revised
design for judgment calls left implicit — decisions no amount of source
reading resolves, because they're about what the system *should* do, not
what it *does*. One fact-check ran alongside: a background sub-agent
confirmed **no existing test relies on the current no-op** — of the 3 tests
that set `active_parameters` and drive a real fit, all are `two_component`
(already correctly scoped, unaffected), and none assert that a
fixed/excluded parameter's value moved. Golden parity fixtures
(`stratified_residual_jit.npz`, `laminar_flow_end_to_end.npz`) confirmed to
have both keys unset. The fix is safe to land without a migration step.

| # | Decision | Resolution |
|---|---|---|
| Q1 | Is homodyne's physical-only scope cut (Codex item 2) permanent, or a v1 cut to backfill later? | **Permanent.** `per_angle_scaling` already owns scaling control; a second path to the same value is redundant, not a gap. |
| Q2 | Bundle the `active_parameters: []` truthiness bug (homodyne) into this plan, or file separately? | **Bundle.** One-line fix in code this plan already opens, same function, same config-key family. |
| Q3 | Should every tier converge on one "all parameters fixed" behavior, or preserve `sequential.py`'s existing tolerant exception? | **Preserve the split.** Sequential's zero-covariance tolerance is existing, tested behavior for its own per-angle use case; new tiers get the safer `ValueError` default with no precedent to preserve. |
| Q4 | Does `adapter.fit()`/`wrapper.fit()` derive free/fixed from `config` themselves, or does `core.py` pass an explicit descriptor? | **Explicit descriptor from `core.py`.** Deriving from `config` inside both files would duplicate the resolution logic this RCA started by eliminating. Confirmed safe: `NLSQAdapter`/`NLSQWrapper` are called extensively outside `core.py` (heavily by `heterodyne_core.py` and friends, plus many tests) — a new optional parameter defaulting to "all free" doesn't touch any of them. |
| Q5 | Scaling parameter in homodyne's `fixed_parameters`: warn-and-ignore, or hard error? | **Hard `ValueError`.** A warning is exactly the failure mode this RCA exists to eliminate. |
| Q6 | Heterodyne: loosen `_apply_initial_parameters`'s early-return (Codex item 10), or enforce the flat-`parameter_names`/`values` precondition instead? | **Loosen.** The coupling is accidental, not structural — a config setting only `fixed_parameters` and defaulting everything else is reasonable. |
| Q7 | Heterodyne: does `fixed_parameters` mirror `active_parameters`'s *existing* scope (which already includes top-level `contrast`/`offset` via `ALL_PARAM_NAMES_WITH_SCALING`), or does homodyne's physical-only rule (Q1) transfer? | **Mirrors `active_parameters`'s existing scope — includes `contrast`/`offset`.** Verified: `ALL_PARAM_NAMES_WITH_SCALING = ALL_PARAM_NAMES + SCALING_PARAMS` where `SCALING_PARAMS = ("contrast", "offset")` (`heterodyne_parameter_names.py:47,83`), and heterodyne's `active_parameters` already iterates over it. Homodyne's rule exists because homodyne's *own* `active_parameters` already excludes scaling by contract; no such contract exists on heterodyne's side, and having `fixed_parameters` reject a name `active_parameters` accepts, in the same config block, would be the more confusing inconsistency. |
| Q8 | Validation timing for Q5's hard error: config-load time (new validation point, symmetric with heterodyne) or fit-time only? | **Fit-time only**, inside `core.py`. Matches the project's existing deferred-validation convention (root `CLAUDE.md`: `ConfigManager` already validates `analysis_mode` late, not at construction) — not worth a second validation entry point in an already-large plan. |
| Q9 | Heterodyne has no guard against zero varying parameters today (`active_parameters: []` can already reach it; Q7 widens the surface via `fixed_parameters` too). Add a matching guard, or leave as a pre-existing, out-of-scope gap? | **Add it.** Verified no guard exists (`grep` across `heterodyne_parameter_manager.py`/`heterodyne_parameter_space.py`/`heterodyne_core.py` for zero-varying checks: no hits). This plan widens the ways the gap is reached, so it should close the gap it widens. |

Two silent corollaries, applied without a separate question (single sane
answer, not a judgment call):
- `parameter_manager.py::get_fixed_parameters()`'s docstring example
  (`{"contrast": 0.5, "offset": 1.0}`) becomes actively wrong under Q1/Q5 for
  homodyne — corrected to a physical-only example as part of Component 1.
- Q7's fixed-value write (Codex finding #11) applies uniformly whether the
  heterodyne name is physical or scaling — no special-casing needed in
  Component 5.

Design section below reflects all nine resolutions.

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

**Homodyne**: `core.py` resolves, once per fit, which *physical* parameters
are free vs. fixed and their values (a small descriptor, not a
length-reduced array), and passes that descriptor down to
`NLSQAdapter.fit()` / `NLSQWrapper.fit()` / CMA-ES / multistart. Each of
those applies strip-before-solve / restore-after-solve to the **trailing
physical-parameter slice** of whichever vector it actually hands to its own
`curve_fit()` call — the compact `[contrast, offset, *physical]` vector for
paths that don't expand per-angle, or the per-angle-expanded
`[contrast_0..N, offset_0..N, *physical]` vector for the standard
`per_angle_scaling=True` path — reusing the `strip_fixed_parameters`/
`restore_fixed_parameters` primitives already proven in
`strategies/sequential.py`. Physical parameters occupy the same trailing
`n_physical`-length slice in both vector shapes (per-angle expansion only
grows the leading scaling prefix), so one shared helper handles both.

**Heterodyne**: `fixed_parameters` sets `space.vary[name] = False` **and**
`space.values[canonical] = value`, mirroring the existing `active_parameters`
code path in `heterodyne_parameter_space.py::_apply_initial_parameters` plus
the value-write `expand_varying_to_full` actually depends on. Unlike
homodyne, this is **not** scoped to physical parameters only — it mirrors
`active_parameters`'s existing scope, which already legitimately includes
the top-level `contrast`/`offset` names via `ALL_PARAM_NAMES_WITH_SCALING`
(grilling session, Q7). Every tier already reads
`varying_names`/`expand_varying_to_full` off the same `ParameterManager`
instance, so this one function-level change propagates everywhere
automatically. A new guard (Q9) raises if `fixed_parameters`/
`active_parameters` combine to leave zero varying parameters.

### Components

1. **`xpcsjax/optimization/nlsq/parameter_utils.py`** (existing file,
   extend): relocate `strip_fixed_parameters`/`restore_fixed_parameters`
   here from `strategies/sequential.py` (re-exported from there for
   back-compat, signatures unchanged). Add:

   ```python
   @dataclass
   class ResolvedPhysicalParameters:
       physical_names: list[str]      # _get_physical_param_names(analysis_mode) order
       values_full: np.ndarray        # length n_physical, from initial_params
       lower_full: np.ndarray
       upper_full: np.ndarray
       free_mask: np.ndarray          # length n_physical; True where optimized

   def resolve_optimized_physical_parameters(
       param_manager: ParameterManager,
       analysis_mode: AnalysisMode,
       values_full: np.ndarray,
       lower_full: np.ndarray,
       upper_full: np.ndarray,
   ) -> ResolvedPhysicalParameters: ...
   ```

   `free_mask` comes from `param_manager.get_optimizable_parameters()`
   (already correctly computes active-minus-fixed, and is physics-only by
   contract — no scaling-parameter special-casing needed given the scope cut
   above) mapped onto `physical_names` position order. When
   `active_parameters`/`fixed_parameters` are both unset (template default),
   `free_mask` is all-`True` — byte-identical arrays to today, a provable
   no-op for every currently-passing config. Raise `ValueError` if
   `free_mask` is all-`False` (every physical parameter fixed) *unless* the
   caller is `strategies/sequential.py`, which keeps its existing, tested
   zero-length-covariance convention for that case (`sequential.py:550-564`)
   — this plan does not change that behavior, only reuses its stripping
   primitive elsewhere. Raise `ValueError` (fit-time, not config-load —
   grilling Q8) if `fixed_parameters` names a scaling parameter for
   homodyne — a hard error, not a warning (grilling Q5). Correct the
   `{"contrast": 0.5, "offset": 1.0}` example in
   `parameter_manager.py::get_fixed_parameters()`'s docstring to a
   physical-only example (`{"D_offset": 10.0}`), since it would now raise if
   followed literally.

2. **`core.py`**: all three entry points (`fit_nlsq_jax`, `fit_nlsq_cmaes`,
   `fit_nlsq_multistart`) call `resolve_optimized_physical_parameters` on the
   physical slice of their initial-values/bounds, and thread the resulting
   descriptor (not a masked array) into `NLSQAdapter.fit()` /
   `NLSQWrapper.fit()` (new optional parameter on both, defaulting to "all
   free" so every other caller of these methods is unaffected) and into
   CMA-ES/multistart's own vector construction. `_get_param_names` /
   `get_parameter_bounds()` calls stay as they are today — they still
   describe the *full* problem; only the *free* subset handed to the solver
   changes.
   - `active_parameters: []` (explicit empty list, "fix everything") is
     currently swallowed by a truthiness check in
     `ParameterManager.get_active_parameters()` (`parameter_manager.py:515`,
     `if active_params_config and isinstance(...)`) — fixed to
     `is not None`, matching the pattern heterodyne's
     `_apply_initial_parameters` already uses correctly.
   - `_SingleFitWorker` (`core.py:1399-1409`, multistart's per-start worker)
     samples and zips against the *free* physical subset instead of the full
     `_get_param_names` list, then expands each sampled start back to a full
     `initial_params` dict (fixed slots filled from the resolved descriptor,
     not from noise) before its existing recursive call into `fit_nlsq_jax`
     — which re-applies `fixed_parameters` internally regardless, so this
     change is about correct sampling scope and avoiding a length mismatch,
     not a second, independent correctness mechanism.

3. **`adapter.py`**: inside `fit()`, between `_build_model_function()`
   (`:1342`) and `self._fitter.curve_fit()` (`:1393`) — the same place the
   existing compact→per-angle expansion already happens — apply
   `strip_fixed_parameters` to the trailing physical slice of the
   (possibly-expanded) `p0`/`bounds` actually being sent to `curve_fit`, and
   wrap `model_func` (`:968`) so it restores the fixed physical values before
   evaluating the physics kernel. Restore fixed values into
   `OptimizationResult.parameters` and zero-pad the covariance/uncertainty at
   those positions after `curve_fit` returns.

4. **`wrapper.py`**: same shape, positioned after Step 6.6's per-angle
   expansion (`:1821`) and before whatever call actually invokes the
   solver — wrap the model function built at Step 6 (`:1809`,
   `_create_residual_function`) so it restores fixed physical values, strip
   the physical slice from the expanded vector handed to the solver, restore
   + zero-pad on the way out.

5. **`heterodyne_parameter_space.py::_apply_initial_parameters`**: add a
   `fixed_parameters` block, same shape as the adjacent `active_parameters`
   block, writing **both** the vary flag and the value:
   ```python
   fixed_raw = initial.get("fixed_parameters")
   if fixed_raw is not None and isinstance(fixed_raw, dict):
       for name, value in fixed_raw.items():
           canonical = ...  # same name-mapping chain as active_parameters
           space.vary[canonical] = False
           space.values[canonical] = coerce_finite_float(value, context=...)
   ```
   applied **after** the existing `active_parameters` block (fixed wins on
   conflict, warning logged — falls out of ordering). Loosen the function's
   early return (`:504-513`) so `active_parameters`/`fixed_parameters` are
   still processed when a config supplies them without flat
   `parameter_names`/`values` (currently silently skipped in that case;
   every template happens to supply both together, but nothing should rely
   on that).

6. **`heterodyne_parameter_space.py::_apply_tied_parameters`**: new
   validation — mirroring the existing active-vs-tied conflict check in
   *style*, not reusing existing code (it doesn't exist yet) — reject a tied
   child that also appears in `fixed_parameters` with a `ValueError` in the
   same message style as the existing tied-parameters checks (`:635-675`).

7. **`strategies/sequential.py`**: `strip_fixed_parameters`/
   `restore_fixed_parameters` become thin re-exports from
   `parameter_utils.py`. No behavior change — deduplication only.

8. **`heterodyne_parameter_manager.py`** (new, grilling Q9): raise
   `ValueError` when `len(self.varying_indices) == 0` — no guard exists
   today (verified: no zero-varying check anywhere in
   `heterodyne_parameter_manager.py`/`heterodyne_parameter_space.py`/
   `heterodyne_core.py`), and `active_parameters: []` could already reach
   it before this plan; Q7 widens the surface further via
   `fixed_parameters`. Natural placement: alongside `varying_names`
   (`:134-138`) or wherever `varying_indices` is first computed after
   `_apply_initial_parameters`/`_apply_tied_parameters`/`_apply_fixed...`
   run, so it fires once per config resolution, not once per solver call.

### Data flow (homodyne, `fixed_parameters` set, standard per-angle-scaling path)

```
config.initial_parameters.fixed_parameters
  -> ParameterManager.get_optimizable_parameters()             (already correct,
                                                                  physics-only)
  -> resolve_optimized_physical_parameters(...) builds free_mask
     over the physical slice only
  -> core.py entry point passes the descriptor (not a masked array) into
     NLSQAdapter.fit() / NLSQWrapper.fit() / CMA-ES / multistart
  -> adapter.py / wrapper.py expand compact -> per-angle as they do today,
     THEN strip the physical slice's fixed dims immediately before curve_fit
  -> solver sees only strictly lower < upper free dimensions        (satisfies
                                                                       the spike's
                                                                       hard constraint)
  -> model closure restores fixed physical values before every residual eval
  -> OptimizationResult: fixed values restored, uncertainty=0.0 at those
     positions
```

### Error handling

- **Homodyne**: `fixed_parameters` naming a scaling parameter (`contrast`/
  `offset`, including per-angle `contrast_N`/`offset_N`) raises `ValueError`
  at fit time, inside `core.py`'s resolution step (grilling Q5, Q8) — not a
  warning, not config-load time.
- **Homodyne**: `fixed_parameters` naming an unknown parameter: existing
  `ParameterManager` warnings already cover this (unchanged).
- **Homodyne**: `fixed_parameters` reducing the free physical set to zero:
  raise `ValueError` before calling the solver, **except** in
  `strategies/sequential.py`, which keeps its existing tested
  zero-length-covariance convention for that case unchanged (grilling Q3).
- **Heterodyne**: no scaling-parameter restriction — `fixed_parameters`
  targeting `contrast`/`offset` is accepted, mirroring `active_parameters`'s
  existing scope (grilling Q7).
- **Heterodyne**: `fixed_parameters`/`active_parameters` combining to leave
  zero varying parameters: raise `ValueError` (grilling Q9, new guard in
  `heterodyne_parameter_manager.py`).
- **Heterodyne**: fixed-child-of-a-tie conflict: `ValueError` at config-load
  time, in the same validation pass and message style as the existing
  tied-parameters checks.

### Testing

- New integration test per mode (the exact coverage gap the RCA found): a
  real `fit_nlsq_jax` (static, laminar_flow) / heterodyne `fit_nlsq_multi_phi`
  call on synthetic data with `fixed_parameters` set on a physical parameter.
  Assert the fixed parameter's fitted value equals the configured value
  exactly and its reported uncertainty is `0.0`.
- One test per strategy tier confirming a fixed parameter survives: CMA-ES,
  multistart (including that `_SingleFitWorker`'s sampling scope is correct),
  hybrid-streaming, stratified-LS, sequential (regression — must still pass
  unchanged), out-of-core.
- `active_parameters` regression test, static/laminar_flow: a config with a
  restricted `active_parameters` list, including the `active_parameters: []`
  ("fix everything") edge case; assert excluded physical parameters never
  move from their initial value.
- Heterodyne: fixed-value-is-honored test (not just vary=False — assert the
  fitted value equals the configured `fixed_parameters` value, not whatever
  the flat `parameter_names`/`values` list happened to set); tied-child+fixed
  conflict raises `ValueError`; `active_parameters`+`fixed_parameters` set
  without a flat `parameter_names`/`values` pair still applies both.
- Heterodyne, grilling Q7: `fixed_parameters: {contrast: 0.5}` (and
  `offset`) is honored — fitted value equals configured value, unlike
  homodyne where the same config raises.
- Heterodyne, grilling Q9: a config where `active_parameters`/
  `fixed_parameters` combine to leave zero varying parameters raises
  `ValueError` before reaching the solver.
- Homodyne, grilling Q5: `fixed_parameters: {contrast: 0.5}` raises
  `ValueError` at fit time (not a warning, not silently ignored).
- No regression-suite migration needed: a background fact-check (this
  session) confirmed no existing test currently passes by relying on the
  no-op behavior being fixed — see Grilling revision note.
- Full `tests/parity/_golden/` (`rtol=1e-10`) and
  `test_phase5_default_no_worse.py` re-run — must be untouched, since
  `fixed_parameters: null` / `active_parameters: null` (the template
  default) is provably the identical code path as today (`free_mask`
  all-`True`).
- `make test-optimization`, `make test-heterodyne`,
  `XPCSJAX_RUN_CHARACTERIZATION=1 XPCSJAX_RUN_ENGINE_PARITY=1 make test-full-local`
  (forced locally per root `CLAUDE.md`, since this touches the engine seam).

### Out of scope

- **Homodyne only**: fixing/honoring `fixed_parameters`/`active_parameters`
  for scaling parameters (`contrast`/`offset`, including per-angle
  `contrast_N`/`offset_N`) — permanent scope cut, grilling Q1. No homodyne
  template example fixes a scaling parameter this way; the existing
  `per_angle_scaling` initial-value block and `constant`/individual
  per-angle-mode machinery already cover that need. **Heterodyne is
  explicitly the opposite** — see grilling Q7; do not generalize this
  bullet across both modes.
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
