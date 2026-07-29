# Heterodyne tied parameters (equality constraints) — design

**Date:** 2026-07-29
**Status:** Approved (brainstorming) — pending implementation plan
**Author:** Wei Chen (with Claude Code)

## Goal

Add a general, config-driven mechanism to force two named `two_component` (heterodyne) physics
parameters to be numerically equal throughout optimization — e.g. `D0_ref == D0_sample`,
`alpha_ref == alpha_sample`, `D_offset_ref == D_offset_sample` — with correct joint-optimizer
coupling (the tied pair behaves as ONE free variable, not two independently-fitted values
mirrored post-hoc) and correct downstream reporting (result JSON/plots see the full physics
vector, no shape mismatch).

**Scope: `two_component` (heterodyne) only.** Static (`static_anisotropic`/`static_isotropic`)
and `laminar_flow` (homodyne) modes do not need this — confirmed explicitly, not inferred.

## Background (verified findings that scope the task)

### Why the naive approach fails

The obvious workaround — set `active_parameters` to exclude one side of a would-be-tied pair,
manually pin its value to match the other side, refit, repeat — was tried on a real dataset
(C045) and found broken two ways:

1. **Not real coupling.** Excluding a parameter via `active_parameters`/`vary=False` fixes it to
   a *constant*; the optimizer never sees a dependency on the other (still-free) parameter. Any
   resulting equality is coincidental/manual (an outer bootstrap loop), not a property the
   optimizer enforces, and the reported uncertainty on the fixed side is meaningless (no
   covariance was computed for it).
2. **Breaks viz for `individual` mode.** `xpcsjax/viz/nlsq_plots.py:1651` derives
   `n_physical = len(model.parameter_names)` (not a hardcoded `14`), and the length check at
   `:1657` is already bypassed when `per_angle_mode in ("averaged", "constant")`. It still enforces
   `n_physical + 2·n_phi` for `individual` mode (and for diagnostics-less results), which is where
   fixing physics parameters via `active_parameters` — shrinking the *reported* result vector
   instead of filling it with the constant — still throws
   `NotImplementedError: Heterodyne result has N parameters but xpcsjax viz expects M`.

### The existing scatter idiom (foundation to build on)

The codebase already has an established, JIT-safe pattern for "reduced free-vector → full
14-physics-array", used identically in every in-memory/streaming/stratified residual closure:

```python
fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)   # (14,)
varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)       # static
...
full_jax = fixed_values_jax.at[varying_indices_jax].set(physics_varying)              # in closure
```

Confirmed occurrences of this exact idiom:
- `xpcsjax/optimization/nlsq/heterodyne_core.py:1313` — `_fit_joint_averaged_multi_phi`
- `xpcsjax/optimization/nlsq/heterodyne_core.py:3102` — `_build_joint_problem` (individual mode)
- `xpcsjax/optimization/nlsq/heterodyne_core.py:3992`, `:4343`, `:4552` — per-angle/single-phi
  variants
- `xpcsjax/optimization/nlsq/heterodyne_constant_mode.py:222` — `_fit_joint_constant_multi_phi`
- `xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:362` — ≥1M stratified-LS residual
- `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:300` — hybrid-streaming
  `model_fn`

`varying_indices` itself is already a shared, cached property
(`xpcsjax/config/heterodyne_parameter_manager.py:139-145`) — the *index computation* is
centralized even though the *scatter call* is duplicated per closure. This is documented as the
deliberate JIT-safe pattern (comments at `heterodyne_core.py:1306-1307`, `:3079-3083`): NLSQ's
`masked_residual_func` JIT-traces the closure, so a Python-level static `jnp.int32` index array
closed over inside the traced function is required — `np.asarray()` on a traced value raises
`TracerArrayConversionError`.

### Bounds/p0/CMA-ES normalization already restrict to varying parameters — no extra work needed

`get_bounds()` and `get_initial_values()` (`heterodyne_parameter_manager.py:156-206`) already
restrict to `varying_indices` (distinct from `get_bounds_as_arrays()`/`get_bounds_as_tuples()`,
which default to all 16 unless `parameter_names` is passed explicitly). Every real call site uses
the restricting pair. CMA-ES normalization (`cmaes_wrapper.py`) derives its scale/offset purely
from whatever bounds array is passed in — no independent length-14 array exists there.
**Consequence: setting a tied child's `vary=False` (same lever as an ordinary fixed parameter)
automatically shrinks bounds/p0 for every optimizer path with zero additional code.**

### Escapes and L2 hierarchical are not independent touch points

- `_fit_joint_cmaes_multi_phi` and `_fit_joint_multistart` (`heterodyne_core.py:1846`, `:2116`)
  both call `_build_joint_problem` and reuse its `joint_residual_fn`/`lb`/`ub` verbatim — they do
  not construct their own physics vector. Wiring tying into `_build_joint_problem` covers both
  escapes automatically.
- `hierarchical.py`'s `HierarchicalOptimizer` (L2) never touches the physics-array expansion at
  all. It operates on the already-reduced `[scaling | physics_varying]` buffer and calls back
  into the SAME `loss_fn`/`residual_fn` the plain solve path built (confirmed in both heterodyne
  callers: `heterodyne_stratified_ls.py:488-538`, `heterodyne_hybrid_streaming.py:680-728`). Not a
  touch point.
- `_build_joint_fourier` does not exist anywhere in the codebase (the CLAUDE.md reference to it is
  stale). There are only 3 per-angle scaling modes — `constant`/`averaged`/`individual`
  (`per_angle_mode.py:19`).

### Result assembly: no path currently reports the expanded vector — this is the design's central gap

`expand_varying_to_full()` (`heterodyne_parameter_manager.py:208-239`, plain numpy/Python, not
JIT-traced) IS called on three of the four result-assembly paths — but only for a side effect, not
for the reported result:
- `heterodyne_core.py:1487` (averaged) and `:3512` (`_build_joint_result`, individual + escapes)
- `heterodyne_constant_mode.py:337` (constant)

In every one of these three call sites, `expand_varying_to_full()`'s return value (`full_fitted`) is
used **only** for `model.set_params(full_fitted)` — a side effect on the model object. The actual
`OptimizationResult(parameters=...)` construction at each site passes the **raw reduced optimizer
vector**, never the expanded output: `heterodyne_core.py:1711`
(`parameters=np.asarray(fitted_all, ...)`), `heterodyne_core.py:3759`
(`parameters=np.asarray(fitted_params_full, ...)`), and `heterodyne_constant_mode.py:452`
(`parameters=fitted_physics`). So making `expand_varying_to_full()` tying-aware, by itself, changes
nothing about what any of these three sites report — each `OptimizationResult(...)` construction
needs its own edit to consume the full-expansion output instead of the reduced vector. This makes
result assembly a **per-call-site fix**, not a single centralized one.

The 4th pattern — `heterodyne_stratified_ls.py:1287` → `build_hybrid_streaming_result`
(`heterodyne_result_builder.py:742` for the `parameters=popt` assignment) — does **not** call
`expand_varying_to_full` at all; it returns the raw reduced `popt` vector directly as
`OptimizationResult.parameters`. This is the exact bug class behind the viz crash that motivated
this work, and needs fixing independent of tying (see "Tier 0" below).

### A related, pre-existing bug — confirmed real, explicitly out of scope

`xpcsjax/optimization/nlsq/heterodyne_engine_route.py` (`fit_two_component_via_engine`) is the
production-default router for in-memory `two_component` fits (<1M points, non-escape,
constant/averaged/individual modes — `_fit_nlsq_heterodyne` routes here first per
`xpcsjax/optimization/nlsq/__init__.py:1130-1160`). It has **no expansion mechanism at all**:
`phys_names = model.param_manager.varying_names` (`:395`) is passed straight through to
`StratifiedResidualFunctionJIT` → `HeterodynePointEvaluator.eval_points`
(`model_adapter.py:163` imports and calls `compute_c2_heterodyne` — `heterodyne_jax_backend.py:164`,
a shim into `compute_c2_unified` → `_compute_c2_meshgrid`, `heterodyne_physics_kernel.py:410-414`),
which unpacks `params[0]`..`params[13]` positionally with no length check.

**Confirmed:** whenever ANY `two_component` physics parameter is fixed (via `active_parameters` —
as C045's current config already does), `physical_params` is shorter than 14. Empirically verified
against JAX's actual indexing semantics: `@jax.jit`-traced static out-of-bounds indexing (e.g.
`params[13]` on an 11-element traced array) does **not** raise `IndexError` — JAX silently clips
the index to the last valid element and returns a value. So `compute_c2_heterodyne` does not throw;
it silently reads clipped/wrong parameter values (e.g. `phi0` reads whatever the last present
element is) and produces a wrong-but-plausible fit. This is **never caught** by the existing
best-effort exception handler in `_fit_nlsq_heterodyne`
(`xpcsjax/optimization/nlsq/__init__.py:1130-1160`), because no exception occurs — the fallback to
`fit_nlsq_multi_phi` never triggers. This is silently-wrong physics, not silently slower: real and
pre-existing, independent of this feature. C045's fit almost certainly already produced a
wrong-but-plausible result via this path, not a fallback.

**Decision: still excluded from this design's implementation, but the "degrades gracefully"
justification above is false and must be re-examined before shipping.** Given the exclusion no
longer degrades gracefully, whether `heterodyne_engine_route.py` can safely stay out of scope (as
opposed to needing at least a length-guard fix as a prerequisite) is an open question for the
implementation plan — track separately, but do not assume the fallback protects users today.

## Decisions (from brainstorming)

1. **General mechanism, not a one-off hardcode.** `tied_parameters: {child: parent}` config dict,
   works for any physics-parameter pair in any `two_component` config — not baked into
   `heterodyne_core.py` as a special case for the ref/sample triplet. (Rejected: narrow hardcode —
   zero reuse for future tie needs.)
2. **All production joint-fit entry points, not just the in-memory path.** In-memory joint
   (constant/averaged/individual), hybrid-streaming, and stratified-LS (≥1M) must all honor
   `tied_parameters` correctly — not silently ignore it on some paths. (See "Non-goals" for the
   one explicitly excluded router.)
3. **Bundle the Tier-0 full-vector-result fix into this same design.** Tying needs the same
   "always emit the full 14-physics result" guarantee that fixes today's viz crash — building it
   once serves both. (Rejected: ship Tier 0 as a fully separate prior spec — real option, but
   tying's `expand_varying_to_full` fix and Tier 0's streaming-result-builder fix are close enough
   in mechanism that splitting them would mean touching `heterodyne_parameter_manager.py` twice
   for the same underlying guarantee.)
4. **Tied child's reported uncertainty mirrors the parent's exactly.** Since child and parent are
   the same free variable, `result.D0_ref.value == result.D0_sample.value` and
   `result.D0_ref.uncertainty == result.D0_sample.uncertainty` — not NaN, not independently
   computed. `nlsq_diagnostics["tied_parameters"]` records the child→parent map so consumers know
   the child's stats are derived, not independent.
5. **`heterodyne_engine_route.py` is out of scope for this design's implementation.** See
   "Background" above — pre-existing gap, does NOT degrade gracefully (JAX clips rather than
   raising, so the fallback never triggers and the route silently produces wrong-but-plausible
   physics), whether it can safely stay excluded is an open question tracked separately.

## Architecture

### Component 1 — config schema

New key `initial_parameters.tied_parameters: {child: parent}`, e.g.:

```yaml
initial_parameters:
  tied_parameters:
    D0_ref: D0_sample
    alpha_ref: alpha_sample
    D_offset_ref: D_offset_sample
```

Validated at config-load time (fail fast, `ValueError`, matching the strictness of existing
`initial_parameters` validation):
- both `child` and `parent` must be in the 14 physics names (`ALL_PARAM_NAMES`) — scaling
  (`contrast`/`offset`) is not tie-able in v1 (no known use case; YAGNI)
- no self-tie (`child == parent`)
- no chains — a `parent` must not itself appear as a `child` key in the same map (keeps
  resolution single-hop, no transitive-closure logic needed)
- `parent` must be varying (not itself fixed via `active_parameters`/grouped `vary: false`) —
  tying to a frozen constant is a different, already-existing feature
- if `child` also appears in an explicit `active_parameters` whitelist, the tie wins: log a
  warning and force `vary=False` for `child` regardless (a tied parameter can never
  independently vary — contradictory config, not a hard error, since the resolution is
  unambiguous)
- if `child`'s configured initial value or bounds differ from `parent`'s, log a warning (not a hard
  error — the tie is still unambiguous) and set `space.values[child] = space.values[parent]` for
  consistency of any code that reads `space.values` directly (e.g. `to_config()`). Without this,
  `get_initial_values()`/`get_bounds()` (`heterodyne_parameter_manager.py:156-206`, both restricted
  to `varying_indices`) silently drop the child's own configured value/bounds once `vary[child]` is
  forced `False`, with no warning if they diverged from the parent's.

### Component 2 — `ParameterSpace.tied: dict[str, str]`

New `_apply_tied_parameters()` in `xpcsjax/config/heterodyne_parameter_space.py`. Parses
`tied_parameters`, runs the Component 1 validations, sets `space.tied = {child: parent, ...}`, and
forces `space.vary[child] = False` for every child (reusing the existing fixed-parameter lever —
this is what makes bounds/p0/CMA-ES normalization shrink automatically per the "Background"
findings above, no separate array plumbing needed).

**Must run last, after every other config overlay — not as an early sibling of
`_apply_initial_parameters()`.** In `from_config` (`heterodyne_parameter_space.py:334-348`),
`_apply_initial_parameters` and `_apply_parameter_space_bounds` run first, then the grouped-format
block (lines 367-440) runs *after* and unconditionally overwrites `space.vary[param_name] =
new_vary` whenever a `vary` key is present in that group's config (line 440). If
`_apply_tied_parameters()` ran early, a grouped-format config block that sets
`vary: true`/`vary: false` for a tied child (even the registry default) would silently re-enable it
as an independently-varying parameter, or leave a tied-away parameter stuck, with no error.
`_apply_tied_parameters()` must therefore be the final step in `from_config`, applied after the
flat, bounds, and grouped overlays have all resolved `space.vary`, so its `vary[child] = False`
cannot be silently undone.

**`to_config()` must also serialize the tie.** `to_config()` (`heterodyne_parameter_space.py:135-141`)
currently only emits `parameter_names`, `values`, `active_parameters` — it never round-trips
`space.tied`. Any code path that reconstructs a config from a `ParameterSpace` (persistence,
provenance, GUI round-trip) would silently drop the tie. `to_config()` must also emit
`initial_parameters.tied_parameters` from `space.tied`.

### Component 3 — shared expansion helper (the correctness core)

A pure, JIT-safe function — conceptually `expand_theta(free_vector, space) -> full_14_array` —
built as a straightforward extension of the existing scatter idiom:

```python
full = fixed_values_jax.at[varying_indices_jax].set(physics_varying)   # existing idiom
for child_idx, parent_idx in tied_idx_pairs:                            # NEW, static pairs
    full = full.at[child_idx].set(full[parent_idx])
```

`tied_idx_pairs` (physics-array index pairs, e.g. `(0, 3)` for `D0_ref` ← `D0_sample`) are
computed once at config-load time from `space.tied` — static Python ints, safe to close over
inside a JIT-traced function exactly like `varying_indices_jax` already is.

Because the overwrite happens *inside* the traced residual closure (not post-hoc), every
evaluation during optimization sees `D0_ref == D0_sample` by construction. Whatever
Jacobian/gradient scheme the solver actually uses (NLSQ's internal AD or finite-difference; JAX
`grad` on the L2 hierarchical branch) will differentiate/perturb through the composed function
correctly, since it's a black-box function of the reduced free-vector either way — this doesn't
require JAX autodiff specifically, just that the tie is applied before the kernel call rather than
after the fit converges.

### Component 4 — wiring into the required residual closures

Extend the existing scatter idiom (add the `tied_idx_pairs` overwrite loop) at each confirmed real
touch point:
1. `heterodyne_core.py:1313` — averaged mode (also covers both joint escapes via delegation)
2. `heterodyne_core.py:3102` — individual mode (`_build_joint_problem`, also covers both escapes)
3. `heterodyne_constant_mode.py:222` — constant mode
4. `heterodyne_stratified_ls.py:362` — ≥1M stratified-LS
5. `heterodyne_hybrid_streaming.py:300` — hybrid streaming

6. `heterodyne_core.py:3992` — `_fit_cmaes` per-angle CMA-ES escape
7. `heterodyne_core.py:4343` — `_fit_local` per-angle residual builder
8. `heterodyne_core.py:4552` — per-angle residual builder (single-phi variant)

These three are confirmed real, reachable production entry points (per-angle CMA-ES escapes and
per-angle local fits, part of this codebase's per-angle-mode machinery) and a tied config can reach
them the same way it reaches the joint paths — they are wired into this touch-point list above, not
deferred to planning. Before writing the implementation plan, still grep
`.at[varying_indices_jax].set(` to confirm this list is exhaustive (no new occurrence introduced
since this design was written).

Config-gated: when `tied_parameters` is absent (default), `tied_idx_pairs` is empty and the added
loop is a no-op — byte-identical to current behavior. Existing `rtol=1e-10` golden/parity tests
for untied configs are unaffected.

### Component 5 — result reconstruction

Make `expand_varying_to_full()` (`heterodyne_parameter_manager.py:208-239`) tying-aware: after its
existing fill-from-`get_full_values()`-then-overwrite-varying logic, add the same
`full[child_idx] = full[parent_idx]` overwrite. This makes the *expansion helper itself* correct,
but — per the Background finding above — it does **not** automatically fix any reported result,
because none of the three existing call sites currently consume its return value for
`OptimizationResult.parameters`. Each site needs its own edit to use the (now tying-aware)
full-expansion output instead of the raw reduced vector:
- `heterodyne_core.py:1711` — change `parameters=np.asarray(fitted_all, ...)` to use the expanded
  full-14 array (the same one already computed at `:1487` for `model.set_params`, not a fresh call)
- `heterodyne_core.py:3759` — change `parameters=np.asarray(fitted_params_full, ...)` to use the
  expanded array from `:3512`
- `heterodyne_constant_mode.py:452` — change `parameters=fitted_physics` to use the expanded array
  from `:337`

**Covariance/uncertainty expansion is a separate, required piece — not automatic.**
`OptimizationResult.__post_init__` (`results.py:231-245`) hard-enforces
`uncertainties.size == parameters.size` and `covariance.shape == (parameters.size, parameters.size)`,
raising `ValueError` otherwise. The stored covariance from the solves (e.g.
`heterodyne_core.py:3589-3593`) is sized to the **reduced** vector, so expanding only `parameters`
to full `14 + 2·n_phi` length — without also expanding `uncertainties`/`covariance` to match — will
raise `ValueError` on every tied or fixed-parameter result at construction time. Add an explicit
covariance/uncertainty expansion helper alongside the parameter expansion:
- full-length ordering matching the expanded `parameters` array
- for tied children: mirror the parent's row/column into the child's row/column (per Decision 4) —
  `result.D0_ref.uncertainty == result.D0_sample.uncertainty`, not NaN, not independently computed
- for ordinary fixed (non-tied) physics parameters: an explicit NaN/placeholder row/column (no
  covariance was computed for a constant)

Add `nlsq_diagnostics["tied_parameters"]` recording the map used for this fit.

### Component 6 — Tier 0: streaming result builder

Fix `build_hybrid_streaming_result` (`heterodyne_result_builder.py:742`, the `parameters=popt`
assignment) to call the (now tying-aware) `expand_varying_to_full()` before constructing
`OptimizationResult.parameters`, instead of returning the raw reduced `popt` vector directly.

**`popt` cannot be passed to `expand_varying_to_full()` directly.** Per
`heterodyne_hybrid_streaming.py:297-300`, the layout is confirmed scaling-first
(`physics = params_all[_physics_block]`, tail) for `averaged`/`individual` modes — so `popt` is
`[scaling_head | physics_tail]`, not a physics-only vector. Passing it straight to
`expand_varying_to_full()` would fail that function's strict `len(varying_params) !=
len(self.varying_indices)` check. The fix must: split `popt` into its scaling head and physics tail
per the resolved per-angle mode → expand the physics tail via `expand_varying_to_full()` →
reassemble scaling head + expanded full-14 physics into the final `parameters` array.

This is required independent of tying (any fixed physics param already breaks this path the same
way), and is a prerequisite for the ≥1M stratified path to support tying correctly.

### Component 7 — viz

No viz code changes needed. Components 5 and 6 together guarantee the *physics block* of every
`two_component` result — tied or not, fixed or not — is always full-length 14, matching what
`nlsq_plots.py` (`:1651`, `:1657`) already expects per mode. Per `nlsq_plots.py:279-313`, each mode
continues to report its own existing overall layout: `constant` carries physics-only (`n_physical`,
no scaling params), `averaged` carries `n_physical + 2` (one scaling pair, not `2·n_phi`), and only
`individual` carries the full `n_physical + 2·n_phi` (physics + per-angle scaling). Components 5/6
do not change these per-mode shapes — they only guarantee the physics portion is never short.

## Data flow (one fit call, tied config)

1. Config load → `ParameterSpace` with `.tied = {"D0_ref": "D0_sample", ...}`,
   `.vary["D0_ref"] = False` (and the other tied children)
2. Optimizer builds its free-vector from `varying_names` (11 names for a 3-tie case instead of
   14) + scaling — automatically reduced, no separate code path
3. Every residual/model evaluation scatters the trial vector into the full 14-array, then
   overwrites tied children from their parents — `D0_ref == D0_sample` holds on every iteration
4. Whatever gradient/Jacobian machinery the solver uses differentiates through the composed
   function — the shared free variable correctly receives the combined sensitivity from both its
   ref-slot and sample-slot usages
5. Converged free-vector → `expand_varying_to_full()` (now tying-aware) → full result + mirrored
   covariance → `nlsq_diagnostics.tied_parameters` → JSON/HDF5/viz all see a normal-shaped,
   internally-consistent result

## Error handling

- Invalid tie config (chain, self-tie, parent not varying, unknown name) → `ValueError` at config
  load, fail fast — same strictness tier as other `initial_parameters` validation.
- `child` also present in `active_parameters` → warning logged, tie takes precedence
  (`vary=False` forced), not a hard error.
- `heterodyne_engine_route.py` (out of scope, Background section) → unchanged existing behavior:
  no exception is raised (JAX clips the out-of-bounds static index rather than raising), so the
  best-effort fallback to `fit_nlsq_multi_phi` never triggers — the route silently produces
  wrong-but-plausible physics for any fixed/tied physics parameter today. Not touched by this
  design; tracked separately as a prerequisite question, not a safe-to-ignore gap.

## Testing

- Config parsing: valid `tied_parameters` round-trips through `ParameterSpace.tied`; each
  rejection case (chain, self-tie, parent-not-varying, unknown name) raises `ValueError`.
- Expansion helper: given a free vector, output array has `full[child_idx] == full[parent_idx]`
  for every tied pair.
- Finite-difference check: numerically verify the SSR gradient w.r.t. the shared free variable
  equals the sum of the two partial derivatives (from the child-slot and parent-slot usages) —
  proves real coupling, not cosmetic mirroring.
- One synthetic `two_component` end-to-end fit per wired path (in-memory constant/averaged/
  individual, hybrid-streaming, stratified-LS ≥1M) with a tied triplet — assert converged
  `D0_ref == D0_sample` exactly (bit-identical, since it's a literal array copy), covariance
  mirrored, full-length `14 + 2·n_phi` result (viz-compatible).
- Existing untied `rtol=1e-10` golden tests re-run unchanged (`tied_parameters` absent by default)
  — must stay green, proves zero regression to the untied path.
- Streaming result builder fix (Component 6): synthetic ≥1M-point fit with a fixed (non-tied)
  physics parameter, assert `OptimizationResult.parameters` has full 14-physics length — this
  regression-tests the pre-existing viz-crash bug independent of tying.

## Non-goals / explicitly out of scope

- `heterodyne_engine_route.py` wiring (see Background) — separate follow-up.
- Tying scaling parameters (`contrast`/`offset`) — no known use case.
- Tying with a scale factor or affine relationship (`child = k * parent + b`) — YAGNI; only
  strict equality is needed today. A linear reduction-matrix generalization was considered and
  rejected as unnecessary complexity for the current requirement.
- Static/homodyne (`laminar_flow`) modes — confirmed not needed.

## Rollout / migration

C045's current `active_parameters`-based workaround config gets replaced with a clean
`tied_parameters` block once this ships. In the meantime, `xpcsjax_config.yaml.pre-tie-backup.yaml`
(already saved) is available as a known-good untied baseline.
