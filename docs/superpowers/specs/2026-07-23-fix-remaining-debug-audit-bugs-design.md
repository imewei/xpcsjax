# Fix remaining debug-audit findings — design

Date: 2026-07-23
Status: Implemented

## Context

Two whole-codebase debug-audit passes (PR #14) confirmed 25 bugs in a second
pass on top of an earlier 30-bug pass. 19 of the 25 were fixed and merged
immediately (low risk, no basin/behavior concerns). The remaining 6 were held
back because each needs a real design decision, not a mechanical patch. This
spec covers fixing all 6, per user request, after a Q&A round settling the
ambiguous design point on each.

## Findings and decisions

### 1. `xpcs_loader.py` — NaN-tolerant `wavevector_q_list`

**Problem:** `_validate_loaded_arrays` (I/O-boundary hard-fail validator)
rejects any non-finite value in `wavevector_q_list`, contradicting the same
function's own docstring three lines above ("Monotonicity is intentionally
NOT asserted on `wavevector_q_list`... legitimately non-monotonic") and older
nan-safe consumer code (`np.nanmean`/`np.nanstd` in cache metadata,
`np.nanmin`/`np.nanmax` in `_validate_physics_parameters`) that expects NaN
there from masked/bad detector pixels.

**Decision:** NaN in `wavevector_q_list` is legitimate (bad-pixel masking),
but `inf`/`-inf` is not — inf still indicates corrupt data and must keep
hard-failing. Additionally, NLSQ's scalar-extraction call sites
(`xpcsjax/optimization/nlsq/core.py:715`, `core.py:1802`, and
`xpcsjax/optimization/nlsq/adapter.py:875`) take `wavevector_q_list[0]`
unguarded and feed it straight into the JAX physics model as the fitted
`q` — a plain "exclude the key from the finite check" change would let a
bad-pixel NaN silently poison a fit whenever it lands at (or is the only
entry reaching) that extracted index. This is a real gap, not a documented
non-issue: the extraction sites need their own guard.

**Change:**
- In `_validate_loaded_arrays` (xpcsjax/data/xpcs_loader.py), replace the
  blanket `np.all(np.isfinite(...))` check on `wavevector_q_list` with one
  that still raises on `inf`/`-inf` but tolerates `NaN` (e.g. check
  `np.isinf(arr).any()` instead of `not np.all(np.isfinite(arr))` for this
  key only). Keep `c2_exp`, `t1`, `t2`, `phi_angles_list` on the existing
  all-finite check, unchanged. Add a short comment next to the check
  explaining why, matching the existing monotonicity carve-out comment
  style.
- Add a finite guard at each scalar-extraction call site that pulls
  `wavevector_q_list[0]` (`core.py:715`, `core.py:1802`, `adapter.py:875`):
  if the extracted value is NaN, raise (do not silently proceed) rather than
  handing a NaN `q` to the JAX model — a bad-pixel NaN elsewhere in the list
  must never be able to reach the fitted-q scalar unnoticed.

**No other code changes beyond the guarded extraction sites above** — the
aggregate nan-safe consumers (`nanmean`/`nanstd` in cache metadata,
`nanmin`/`nanmax` in `_validate_physics_parameters`) already handle NaN
correctly elsewhere in the array and need no change.

### 2. `xpcs_loader.py` — skip mandatory diagonal correction when preprocessing already ran it

**Problem:** When a user opts into `preprocessing.enabled=True` and
configures a `correct_diagonal` stage method (statistical/interpolation),
the loader unconditionally re-applies a mandatory `'basic'` diagonal
correction afterward (`apply_diagonal_correction_batch`, no `method=`
kwarg), silently overwriting the configured method's result.

**Decision:** Skip the mandatory correction when preprocessing already
corrected the diagonal.

**Change:**
- `_apply_preprocessing_pipeline` (xpcs_loader.py) sets a marker, e.g.
  `data["_diagonal_corrected"] = True`, when the `CORRECT_DIAGONAL` stage
  runs successfully (mirrors the existing `_preprocessing_degraded`
  convention for pipeline-to-loader state passing).
- `load_experimental_data`'s mandatory-correction call site checks
  `data.get("_diagonal_corrected", False)` before calling
  `apply_diagonal_correction_batch`; skips if already `True`.
- Default (preprocessing disabled, the documented default path) is
  unaffected — the marker is never set, so the mandatory correction still
  runs exactly as today.

### 3. `fallback_chain.py` — STREAMING soft-failure escalates to next strategy

**Problem:** `execute_optimization_with_fallback`'s STREAMING branch sets
`convergence_status = "partial"` when `info.get("success", False)` is
`False` (no exception raised), then falls through to the loop's
unconditional `break` — never reaching `get_fallback_strategy()`, the only
place the CHUNKED→LARGE→STANDARD degradation is implemented. This
contradicts the function's own docstring, which explicitly promises
degrading "until one succeeds or all are exhausted." The same silent-break
gap also exists on the `enable_recovery=True` branch: `execute_with_recovery`
(recovery.py, ~line 280) can return `convergence_status="failed"` directly
(a plain `return`, not a raise) after its own internal retry loop is
exhausted, and that result hits the identical unconditional `break` — so a
recovery soft-failure never reaches escalation either. `enable_recovery`
defaults to `True` in `config.py`, `wrapper.py`, and `heterodyne_config.py`,
so this is the default flow, not an edge case — the fix must cover both
branches with one shared rule, not just STREAMING.

**Decision:** Escalate to match the docstring (not the other option of
fixing the docstring to match current behavior), applying one shared success
predicate to both the STREAMING branch and the recovery branch.

**Change:** Define a single success predicate — e.g. `success is True and
convergence_status not in {"failed"}` — and apply it uniformly at every
point where a strategy attempt currently falls straight through to the
loop's unconditional `break` without escalating: the STREAMING soft-failure
case above, and `enable_recovery`'s `execute_with_recovery` returning
`"failed"`. Where the predicate is false, route through the same escalation
path the `except` block already uses — either raise a `RuntimeError` so it's
caught by the existing `except (ValueError, RuntimeError, ...)` block below
(simplest, reuses existing machinery), or directly call
`get_fallback_strategy(current_strategy)` and `continue` the loop. Prefer
whichever is the smaller diff once the code is in front of you — check
whether the `except` block does anything STREAMING-specific (or
recovery-specific) that would need duplicating if bypassed via a raise.

**Guardrail — corrected (the original text below was factually wrong about
current behavior and must not be relied on):** `execute_optimization_with_fallback`
does NOT return a fallback/partial result once all strategies are exhausted.
Verified directly against the code (fallback_chain.py, the
`except (ValueError, RuntimeError, TypeError, AttributeError, OSError,
MemoryError)` block, roughly lines 425-458): when
`get_fallback_strategy(current_strategy)` returns `None`, the function
always `raise`s — either re-raising the original `RuntimeError` (when it
already carries "Recovery actions"/"Suggestions" text) or wrapping it in a
new `RuntimeError(f"Optimization failed with all strategies: ...")`. There
is no `return` anywhere in that branch; this matches the function's own
docstring ("Raises: RuntimeError — If every strategy in the fallback chain
fails."). Consequence: once STREAMING's and recovery's soft failures are
routed into this same escalation path, a run where CHUNKED/LARGE/STANDARD
also fail will now raise `RuntimeError` and discard the STREAMING/recovery
partial result, instead of the previous (STREAMING-only-reachable-today)
"partial" return — a real, user-visible behavior change (crash instead of a
degraded-but-labeled result). This spec accepts that change as the intended
consequence of honoring the docstring's "until one succeeds or all are
exhausted" contract; do not assume, as an earlier draft of this guardrail
incorrectly did, that a graceful non-crashing fallback already exists for
the fully-exhausted case.

### 4. L2 hierarchical sigma-weighting — both `laminar_flow` and `two_component`

**Problem:** In both `xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py`
(laminar_flow) and `.../heterodyne_hybrid_streaming.py` (two_component), the
L2 hierarchical branch's loss closures (`_hier_loss`/`loss_fn`,
`_loss_jax`/`grad_fn`) compute an unweighted `jnp.mean(residuals**2)`, never
referencing `sigma` — even though the sibling plain-path branch in the same
function correctly threads `sigma` into `optimizer.fit(sigma=sigma, ...)`.
This is a shared, pre-existing gap across both modes (heterodyne's plain
path is actually more complete than laminar's), not a heterodyne-specific
regression.

**Decision:** Fix both modes for parity, not just the audited one.

**Change (per file — NOT the same shape / independent edits as originally
stated; laminar needs a prerequisite plumbing step heterodyne already has):**
- **heterodyne** (`heterodyne_hybrid_streaming.py`): `sigma` is already
  constructed and in scope — `build_heterodyne_pointwise_model` builds an
  aligned `meta_sigma` (masked 1:1 with `x_data`/`y_data`), and
  `fit_with_stratified_hybrid_streaming_heterodyne` derives a local `sigma`
  from it that's already visible to both `_hier_loss`/`_loss_jax` and the
  plain-path `optimizer.fit(sigma=sigma, ...)` call. Only the divide-by-sigma
  edit below is needed here.
- **laminar_flow** (`strategies/hybrid_streaming.py`): the hierarchical
  closures to fix are `fit_with_stratified_hybrid_streaming`'s `loss_fn` and
  `grad_fn` — NOT `fit_with_hybrid_streaming_optimizer`'s unrelated
  non-hierarchical `sigma=None` plain path. Neither `loss_fn` nor `grad_fn`
  has any `sigma` name in scope today: `stratified_data.sigma =
  original_data.sigma` is populated upstream but never threaded into this
  function, and this function's own plain-path sibling branch doesn't pass
  `sigma=` either. **Prerequisite step (required before the closure edit can
  do anything):** extract `stratified_data.sigma`, align it to
  `x_data`/`y_data`'s flattened/masked point order (mirroring what
  `build_heterodyne_pointwise_model` already does for heterodyne), and pass
  it into `loss_fn`/`grad_fn`'s closure scope. Implementing only the
  divide-by-sigma bullet below without this step is a no-op for laminar (in
  fact a `NameError`, since `sigma` would not exist in the closure at all).
- In each hierarchical loss closure, once `sigma` is genuinely in scope and
  not `None`, divide `residuals` by `sigma` (guarded against near-zero,
  mirroring the `safe_sigma`/`valid_sigma` convention already used in
  `strategies/residual_jit.py`) before squaring/summing.
- When `sigma is None`, behavior is unchanged (unweighted, as today) — this
  only activates the weighting path when real per-point weights exist.

**Risk:** This changes the actual optimizer loss surface on the L2
hierarchical path for anyone currently fitting with non-uniform `sigma` in
`individual`-mode streaming — a real numeric change, not diagnostic-only.
No `rtol` golden gate currently covers this combination (per the original
audit finding: "no test... exercises non-uniform sigma together with the
hierarchical path"). Needs careful review at implementation time; see
Testing section.

### 5. `cmaes_wrapper.py` — wire `cmaes_seed` end-to-end

**Problem:** `CMAESWrapperConfig.from_nlsq_config` (cmaes_wrapper.py — the
classmethod that builds a `CMAESWrapperConfig` from an incoming
`NLSQConfig`) reads `getattr(config, "cmaes_seed", None)`, but `NLSQConfig`
never declares, parses, or serializes a `cmaes_seed` field — the getattr
always silently falls back to `None`, so a user-supplied `cmaes.seed` in
YAML has no effect. (Correction: `to_cmaes_config` is a *different*,
unrelated instance method that only converts an already-built
`CMAESWrapperConfig` into NLSQ's own `CMAESConfig` type via direct `self.*`
access — it takes no `NLSQConfig` argument and is not the getattr call site;
an earlier draft of this finding misattributed the getattr call to it.)
`MultiStartConfig` has the equivalent `multi_start_seed` fully wired
(dataclass field + `from_dict` parsing + `to_dict` serialization) — proving
per-strategy seed fields are normally plumbed end-to-end in this codebase;
this one was simply never finished.

**Decision:** Wire it, default `None` (not `multi_start_seed`'s default of
`42`) — since the field never existed before, no one has ever relied on an
implicit fixed seed here, and `None` preserves current (nondeterministic
unless explicitly set) effective behavior for every existing config.

**Change:**
- Add `cmaes_seed: int | None = None` to `NLSQConfig` (config.py).
- Parse in `from_dict()`: `cmaes_seed=cmaes.get("seed")` (mirrors how
  `multi_start_seed` is parsed from its own block).
- Emit in `to_dict()`'s `cmaes` block for YAML round-tripping.
- In `CMAESWrapperConfig.from_nlsq_config`, KEEP reading the field via
  `getattr(config, "cmaes_seed", None)` — do not switch to a bare
  `config.cmaes_seed` attribute access. Every one of this method's 20+ field
  reads (`preset`, `max_generations`, `popsize`, `sigma`, `tol_fun`,
  `tol_x`, `restart_strategy`, ... ) uses the same
  `getattr(config, "cmaes_...", default)` pattern, per the method's own
  inline comment ("`NLSQConfig` might not have all fields if it's an older
  version or partial ... Use getattr with defaults where appropriate.");
  making `cmaes_seed` the one bare-access field would both break that
  uniform convention and raise `AttributeError` for any `NLSQConfig`-like
  object that lacks the field — exactly the failure mode getattr-with-default
  exists to prevent. (`to_cmaes_config`'s direct `self.*` reads are a
  separate, legitimately-different case: it operates on an already-built
  `CMAESWrapperConfig`, not a raw `NLSQConfig`, so there is nothing to
  change there either.) The actual fix is simply that `cmaes_seed` now
  exists as a real field on `NLSQConfig` (previous bullets) — the existing
  `getattr` call in `from_nlsq_config` then resolves to the real value
  instead of always falling back to `None`.

### 6. `quality_controller.py` — gate negative-correlation repair on normalization state

**Problem:** `_repair_negative_correlations` clamps every negative
`c2_exp` value to `1e-6` whenever aggressive auto-repair is enabled and
*any* issue triggers the repair pass (not specifically a negative-correlation
issue). If upstream preprocessing already ran `NormalizationMethod.STATISTICAL`
(z-score) or `ROBUST` (IQR-scaled) normalization — both of which legitimately
produce negative values by design — this repair silently collapses the
entire negative half of the distribution to a near-zero constant.

**Decision:** Track normalization state explicitly in the data dict (more
precise than gating on pipeline stage alone) — and at **per-matrix**, not
dataset-level, granularity. STATISTICAL/ROBUST normalization
(`xpcsjax/data/preprocessing.py`'s `_normalize_data` method) loops
per-matrix-index and explicitly *skips* the transform (logging "Zero
standard deviation ... skipping normalization" / "No variance in percentile
range ... skipping normalization") for any matrix with near-zero
variance/IQR, without failing the overall stage —
`PreprocessingPipeline.process`'s `stage_results[stage] = True` only means
"the stage didn't raise," not "every matrix was actually normalized." A
single dataset-level `data["_normalized"] = True` flag would therefore
suppress repair on skipped matrices too, even though those still hold raw,
never-transformed values and are not the "legitimate by design" negatives
the fix is meant to protect.

**Change:**
- In `xpcsjax/data/preprocessing.py`'s `_normalize_data` method (where the
  `STATISTICAL`/`ROBUST` branches and their per-matrix skip logic already
  live), track normalization applied-vs-skipped **per matrix index**, not
  just overall stage success — e.g. `data["_normalized_mask"]`, a per-frame
  boolean array/list aligned to `c2_exp`'s leading axis, set `True` only for
  matrices actually transformed and `False` for matrices that hit the
  zero-variance/zero-IQR skip branch (mirrors the existing
  `_preprocessing_degraded` state-passing convention between preprocessing
  and quality-control stages, but at finer granularity).
- `_repair_negative_correlations` (quality_controller.py) checks the
  per-matrix marker and skips the clamp only for matrices where
  `_normalized_mask[i]` is `True`, still clamping negative values on
  matrices where it's `False` (skipped/never transformed) — logging which
  matrices were skipped due to normalized data vs. still repaired.
- Unnormalized data (the default path, no preprocessing or a non-normalizing
  preprocessing config) is unaffected — repair behavior unchanged.

## Testing

Each fix gets a regression test at a real seam (calling the actual function,
not restating its logic in a test-local closure — the standard this session
converged on after PR #14's review caught exactly that anti-pattern once):

1. NaN-q: construct a data dict with `wavevector_q_list` containing NaN,
   assert `_validate_loaded_arrays` does not raise (and still raises for NaN
   in `c2_exp`, proving the exemption is scoped correctly). Extend this test
   to also cover: (a) `_validate_loaded_arrays` still raises for `inf`/`-inf`
   in `wavevector_q_list` — the exemption is NaN-only, not blanket; (b) an
   all-NaN `wavevector_q_list` is handled explicitly rather than silently
   sailing into `nanmin`/`nanmax` uncontested; and (c) the scalar q value
   actually extracted and used by the NLSQ fit (the guarded extraction sites
   added in Finding 1) is verified finite even when the raw list contains
   NaN elsewhere, including at index 0.
2. Diagonal correction: run a full preprocessing-enabled load with a
   non-default `correct_diagonal.method`, assert the final `c2_exp`
   diagonal reflects that method's output, not `'basic'`'s.
3. STREAMING escalation: force `fit_with_hybrid_streaming_fn` to return
   `success=False` (monkeypatch/stub), assert the fallback loop actually
   invokes the CHUNKED path next (not an immediate terminal "partial").
   Also cover the recovery-branch half of this same fix (the default
   `enable_recovery=True` flow, not an edge case): stub
   `execute_with_recovery_fn` to return `convergence_status="failed"`
   (its real return value on retry exhaustion, per `recovery.py:280` — a
   `return`, not a raised exception) and assert this also escalates to the
   next fallback strategy under the shared success predicate, not just the
   STREAMING path.
4. Sigma-weighting (both files): construct a small synthetic per-angle
   dataset with non-uniform `sigma`, assert the L2 hierarchical loss value
   changes when `sigma` is non-uniform vs uniform (proving it's now
   consulted), for both `hybrid_streaming.py` and
   `heterodyne_hybrid_streaming.py` independently.
5. `cmaes_seed`: assert a config with `cmaes.seed: 123` produces a
   `CMAESWrapperConfig.seed == 123` (not `None`), via `NLSQConfig.from_dict`
   → `CMAESWrapperConfig.from_nlsq_config` — this is the function that
   actually reads `cmaes_seed` off the `NLSQConfig` and sets `.seed`, not
   `to_cmaes_config` (that method only converts an already-built
   `CMAESWrapperConfig` into NLSQ's own `CMAESConfig`; include it as an
   additional step only if also verifying the seed reaches NLSQ's own
   `CMAESConfig` kwargs, not as the primary assertion target).
6. Negative-correlation repair: run the actual preprocessing stage
   end-to-end (not a hand-built `data["_normalized"] = True` dict) covering
   two cases: (a) `NormalizationMethod.STATISTICAL`/`ROBUST` genuinely
   normalizes a matrix, the per-matrix marker correctly propagates through
   the pipeline into `DataQualityController._repair_negative_correlations`,
   and that matrix's negative `c2_exp` values are NOT clamped; (b) a matrix
   hits the per-frame zero-variance/zero-IQR skip branch inside
   `_normalize_data` (never actually transformed) and its negative values
   ARE still clamped despite the dataset having run the normalizing method.
   Also keep the existing regression check that a fully-unmarked dataset is
   still clamped.

After all 6: run `make verify` (ruff + advisory mypy + parallel smoke), plus
the relevant domain-scoped targets (`make test-optimization`,
`make test-heterodyne`) given #3 and #4 touch actual solve paths with no
`rtol` golden gate. If a real (non-synthetic) XPCS dataset is available,
sanity-check #3 and #4's fitted results against it before merging, per the
project's own convention of preferring real-data verification for anything
touching solve-path behavior.

## Risk summary

| # | Fix | Risk | Basin-risk? |
|---|-----|------|-------------|
| 1 | NaN-q exemption + inf-still-rejected + extraction-site guard | behavior — without the extraction-site guard, a bad-pixel NaN could silently reach the JAX model as the fitted `q`; the guard closes that gap, but this is a real correctness-relevant change, not a risk-free cleanup | no |
| 2 | Skip double diagonal-correction | behavior — changes the final fitted `c2_exp` for every run with `preprocessing.enabled=True` where the (default-enabled, default-method `'statistical'`) `CORRECT_DIAGONAL` stage runs; this is the common case for anyone using preprocessing, not a narrow secondary opt-in, and needs a regression/parity check, not just a no-risk label | no |
| 3 | STREAMING + recovery escalation | behavior — per the corrected Guardrail above, a fully-exhausted fallback chain now raises `RuntimeError` instead of returning a "partial" result; also now covers the `enable_recovery` path (default `True`), not just STREAMING | possibly (routes to a different strategy) |
| 4 | Sigma-weighting (both modes) | behavior — laminar requires new sigma-propagation plumbing before the closure edit does anything, not just a symmetric closure edit | yes (changes L2 loss surface) |
| 5 | `cmaes_seed` wiring | none (opt-in field, default `None`, no effect on any existing config unless explicitly set) | no |
| 6 | Negative-correlation repair gating | behavior — must be tracked per-matrix, not dataset-level, or a coarse marker risks suppressing repair on matrices that were never actually normalized | no (only affects already-normalized matrices) |

\#1, #3, and #4 need the most care during implementation and review: #1
because the extraction-site guard is a new correctness-critical code path
(not just a validator relaxation), #3 because the corrected guardrail means
a previously-impossible crash path becomes reachable, and #4 because laminar
needs new plumbing before its fix has any effect. #2 and #6 are real data/
behavior changes for their affected populations (see risk column) but are
scoped and testable. #5 remains genuinely risk-free — a new field that
defaults to `None` and has no effect on any config that doesn't set it.
