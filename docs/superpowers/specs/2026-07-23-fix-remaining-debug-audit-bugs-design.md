# Fix remaining debug-audit findings — design

Date: 2026-07-23
Status: Approved

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

**Decision:** NaN in `wavevector_q_list` is legitimate (bad-pixel masking).
Exclude it from the hard finite-value check.

**Change:** In `_validate_loaded_arrays` (xpcsjax/data/xpcs_loader.py),
remove `wavevector_q_list` from the set of keys checked for finiteness (keep
`c2_exp`, `t1`, `t2`, `phi_angles_list` checked as today). Add a short
comment next to the check explaining why, matching the existing
monotonicity carve-out comment style.

**No other code changes** — the nan-safe consumers already handle this
correctly; only the newer hard-fail gate was the outlier.

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
degrading "until one succeeds or all are exhausted."

**Decision:** Escalate to match the docstring (not the other option of
fixing the docstring to match current behavior).

**Change:** In the STREAMING branch, when `info.get("success", False)` is
`False`, route through the same escalation path the `except` block already
uses — either raise a `RuntimeError` so it's caught by the existing
`except (ValueError, RuntimeError, ...)` block below (simplest, reuses
existing machinery), or directly call `get_fallback_strategy(current_strategy)`
and `continue` the loop. Prefer whichever is the smaller diff once the code
is in front of you — check whether the `except` block does anything
STREAMING-specific that would need duplicating if bypassed via a raise.

**Guardrail:** If all strategies are exhausted (CHUNKED, LARGE, STANDARD all
also fail or aren't applicable), the function's existing final-fallback
behavior is unchanged — still returns the last honestly-labeled result
rather than crashing.

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

**Change (per file, same shape, independent edits):**
- In each hierarchical loss closure, when `sigma is not None`, divide
  `residuals` by `sigma` (guarded against near-zero, mirroring the
  `safe_sigma`/`valid_sigma` convention already used in
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

**Problem:** `to_cmaes_config` reads `getattr(config, "cmaes_seed", None)`,
but `NLSQConfig` never declares, parses, or serializes a `cmaes_seed`
field — the getattr always silently falls back to `None`, so a user-supplied
`cmaes.seed` in YAML has no effect. `MultiStartConfig` has the equivalent
`multi_start_seed` fully wired (dataclass field + `from_dict` parsing +
`to_dict` serialization) — proving per-strategy seed fields are normally
plumbed end-to-end in this codebase; this one was simply never finished.

**Decision:** Wire it, default `None` (not `multi_start_seed`'s default of
`42`) — since the field never existed before, no one has ever relied on an
implicit fixed seed here, and `None` preserves current (nondeterministic
unless explicitly set) effective behavior for every existing config.

**Change:**
- Add `cmaes_seed: int | None = None` to `NLSQConfig` (config.py).
- Parse in `from_dict()`: `cmaes_seed=cmaes.get("seed")` (mirrors how
  `multi_start_seed` is parsed from its own block).
- Emit in `to_dict()`'s `cmaes` block for YAML round-tripping.
- Change `cmaes_wrapper.py`'s `to_cmaes_config` to reference
  `config.cmaes_seed` directly instead of `getattr(..., None)` — surfaces a
  typo'd/missing field as an `AttributeError` at the real call site instead
  of silently masking it (matches how the rest of `to_cmaes_config` already
  reads other fields directly, not via `getattr`).

### 6. `quality_controller.py` — gate negative-correlation repair on normalization state

**Problem:** `_repair_negative_correlations` clamps every negative
`c2_exp` value to `1e-6` whenever aggressive auto-repair is enabled and
*any* issue triggers the repair pass (not specifically a negative-correlation
issue). If upstream preprocessing already ran `NormalizationMethod.STATISTICAL`
(z-score) or `ROBUST` (IQR-scaled) normalization — both of which legitimately
produce negative values by design — this repair silently collapses the
entire negative half of the distribution to a near-zero constant.

**Decision:** Track normalization state explicitly in the data dict (more
precise than gating on pipeline stage alone).

**Change:**
- When `NormalizationMethod.STATISTICAL` or `ROBUST` runs in the
  preprocessing pipeline, set `data["_normalized"] = True` (new key,
  mirrors the existing `_preprocessing_degraded` state-passing convention
  between preprocessing and quality-control stages).
- `_repair_negative_correlations` checks `data.get("_normalized", False)`
  and skips the clamp when `True`, logging that the repair was skipped due
  to normalized data rather than silently doing nothing.
- Unnormalized data (the default path, no preprocessing or a non-normalizing
  preprocessing config) is unaffected — repair behavior unchanged.

## Testing

Each fix gets a regression test at a real seam (calling the actual function,
not restating its logic in a test-local closure — the standard this session
converged on after PR #14's review caught exactly that anti-pattern once):

1. NaN-q: construct a data dict with `wavevector_q_list` containing NaN,
   assert `_validate_loaded_arrays` does not raise (and still raises for NaN
   in `c2_exp`, proving the exemption is scoped correctly).
2. Diagonal correction: run a full preprocessing-enabled load with a
   non-default `correct_diagonal.method`, assert the final `c2_exp`
   diagonal reflects that method's output, not `'basic'`'s.
3. STREAMING escalation: force `fit_with_hybrid_streaming_fn` to return
   `success=False` (monkeypatch/stub), assert the fallback loop actually
   invokes the CHUNKED path next (not an immediate terminal "partial").
4. Sigma-weighting (both files): construct a small synthetic per-angle
   dataset with non-uniform `sigma`, assert the L2 hierarchical loss value
   changes when `sigma` is non-uniform vs uniform (proving it's now
   consulted), for both `hybrid_streaming.py` and
   `heterodyne_hybrid_streaming.py` independently.
5. `cmaes_seed`: assert a config with `cmaes.seed: 123` produces a
   `CMAESWrapperConfig.seed == 123` (not `None`), via `NLSQConfig.from_dict`
   → `to_cmaes_config`, end to end.
6. Negative-correlation repair: assert a dataset with
   `data["_normalized"] = True` and negative `c2_exp` values is NOT clamped
   by `_repair_negative_correlations`, while an unmarked dataset still is
   (regression-guards the existing behavior too).

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
| 1 | NaN-q exemption | none | no |
| 2 | Skip double diagonal-correction | none (opt-in path only) | no |
| 3 | STREAMING escalation | behavior | possibly (routes to a different strategy) |
| 4 | Sigma-weighting (both modes) | behavior | yes (changes L2 loss surface) |
| 5 | `cmaes_seed` wiring | none (opt-in field) | no |
| 6 | Negative-correlation repair gating | behavior | no (only affects already-normalized data) |

\#3 and #4 are the two that need the most care during implementation and
review — everything else is either purely additive (new opt-in fields/flags)
or fixes a validator that was over-eager on genuinely inert paths.
