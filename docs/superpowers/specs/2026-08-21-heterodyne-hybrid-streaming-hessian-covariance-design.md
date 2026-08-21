# Design: Real Hessian Covariance for Heterodyne Hybrid-Streaming L2 Branch

**Date:** 2026-08-21
**Status:** Approved for implementation planning
**Scope:** `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py`, L2/hierarchical branch only

## Problem

`fit_with_stratified_hybrid_streaming_heterodyne`'s L2/hierarchical branch (triggered
when the heterodyne `two_component` fit is large enough to hit the hybrid-streaming
memory tier AND `per_angle_mode` resolves to `individual`/`fourier`) never computes a
real parameter covariance. It unconditionally sets:

```python
pcov = np.eye(n)  # Hessian covariance is optional; identity placeholder
info["covariance_is_placeholder"] = True
```

This was already found and partially remediated in a prior fix
(`heterodyne_result_builder.py:558-568`, commit `391dd21`): the uncertainty
computation now honors `covariance_is_placeholder` and reports NaN instead of the
previously fabricated `±1.0` on every parameter. That fix is a safety net — it stops
a wrong number from being reported, but it doesn't recover the *real* uncertainty.
This spec addresses the root gap: give this branch a genuine covariance the way its
laminar (homodyne) sibling already has.

## What `covariance_is_placeholder` means and who reads it

The flag means "this `pcov` is a fabricated `eye(n)`, not a curvature measurement —
do not treat its derived numbers as real." Two call sites already branch on it:

1. `heterodyne_hybrid_streaming.py:994-1002` (L4 gradient-monitor post-solve
   condition-number fallback) — reports NaN instead of `cond(eye)=1.0`.
2. `heterodyne_result_builder.py:566` (uncertainty computation, fixed in commit
   `391dd21`) — reports NaN instead of `sqrt(diag(eye))=1.0`.

Both already do the right thing *given* the flag. Flipping the flag to `False` on a
successful real-Hessian computation routes both consumers onto the real numbers with
**no changes required at either site**.

Out of scope for this spec: `heterodyne_stratified_ls.py`'s L2-accepted branch
(≥1M-point path) sets an equivalent `_cov_placeholder=True`, but that file already
has a working real-covariance mechanism (`_chunked_jacfwd_dense` → JTJ, used by its
sibling L3-only branch) that the L2 branch simply isn't wired into
(`_invalidate_adapter_cov` is never set to `True` on L2 accept). That is a small,
independent wiring bug — not a Hessian-derivation problem — and is tracked as a
separate follow-up fix, not part of this spec.

## Why the laminar recipe transfers directly

Laminar's (homodyne) equivalent L2 branch, `strategies/hybrid_streaming.py:1590-1623`
(the BUG-15/H-5 fix), faced the identical constraint: no residual-vector Jacobian
machinery is available in that branch, only a scalar loss. It solves this with
`jax.hessian` on the scalar loss function passed to the hierarchical optimizer:

```python
n_hier_data = len(y_data)
n_hier_params = len(hier_result.x)
s2_hier = hier_result.fun / max(n_hier_data - n_hier_params, 1)
try:
    popt_jax = jnp.asarray(hier_result.x)
    H = np.asarray(jax.hessian(loss_fn)(popt_jax))
except Exception as e:
    logger.warning(f"Could not compute Hessian: {e}. Using identity placeholder.")
    H = None

covariance_is_placeholder = False
if H is not None:
    try:
        pcov_hier = 2.0 * s2_hier * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        pcov_hier = 2.0 * s2_hier * np.linalg.pinv(H)
else:
    pcov_hier = np.eye(n_hier_params)
    covariance_is_placeholder = True
```

The heterodyne L2 branch is in exactly the same position: no `residual_fn`, only a
scalar loss. It already has the necessary pure-JAX scalar loss function defined —
`_loss_jax(ph)` at `heterodyne_hybrid_streaming.py:847` — built for the existing
`_hier_grad` autodiff call. This is the same function, with the same math (sigma
weighting + optional L3 regularization), just needing one more autodiff transform
(`jax.hessian` instead of `jax.value_and_grad`) applied to it.

**Approaches considered:**

- **A (recommended): port the laminar recipe verbatim, reusing `_loss_jax`.**
  Matches an established, tested pattern exactly. No new abstractions.
- **B: build a residual-vector version of the streamed loss and use a JTJ-based
  covariance** (like `heterodyne_stratified_ls.py`'s mechanism). Rejected — this
  branch's loss is fundamentally scalar (weighted MSE reduction over the streamed
  chunk), not a residual vector; constructing one would require restructuring the
  streaming evaluation path for no benefit over A, since laminar already proved A
  works for the structurally identical case.
- **C: always report NaN, never attempt a real covariance.** Rejected — regresses
  behavior versus laminar's own branch, which successfully computes a real
  covariance in the common case; NaN should be the fallback, not the default.

## Design

### Component

`fit_with_stratified_hybrid_streaming_heterodyne` in
`xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py`, replacing the
current lines ~886-905 (the `pcov = np.eye(n)` block inside the
`hierarchical_optimizer is not None` branch).

### Mechanism

After `hierarchical_optimizer.fit(...)` returns `hier_result`:

1. `n = len(popt)` (already computed).
2. `s2 = hier_result.fun / max(n_data - n, 1)`, where `n_data = len(y_data)`
   (mirrors laminar's `s2_hier` exactly — `hier_result.fun` is the final scalar loss
   value, which laminar treats as SSR-equivalent for the Gauss-Newton scaling; this
   is consistent because `_loss_jax` returns the same weighted-MSE-times-N quantity
   laminar's `loss_fn` returns).
3. `H = np.asarray(jax.hessian(_loss_jax)(jnp.asarray(popt)))`, wrapped in
   `try/except Exception` — any failure (tracing error, shape mismatch, OOM) is
   caught and logged via `logger.warning`, `H = None`.
4. If `H is not None`: `pcov = 2.0 * s2 * np.linalg.inv(H)`, falling back to
   `np.linalg.pinv(H)` on `np.linalg.LinAlgError`, with a `logger.warning` on that
   fallback path (matches laminar's singular-Hessian handling).
5. If `H is None`: `pcov = np.eye(n)`, `covariance_is_placeholder = True`,
   `logger.error(...)` stating uncertainties are not meaningful (matches laminar's
   `H-5` log message).
6. `info["covariance_is_placeholder"]` is set from the computed
   `covariance_is_placeholder` value (replacing the current unconditional `True`).

### Regularization interaction

`_loss_jax` already includes the L3 adaptive-regularization term when
`adaptive_regularizer is not None` (line 852-854). The Hessian is therefore computed
on the *regularized* loss, exactly matching what `_hier_grad` already differentiates
for the optimizer's own gradient steps — internally consistent, no special-casing
needed for the L3-active case.

### Error handling / failure modes

- `jax.hessian` raising for any reason (tracer errors, unexpected shapes,
  unsupported ops introduced by future changes to `model_fn`) → caught, logged,
  falls back to placeholder. Fit result (`popt`, `chi_squared`) is never affected —
  only `pcov`/`uncertainties`/`covariance_is_placeholder`.
- Singular `H` (e.g. a parameter direction the fit is genuinely insensitive to) →
  `pinv` fallback, consistent with laminar and with the existing `elif`/`else`
  Jacobian-based paths elsewhere in the codebase (`heterodyne_result_builder.py`'s
  sibling files, `heterodyne_stratified_ls.py`).
- No change to `hierarchical_active`, `regularization_active`, or any other
  diagnostics key — this is purely a `pcov`/`covariance_is_placeholder` fix.

### Downstream propagation (no code changes required)

- `heterodyne_result_builder.py:566` — already branches on
  `info["covariance_is_placeholder"]`; `False` on success routes through the normal
  `sqrt(clip(diag(pcov)))` path with the real Hessian-derived values.
- `heterodyne_hybrid_streaming.py:994-1002` (L4 fallback) — already branches on the
  same flag; `False` on success computes the real `cond(pcov)` instead of skipping
  to NaN.
- `nlsq_diagnostics["covariance_is_placeholder"]` on the public `OptimizationResult`
  — automatically reflects the real per-fit outcome.

## Testing

All new tests live in `tests/optimization/test_heterodyne_hybrid_streaming.py`
(existing file, existing L2/hierarchical-branch test fixtures already present).

1. **Real covariance on a normal successful fit** — run the L2 branch (existing
   `individual`-mode fixture pattern in this file) through a small, well-posed
   synthetic problem; assert `pcov` is finite, `pcov.shape == (n, n)`, `pcov` is
   NOT `np.eye(n)` (e.g. `not np.allclose(pcov, np.eye(n))`), and
   `info["covariance_is_placeholder"] is False`.
2. **Fallback preserved on Hessian failure** — monkeypatch `jax.hessian` (or the
   module-level reference used in `heterodyne_hybrid_streaming.py`) to raise;
   assert `pcov == np.eye(n)`, `info["covariance_is_placeholder"] is True`, and a
   `logger.error` call fires (matches today's exact behavior — this is a
   regression test for the fallback path, not new behavior).
3. **Singular Hessian → pinv fallback** — construct or mock a degenerate `H`
   (e.g. rank-deficient) and assert `pcov` is computed via `pinv` without raising,
   `covariance_is_placeholder is False` (a pinv-derived covariance is still real,
   just ill-conditioned — distinct from the "Hessian call itself failed" case).
4. **Full regression pass** — rerun `test_heterodyne_hybrid_streaming.py` (59
   existing tests) and `test_heterodyne_tied_result_assembly.py` in full; both
   should remain green (existing assertions are shape-only on `pcov`, not
   value-pinned to identity).
5. **No parity impact** — `tests/parity/` golden/`rtol=1e-10` tests pin `popt` and
   `chi_squared`, not `pcov`; no golden regeneration expected or required.

## Non-goals

- `heterodyne_stratified_ls.py`'s L2-accepted-branch wiring fix (separate, small,
  tracked independently — see "Out of scope" above).
- Any change to the `hierarchical_optimizer.fit(...)` call itself, its convergence
  criteria, or the `popt` it returns.
- Any change to the plain (non-hierarchical) hybrid-streaming branch, which already
  receives a real `pcov` from `AdaptiveHybridStreamingOptimizer.fit(...)`.
- Performance/size gating on `jax.hessian` — per project decision, this mirrors
  laminar's unconditional-attempt behavior with no new threshold.
