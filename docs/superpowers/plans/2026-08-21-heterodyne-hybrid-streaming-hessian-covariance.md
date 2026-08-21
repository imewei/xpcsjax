# Heterodyne Hybrid-Streaming L2 Hessian Covariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional `pcov = np.eye(n)` placeholder in the heterodyne hybrid-streaming L2/hierarchical branch with a real Hessian-based Gauss-Newton covariance, ported from laminar's proven recipe.

**Architecture:** In `fit_with_stratified_hybrid_streaming_heterodyne`'s L2 branch, after `hierarchical_optimizer.fit(...)` converges, compute `H = jax.hessian(_loss_jax)(popt)` (the pure-JAX scalar loss already defined in this function for autodiff), scale by Gauss-Newton `pcov = 2*s²*inv(H)`, and fall back to `eye(n)` + `covariance_is_placeholder=True` only if the Hessian computation itself fails or is singular beyond `pinv` recovery.

**Tech Stack:** JAX (`jax.hessian`), NumPy (`linalg.inv`/`linalg.pinv`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-heterodyne-hybrid-streaming-hessian-covariance-design.md`

## Global Constraints

- No size gate on `jax.hessian` — attempt unconditionally, matching laminar's `strategies/hybrid_streaming.py:1590-1623` behavior exactly (per spec "Approaches considered", Approach A).
- `popt` and `chi_squared` (`hier_result.x`, `hier_result.fun`) must be byte-identical to today's output — this change touches only `pcov`/`uncertainties`/`covariance_is_placeholder`.
- No changes outside `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py` and its test file — `heterodyne_result_builder.py`'s NaN-guard (commit `391dd21`) and the L4 fallback (`heterodyne_hybrid_streaming.py:994-1002`) already branch correctly on `covariance_is_placeholder` and need no edits (spec, "Downstream propagation").
- `heterodyne_stratified_ls.py`'s separate placeholder-covariance wiring bug is explicitly out of scope (spec, "Non-goals").

---

## Task 1: Real Hessian covariance for the L2/hierarchical branch

**Files:**
- Modify: `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:886-905`
- Test: `tests/optimization/test_heterodyne_hybrid_streaming.py` (append new tests after `test_streaming_l2_ssr_finite_for_individual`, ~line 1330)

**Interfaces:**
- Consumes: `_loss_jax(ph: jnp.ndarray) -> jnp.ndarray` (already defined at `heterodyne_hybrid_streaming.py:847`, closes over `x_data_jax`, `y_data_jax`, `sigma`, `adaptive_regularizer`), `hier_result` (a `HierarchicalResult` from `hierarchical_optimizer.fit(...)`, attributes used: `.x`, `.fun`, `.success`, `.n_outer_iterations`, `.message`, `.history` — all already referenced in the current code at lines 886-904, unchanged by this task), `y_data` (numpy array, function parameter already in scope), `logger` (module-level `get_logger(__name__)` instance already in scope).
- Produces: `pcov: np.ndarray` shape `(n, n)`, `info["covariance_is_placeholder"]: bool` — both already part of this function's return contract (`popt, pcov, info` tuple returned by `fit_with_stratified_hybrid_streaming_heterodyne`); this task changes their VALUES on the L2 branch, not the function's external signature.

### Current code being replaced (lines 886-905)

```python
        popt = np.asarray(hier_result.x, dtype=np.float64)
        n = len(popt)
        pcov = np.eye(n)  # Hessian covariance is optional; identity placeholder
        info: dict[str, Any] = {
            "success": bool(hier_result.success),
            "nit": int(hier_result.n_outer_iterations),
            "message": hier_result.message,
            # Approximate function-evaluation count: HierarchicalOptimizer does
            # not surface a true inner-iteration tally, so we estimate ~150 inner
            # evaluations per outer step (physical + per-angle alternations),
            # mirroring laminar's same approximation. Diagnostic only — not exact.
            "function_evaluations": hier_result.n_outer_iterations * 150,
            "covariance_is_placeholder": True,
            "hybrid_streaming_diagnostics": {
                "phase_iterations": {"phase1": 0, "phase2": hier_result.n_outer_iterations},
                "warmup_diagnostics": {},
                "gauss_newton_diagnostics": {"final_cost": hier_result.fun},
                "hierarchical_history": hier_result.history,
            },
        }
```

- [ ] **Step 1: Write the failing test — real covariance on a successful fit**

Add to `tests/optimization/test_heterodyne_hybrid_streaming.py`, right after `test_streaming_l2_ssr_finite_for_individual`:

```python
def test_streaming_l2_real_hessian_covariance_on_success():
    """L2 hierarchical branch must report a real Hessian-derived covariance,
    not the identity placeholder, on a normal successful fit."""
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        fit_with_stratified_hybrid_streaming_heterodyne,
    )

    model, c2, phi = _make_synthetic_heterodyne(n_phi=4, n_t=6)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    lo, hi = model.param_manager.get_bounds()

    popt, pcov, info = fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={
            "warmup_iterations": 5,
            "max_warmup_iterations": 10,
            "gauss_newton_max_iterations": 5,
            "verbose": 0,
        },
        anti_degeneracy_config={
            "per_angle_mode": "individual",
            "hierarchical": {"enable": True, "max_outer_iterations": 2},
        },
    )

    n = popt.shape[0]
    assert pcov.shape == (n, n)
    assert np.all(np.isfinite(pcov)), "real covariance must be finite"
    assert not np.allclose(pcov, np.eye(n)), (
        "pcov must not be the identity placeholder on a successful Hessian solve"
    )
    assert info["covariance_is_placeholder"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_real_hessian_covariance_on_success -v`
Expected: FAIL on `assert not np.allclose(pcov, np.eye(n))` — current code always returns the identity placeholder, so this assertion fails against today's implementation.

- [ ] **Step 3: Write the implementation**

Replace lines 886-905 of `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py` (the block shown under "Current code being replaced" above) with:

```python
        popt = np.asarray(hier_result.x, dtype=np.float64)
        n = len(popt)

        # Real Hessian-based Gauss-Newton covariance (mirrors laminar's
        # BUG-15/H-5 fix at strategies/hybrid_streaming.py:1590-1623). Uses
        # the pure-JAX scalar loss `_loss_jax` already defined above for the
        # optimizer's own gradient calls. Falls back to an identity
        # placeholder + covariance_is_placeholder=True only if the Hessian
        # computation itself fails.
        n_hier_data = len(y_data)
        s2_hier = float(hier_result.fun) / max(n_hier_data - n, 1)
        try:
            popt_jax = jnp.asarray(popt)
            H = np.asarray(jax.hessian(_loss_jax)(popt_jax))
        except Exception as e:
            logger.warning("Could not compute Hessian: %s. Using identity placeholder.", e)
            H = None

        covariance_is_placeholder = False
        if H is not None:
            try:
                pcov = 2.0 * s2_hier * np.linalg.inv(H)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Singular Hessian in heterodyne L2 path, using pseudo-inverse"
                )
                pcov = 2.0 * s2_hier * np.linalg.pinv(H)
        else:
            logger.error(
                "Hessian computation failed in heterodyne L2 path; covariance is "
                "an identity placeholder — reported uncertainties are NOT meaningful."
            )
            pcov = np.eye(n)
            covariance_is_placeholder = True

        info: dict[str, Any] = {
            "success": bool(hier_result.success),
            "nit": int(hier_result.n_outer_iterations),
            "message": hier_result.message,
            # Approximate function-evaluation count: HierarchicalOptimizer does
            # not surface a true inner-iteration tally, so we estimate ~150 inner
            # evaluations per outer step (physical + per-angle alternations),
            # mirroring laminar's same approximation. Diagnostic only — not exact.
            "function_evaluations": hier_result.n_outer_iterations * 150,
            "covariance_is_placeholder": covariance_is_placeholder,
            "hybrid_streaming_diagnostics": {
                "phase_iterations": {"phase1": 0, "phase2": hier_result.n_outer_iterations},
                "warmup_diagnostics": {},
                "gauss_newton_diagnostics": {"final_cost": hier_result.fun},
                "hierarchical_history": hier_result.history,
            },
        }
```

Note: `jax` and `jax.numpy as jnp` are already imported at the top of this file (lines 20-21) — no new imports needed. `_loss_jax` is defined at line 847, before this block, and is in scope via closure.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_real_hessian_covariance_on_success -v`
Expected: PASS

- [ ] **Step 5: Write the Hessian-failure fallback regression test**

This test exercises the new fallback branch added in Step 3, so it is meaningful only once that branch exists — write and verify it green immediately after Step 4, as a regression guard rather than a red/green pair. Add to the same test file:

```python
def test_streaming_l2_hessian_failure_falls_back_to_placeholder(monkeypatch):
    """If jax.hessian raises, the L2 branch must fall back to the identity
    placeholder and set covariance_is_placeholder=True (today's behavior),
    not propagate the exception."""
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import (
        heterodyne_hybrid_streaming as hs,
    )

    def _raise_hessian(fn):
        raise RuntimeError("forced hessian failure for test")

    monkeypatch.setattr(hs.jax, "hessian", _raise_hessian)

    model, c2, phi = _make_synthetic_heterodyne(n_phi=4, n_t=6)
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
    lo, hi = model.param_manager.get_bounds()

    popt, pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={
            "warmup_iterations": 5,
            "max_warmup_iterations": 10,
            "gauss_newton_max_iterations": 5,
            "verbose": 0,
        },
        anti_degeneracy_config={
            "per_angle_mode": "individual",
            "hierarchical": {"enable": True, "max_outer_iterations": 2},
        },
    )

    n = popt.shape[0]
    assert np.array_equal(pcov, np.eye(n))
    assert info["covariance_is_placeholder"] is True
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_hessian_failure_falls_back_to_placeholder -v`
Expected: PASS

- [ ] **Step 7: Write the singular-Hessian pinv-fallback test**

```python
def test_streaming_l2_singular_hessian_uses_pinv():
    """A singular Hessian must fall back to pinv and still produce a real
    (non-placeholder) covariance, not raise or silently degrade to eye(n)."""
    import jax.numpy as jnp

    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import (
        heterodyne_hybrid_streaming as hs,
    )

    def _singular_hessian(fn):
        def _zero_hessian(p):
            n = p.shape[0]
            return jnp.zeros((n, n))

        return _zero_hessian

    orig_hessian = hs.jax.hessian

    def _patched_hessian(fn):
        # First call in the branch is always the covariance Hessian in this
        # test's code path (no other jax.hessian call happens between fit()
        # returning and this computation) — return the singular stub.
        return _singular_hessian(fn)

    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(hs.jax, "hessian", _patched_hessian)
    try:
        model, c2, phi = _make_synthetic_heterodyne(n_phi=4, n_t=6)
        strat = build_heterodyne_stratified_data(model, c2, phi, weights=None)
        lo, hi = model.param_manager.get_bounds()

        popt, pcov, info = hs.fit_with_stratified_hybrid_streaming_heterodyne(
            stratified_data=strat,
            model=model,
            physical_param_names=list(model.param_manager.varying_names),
            initial_params=np.asarray(
                model.param_manager.get_initial_values(), dtype=np.float64
            ),
            bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
            hybrid_config={
                "warmup_iterations": 5,
                "max_warmup_iterations": 10,
                "gauss_newton_max_iterations": 5,
                "verbose": 0,
            },
            anti_degeneracy_config={
                "per_angle_mode": "individual",
                "hierarchical": {"enable": True, "max_outer_iterations": 2},
            },
        )
    finally:
        monkeypatch.undo()
        hs.jax.hessian = orig_hessian

    n = popt.shape[0]
    assert np.all(np.isfinite(pcov))
    assert info["covariance_is_placeholder"] is False
```

Note: this test uses an explicit `pytest.MonkeyPatch()` instance (not the `monkeypatch` fixture) because the patch must be undone inside a `finally` block around the fit call — a zero Hessian only needs to apply to this one test's fit call, and leaving `orig_hessian` restored defensively avoids any cross-test leakage even though pytest's fixture-based `monkeypatch` would already undo it automatically at test teardown.

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_singular_hessian_uses_pinv -v`
Expected: PASS

- [ ] **Step 9: Run full regression suite**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py tests/optimization/test_heterodyne_tied_result_assembly.py -v`
Expected: all tests PASS, including the 59 pre-existing tests plus the 3 new ones added in this task (62 total). No existing assertion is value-pinned to `pcov == eye(n)` on the L2 branch (confirmed during spec investigation — existing `pcov=np.eye(n)` occurrences in this file are mock `optimizer.fit()` return values for the plain-path tests, not L2-branch value assertions).

- [ ] **Step 10: Commit**

```bash
git add xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py tests/optimization/test_heterodyne_hybrid_streaming.py
git commit -m "$(cat <<'EOF'
fix(optimization): real Hessian covariance for heterodyne L2 hybrid-streaming

Ports laminar's BUG-15/H-5 Gauss-Newton covariance recipe
(strategies/hybrid_streaming.py:1590-1623) into the heterodyne
hybrid-streaming L2/hierarchical branch, which previously shipped an
unconditional pcov=eye(n) placeholder. Uses the pure-JAX scalar loss
_loss_jax already defined in this function for jax.hessian, falling back
to the identity placeholder + covariance_is_placeholder=True only if the
Hessian computation itself fails. Downstream consumers (L4 post-solve
fallback, heterodyne_result_builder.py's uncertainty NaN-guard from
commit 391dd21) already branch on the flag and require no changes.

EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Every element of the spec's "Design" section maps to Step 3's implementation (mechanism steps 1-6 of the spec correspond 1:1 to the `s2_hier`/`try`/`except`/`if H is not None` structure above). The spec's "Testing" section's 4 numbered cases map to Steps 1, 5, 7, and 9 of this task. The spec's "Regularization interaction" point required no separate task — `_loss_jax` already includes the L3 term unconditionally, so no extra code path is needed; this is called out in Step 3's comment instead of a separate task.
- **Placeholder scan:** No TBD/TODO markers; every step has full runnable code.
- **Type consistency:** `pcov: np.ndarray`, `info: dict[str, Any]`, `covariance_is_placeholder: bool` — all match the pre-existing return contract of `fit_with_stratified_hybrid_streaming_heterodyne` (unchanged signature) and the consumers named in "Interfaces" (`heterodyne_result_builder.py:566`, `heterodyne_hybrid_streaming.py:994-1002`), neither of which is modified by this plan.
