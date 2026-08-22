# Heterodyne Hybrid-Streaming L2 Hessian Covariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional `pcov = np.eye(n)` placeholder in the heterodyne hybrid-streaming L2/hierarchical branch with a real Hessian-based Gauss-Newton covariance, ported from laminar's proven recipe.

**Architecture:** In `fit_with_stratified_hybrid_streaming_heterodyne`'s L2 branch, after `hierarchical_optimizer.fit(...)` converges, compute `H = jax.hessian(_loss_jax)(popt)` (the pure-JAX scalar loss already defined in this function for autodiff), scale by Gauss-Newton `pcov = 2*s²*inv(H)`, and fall back to `eye(n)` + `covariance_is_placeholder=True` if the Hessian computation fails, is non-finite, or the resulting covariance is non-finite.

**Tech Stack:** JAX (`jax.hessian`), NumPy (`linalg.inv`/`linalg.pinv`), pytest (incl. `monkeypatch`, `caplog`).

**Spec:** `docs/superpowers/specs/2026-08-21-heterodyne-hybrid-streaming-hessian-covariance-design.md`

## Global Constraints

- No size gate on `jax.hessian` — attempt unconditionally, matching laminar's `strategies/hybrid_streaming.py:1590-1623` behavior exactly (per spec "Approaches considered", Approach A).
- `popt` and `chi_squared` (`hier_result.x`, `hier_result.fun`) must be byte-identical to today's output — this change touches only `pcov`/`uncertainties`/`covariance_is_placeholder`.
- No changes outside `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py` and its test file — `heterodyne_result_builder.py`'s NaN-guard (commit `391dd21`) and the L4 fallback (`heterodyne_hybrid_streaming.py:994-1002`) already branch correctly on `covariance_is_placeholder` and need no edits (spec, "Downstream propagation").
- `heterodyne_stratified_ls.py`'s separate placeholder-covariance wiring bug is explicitly out of scope (spec, "Non-goals").
- **`n_data <= n_params` (underdetermined fit) is a knowingly inherited edge case, not fixed here.** `max(n_hier_data - n, 1)` prevents division-by-zero but does not detect or flag a statistically-undefined covariance in that regime — this is inherited byte-for-byte from laminar's origin (`hybrid_streaming.py:1593`), not introduced by this port. Per the "exact mirror of laminar, no new gating" decision, this plan does not add new detection logic for it. Flagged here (three-brain review, Codex finding #3, 2026-08-22) as a known limitation, not silently glossed over.

---

## Task 1: Real Hessian covariance for the L2/hierarchical branch

**Files:**
- Modify: `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:886-905`
- Test: `tests/optimization/test_heterodyne_hybrid_streaming.py` (append new tests after `test_streaming_l2_ssr_finite_for_individual`, ~line 1330)

**Interfaces:**
- Consumes: `_loss_jax(ph: jnp.ndarray) -> jnp.ndarray` (already defined at `heterodyne_hybrid_streaming.py:847`, closes over `x_data_jax`, `y_data_jax`, `sigma`, `adaptive_regularizer`), `hier_result` (a `HierarchicalResult` from `hierarchical_optimizer.fit(...)`, attributes used: `.x`, `.fun`, `.success`, `.n_outer_iterations`, `.message`, `.history` — all already referenced in the current code at lines 886-904, unchanged by this task), `y_data` (numpy array, function parameter already in scope), `logger` (module-level `get_logger(__name__)` instance already in scope — `xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming`, confirmed a plain `logging.Logger` compatible with pytest's `caplog`).
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

**Review note (Codex, 2026-08-22):** the naive version of this test — running the L2 branch on `_make_synthetic_heterodyne`'s default fixture unmodified — is a false positive. That fixture generates `c2` *exactly* at the model's initial parameters (zero residual by construction, per its own docstring). Codex empirically verified that at `n_phi=4, n_t=6` this makes `s2_hier = hier_result.fun / dof = 0` and the un-noised Hessian rank-deficient (rank 11/22), so the implementation would compute `pcov = 2*0*pinv(H) = 0` — an all-zero matrix that trivially satisfies "finite", "not eye(n)", and "placeholder=False" without ever exercising the real `inv()` path or a nonzero covariance. Fix: inject small Gaussian noise into `c2` (matching the `1e-3` scale already used by `test_heterodyne_tied_result_assembly.py`'s `_build_synthetic_c2`) so the fit lands away from the exact-zero-residual point, and assert the result is not all-zero.

Add to `tests/optimization/test_heterodyne_hybrid_streaming.py`, right after `test_streaming_l2_ssr_finite_for_individual`:

```python
def test_streaming_l2_real_hessian_covariance_on_success():
    """L2 hierarchical branch must report a real, well-conditioned Hessian-derived
    covariance on a normal successful fit -- not the identity placeholder, and not
    a degenerate all-zero matrix from an exact-zero-residual fixture."""
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming import (
        fit_with_stratified_hybrid_streaming_heterodyne,
    )

    # _make_synthetic_heterodyne generates c2 EXACTLY at the model's initial
    # parameters (residual ~= 0 by construction). Left un-noised, this collapses
    # s2 to 0 and the Hessian to rank-deficient (verified during three-brain
    # review: at n_phi=4, n_t=6 the un-noised Hessian is rank 11/22), giving an
    # all-zero pcov that would trivially pass "finite" / "not eye" /
    # "placeholder=False" without exercising the real inv() path. Inject noise
    # so the fit lands away from the exact optimum and s2 > 0.
    model, c2, phi = _make_synthetic_heterodyne(n_phi=4, n_t=6)
    rng = np.random.default_rng(seed=20260821)
    c2_noisy = c2 + rng.normal(0.0, 1e-3, size=c2.shape)

    strat = build_heterodyne_stratified_data(model, c2_noisy, phi, weights=None)
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
    assert not np.allclose(pcov, 0.0), (
        "pcov must not degenerate to all-zero -- this would indicate the noise "
        "injection failed to move the fit off the exact-zero-residual point"
    )
    assert info["covariance_is_placeholder"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_real_hessian_covariance_on_success -v`
Expected: FAIL on `assert not np.allclose(pcov, np.eye(n))` — current code always returns the identity placeholder, so this assertion fails against today's implementation.

- [ ] **Step 3: Write the implementation**

**Review note (Agy + Codex, 2026-08-22):** the original draft of this step left `np.linalg.pinv(H)` outside the outer `try/except` — if `pinv` itself raised (e.g. on pathological non-finite input), the exception would escape uncaught and crash the fit instead of falling back to the placeholder. Codex separately found that neither `H` nor the final `pcov` were checked for finiteness — `np.linalg.inv`/`pinv` can return NaN-filled output *without raising*, silently leaving `covariance_is_placeholder=False` on a meaningless covariance. Fixed below: a single `try/except` wraps the Hessian computation, the inv/pinv attempt, AND explicit `np.isfinite` checks on both `H` and the final `pcov` — any failure at any of those points raises inside the `try` and is caught by one `except`, guaranteeing the placeholder+flag are only skipped when the covariance is genuinely finite and real.

Replace lines 886-905 of `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py` (the block shown under "Current code being replaced" above) with:

```python
        popt = np.asarray(hier_result.x, dtype=np.float64)
        n = len(popt)

        # Real Hessian-based Gauss-Newton covariance (mirrors laminar's
        # BUG-15/H-5 fix at strategies/hybrid_streaming.py:1590-1623). Uses
        # the pure-JAX scalar loss `_loss_jax` already defined above for the
        # optimizer's own gradient calls. Any failure along this path --
        # jax.hessian raising, a non-finite Hessian, inv/pinv raising, or a
        # non-finite result -- falls back to an identity placeholder +
        # covariance_is_placeholder=True (single unified except below; a
        # singular-but-finite Hessian still yields a real pinv-based
        # covariance and stays non-placeholder). n_data <= n_params is
        # guarded against division-by-zero the same way laminar's origin
        # does (max(..., 1)) and is not separately detected as a
        # placeholder case, matching the "exact mirror of laminar" decision
        # (spec, "Approaches considered"; see Global Constraints above).
        n_hier_data = len(y_data)
        s2_hier = float(hier_result.fun) / max(n_hier_data - n, 1)
        covariance_is_placeholder = False
        try:
            popt_jax = jnp.asarray(popt)
            H = np.asarray(jax.hessian(_loss_jax)(popt_jax))
            if not np.all(np.isfinite(H)):
                raise ValueError("Hessian contains non-finite entries")
            try:
                pcov = 2.0 * s2_hier * np.linalg.inv(H)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Singular Hessian in heterodyne L2 path, using pseudo-inverse"
                )
                pcov = 2.0 * s2_hier * np.linalg.pinv(H)
            if not np.all(np.isfinite(pcov)):
                raise ValueError("Covariance contains non-finite entries")
        except Exception as e:
            logger.error(
                "Hessian covariance failed in heterodyne L2 path (%s); covariance "
                "is an identity placeholder — reported uncertainties are NOT "
                "meaningful.",
                e,
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

This test exercises the fallback branch added in Step 3, so it is meaningful only once that branch exists — write and verify it green immediately after Step 4, as a regression guard rather than a red/green pair.

**Review note (Codex, 2026-08-22):** the spec requires the fallback path to log an ERROR explaining that reported uncertainties are not meaningful (spec, "Error handling / failure modes"). The original draft of this test checked only `pcov`/`covariance_is_placeholder`, not the log — add a `caplog` assertion so a future refactor that silently drops the log call is caught. Uses this repo's existing `caplog.at_level(..., logger=...)` pattern (see `tests/optimization/test_heterodyne_config.py:233`).

Add to the same test file:

```python
def test_streaming_l2_hessian_failure_falls_back_to_placeholder(
    monkeypatch, caplog
):
    """If jax.hessian raises, the L2 branch must fall back to the identity
    placeholder, set covariance_is_placeholder=True, and log an ERROR
    explaining the reported uncertainties are not meaningful."""
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

    with caplog.at_level(
        "ERROR",
        logger="xpcsjax.optimization.nlsq.strategies.heterodyne_hybrid_streaming",
    ):
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

    n = popt.shape[0]
    assert np.array_equal(pcov, np.eye(n))
    assert info["covariance_is_placeholder"] is True
    assert any(
        "identity placeholder" in r.getMessage() and r.levelname == "ERROR"
        for r in caplog.records
    ), "expected an ERROR log explaining the covariance is a placeholder"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_hessian_failure_falls_back_to_placeholder -v`
Expected: PASS

- [ ] **Step 7: Write the singular-Hessian pinv-fallback test**

**Review note (Agy + Codex, 2026-08-22):** the original draft manually instantiated `pytest.MonkeyPatch()` with a hand-rolled `try/finally` cleanup — both reviewers independently flagged this as an unnecessary deviation from the standard `monkeypatch` fixture already used correctly in Step 5, which pytest guarantees to undo at teardown (including on failure/timeout) without any hand-written cleanup code. Fixed below to take `monkeypatch` as a normal fixture parameter, matching Step 5's pattern.

```python
def test_streaming_l2_singular_hessian_uses_pinv(monkeypatch):
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

    monkeypatch.setattr(hs.jax, "hessian", _singular_hessian)

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
    assert np.all(np.isfinite(pcov))
    assert info["covariance_is_placeholder"] is False
```

Note: this test forces `H = zeros((n, n))` regardless of input, so `pinv(H) = zeros((n, n))` (well-defined, does not raise) — the point of this test is that a singular-but-finite Hessian takes the `pinv` branch without crashing and without falling back to the placeholder, not that the resulting covariance is nonzero (Step 1 already covers the nonzero/well-conditioned case).

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py::test_streaming_l2_singular_hessian_uses_pinv -v`
Expected: PASS

- [ ] **Step 9: Run full regression suite**

**Review note (Codex, 2026-08-22):** verified via `uv run pytest --collect-only -q tests/optimization/test_heterodyne_hybrid_streaming.py` that this file currently collects **50** tests (not 59 — the earlier "59" referred to the combined total across this file and `test_heterodyne_tied_result_assembly.py`, which was imprecise here). Stated precisely below.

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py tests/optimization/test_heterodyne_tied_result_assembly.py -v`
Expected: all tests PASS — 50 pre-existing tests in `test_heterodyne_hybrid_streaming.py` + 3 new tests added in this task (53 in that file) + 9 pre-existing tests in `test_heterodyne_tied_result_assembly.py` = **62 total** across both files. No existing assertion is value-pinned to `pcov == eye(n)` on the L2 branch (confirmed independently by both this plan's original investigation and Codex's review — existing `pcov=np.eye(n)` occurrences in this file are mock `optimizer.fit()` return values for the plain-path tests, not L2-branch value assertions).

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
to the identity placeholder + covariance_is_placeholder=True if the
Hessian computation fails, is non-finite, or the resulting covariance is
non-finite. Downstream consumers (L4 post-solve fallback,
heterodyne_result_builder.py's uncertainty NaN-guard from commit
391dd21) already branch on the flag and require no changes.

EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Every element of the spec's "Design" section maps to Step 3's implementation (mechanism steps 1-6 of the spec correspond 1:1 to the `s2_hier`/`try`/`except`/finiteness-check structure above). The spec's "Testing" section's 4 numbered cases map to Steps 1, 5, 7, and 9 of this task. The spec's "Regularization interaction" point required no separate task — `_loss_jax` already includes the L3 term unconditionally, so no extra code path is needed; this is called out in Step 3's comment instead of a separate task.
- **Placeholder scan:** No TBD/TODO markers; every step has full runnable code.
- **Type consistency:** `pcov: np.ndarray`, `info: dict[str, Any]`, `covariance_is_placeholder: bool` — all match the pre-existing return contract of `fit_with_stratified_hybrid_streaming_heterodyne` (unchanged signature) and the consumers named in "Interfaces" (`heterodyne_result_builder.py:566`, `heterodyne_hybrid_streaming.py:994-1002`), neither of which is modified by this plan.

## Three-Brain Review (2026-08-22)

Codex (correctness/edge-cases) and Agy (architecture/design-pattern) independently reviewed the spec and this plan against the live source. Findings applied above:

1. **[Agy + Codex, CONFIRMED]** `pinv` failure and non-finite `H`/`pcov` were not fully caught by the original nested try/except — fixed in Step 3 with a single unified `try/except` plus explicit `np.isfinite` checks on both `H` and the final `pcov`.
2. **[Codex, CONFIRMED, empirically verified]** the original Step 1 test used an un-noised, exact-zero-residual fixture that produces a degenerate all-zero `pcov` (rank-11/22 Hessian, `s2=0`), passing all original assertions without exercising the real `inv()` path — fixed by injecting `1e-3` Gaussian noise (matching this repo's existing convention) and adding a not-all-zero assertion.
3. **[Codex, INFORMATIONAL]** `n_data <= n_params` is an inherited, undetected edge case from laminar's own origin — documented in Global Constraints, not newly fixed here per the "exact mirror of laminar" decision.
4. **[Agy + Codex, CONFIRMED]** the original Step 7 test's manual `pytest.MonkeyPatch()` instance + hand-rolled `try/finally` cleanup was an unnecessary anti-pattern — fixed by switching to the standard `monkeypatch` fixture, matching Step 5.
5. **[Codex, CONFIRMED]** the original Step 5 fallback test did not assert the spec-required ERROR log — fixed by adding a `caplog`-based assertion.
6. **[Codex, CONFIRMED, minor]** the plan's test-count claim ("59 existing / 62 total") conflated this file's own count (50, verified via `pytest --collect-only`) with the two-file combined count — reworded in Step 9 for precision.

Checks that passed review unchanged: the `2*s²*inv(H)` Gauss-Newton factor/sign is correct for `_loss_jax`'s weighted-SSE form (`H ≈ 2·JᵀJ`); no closure/scoping issue over `_loss_jax`'s free variables; all three tests' import paths and call signatures are valid against the current source; the claim that no existing test value-asserts `pcov == eye(n)` on the L2 branch is true.
