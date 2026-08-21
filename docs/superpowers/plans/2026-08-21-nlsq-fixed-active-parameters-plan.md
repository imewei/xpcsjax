# NLSQ `fixed_parameters` / `active_parameters` Correctness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `initial_parameters.fixed_parameters` actually fix a parameter during the real NLSQ solve (currently a silent no-op in all three analysis-mode families), and make `initial_parameters.active_parameters` actually restrict the optimized parameter set in `static`/`laminar_flow` modes (currently a silent no-op there too, though already correct in `two_component`).

**Architecture (v2 — corrected after two rounds of adversarial Codex review against actual source):** Homodyne: an explicit **boolean `free_mask`** — never inferred from bounds equality — is computed once by a resolver in `parameter_utils.py`, which also substitutes the *configured fixed value* into the values array (v1 of this plan forgot that step entirely). `wrapper.py` and `adapter.py` each strip the trailing physical-parameter slice **once, immediately before their own solver dispatch**, wrap the model/residual closure with a **JAX-safe** restore (`.at[].set()`, never NumPy indexed assignment — v1 would have crashed under JIT tracing), and restore the *raw* solver output **once, immediately after dispatch returns** — before any of the existing inverse-transform / label / covariance-adjustment / compressed-scaling-expansion machinery runs, so all of that pre-existing code stays completely unaware anything was ever reduced. `fit_nlsq_cmaes` gets the same treatment inline (it calls neither adapter nor wrapper). `fit_nlsq_multistart` needs no code change, *conditional on* the resolver's value-substitution fix landing first — verified by a dedicated test, not assumed. Heterodyne: `fixed_parameters` is extracted into its own `_apply_fixed_parameters` step that runs **last** in `ParameterSpace.from_config()`'s call sequence (after `parameter_space.bounds` and grouped `parameters.*` overlays, which v1 missed and which can otherwise re-set `vary=True` after a fixed_parameters write), sets both `vary=False` and the value, and a companion guard rejects zero varying parameters before `ParameterManager.from_config()` returns.

**Tech Stack:** Python 3.12+, JAX (`JAX_ENABLE_X64=1`), upstream `nlsq>=0.6.10` (`CurveFit`/`curve_fit`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-nlsq-fixed-active-parameters-design.md` (two revision rounds recorded there — a Codex source-accuracy review, then a `/grilling` session with 9 numbered decisions Q1-Q9 — read both before starting). **This plan itself has been through two additional adversarial Codex review rounds** (source-accuracy, then spec-compliance + remaining exact-code facts) after the first plan draft was found to contain a non-functional core mechanism — see "What changed from v1" below.

## What changed from v1 (do not resurrect these bugs)

1. v1's resolver computed `free_mask` but never connected it to anything — `strip_fixed_parameters` derives its *own* mask from `lower_bounds < upper_bounds` on bounds that were never actually narrowed. v2 threads an **explicit** `free_mask` boolean array everywhere; no function infers freedom from bounds equality except the pre-existing `sequential.py` helpers, which are untouched and still used only by `sequential.py` itself.
2. v1's resolver left `values_full` unchanged — a `fixed_parameters: {D_offset: 999}` config with a flat initial value of `10000` would have restored `10000`. v2's resolver overwrites `values_full` at fixed positions with the configured value. **This is the single most important correctness fix in this plan — verify it explicitly in Task 1's tests, not indirectly through a test where the fixed value happens to equal the flat initial value.**
3. v1's restore function did NumPy indexed assignment, which breaks inside a JAX-traced closure (the model functions passed to `curve_fit`/CMA-ES *are* traced). v2 uses a separate JAX-native restore (`.at[].set()`) inside any closure that gets JIT-traced, and reserves the NumPy version for already-concrete, post-solve results only.
4. v1 had `core.py` call `adapter.fit(..., resolved_physical=...)`/`wrapper.fit(..., resolved_physical=...)` in a task ordered *before* the tasks that add that parameter to their signatures — guaranteed `TypeError`. v2 reorders: wrapper.py and adapter.py get their new parameter and internal wiring *first*, each independently testable via a direct `.fit(resolved_physical=...)` call; `core.py`'s wiring (which merely computes the descriptor and threads it through) comes after, once both already accept it.
5. v1 assumed a single formula for where physical parameters sit in wrapper.py's vector and where to strip, missing that **shear transforms run between the per-angle assembly and the point labels/`x_scale` are built** (`wrapper.py:1974-1983`), and that labels/`x_scale` are mode-dependent-length. v2's insertion point is the single spot *after* shear transforms, labels, and `x_scale` are all finalized and *before* the solver dispatch (`wrapper.py:2155-2172`) — the only place where stripping once covers everything correctly.
6. v1 claimed adapter.py expands compact→per-angle between model-build and `curve_fit()`. It does not — confirmed twice now. v2's adapter.py task strips the compact vector directly; there is no expansion step to reason about.
7. v1 used placeholder attribute names (`result.popt`/`result.pcov`) that don't exist for adapter's tuple-return branches. v2 restores on the local `popt`/`pcov` variables *after* adapter's existing tuple-vs-object extraction branching completes, which normalizes both cases to those two local names already.
8. v1's CMA-ES task referenced an undefined `physical_names` local and restored between the warm-start and CMA-ES phases, which would desync `x0`'s length from `bounds` (which is never reassigned across phases). v2 computes `physical_names` explicitly, strips `x0`/`bounds` once before phase 1, does **not** restore between phases (the warm-start's reduced-length output feeds directly into phase 2, matching the still-reduced `bounds`), and restores only the final result once.
9. v1's heterodyne `fixed_parameters` block ran inside `_apply_initial_parameters`, before `parameter_space.bounds` and grouped `parameters.*` overlays in `ParameterSpace.from_config()`'s actual call order — either overlay could re-set `vary=True` afterward, silently un-fixing it. v2 extracts a standalone `_apply_fixed_parameters` step invoked last, after `_apply_tied_parameters`.
10. v1's Task 10 referenced a nonexistent `instance` variable — `ParameterManager.from_config()` returns `cls(space=space)` directly. v2 guards `space.n_varying` before that return.

## Global Constraints

- `JAX_ENABLE_X64=1` is set by `xpcsjax/__init__.py` before any JAX import — never set it elsewhere.
- No `from module import *` (ruff `F` rule).
- `tests/parity/_golden/` goldens are pinned at `rtol=1e-10` — every code path this plan touches must be a **provable no-op** when `fixed_parameters`/`active_parameters` are unset (the template default): `free_mask` all-`True`, every strip/restore call short-circuited by an explicit `if not free_mask.all():` guard so the untouched path is byte-identical to today.
- **Invariant, tested at every layer that touches it:** a fixed parameter's value in the final `OptimizationResult.parameters` must equal the *configured* `fixed_parameters` value exactly (not the flat `initial_parameters.values` entry, if the two differ), and its uncertainty/covariance-diagonal entry must be exactly `0.0`.
- Homodyne `fixed_parameters`/`active_parameters` are scoped to **physical parameters only** (never `contrast`/`offset`) — permanent, not a v1 cut (spec, grilling Q1). A scaling parameter named in homodyne's `fixed_parameters` is a hard `ValueError` at fit time (spec, grilling Q5, Q8) — unchanged, existing `ParameterManager` behavior for an *unknown* name (neither physical nor scaling) stays a warning, not touched by this plan.
- Heterodyne `fixed_parameters` is **not** scoped that way — it mirrors `active_parameters`'s existing scope, which already includes `contrast`/`offset` (spec, grilling Q7). Test both scaling names, not just `contrast`.
- `strategies/sequential.py` keeps its existing, tested zero-length-covariance convention when every physical parameter is fixed — every *other* new call site raises `ValueError` instead (spec, grilling Q3). Multistart's own pre-existing `check_zero_volume_bounds` → single-start-fallback convention (`multistart.py:944-964`) is unrelated and untouched by this plan — it only ever triggers on a literal degenerate `parameter_space.bounds`, which this plan never produces.
- All strategy tiers must honor `fixed_parameters`/`active_parameters` (spec scope decision 1): CMA-ES and multistart get dedicated tasks (6, 7); hybrid-streaming/stratified-LS/out-of-core/chunking inherit correctness by construction because they are internal size-based dispatch branches inside `NLSQWrapper.fit()`'s call graph that receive the already-stripped `validated_params`/`nlsq_bounds` (confirmed: `wrapper.py:2155-2172` hands the same reduced values into `_execute_optimization_with_fallback`, which forwards them unchanged into `fallback_chain.py:321-424`'s streaming/recovery/large/standard branches) — verify this with a regression run (Task 3, Step 8), not a new mechanism.
- Heterodyne's new fixed-child-of-a-tie validation (Task 10) reads the raw `initial_parameters.fixed_parameters` config dict directly, independent of when `_apply_fixed_parameters` actually mutates `space` — so it is correct regardless of call order between `_apply_tied_parameters` and `_apply_fixed_parameters`.
- Run `make lint` and `uv run mypy xpcsjax` (advisory) before each commit; `make test-optimization`/`make test-heterodyne` must pass before moving to the next task.

---

## File Structure

| File | Responsibility |
|---|---|
| `xpcsjax/optimization/nlsq/parameter_utils.py` | **Modify.** Gains `ResolvedPhysicalParameters` (with real value substitution), `resolve_optimized_physical_parameters()`, mask-based `strip_by_mask`/`restore_by_mask_numpy`/`restore_by_mask_jax`. Also gains the relocated `strip_fixed_parameters`/`restore_fixed_parameters` (moved from `strategies/sequential.py`, **unchanged** semantics — bounds-equality based, used only by `sequential.py`). |
| `xpcsjax/optimization/nlsq/strategies/sequential.py` | **Modify.** `strip_fixed_parameters`/`restore_fixed_parameters` definitions removed, replaced with a re-export import. No behavior change. |
| `xpcsjax/config/parameter_manager.py` | **Modify.** Fix `active_parameters: []` truthiness bug; fix a misleading docstring example. |
| `xpcsjax/optimization/nlsq/wrapper.py` | **Modify.** `NLSQWrapper.fit()` accepts a new optional `resolved_physical` parameter; strips once right before solver dispatch, restores once right after. |
| `xpcsjax/optimization/nlsq/adapter.py` | **Modify.** `NLSQAdapter.fit()` accepts the same new parameter; strips the compact vector before `curve_fit()`, restores the normalized `popt`/`pcov` locals after. |
| `xpcsjax/optimization/nlsq/core.py` | **Modify.** `fit_nlsq_jax` computes the resolved-parameters descriptor and threads it into `adapter.fit()`/`wrapper.fit()`. `fit_nlsq_cmaes` strips/restores directly around its own `model_for_cmaes` closure and two solver-phase calls. |
| `xpcsjax/config/heterodyne_parameter_space.py` | **Modify.** New standalone `_apply_fixed_parameters` step invoked last in `ParameterSpace.from_config()`; `_apply_tied_parameters` gains a fixed-child conflict check; `_apply_initial_parameters`'s early-return precondition loosened. |
| `xpcsjax/config/heterodyne_parameter_manager.py` | **Modify.** New zero-varying-parameters guard before `from_config()` returns. |
| `tests/optimization/test_parameter_utils_resolve.py` | **Create.** Unit tests for the resolver and mask-based strip/restore, including the value-substitution invariant. |
| `tests/optimization/test_fixed_parameters_integration.py` | **Create.** Real-fit integration tests, homodyne (wrapper, adapter, CMA-ES, multistart) + heterodyne. |
| `tests/config/test_active_parameters_empty_list.py` | **Create.** Regression test for the truthiness fix. |
| `tests/config/test_heterodyne_fixed_parameters.py` | **Create.** Heterodyne-specific: value-write, overlay-ordering win, tied conflict, zero-varying guard, both scaling names. |

---

### Task 1: Add the resolver and mask-based strip/restore primitives to `parameter_utils.py`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/parameter_utils.py`
- Modify: `xpcsjax/optimization/nlsq/strategies/sequential.py:804-880`
- Test: `tests/optimization/test_parameter_utils_resolve.py`

**Interfaces:**
- Consumes: `xpcsjax.config.parameter_manager.ParameterManager.get_optimizable_parameters() -> list[str]`, `.get_fixed_parameters() -> dict[str, float]` (both existing, unmodified).
- Produces (used by Tasks 3-7):
  ```python
  @dataclass
  class ResolvedPhysicalParameters:
      physical_names: list[str]
      values_full: np.ndarray   # fixed positions already hold the CONFIGURED fixed value
      lower_full: np.ndarray
      upper_full: np.ndarray
      free_mask: np.ndarray

  def resolve_optimized_physical_parameters(
      param_manager, physical_names, values_full, lower_full, upper_full,
      *, allow_all_fixed: bool = False,
  ) -> ResolvedPhysicalParameters: ...

  def strip_by_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

  def restore_by_mask_numpy(
      free_values: np.ndarray, full_values: np.ndarray, mask: np.ndarray,
  ) -> np.ndarray: ...

  def restore_by_mask_jax(free_values, full_values: np.ndarray, mask: np.ndarray): ...
  ```
  Plus the relocated, **semantically unchanged** `strip_fixed_parameters`/`restore_fixed_parameters` (bounds-equality based — used only by `sequential.py`, never by the new mask-based call sites).

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/test_parameter_utils_resolve.py`:

```python
"""Tests for resolve_optimized_physical_parameters and mask-based strip/restore
(fixed/active physical-parameter resolution for homodyne)."""

import numpy as np
import pytest

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    resolve_optimized_physical_parameters,
    restore_by_mask_jax,
    restore_by_mask_numpy,
    strip_by_mask,
)

PHYSICAL_NAMES_LAMINAR = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]


def _base_arrays():
    values = np.array([8000.0, -1.2, 50.0, 0.01, 0.1, 0.0, 0.0])
    lower = np.array([100.0, -2.0, -1e5, 1e-6, -2.0, -0.1, -10.0])
    upper = np.array([1e5, 2.0, 1e5, 0.5, 2.0, 0.1, 10.0])
    return values, lower, upper


def test_no_config_is_all_free_and_byte_identical():
    pm = ParameterManager({}, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)
    assert isinstance(resolved, ResolvedPhysicalParameters)
    np.testing.assert_array_equal(resolved.free_mask, np.ones(7, dtype=bool))
    np.testing.assert_array_equal(resolved.values_full, values)


def test_fixed_parameter_excluded_from_free_mask_AND_value_substituted():
    """The critical v1 regression: the resolved value must be the CONFIGURED
    fixed value, not whatever the flat initial-values array happened to have --
    use DIFFERENT numbers so a bug that leaves values_full unchanged is caught."""
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"D_offset": 12.5}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()  # values[2] (D_offset) == 50.0, NOT 12.5
    resolved = resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)
    d_offset_idx = PHYSICAL_NAMES_LAMINAR.index("D_offset")
    assert resolved.free_mask[d_offset_idx] == False  # noqa: E712
    assert resolved.free_mask.sum() == 6
    assert resolved.values_full[d_offset_idx] == 12.5  # NOT 50.0 -- this is the v1 bug
    # Every OTHER position must be untouched.
    assert resolved.values_full[0] == values[0]


def test_scaling_name_in_fixed_parameters_raises():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": {"contrast": 0.5}}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="contrast"):
        resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)


def test_all_physical_fixed_raises_by_default():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|all.*fixed"):
        resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)


def test_all_physical_fixed_tolerated_when_allowed():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper, allow_all_fixed=True,
    )
    assert resolved.free_mask.sum() == 0


def test_strip_by_mask():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    np.testing.assert_array_equal(strip_by_mask(values, mask), [1.0, 3.0])


def test_restore_by_mask_numpy_round_trip():
    full_values = np.array([1.0, 99.0, 3.0, 99.0])  # positions 1,3 hold the FIXED values
    mask = np.array([True, False, True, False])
    free_result = np.array([10.0, 30.0])  # solver's fitted values for the two free positions
    restored = restore_by_mask_numpy(free_result, full_values, mask)
    np.testing.assert_array_equal(restored, [10.0, 99.0, 30.0, 99.0])


def test_restore_by_mask_jax_matches_numpy_and_is_traceable():
    import jax

    full_values = np.array([1.0, 99.0, 3.0, 99.0])
    mask = np.array([True, False, True, False])
    free_result = np.array([10.0, 30.0])

    @jax.jit
    def traced(free):
        return restore_by_mask_jax(free, full_values, mask)

    result = np.asarray(traced(free_result))
    np.testing.assert_allclose(result, [10.0, 99.0, 30.0, 99.0])


def test_relocated_strip_and_restore_unchanged():
    """sequential.py's own bounds-equality helpers, relocated but not altered."""
    from xpcsjax.optimization.nlsq.parameter_utils import restore_fixed_parameters, strip_fixed_parameters

    p = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 2.0, 0.0])
    hi = np.array([5.0, 2.0, 5.0])
    free, free_lo, free_hi, mask = strip_fixed_parameters(p, lo, hi)
    np.testing.assert_array_equal(free, [1.0, 3.0])
    np.testing.assert_array_equal(mask, [True, False, True])
    restored = restore_fixed_parameters(np.array([9.0, 8.0]), p, mask)
    np.testing.assert_array_equal(restored, [9.0, 2.0, 8.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/optimization/test_parameter_utils_resolve.py -v`
Expected: FAIL — `ImportError: cannot import name 'ResolvedPhysicalParameters'`.

- [ ] **Step 3: Relocate `strip_fixed_parameters`/`restore_fixed_parameters` (unchanged) into `parameter_utils.py`**

Read `xpcsjax/optimization/nlsq/strategies/sequential.py:804-880` to copy the exact existing bodies byte-for-byte (docstrings included) into `xpcsjax/optimization/nlsq/parameter_utils.py`, after its existing imports. Add `"strip_fixed_parameters"`, `"restore_fixed_parameters"` to `__all__`. Then modify `strategies/sequential.py`: delete the two function bodies, add to its import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    restore_fixed_parameters,
    strip_fixed_parameters,
)
```

Leave every call site inside `optimize_per_angle_sequential` completely unchanged.

- [ ] **Step 4: Run the relocation regression test**

Run: `uv run pytest tests/optimization/test_laminar_streaming_diag.py -v`
Expected: PASS (unchanged — exercises `optimize_per_angle_sequential`, proving the relocation didn't change behavior).

- [ ] **Step 5: Add the resolver and mask-based primitives**

Add to `xpcsjax/optimization/nlsq/parameter_utils.py` (add `from dataclasses import dataclass` to imports if not present; add `TYPE_CHECKING` guard for the `ParameterManager` type hint to avoid a circular import):

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xpcsjax.config.types import SCALING_PARAM_NAMES

if TYPE_CHECKING:
    import jax.numpy as jnp

    from xpcsjax.config.parameter_manager import ParameterManager


@dataclass
class ResolvedPhysicalParameters:
    """Which physical parameters are free vs. fixed for one NLSQ solve.

    ``free_mask`` comes from ``ParameterManager.get_optimizable_parameters()``
    (active-minus-fixed, physics-only by contract). ``values_full`` has the
    CONFIGURED fixed value substituted in at every fixed position -- it is
    NOT simply the caller's original values array, which may hold a stale
    flat-list value at that position instead.

    When neither ``active_parameters`` nor ``fixed_parameters`` is set (the
    template default), ``free_mask`` is all-``True`` and ``values_full``
    equals the input unchanged -- a provable no-op.
    """

    physical_names: list[str]
    values_full: np.ndarray
    lower_full: np.ndarray
    upper_full: np.ndarray
    free_mask: np.ndarray


def resolve_optimized_physical_parameters(
    param_manager: "ParameterManager",
    physical_names: list[str],
    values_full: np.ndarray,
    lower_full: np.ndarray,
    upper_full: np.ndarray,
    *,
    allow_all_fixed: bool = False,
) -> ResolvedPhysicalParameters:
    """Resolve the free/fixed split AND substitute fixed values for homodyne.

    Physical-parameter-only by design (grilling Q1) -- a scaling name in
    ``fixed_parameters`` is a hard fit-time error (grilling Q5, Q8), not a
    silent ignore.

    Raises
    ------
    ValueError
        If ``fixed_parameters`` names a scaling parameter, or if the
        resulting free set is empty and ``allow_all_fixed`` is False.
    """
    fixed_params = param_manager.get_fixed_parameters()
    scaling_fixed = [name for name in fixed_params if name in SCALING_PARAM_NAMES]
    if scaling_fixed:
        raise ValueError(
            f"fixed_parameters names scaling parameter(s) {scaling_fixed!r}, "
            "which is not supported for this analysis mode -- fixed_parameters "
            "only constrains physical parameters here. Use 'per_angle_scaling' "
            "initial values to control contrast/offset instead."
        )

    optimizable = set(param_manager.get_optimizable_parameters())
    free_mask = np.array([name in optimizable for name in physical_names], dtype=bool)

    values_full = np.array(values_full, dtype=np.float64)  # copy -- never mutate caller's array
    for i, name in enumerate(physical_names):
        if name in fixed_params:
            values_full[i] = fixed_params[name]

    if not allow_all_fixed and not free_mask.any():
        raise ValueError(
            "fixed_parameters/active_parameters leave nothing left to "
            f"optimize: every physical parameter in {physical_names!r} is "
            "fixed or excluded. Free at least one parameter."
        )

    return ResolvedPhysicalParameters(
        physical_names=list(physical_names),
        values_full=values_full,
        lower_full=np.asarray(lower_full, dtype=np.float64),
        upper_full=np.asarray(upper_full, dtype=np.float64),
        free_mask=free_mask,
    )


def strip_by_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return only the entries of ``values`` where ``mask`` is True."""
    return np.asarray(values)[mask]


def restore_by_mask_numpy(
    free_values: np.ndarray, full_values: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Re-insert solved free values into a full-length array (post-solve only).

    ``full_values`` supplies the values at fixed positions -- typically
    ``ResolvedPhysicalParameters.values_full``, which already carries the
    CONFIGURED fixed value there, not a stale initial guess.

    NumPy indexed assignment -- safe only on already-concrete arrays. Do NOT
    call this inside a JAX-traced closure (a function passed to
    ``curve_fit``/CMA-ES); use :func:`restore_by_mask_jax` there instead.
    """
    result = np.array(full_values, dtype=np.float64)
    result[mask] = np.asarray(free_values)
    return result


def restore_by_mask_jax(free_values, full_values: np.ndarray, mask: np.ndarray):
    """JAX-traceable equivalent of :func:`restore_by_mask_numpy`.

    Uses immutable ``.at[].set()`` indexed update -- safe inside a function
    that JAX traces for JIT compilation or automatic differentiation (the
    model/residual closures passed to ``curve_fit``/CMA-ES). ``free_values``
    may be a JAX tracer; ``full_values``/``mask`` must be concrete NumPy
    arrays captured by closure (not themselves traced).
    """
    import jax.numpy as jnp

    full_jnp = jnp.asarray(full_values)
    free_idx = jnp.asarray(np.where(mask)[0])
    return full_jnp.at[free_idx].set(jnp.asarray(free_values))
```

Add `"ResolvedPhysicalParameters"`, `"resolve_optimized_physical_parameters"`, `"strip_by_mask"`, `"restore_by_mask_numpy"`, `"restore_by_mask_jax"` to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_parameter_utils_resolve.py -v`
Expected: PASS, all 9 tests — in particular `test_fixed_parameter_excluded_from_free_mask_AND_value_substituted`, which is the direct regression test for v1's critical bug.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git add xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git commit -m "feat(optimization): add resolve_optimized_physical_parameters with value substitution, mask-based strip/restore"
```

---

### Task 2: Fix `active_parameters: []` truthiness bug and misleading docstring in `parameter_manager.py`

*(Unchanged from v1 — Codex CONFIRMED this task's line references and code exactly as written both review rounds.)*

**Files:**
- Modify: `xpcsjax/config/parameter_manager.py:517` (truthiness check), `:771` (docstring example)
- Test: `tests/config/test_active_parameters_empty_list.py`

- [ ] **Step 1: Write the failing test**

Create `tests/config/test_active_parameters_empty_list.py`:

```python
"""Regression: initial_parameters.active_parameters: [] must mean 'fix everything',
not 'absent -> fall back to mode defaults'."""

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


def test_empty_active_parameters_list_means_none_active():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"active_parameters": []}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert pm.get_active_parameters() == []


def test_missing_active_parameters_key_falls_back_to_defaults():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert len(pm.get_active_parameters()) == 7  # unchanged: absent key still uses mode defaults
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_active_parameters_empty_list.py::test_empty_active_parameters_list_means_none_active -v`
Expected: FAIL.

- [ ] **Step 3: Fix the truthiness check**

Modify `xpcsjax/config/parameter_manager.py` line 517:

```python
# Before:
            if active_params_config and isinstance(active_params_config, list):
# After:
            if active_params_config is not None and isinstance(active_params_config, list):
```

- [ ] **Step 4: Fix the misleading docstring example**

Modify `get_fixed_parameters()`'s docstring (around line 767-776):

```python
# Before:
        >>> config = {
        ...     "initial_parameters": {
        ...         "fixed_parameters": {"contrast": 0.5, "offset": 1.0}
        ...     }
        ... }
        >>> pm = ParameterManager(config)
        >>> pm.get_fixed_parameters()
        {'contrast': 0.5, 'offset': 1.0}
# After:
        >>> config = {
        ...     "initial_parameters": {
        ...         "fixed_parameters": {"D_offset": 10.0}
        ...     }
        ... }
        >>> pm = ParameterManager(config)
        >>> pm.get_fixed_parameters()
        {'D_offset': 10.0}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_active_parameters_empty_list.py -v`
Expected: PASS.

- [ ] **Step 6: Run the existing `ParameterManager` suite for regressions**

Run: `uv run pytest tests/config/ -v -k "parameter_manager or active_parameters or fixed_parameters"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/config/parameter_manager.py tests/config/test_active_parameters_empty_list.py
git commit -m "fix(config): active_parameters: [] now means none-active, not absent"
```

---

### Task 3: Wire strip/restore into `wrapper.py` — self-contained, testable directly

**Files:**
- Modify: `xpcsjax/optimization/nlsq/wrapper.py` — `NLSQWrapper.fit()` signature; a single strip point after `x_scale`/label construction (`:2022`) and before solver dispatch (`:2155`); a single restore point immediately after dispatch returns, before any inverse-transform/covariance-adjustment/compressed-scaling-expansion code runs.

**Interfaces:**
- Consumes: `ResolvedPhysicalParameters`, `strip_by_mask`, `restore_by_mask_numpy`, `restore_by_mask_jax` (Task 1).
- Produces: `NLSQWrapper.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)` — `None` (every caller not updated by this plan) is a complete no-op.

**Why the insertion points are exactly here (confirmed via source read, not assumed):** `wrapper.py:1974-1983` applies forward shear transforms to `validated_params`/`nlsq_bounds` (value-only, does not change vector length) — this must run *before* any physical-parameter reduction, since it indexes by `physical_index_map` computed for the full vector. `wrapper.py:1998-2022` then builds `param_labels` (mode-dependent: `"averaged"`/`"constant"`/default all use different label lists) and `x_scale_value` from the *full* vector — reduction must happen *after* this, or labels/`x_scale` would carry stale entries for removed dimensions. `wrapper.py:2155-2172` is the dispatch call into `_execute_optimization_with_fallback(..., wrapped_residual_fn=wrapped_residual_fn, xdata=..., ydata=..., validated_params=..., nlsq_bounds=..., loss_name=..., x_scale_value=..., ...)` — the single point after which nothing else touches the full-length vector before the solver runs. On the way out, `wrapper.py:2344-2359` inverse-transforms `popt`/`pcov` using `physical_index_map` again (full-length expected), `:2382-2393` computes final residuals from `popt`, and `:2524-2541` may expand a compressed scaling mode — restoring `popt`/`pcov` to full length *before* any of this (i.e., immediately when `_execute_optimization_with_fallback` returns) means every one of those existing stages runs exactly as it does today, unaware a reduction ever happened.

- [ ] **Step 1: Read the current region before editing**

Read `xpcsjax/optimization/nlsq/wrapper.py` lines 1953-2200 (strip region) and 2340-2545 (restore region) in full to confirm the exact current variable names (`validated_params`, `nlsq_bounds`, `param_labels`, `x_scale_value`, `wrapped_residual_fn`, `physical_param_names`, `popt`, `pcov`) immediately before writing the diff — these are the names confirmed via plan research, but re-read them live since this is a large, actively-developed file.

- [ ] **Step 2: Add `resolved_physical` to `NLSQWrapper.fit()`'s signature and import**

```python
    def fit(
        self,
        # ... existing parameters unchanged ...
        *,
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

Add to `wrapper.py`'s import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    restore_by_mask_jax,
    restore_by_mask_numpy,
    strip_by_mask,
)
```

- [ ] **Step 3: Strip once, immediately before dispatch**

Insert immediately after the `x_scale_value`/`param_labels` construction block (after line ~2022) and before the dispatch call (before line ~2155):

```python
        _phys_free_mask = None
        if resolved_physical is not None and not resolved_physical.free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            _phys_free_mask = resolved_physical.free_mask
            phys_slice = validated_params[-n_physical:]
            validated_params = np.concatenate(
                [validated_params[:-n_physical], strip_by_mask(phys_slice, _phys_free_mask)]
            )
            nlsq_bounds = (
                np.concatenate(
                    [nlsq_bounds[0][:-n_physical], strip_by_mask(nlsq_bounds[0][-n_physical:], _phys_free_mask)]
                ),
                np.concatenate(
                    [nlsq_bounds[1][:-n_physical], strip_by_mask(nlsq_bounds[1][-n_physical:], _phys_free_mask)]
                ),
            )
            param_labels = param_labels[:-n_physical] + [
                name for name, free in zip(resolved_physical.physical_names, _phys_free_mask, strict=True) if free
            ]
            x_scale_value = np.concatenate(
                [
                    np.asarray(x_scale_value)[:-n_physical],
                    strip_by_mask(np.asarray(x_scale_value)[-n_physical:], _phys_free_mask),
                ]
            )
            _fixed_physical_full = resolved_physical.values_full
            _base_wrapped_residual_fn = wrapped_residual_fn

            def wrapped_residual_fn(params, *args, **kwargs):
                n_prefix = len(params) - int(_phys_free_mask.sum())
                full_physical = restore_by_mask_jax(params[n_prefix:], _fixed_physical_full, _phys_free_mask)
                full_params = jnp.concatenate([params[:n_prefix], full_physical])
                return _base_wrapped_residual_fn(full_params, *args, **kwargs)
```

(`x_scale_value` may be a scalar in some modes rather than a per-parameter array — confirm from Step 1's read whether `per_param_x_scale is not None` was actually taken for the config under test before assuming the array branch; guard with `if np.ndim(x_scale_value) > 0:` around the `x_scale_value` reduction if the scalar case is reachable for a per-angle-scaling fit, which the docstring at `wrapper.py:288-291` says is mandatory — confirm scalar `x_scale` is not reachable on the tested path, or add the guard.)

- [ ] **Step 4: Restore once, immediately after dispatch returns**

Immediately after the `_execute_optimization_with_fallback(...)` call returns `popt, pcov, info, recovery_actions, convergence_status` (or however many values it actually unpacks — confirm from Step 1's read), insert, *before* any inverse-transform code runs:

```python
        if resolved_physical is not None and _phys_free_mask is not None and not _phys_free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            n_prefix = len(popt) - int(_phys_free_mask.sum())
            popt = restore_by_mask_numpy(popt[n_prefix:], resolved_physical.values_full, _phys_free_mask)
            popt = np.concatenate([popt[:n_prefix], popt[-n_physical:]])
            if pcov is not None:
                n_reduced_physical = int(_phys_free_mask.sum())
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [
                    n_prefix + i for i, free in enumerate(_phys_free_mask) if free
                ]
                full_cov[np.ix_(free_idx, free_idx)] = pcov
                pcov = full_cov
```

(This restores `popt`/`pcov` to full length *before* line ~2344's inverse-transform block runs — nothing downstream needs any further change.)

- [ ] **Step 5: Write a direct (not-yet-core.py-wired) test**

Append to `tests/optimization/test_fixed_parameters_integration.py` (create the file if Task 1 hasn't already — check first):

```python
"""Integration tests: fixed_parameters/active_parameters actually constrain
the real NLSQ solve (not just ParameterManager in isolation)."""

import numpy as np
import pytest

from xpcsjax.config import ConfigManager
from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.core.jax_backend import compute_g2_scaled
from xpcsjax.optimization.nlsq.core import _get_physical_param_names, _params_to_array
from xpcsjax.optimization.nlsq.parameter_utils import resolve_optimized_physical_parameters
from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper

TRUE_PHYSICAL = {
    "D0": 8000.0, "alpha": -1.2, "D_offset": 50.0,
    "gamma_dot_t0": 0.01, "beta": 0.1, "gamma_dot_t_offset": 0.0, "phi0": 0.0,
}
CONTRAST, OFFSET, Q, L, DT = 0.3, 0.8, 0.005, 2_000_000.0, 0.001


def _synthetic_laminar_data(n_t=10, n_phi=3, seed=0):
    import jax.numpy as jnp

    t = np.arange(1, n_t + 1) * DT
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    phi = np.array([0.0, 45.0, 90.0])[:n_phi]
    params_vec = jnp.array(list(TRUE_PHYSICAL.values()))
    g2 = np.stack(
        [
            np.asarray(
                compute_g2_scaled(
                    params_vec, jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(p),
                    Q, L, CONTRAST, OFFSET, DT,
                )
            )
            for p in phi
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    g2_noisy = g2 + rng.normal(scale=1e-4, size=g2.shape)
    return {
        "phi": phi, "g2": g2_noisy, "t1": t1, "t2": t2,
        "q": Q, "L": L, "dt": DT, "sigma": 1e-4 * np.ones_like(g2_noisy),
    }


def test_wrapper_direct_fixed_physical_parameter():
    """Direct NLSQWrapper.fit() call with resolved_physical -- proves Task 3's
    wiring in isolation, before core.py threads it through in Task 5."""
    data = _synthetic_laminar_data()
    fixed_value = 37.5  # deliberately different from TRUE_PHYSICAL's 50.0 --
    # a fixed fit should converge to 37.5, not drift toward the true 50.0.
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"D_offset": fixed_value}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    physical_names = _get_physical_param_names(AnalysisMode.LAMINAR_FLOW)
    x0_full = _params_to_array(
        {**TRUE_PHYSICAL, "contrast": CONTRAST, "offset": OFFSET}, AnalysisMode.LAMINAR_FLOW,
    )
    physical_x0 = np.asarray(x0_full)[-len(physical_names):]
    lower = np.array([100.0, -2.0, -1e5, 1e-6, -2.0, -0.1, -10.0])
    upper = np.array([1e5, 2.0, 1e5, 0.5, 2.0, 0.1, 10.0])
    resolved = resolve_optimized_physical_parameters(pm, physical_names, physical_x0, lower, upper)

    wrapper = NLSQWrapper(
        parameter_names=["contrast", "offset", *physical_names],
    )
    result = wrapper.fit(
        data=data,
        initial_params=np.asarray(x0_full),
        bounds=(
            np.concatenate([[0.0, 0.5], lower]),
            np.concatenate([[1.0, 1.5], upper]),
        ),
        analysis_mode=AnalysisMode.LAMINAR_FLOW,
        per_angle_scaling=True,
        resolved_physical=resolved,
    )
    names = ["contrast", "offset", *physical_names]
    d_offset_idx = names.index("D_offset")
    params = np.asarray(result.parameters).ravel()
    assert abs(params[d_offset_idx] - fixed_value) < 1e-9
    if result.uncertainties is not None:
        unc = np.asarray(result.uncertainties).ravel()
        assert unc[d_offset_idx] == 0.0
```

Adjust `NLSQWrapper.fit()`'s actual call signature (`data=`, `initial_params=`, `bounds=`, etc.) to match what Step 1's read confirmed — this sketch uses the names from `core.py`'s own call site as a starting point; verify against `wrapper.py`'s real `fit()` signature before finalizing.

- [ ] **Step 6: Run test to verify it fails, then implement, then verify it passes**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_wrapper_direct_fixed_physical_parameter -v` — FAIL before Steps 2-4 land, PASS after.

- [ ] **Step 7: Run the wrapper regression suite**

Run: `uv run pytest tests/optimization/test_phase5_model_function_modes.py -v` (and `grep -rl "NLSQWrapper(" tests/optimization/` for the full list)
Expected: PASS unchanged — `resolved_physical=None` default preserves the old path.

- [ ] **Step 8: Verify large-dataset memory-routing tiers inherit the strip**

Read `xpcsjax/optimization/nlsq/strategies/executors.py`'s dispatch logic and `xpcsjax/optimization/nlsq/fallback_chain.py:321-424` to confirm they receive the already-stripped `validated_params`/`nlsq_bounds` from Step 3 with no separate vector reconstruction of their own (confirmed via plan research: `fallback_chain.py:321-424` forwards the same values into streaming/recovery/large/standard branches). Then run:

```bash
ls tests/optimization/ | grep -iE "hybrid_streaming|stratified_ls|out_of_core|chunking"
uv run pytest tests/optimization/test_strategy_chunking.py -v  # adjust filenames to whatever exists
```
Expected: PASS unchanged. If the read in this step finds an independent vector construction in any of these files, stop and add a sub-task before proceeding.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/wrapper.py
git add xpcsjax/optimization/nlsq/wrapper.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQWrapper.fit()"
```

---

### Task 4: Wire strip/restore into `adapter.py` — self-contained, testable directly

**Files:**
- Modify: `xpcsjax/optimization/nlsq/adapter.py` — `NLSQAdapter.fit()` signature; strip before `curve_fit()`; restore after the existing tuple-vs-object result-normalization block.

**Interfaces:**
- Consumes: same as Task 3.
- Produces: `NLSQAdapter.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)`.

**Confirmed (twice, via source read): adapter.py does NOT expand compact→per-angle between model construction and `curve_fit()`.** `initial_params`/`bounds` are passed to `curve_fit()` exactly as received — this task's strip operates directly on the compact vector, no expansion complication.

- [ ] **Step 1: Read the current region before editing**

Read `xpcsjax/optimization/nlsq/adapter.py` lines 1256-1430 in full. Confirmed exact code at the result-extraction point:

```python
fit_kwargs: dict[str, Any] = {
    "p0": initial_params, "bounds": bounds, "method": "trf", "loss": loss,
    "ftol": ftol, "gtol": gtol, "xtol": xtol,
}
if max_nfev is not None:
    fit_kwargs["max_nfev"] = max_nfev
result = self._fitter.curve_fit(f=model_func, xdata=xdata, ydata=ydata, **fit_kwargs)
if isinstance(result, tuple):
    if len(result) == 2:
        popt, pcov = result
        info: dict[str, Any] = {}
    elif len(result) == 3:
        popt, pcov, info = result
    else:
        raise TypeError(f"Unexpected tuple length: {len(result)}")
elif hasattr(result, "popt"):
    popt = result.popt
    pcov = result.pcov
    info = dict(result) if isinstance(result, dict) else {}
else:
    raise TypeError(f"Unexpected result type: {type(result)}")
```

Restore must run *after* this entire block, operating on the local `popt`/`pcov` — never mutating `result` (which doesn't exist in a uniform shape across branches).

- [ ] **Step 2: Add `resolved_physical` to `NLSQAdapter.fit()`'s signature and import**

Same import as Task 3, Step 2, added to `adapter.py`.

```python
    def fit(
        self,
        # ... existing parameters unchanged ...
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

- [ ] **Step 3: Strip before `curve_fit()`, JAX-safe restore inside `model_func`**

Immediately after `model_func, cache_hit, jit_compiled = self._build_model_function(...)` (around line 1342) and before building `fit_kwargs`:

```python
        _phys_free_mask = None
        if resolved_physical is not None and not resolved_physical.free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            _phys_free_mask = resolved_physical.free_mask
            initial_params = np.concatenate(
                [np.asarray(initial_params)[:-n_physical], strip_by_mask(np.asarray(initial_params)[-n_physical:], _phys_free_mask)]
            )
            bounds = (
                np.concatenate([np.asarray(bounds[0])[:-n_physical], strip_by_mask(np.asarray(bounds[0])[-n_physical:], _phys_free_mask)]),
                np.concatenate([np.asarray(bounds[1])[:-n_physical], strip_by_mask(np.asarray(bounds[1])[-n_physical:], _phys_free_mask)]),
            )
            _fixed_physical_full = resolved_physical.values_full
            _base_model_func = model_func

            def model_func(x, *params):
                n_prefix = len(params) - int(_phys_free_mask.sum())
                full_physical = restore_by_mask_jax(
                    jnp.asarray(params[n_prefix:]), _fixed_physical_full, _phys_free_mask,
                )
                full_params = (*params[:n_prefix], *[full_physical[i] for i in range(n_physical)])
                return _base_model_func(x, *full_params)
```

- [ ] **Step 4: Restore after result normalization**

Immediately after the tuple-vs-object extraction block quoted in Step 1 (so this runs regardless of which branch fired), before `NLSQAdapter.fit()` constructs its `OptimizationResult`:

```python
        if resolved_physical is not None and _phys_free_mask is not None and not _phys_free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            n_prefix = len(popt) - int(_phys_free_mask.sum())
            popt = np.concatenate(
                [popt[:n_prefix], restore_by_mask_numpy(popt[n_prefix:], resolved_physical.values_full, _phys_free_mask)]
            )
            if pcov is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [n_prefix + i for i, free in enumerate(_phys_free_mask) if free]
                full_cov[np.ix_(free_idx, free_idx)] = pcov
                pcov = full_cov
```

- [ ] **Step 5: Write a direct test**

Append to `tests/optimization/test_fixed_parameters_integration.py`, mirroring Task 3's `test_wrapper_direct_fixed_physical_parameter` but calling `NLSQAdapter(...).fit(..., resolved_physical=resolved)` directly.

- [ ] **Step 6: Run test to verify it fails, then verify it passes**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -k adapter -v`

- [ ] **Step 7: Run the adapter regression suite**

Run: `uv run pytest tests/optimization/test_adapter_info_extraction.py tests/optimization/test_adapter_cost_default.py tests/optimization/test_adapter_flatten_phi_order.py -v`
Expected: PASS unchanged.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/adapter.py
git add xpcsjax/optimization/nlsq/adapter.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQAdapter.fit()"
```

---

### Task 5: Wire the resolver into `core.py::fit_nlsq_jax` (both adapter.py and wrapper.py already accept it by this point)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_jax`, after the existing bounds-construction block, and at both the `adapter.fit(...)`/`wrapper.fit(...)` call sites).

**Interfaces:**
- Consumes: `resolve_optimized_physical_parameters` (Task 1); `NLSQAdapter.fit`/`NLSQWrapper.fit` now accepting `resolved_physical` (Tasks 3-4 — **must both be committed first**, unlike v1's ordering).

- [ ] **Step 1: Compute the resolved descriptor**

Insert immediately after the existing bounds-construction block (ending around `bounds = (lower_bounds, upper_bounds)`, ~line 463), guarded for the `HAS_PARAMETER_MANAGER=False` fallback branch (v1 assumed `param_manager` unconditionally exists — it doesn't):

```python
    resolved_physical = None
    if HAS_PARAMETER_MANAGER:
        from xpcsjax.optimization.nlsq.parameter_utils import resolve_optimized_physical_parameters

        physical_names = _get_physical_param_names(analysis_mode)
        full_names = _get_param_names(analysis_mode)
        physical_idx = [full_names.index(name) for name in physical_names]
        resolved_physical = resolve_optimized_physical_parameters(
            param_manager,
            physical_names,
            values_full=np.asarray(x0)[physical_idx],
            lower_full=lower_bounds[physical_idx],
            upper_full=upper_bounds[physical_idx],
        )
```

- [ ] **Step 2: Thread `resolved_physical` into both dispatch call sites**

Add `resolved_physical=resolved_physical` as a new keyword to the existing `adapter.fit(...)` call (~line 527) and the existing `NLSQWrapper(...).fit(...)` call (the `not _use_adapter or fallback_occurred` branch further down).

- [ ] **Step 3: Write the end-to-end integration tests**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def _laminar_config(fixed_parameters=None):
    return {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {
            "parameter_names": list(TRUE_PHYSICAL.keys()),
            "values": [7000.0, -1.0, 50.0, 0.008, 0.05, 0.0, 0.0],
            **({"fixed_parameters": fixed_parameters} if fixed_parameters else {}),
        },
    }


@pytest.mark.parametrize("use_adapter", [False, True])
def test_fixed_parameter_survives_real_fit(use_adapter):
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    fixed_value = 37.5  # different from both the true value (50.0) and the flat-list value (50.0 in config)
    cm = ConfigManager(config_override=_laminar_config(fixed_parameters={"D_offset": fixed_value}))
    result = fit_nlsq_jax(data, cm, use_adapter=use_adapter)
    names = ["contrast", "offset", *TRUE_PHYSICAL.keys()]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = names.index("D_offset")
    assert abs(params[d_offset_idx] - fixed_value) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_scaling_parameter_raises():
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    cm = ConfigManager(config_override=_laminar_config(fixed_parameters={"contrast": 0.5}))
    with pytest.raises(ValueError, match="contrast"):
        fit_nlsq_jax(data, cm, use_adapter=False)


def test_unset_fixed_parameters_is_a_noop():
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    cm = ConfigManager(config_override=_laminar_config())
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    assert np.asarray(result.parameters).size == 9


def test_restricted_active_parameters_real_fit():
    """A physical parameter excluded via active_parameters must not move from
    its initial value -- distinct from fixed_parameters, same underlying
    resolver mechanism (grilling Q1/Q2 coverage gap identified in plan review)."""
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    config = _laminar_config()
    config["initial_parameters"]["active_parameters"] = ["D0", "alpha", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 50.0) < 1e-9  # unchanged from its initial value
```

Note the deliberate choice of `fixed_value = 37.5` (matching neither the true simulated value 50.0 nor coincidentally equal to anything else in the config) — this is the exact regression the plan-review process demanded: a test that would fail if Task 1's value-substitution fix were reverted, unlike v1's test where the fixed value equalled the flat-list value.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -k "not adapter and not wrapper_direct" -v`

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): compute and thread resolved physical-parameter descriptor in fit_nlsq_jax"
```

---

### Task 6: Wire strip/restore into `core.py::fit_nlsq_cmaes`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_cmaes`).

**Interfaces:**
- Consumes: same as Task 3. `fit_nlsq_cmaes` calls neither `NLSQAdapter` nor `NLSQWrapper` — it builds its own `model_for_cmaes` JAX closure and calls a distinct CMA-ES-family `wrapper` object's `_run_nlsq_refinement`/`fit` methods with `p0=x0, bounds=bounds` directly (confirmed via source read, both review rounds).

**Confirmed two-phase sequencing (`core.py:2385-2489`):** phase 1 (`wrapper._run_nlsq_refinement(model_func=model_for_cmaes, ..., p0=x0, bounds=bounds, ...)`) returns `warmstart_result["popt"]`/`["pcov"]`; then `x0 = np.asarray(nlsq_warmstart_params)` reassigns `x0` for phase 2, but **`bounds` is never reassigned across phases**. Strip `x0`/`bounds` **once**, before phase 1; do **not** restore between phases (the reduced-length warm-start output feeds directly into the still-reduced-`bounds` phase 2 call, and stays consistent); restore only the final `cmaes_result["popt"]`/`["pcov"]` once, after phase 2 returns.

- [ ] **Step 1: Write the failing test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_cmaes_fit():
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    config = _laminar_config(fixed_parameters={"D_offset": 37.5})
    config["optimization"] = {"nlsq": {"cmaes": {"enable": True}}}
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 37.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`

- [ ] **Step 3: Read the current `fit_nlsq_cmaes` region before editing**

Read `xpcsjax/optimization/nlsq/core.py` lines 1900-2500 to confirm exact current variable names for `x0`/`bounds` at the point the scaling-mode branch finishes (around line 2154, confirmed comment at `:2261-2264`: *"The physics block is ALWAYS the last n_physical entries regardless of the scaling-head layout"*), and the exact `model_for_cmaes` definition (`:2301-2364`) and both solver-phase calls (`:2385-2399`, `:2481-2489`).

- [ ] **Step 4: Compute `physical_names` explicitly and strip once before phase 1**

Insert after the scaling-mode branch finishes producing `x0`/`bounds` (~line 2154), before `model_for_cmaes` is defined (~line 2301):

```python
    from xpcsjax.optimization.nlsq.parameter_utils import (
        resolve_optimized_physical_parameters,
        restore_by_mask_jax,
        restore_by_mask_numpy,
        strip_by_mask,
    )

    physical_names = _get_physical_param_names(analysis_mode)
    n_physical = len(physical_names)
    resolved_physical = resolve_optimized_physical_parameters(
        param_manager, physical_names,
        values_full=x0[-n_physical:], lower_full=bounds[0][-n_physical:], upper_full=bounds[1][-n_physical:],
    )
    _cmaes_phys_free_mask = None
    if not resolved_physical.free_mask.all():
        _cmaes_phys_free_mask = resolved_physical.free_mask
        x0 = np.concatenate([x0[:-n_physical], strip_by_mask(x0[-n_physical:], _cmaes_phys_free_mask)])
        bounds = (
            np.concatenate([bounds[0][:-n_physical], strip_by_mask(bounds[0][-n_physical:], _cmaes_phys_free_mask)]),
            np.concatenate([bounds[1][:-n_physical], strip_by_mask(bounds[1][-n_physical:], _cmaes_phys_free_mask)]),
        )
```

- [ ] **Step 5: Wrap `model_for_cmaes` with a JAX-safe restore (used by both phases)**

Immediately after `model_for_cmaes` is defined (~line 2364):

```python
    if _cmaes_phys_free_mask is not None:
        _base_model_for_cmaes = model_for_cmaes
        _fixed_physical_full = resolved_physical.values_full

        def model_for_cmaes(params_array, *args, **kwargs):
            n_prefix = len(params_array) - int(_cmaes_phys_free_mask.sum())
            full_physical = restore_by_mask_jax(
                params_array[n_prefix:], _fixed_physical_full, _cmaes_phys_free_mask,
            )
            full_params = jnp.concatenate([params_array[:n_prefix], full_physical])
            return _base_model_for_cmaes(full_params, *args, **kwargs)
```

- [ ] **Step 6: Restore only the final result, after phase 2**

Immediately after `cmaes_result = wrapper.fit(...)` returns (~line 2489), before its `popt`/`pcov` (dict-keyed: `cmaes_result["popt"]`/`cmaes_result["pcov"]`, confirmed via source read) are consumed to build the final `OptimizationResult`:

```python
        if _cmaes_phys_free_mask is not None:
            n_prefix = len(cmaes_result["popt"]) - int(_cmaes_phys_free_mask.sum())
            full_popt = restore_by_mask_numpy(
                cmaes_result["popt"][n_prefix:], resolved_physical.values_full, _cmaes_phys_free_mask,
            )
            cmaes_result["popt"] = np.concatenate([cmaes_result["popt"][:n_prefix], full_popt])
            if cmaes_result.get("pcov") is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [
                    n_prefix + i for i, free in enumerate(_cmaes_phys_free_mask) if free
                ]
                full_cov[np.ix_(free_idx, free_idx)] = cmaes_result["pcov"]
                cmaes_result["pcov"] = full_cov
```

Do **not** insert a restore step between phase 1 (`warmstart_result`) and phase 2 (`cmaes_result`) — `nlsq_warmstart_params` stays reduced-length and feeds directly into phase 2's still-reduced `bounds`, which is correct.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`

- [ ] **Step 8: Run the CMA-ES regression suite**

Run: `uv run pytest tests/optimization/test_cmaes_trigger.py tests/optimization/test_heterodyne_cmaes_seed.py -v` (and `grep -rl "fit_nlsq_cmaes" tests/optimization/` for the full homodyne-CMA-ES list)
Expected: PASS unchanged.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): honor fixed physical parameters in fit_nlsq_cmaes"
```

---

### Task 7: Prove `fit_nlsq_multistart` needs no code change — now conditional on Task 1's value fix

**Files:**
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append only).

**Interfaces:**
- Consumes: `fit_nlsq_jax` (Task 5, already fixed by this point).
- Produces: proof, not code. `core.py::_SingleFitWorker.__call__` (`:1399-1410`) samples a start across the *full* parameter dimensionality and recurses into `fit_nlsq_jax(..., _skip_global_selection=True)`. **This claim is only actually safe now that Task 1's resolver substitutes the configured fixed value into `values_full`** — plan review round 2 flagged that under v1's buggy resolver (which left the sampled/flat value untouched), a "fixed" slot's LHS-sampled value would have been restored instead of the real fixed value. With Task 1's fix, `resolve_optimized_physical_parameters` inside the recursive `fit_nlsq_jax` call always overwrites that position with the configured fixed value regardless of what the worker's `params_dict` happened to carry there — so no worker-level change is needed.

- [ ] **Step 1: Write the test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_multistart_fit():
    from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

    data = _synthetic_laminar_data()
    config = _laminar_config(fixed_parameters={"D_offset": 37.5})
    config["optimization"] = {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 37.5) < 1e-9
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_multistart_fit -v`
Expected: PASS **without any production code change** in this task. If it fails, that falsifies the recursion-safety claim above — stop, do not weaken the test, and add a `_SingleFitWorker` sampling-scope fix (narrow the LHS sample to the free physical subset, expand to full before calling `fit_nlsq_jax`) as a new sub-task before proceeding to Task 8.

- [ ] **Step 3: Commit**

```bash
git add tests/optimization/test_fixed_parameters_integration.py
git commit -m "test(optimization): prove fixed_parameters propagates through multistart via recursion, conditional on Task 1's value substitution"
```

---

### Task 8: Heterodyne — loosen `_apply_initial_parameters`'s precondition, extract `_apply_fixed_parameters` as its own function

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::_apply_initial_parameters` (currently lines 481-556) and its `active_parameters` block.
- Test: `tests/config/test_heterodyne_fixed_parameters.py`

**Interfaces:**
- Consumes: `space.vary`, `space.values`, `_INBOUND_NAME_ALIAS`, `PARAMETER_NAME_MAPPING`, `coerce_finite_float` (all existing).
- Produces: `_apply_fixed_parameters(space, config)` — a **new, standalone** function (not nested inside `_apply_initial_parameters`), invoked separately by Task 9's orchestration change, so it can run *after* every overlay.

**Confirmed exact config-application order in `ParameterSpace.from_config()` (source-verified):**
```python
_apply_initial_parameters(space, config)       # :342
_apply_parameter_space_bounds(space, config)   # :351
# inline grouped parameters.{group}.{param} overlay  :353-465
_apply_tied_parameters(space, config)          # :470
```
Any of these can set `space.vary[name] = True` after `_apply_initial_parameters` runs — this is why `fixed_parameters` must be its own step invoked *last* (Task 9), not embedded inside `_apply_initial_parameters`.

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_heterodyne_fixed_parameters.py`:

```python
"""Heterodyne fixed_parameters: vary=False + value write, wins over EVERY
overlay including parameter_space.bounds and grouped parameters (not just
active_parameters within the same initial_parameters block), including
scaling names."""

import pytest

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager


def _config(fixed_parameters, **extra_initial):
    initial = {"fixed_parameters": fixed_parameters, **extra_initial}
    return {"analysis_mode": "two_component", "initial_parameters": initial}


def test_fixed_parameter_value_is_honored_not_flat_list_value():
    config = _config(
        fixed_parameters={"D0_ref": 999.0},
        parameter_names=["D0_ref"], values=[10000.0],
    )
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 999.0
    assert pm.space.vary["D0_ref"] is False
    assert "D0_ref" not in pm.varying_names


def test_fixed_contrast_is_honored():
    pm = ParameterManager.from_config(_config(fixed_parameters={"contrast": 0.42}))
    assert pm.space.values["contrast"] == 0.42
    assert pm.space.vary["contrast"] is False


def test_fixed_offset_is_honored():
    pm = ParameterManager.from_config(_config(fixed_parameters={"offset": 1.05}))
    assert pm.space.values["offset"] == 1.05
    assert pm.space.vary["offset"] is False


def test_fixed_parameters_applies_without_flat_parameter_names():
    config = {"analysis_mode": "two_component", "initial_parameters": {"fixed_parameters": {"D0_ref": 7.0}}}
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 7.0
    assert pm.space.vary["D0_ref"] is False


def test_fixed_wins_over_active_on_conflict():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"active_parameters": ["D0_ref"], "fixed_parameters": {"D0_ref": 3.0}},
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.vary["D0_ref"] is False
    assert pm.space.values["D0_ref"] == 3.0


def test_fixed_wins_over_LATER_parameter_space_bounds_overlay():
    """The v1-review-identified gap: fixed_parameters must win even against
    overlays that run AFTER it in ParameterSpace.from_config()'s call order."""
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"fixed_parameters": {"D0_ref": 5.0}},
        "parameter_space": {"bounds": [{"name": "D0_ref", "vary": True, "min": 0.0, "max": 1e5}]},
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.vary["D0_ref"] is False
    assert pm.space.values["D0_ref"] == 5.0
```

(The exact `parameter_space.bounds` entry shape for setting `vary` may differ from this sketch — read `_apply_parameter_space_bounds` before finalizing this test to use its real config shape for setting `vary=True` on a named parameter; adjust the test's `parameter_space` block accordingly, keeping the assertion intent unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: FAIL on all 6.

- [ ] **Step 3: Read the current `_apply_initial_parameters` and `ParameterSpace.from_config` before editing**

Read `xpcsjax/config/heterodyne_parameter_space.py` lines 294-360 (`from_config`'s orchestration) and 481-557 (`_apply_initial_parameters`) to confirm exact current structure before editing.

- [ ] **Step 4: Loosen `_apply_initial_parameters`'s early-return precondition**

```python
# Before:
    param_names_raw = initial.get("parameter_names")
    param_values = initial.get("values")
    if (not param_names_raw or not isinstance(param_names_raw, list)
        or param_values is None or not isinstance(param_values, list)):
        return
# After:
    param_names_raw = initial.get("parameter_names")
    param_values = initial.get("values")
    has_flat_values = (
        param_names_raw and isinstance(param_names_raw, list)
        and param_values is not None and isinstance(param_values, list)
    )
```

Wrap the existing flat-value-application loop in `if has_flat_values:`. The `active_parameters` block below it must run unconditionally (not gated on `has_flat_values`) — it already does not depend on the flat-values loop's variables, so no further change needed there beyond removing the early `return`.

- [ ] **Step 5: Extract `_apply_fixed_parameters` as a standalone function** (NOT nested inside `_apply_initial_parameters` — Task 9 calls it separately, last)

Add as a new top-level function in `heterodyne_parameter_space.py`, after `_apply_initial_parameters`:

```python
def _apply_fixed_parameters(space: ParameterSpace, config: dict[str, Any]) -> None:
    """Apply ``initial_parameters.fixed_parameters`` to *space*.

    Sets BOTH the vary flag and the value -- expand_varying_to_full() fills
    non-varying positions from space.values, so the value write is required,
    not optional (design spec, Codex review finding #11).

    MUST run LAST in ParameterSpace.from_config()'s call sequence -- after
    _apply_initial_parameters, _apply_parameter_space_bounds, the grouped
    parameters.* overlay, and _apply_tied_parameters -- so a fixed parameter
    always wins regardless of what any other overlay sets (plan review round
    2, Task 8's overlay-ordering gap). Not scoped to physical-only (grilling
    Q7 -- unlike homodyne, heterodyne's fixed_parameters mirrors
    active_parameters' existing scope, which already includes contrast/offset
    via ALL_PARAM_NAMES_WITH_SCALING).
    """
    from xpcsjax.config.types import coerce_finite_float

    initial = config.get("initial_parameters", {})
    if not initial or not isinstance(initial, dict):
        return

    fixed_raw = initial.get("fixed_parameters")
    if fixed_raw is None or not isinstance(fixed_raw, dict):
        return

    for name, value in fixed_raw.items():
        mapped = PARAMETER_NAME_MAPPING.get(str(name), str(name))
        canonical = _INBOUND_NAME_ALIAS.get(mapped, mapped)
        if canonical not in space.values:
            logger.warning("fixed_parameters: unknown parameter '%s', skipping", name)
            continue
        space.values[canonical] = coerce_finite_float(
            value, context=f"initial_parameters.fixed_parameters[{canonical!r}]"
        )
        space.vary[canonical] = False
        logger.debug("fixed_parameters: fixed %s = %.6g", canonical, value)
```

- [ ] **Step 6: Run the value/scaling/no-flat-values tests (overlay-ordering test still fails — wired in Task 9)**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -k "not LATER" -v`
Expected: PASS for the first 5 tests. `test_fixed_wins_over_LATER_parameter_space_bounds_overlay` still FAILS — expected, `_apply_fixed_parameters` isn't wired into `from_config`'s call sequence yet (Task 9).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_space.py
git add xpcsjax/config/heterodyne_parameter_space.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): extract standalone _apply_fixed_parameters for heterodyne, loosen precondition"
```

---

### Task 9: Heterodyne — invoke `_apply_fixed_parameters` last, and validate the tied-child-fixed conflict

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::from_config` (call-sequence) and `_apply_tied_parameters` (currently lines 559-733).
- Test: `tests/config/test_heterodyne_fixed_parameters.py` (append)

**Interfaces:**
- Consumes: `_apply_fixed_parameters` (Task 8).
- Produces: `fixed_parameters` now wins against every overlay; `ValueError` when a tied child also appears in `fixed_parameters`.

- [ ] **Step 1: Wire `_apply_fixed_parameters` to run last in `from_config`**

Modify the call sequence (confirmed exact current order, `heterodyne_parameter_space.py:341-470`):

```python
# Before:
    _apply_initial_parameters(space, config)
    _apply_parameter_space_bounds(space, config)
    # ... inline grouped parameters.{group}.{param} overlay ...
    _apply_tied_parameters(space, config)
# After:
    _apply_initial_parameters(space, config)
    _apply_parameter_space_bounds(space, config)
    # ... inline grouped parameters.{group}.{param} overlay ...
    _apply_tied_parameters(space, config)
    _apply_fixed_parameters(space, config)
```

- [ ] **Step 2: Run the overlay-ordering test**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_fixed_wins_over_LATER_parameter_space_bounds_overlay -v`
Expected: PASS.

- [ ] **Step 3: Write the tied-child-fixed conflict test**

Append to `tests/config/test_heterodyne_fixed_parameters.py`:

```python
def test_tied_child_also_fixed_raises():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "tied_parameters": {"D0_ref": "D0_sample"},
            "fixed_parameters": {"D0_ref": 5.0},
        },
    }
    with pytest.raises(ValueError, match="tied_parameters.*D0_ref.*fixed_parameters"):
        ParameterManager.from_config(config)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_tied_child_also_fixed_raises -v`

- [ ] **Step 5: Add the conflict check to `_apply_tied_parameters`**

Read `xpcsjax/config/heterodyne_parameter_space.py:559-680` — confirmed exact validation order and message style:

```python
if child not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{child}'. Valid names: {list(ALL_PARAM_NAMES)}")
if parent not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{parent}'. Valid names: {list(ALL_PARAM_NAMES)}")
if child == parent: raise ValueError(f"tied_parameters: '{child}' cannot be tied to itself")
if parent in children: raise ValueError(f"tied_parameters: '{parent}' is itself a tied child (tied to '{tied_translated[parent]}') -- chained ties are not supported. Tie '{child}' directly to '{tied_translated[parent]}' instead.")
if not space.vary.get(parent, False): raise ValueError(f"tied_parameters: parent '{parent}' is not varying (fixed via active_parameters or vary: false) -- tying '{child}' to a fixed parent is not supported; fix '{child}' directly instead via active_parameters.")
```

Add, immediately after the `parent in children` check, inside the same per-pair validation loop — reads the **raw config dict** directly (order-independent w.r.t. when `_apply_fixed_parameters` actually mutates `space`):

```python
        fixed_raw = initial.get("fixed_parameters")
        fixed_names: set[str] = set()
        if fixed_raw is not None and isinstance(fixed_raw, dict):
            fixed_names = {
                _INBOUND_NAME_ALIAS.get(PARAMETER_NAME_MAPPING.get(str(n), str(n)), PARAMETER_NAME_MAPPING.get(str(n), str(n)))
                for n in fixed_raw
            }
        if child in fixed_names:
            raise ValueError(
                f"tied_parameters: '{child}' is also listed in fixed_parameters "
                "-- a tied child's value is derived from its parent every "
                f"residual evaluation; fixing it independently is a "
                f"contradiction. Fix '{parent}' instead if you want both pinned."
            )
```

(Compute `fixed_raw`/`fixed_names` once before the per-pair loop begins, not inside it — confirm the loop's exact structure from Step 5's read.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 7.

- [ ] **Step 7: Run the existing tied-parameters regression suite**

Run: `uv run pytest tests/config/test_heterodyne_parameter_manager_tied.py -v`
Expected: PASS unchanged.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_space.py
git add xpcsjax/config/heterodyne_parameter_space.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): apply heterodyne fixed_parameters last, reject tied-child-fixed conflict"
```

---

### Task 10: Heterodyne — zero-varying-parameters guard

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_manager.py`
- Test: `tests/config/test_heterodyne_fixed_parameters.py` (append)

**Interfaces:**
- Consumes: `ParameterSpace.n_varying`/`.varying_names` (existing, scaling-inclusive — confirmed `heterodyne_parameter_space.py:86-94`; do **not** use `ParameterManager.varying_indices`, physics-only).
- Produces: `ValueError` before `ParameterManager.from_config()` returns, when `space.n_varying == 0`.

**Confirmed (v1 bug):** `ParameterManager.from_config()` returns `cls(space=space)` directly — there is no intermediate `instance` variable. The guard must check `space.n_varying` directly, before that return statement, not on a constructed manager afterward.

- [ ] **Step 1: Write the failing tests**

Append to `tests/config/test_heterodyne_fixed_parameters.py`:

```python
from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES_WITH_SCALING


def test_zero_varying_parameters_raises():
    fixed = {name: 0.0 for name in ALL_PARAM_NAMES_WITH_SCALING}
    config = {"analysis_mode": "two_component", "initial_parameters": {"fixed_parameters": fixed}}
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|no varying"):
        ParameterManager.from_config(config)


def test_active_parameters_empty_list_also_raises():
    config = {"analysis_mode": "two_component", "initial_parameters": {"active_parameters": []}}
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|no varying"):
        ParameterManager.from_config(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_zero_varying_parameters_raises tests/config/test_heterodyne_fixed_parameters.py::test_active_parameters_empty_list_also_raises -v`

- [ ] **Step 3: Read `ParameterManager.from_config` before editing**

Read `xpcsjax/config/heterodyne_parameter_manager.py:560-580` to confirm the exact current `return cls(space=space)` (or equivalent) statement.

- [ ] **Step 4: Add the guard immediately before the return**

```python
        if space.n_varying == 0:
            raise ValueError(
                "Nothing left to optimize: active_parameters/fixed_parameters "
                "combine to leave zero varying parameters (physics and "
                "scaling combined). Free at least one parameter."
            )
        return cls(space=space)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 9.

- [ ] **Step 6: Run the full heterodyne config regression suite**

Run: `uv run pytest tests/config/ -k heterodyne -v`
Expected: PASS — no existing config leaves zero varying parameters.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_manager.py
git add xpcsjax/config/heterodyne_parameter_manager.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): guard against zero varying parameters in heterodyne ParameterManager"
```

---

### Task 11: Heterodyne real-fit integration test (reusing existing synthetic-data helpers) + full regression sweep

**Files:**
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append heterodyne case) — no production code changes.

**Interfaces:**
- Consumes: everything from Tasks 1-10.

**Do not hand-roll a `compute_c2_heterodyne` call for synthetic data** — v1's test used a wrong signature (it takes a single `t` array and a single scalar `phi_angle` per call, not separate `t1`/`t2`/an array of `phi`, confirmed: `compute_c2_heterodyne(params, t, q, dt, phi_angle, contrast=1.0, offset=1.0)`). Reuse the existing `_make_synthetic_heterodyne()` helper (`tests/optimization/test_heterodyne_hybrid_streaming.py:9`), which already builds correct heterodyne synthetic data.

- [ ] **Step 1: Read the existing synthetic-heterodyne helper**

Read `tests/optimization/test_heterodyne_hybrid_streaming.py` lines 1-60 to see `_make_synthetic_heterodyne()`'s exact signature and return shape before writing the test.

- [ ] **Step 2: Write the heterodyne real-fit test using the existing helper**

Append to `tests/optimization/test_fixed_parameters_integration.py`, importing and calling `_make_synthetic_heterodyne` (or an equivalent already-existing helper found in Step 1) rather than reconstructing the forward model by hand:

```python
def test_heterodyne_fixed_parameter_survives_real_fit():
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq import fit_nlsq

    data, true_physical_dict = _make_synthetic_heterodyne()  # adjust unpacking to the helper's real return shape
    fixed_name, fixed_value = "D_offset_sample", 0.0  # matches the helper's true value if it uses 0.0; else pick a distinct value
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "parameter_names": list(true_physical_dict.keys()),
            "values": list(true_physical_dict.values()),
            "fixed_parameters": {fixed_name: fixed_value},
        },
    }
    cm = ConfigManager(config_override=config)
    result = fit_nlsq(data, cm)
    names = list(true_physical_dict.keys())
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index(fixed_name)] - fixed_value) < 1e-6


def test_heterodyne_fixed_scaling_parameter_survives_real_fit():
    """grilling Q7 end-to-end: heterodyne can fix a scaling name too, unlike homodyne."""
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq import fit_nlsq

    data, true_physical_dict = _make_synthetic_heterodyne()
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "parameter_names": list(true_physical_dict.keys()),
            "values": list(true_physical_dict.values()),
            "fixed_parameters": {"contrast": 0.35},
        },
    }
    cm = ConfigManager(config_override=config)
    result = fit_nlsq(data, cm)
    assert result.convergence_status is not None  # fit completed without the ValueError homodyne would raise
```

Adjust both tests' exact unpacking/field names once Step 1's read confirms `_make_synthetic_heterodyne()`'s real signature — this sketch assumes it returns `(data_dict, true_physical_dict)`; correct it if the real shape differs (e.g. it may return a data dict plus separate `phi`/`t1`/`t2`/true-parameter-array pieces rather than a name-keyed dict — in that case, build `parameter_names`/`values` from whatever it actually returns, in its actual order).

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -k heterodyne -v`
Expected: PASS.

- [ ] **Step 4: Run the full optimization + config + heterodyne test suites**

```bash
make test-optimization
make test-heterodyne
uv run pytest tests/config/ -v
```
Expected: PASS, zero failures, zero new skips.

- [ ] **Step 5: Full local verification (forces the CPU-microarch-gated synthetic parity tests, per root `CLAUDE.md`)**

```bash
make test-full-local
```
Expected: PASS, `rtol=1e-10` unchanged on `tests/parity/test_homodyne_engine_preservation.py` and `tests/parity/test_engine_heterodyne_fit_parity.py`, and `tests/parity/test_phase5_default_no_worse.py::test_synthetic_default_averaged_no_worse_than_individual` unaffected — every golden/no-worse config has `fixed_parameters`/`active_parameters` unset, which this plan guarantees is a byte-identical no-op path (`ResolvedPhysicalParameters.free_mask` all-`True`, every strip/restore branch short-circuited).

- [ ] **Step 6: Full suite + lint**

```bash
make test
make lint
uv run mypy xpcsjax
```
Expected: `make test` and `make lint` pass; `mypy` advisory (non-blocking per root `CLAUDE.md`).

- [ ] **Step 7: Final commit**

```bash
git add tests/optimization/test_fixed_parameters_integration.py
git commit -m "test(optimization): heterodyne real-fit integration tests for fixed_parameters, including scaling names"
```

---

## Deferred / explicitly out of scope

Per the spec's Out-of-scope section: the separate CLI-override bug in `cli/config_handling.py` (`--initial-*` re-introducing a fixed parameter); dead-code cleanup of `xpcsjax/core/fitting.py::ParameterSpace` and `xpcsjax/config/parameter_space.py::ParameterSpace`; the `config/manager.py::_calculate_midpoint_defaults` misleading comment. File these as separate follow-up issues — do not fold them into this plan's tasks or commits.
