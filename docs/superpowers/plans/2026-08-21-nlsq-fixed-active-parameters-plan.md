# NLSQ `fixed_parameters` / `active_parameters` Correctness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `initial_parameters.fixed_parameters` actually fix a parameter during the real NLSQ solve (currently a silent no-op in all three analysis-mode families), and make `initial_parameters.active_parameters` actually restrict the optimized parameter set in `static`/`laminar_flow` modes (currently a silent no-op there too, though already correct in `two_component`).

**Architecture (v3 — after two Codex review rounds plus a second `/grilling` session):** Homodyne: an explicit **boolean `free_mask`** — never inferred from bounds equality — is computed once by a resolver in `parameter_utils.py`, which also substitutes the *configured fixed value* into the values array. `wrapper.py` and `adapter.py` each strip the trailing physical-parameter slice **once, immediately before their own solver dispatch**, wrap the model/residual closure with a **JAX-safe** restore (`.at[].set()`, never NumPy indexed assignment), and restore the *raw* solver output **once, immediately after dispatch returns** — before any inverse-transform/label/covariance-adjustment/compressed-scaling-expansion machinery runs, so all of it stays unaware anything was reduced. `fit_nlsq_cmaes` gets the same treatment inline (it calls neither adapter nor wrapper, and its CMA-ES-phase result is a `CMAESResult` **dataclass** — `.parameters`/`.covariance` attributes, not dict keys). `fit_nlsq_multistart` needs no code change *conditional on* the resolver's value-substitution fix — verified by a dedicated test, with a pre-written fallback task if that verification fails. Heterodyne: `fixed_parameters` is extracted into its own `_apply_fixed_parameters` step invoked **last** in `ParameterSpace.from_config()`'s call sequence, sets both `vary=False` and the value, and a companion guard rejects zero varying parameters.

**Tech Stack:** Python 3.12+, JAX (`JAX_ENABLE_X64=1`), upstream `nlsq>=0.6.10` (`CurveFit`/`curve_fit`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-nlsq-fixed-active-parameters-design.md` (two revision rounds — Codex review, then a `/grilling` session with 9 decisions Q1-Q9). **This plan has been through two Codex review rounds and two `/grilling` sessions** — see "What changed" sections below; do not resurrect any of these bugs.

## What changed from v1 → v2 (Codex review)

1. v1's resolver computed `free_mask` but never connected it to `strip_fixed_parameters`, which derives its own mask from `lower_bounds < upper_bounds` on bounds never actually narrowed. v2+ threads an **explicit** `free_mask` boolean array everywhere.
2. v1's resolver left `values_full` unchanged. v2+ overwrites fixed positions with the configured value — **the single most important correctness fix in this plan**.
3. v1's restore used NumPy indexed assignment, which breaks inside a JAX-traced closure. v2+ uses a JAX-native restore (`.at[].set()`) inside traced closures, NumPy only for concrete post-solve results.
4. v1 ordered `core.py`'s call sites before the tasks that add the parameter those calls need — guaranteed `TypeError`. v2+ reorders: adapter/wrapper get their signature first.
5. v1 missed that wrapper.py runs shear transforms, then builds mode-dependent labels/`x_scale`, all before dispatch. v2+'s insertion point is after all of that, right before dispatch.
6. v1 claimed adapter.py expands compact→per-angle before `curve_fit()`. It does not (confirmed twice).
7. v1 used wrong result attribute names (`result.popt`/`result.pcov`). v2+ restores on the local `popt`/`pcov` after adapter's existing tuple-vs-object branching normalizes both cases.
8. v1's CMA-ES task referenced an undefined `physical_names` local and restored between phases, desyncing `x0` from `bounds`. v2+ strips once before phase 1, never restores between phases.
9. v1's heterodyne `fixed_parameters` ran before `parameter_space.bounds`/grouped overlays, which could re-set `vary=True` afterward. v2+ extracts a standalone step invoked last.
10. v1's Task 10 referenced a nonexistent `instance` variable — `from_config()` returns `cls(space=space)` directly. v2+ guards `space.n_varying` before that return.

## What changed from v2 → v3 (second `/grilling` session, plus a third fact-finding pass)

1. **Tasks that were "wrapper.py/adapter.py, self-contained, testable directly" dropped their hand-rolled direct `.fit()` calls entirely.** Constructing a valid direct `NLSQWrapper.fit()`/`NLSQAdapter.fit()` call requires replicating a meaningful slice of `core.py`'s own preprocessing (data normalization, bounds construction, sigma defaults) — new, untested scaffolding whose own bugs could fail a test for reasons unrelated to the fix under test, which is exactly the fragility class this plan's whole review history has been fighting. Tasks 3 and 4 are now implementation-only (write the code, lint, import-smoke-check); the actual proof that both paths work is Task 5's real `fit_nlsq_jax` integration tests, parametrized over `use_adapter`.
2. **`x_scale_value` can be the literal string `"jac"` (the default), not always a per-parameter array.** `wrapper.py:903` defaults it to `"jac"`; only `build_per_parameter_x_scale(...)` returning non-`None` (driven by the `optimization.nlsq.x_scale_map` config key) produces an array. Task 3's array-reduction code must guard `isinstance(x_scale_value, np.ndarray)` before slicing — v2's code would have sliced 3 characters off a string. A dedicated test in Task 5 forces the array branch via `x_scale_map`, since the default-config tests never reach it.
3. **`fit_nlsq_cmaes`'s CMA-ES-phase result is a `CMAESResult` dataclass (`.parameters`/`.covariance` attributes), not a dict** — only the warm-start phase's `_run_nlsq_refinement(...)` genuinely returns a dict (`popt`/`pcov`/`infodict`/`mesg`/`ier` keys). v2's Task 6 used `cmaes_result["popt"]`/`["pcov"]` for the CMA-ES phase, which would raise `TypeError: 'CMAESResult' object is not subscriptable`. v3 fixes this to attribute access for that phase only.
4. **`tests/optimization/test_heterodyne_hybrid_streaming.py::_make_synthetic_heterodyne()` returns `(model, c2, phi)`, not a name-keyed `(data_dict, true_physical_dict)` pair, and produces noise-free data.** v2's Task 11 assumed a return shape that doesn't exist. v3 rebuilds the test around the real return: physical values come from `model.param_manager.get_full_values()` (14-name `ALL_PARAM_NAMES` order), `contrast`/`offset` from `model.scaling.get_for_angle(0)`, and — matching the rigor of the homodyne tests, which do inject noise — this plan now adds `rng.normal(scale=1e-4, ...)` onto `c2` before fitting, with `sigma=None` (unweighted), consistent with root `CLAUDE.md`'s documented `sigma=None` sentinel convention.
5. **Homodyne testing was laminar_flow-only.** The spec names `static_anisotropic`/`static_isotropic`/`laminar_flow` as in-scope, and static mode is a genuinely different code shape (`n_physical=3` vs. 7; the shear-transform stage never engages at all for static modes — both its gating flags look up `index_map.get("gamma_dot_t0")`/`.get("beta")`, both `None` for static's 3-name physical set, regardless of config). Task 5's integration test now parametrizes on `analysis_mode` too, not just `use_adapter`.
6. **Every task now has an explicit "re-verify against current source before editing" step**, not just the tasks flagged with prior uncertainty — this codebase has demonstrated real DRIFT between research and a fresh read minutes later across every review round so far.
7. **Task 7's fallback is pre-written as Task 7b**, not left as "improvise a fix if the test fails" — an executor hitting that failure mid-run should have a concrete, scoped task to run, not a blank page.

## What changed from v3 → v4 (third Codex review round)

1. **Tasks 3 and 6's wrapped closures had the calling convention backwards.** The real solver-facing signature is `f(xdata, *params)` — `xdata` first, physical params unpacked as individual scalars (confirmed live: `wrapper.py:3862`'s docstring, `wrapper.py:2388`'s call site, `core.py`'s `model_for_cmaes(xdata_unused, *params)`). v3's closures took the params array *first*, which would silently swap `xdata` and the parameter vector — wrong physics, not a crash, so lint/import-smoke-check (the only verification v3 gave those two tasks) would never catch it. Task 4's closure had this right by accident; Tasks 3 and 6 now match its shape.
2. **The `x_scale_value` guard was too narrow.** `isinstance(x_scale_value, np.ndarray)` misses a manually-configured `optimization.nlsq.x_scale` list, which reaches this point as a plain `list`/`tuple`, not yet an array. Broadened to cover both, normalizing via `np.asarray` before reducing.
3. **Task 6's restore missed the CMA-ES auto-skip branch.** When CMA-ES is skipped, `core.py` constructs a `CMAESResult` directly without ever calling `wrapper.fit(...)` — a restore placed only immediately after that call misses the skip path entirely. Moved to after both branches converge on one `cmaes_result` variable.
4. **Task 11's hand-built heterodyne config would have evaluated against different geometry than the one that generated the synthetic data.** The public `fit_nlsq()` heterodyne dispatcher ignores `t1`/`t2`/`q`/`dt`/`sigma` from the data dict entirely and rebuilds a fresh `HeterodyneModel` from config alone — so the test now loads the *same* template `_make_synthetic_heterodyne()` uses, overriding only `fixed_parameters`, instead of hand-building a minimal config. Also: `result.parameters` is a reduced varying-vector, not full `ALL_PARAM_NAMES` order — the test now reads `result.nlsq_diagnostics["parameter_names"]` (the same mechanism the codebase's own diagnostics code uses) instead of assuming positional order.
5. **Task 5 was missing `static_anisotropic`**, despite the spec naming both static modes; added (same 3-name physical set as `static_isotropic`, confirmed by a live kernel comparison — a 3-slot and a zero-padded 7-slot `compute_g2_scaled` call produce identical output).
6. **Task 7b's worker-sampling-scope fix was contingent on a test failing first, but the spec's Component 2 states it as required behavior**, not just a correctness safety net (it also avoids wasting multistart's sampling budget on locked dimensions). Merged into Task 7 as a mandatory step; the recursion-based correctness argument still stands independently and remains documented, but the narrowing is no longer conditional on it being tested first.
7. **Added best-effort coverage for the spec's per-tier fixed-parameter requirement** (hybrid-streaming, stratified-LS, sequential, out-of-core) — v3 only ran regression suites for these, which prove no *regression* but never asserted the new mechanism actually reaches them. Task 11 now includes an explicit step to force each reachable tier with a small dataset/threshold override, with instructions to document rather than silently skip any tier that can't be forced.

## Global Constraints

- `JAX_ENABLE_X64=1` is set by `xpcsjax/__init__.py` before any JAX import — never set it elsewhere.
- No `from module import *` (ruff `F` rule).
- **Every task begins with a re-read of the exact file:line ranges it cites, before writing any diff.** This codebase is actively developed; a plan snippet that was correct when written may have drifted by the time a task executes. Do not skip this step because a range was "already confirmed" earlier in this plan's history — confirm it again, live.
- `tests/parity/_golden/` goldens are pinned at `rtol=1e-10` — every code path this plan touches must be a **provable no-op** when `fixed_parameters`/`active_parameters` are unset (the template default): `free_mask` all-`True`, every strip/restore call short-circuited by an explicit `if not free_mask.all():` guard.
- **Invariant, tested at every layer that touches it:** a fixed parameter's value in the final `OptimizationResult.parameters` must equal the *configured* `fixed_parameters` value exactly (not the flat `initial_parameters.values` entry, if the two differ), and its uncertainty/covariance-diagonal entry must be exactly `0.0`.
- Homodyne `fixed_parameters`/`active_parameters` are scoped to **physical parameters only** (never `contrast`/`offset`) — permanent (spec, grilling round 1 Q1). A scaling parameter named in homodyne's `fixed_parameters` is a hard `ValueError` at fit time (spec, grilling round 1 Q5, Q8).
- Heterodyne `fixed_parameters` is **not** scoped that way — it mirrors `active_parameters`'s existing scope, which already includes `contrast`/`offset` (spec, grilling round 1 Q7). Test both scaling names.
- `strategies/sequential.py` keeps its existing, tested zero-length-covariance convention when every physical parameter is fixed — every *other* new call site raises `ValueError` instead (spec, grilling round 1 Q3). Multistart's own pre-existing `check_zero_volume_bounds` → single-start-fallback convention (`multistart.py:944-964`) is unrelated and untouched — it only ever triggers on a literal degenerate `parameter_space.bounds`, which this plan never produces.
- All strategy tiers must honor `fixed_parameters`/`active_parameters`: CMA-ES and multistart get dedicated tasks (6, 7); hybrid-streaming/stratified-LS/out-of-core/chunking inherit correctness by construction (internal size-based dispatch branches inside `NLSQWrapper.fit()`'s call graph, confirmed to receive the already-stripped `validated_params`/`nlsq_bounds` unchanged) — verified by a regression run, not a new mechanism.
- `x_scale_value` is the string `"jac"` by default; `optimization.nlsq.x_scale_map` produces a per-parameter array, but a raw `optimization.nlsq.x_scale` config can also be a manual `list[float]` (`config.py:219`) that reaches this point as a plain list, not yet an array. Any code that reduces `x_scale_value` by slicing must guard on it being a numeric sequence (`isinstance(x_scale_value, (np.ndarray, list, tuple))`, normalizing to `np.asarray(...)` first) and otherwise leave the value untouched — never assume `np.ndarray` is the only non-string case (round 3 Codex finding #1).
- Heterodyne's fixed-child-of-a-tie validation reads the raw `initial_parameters.fixed_parameters` config dict directly, independent of when `_apply_fixed_parameters` actually mutates `space`.
- Run `make lint` and `uv run mypy xpcsjax` (advisory) before each commit; `make test-optimization`/`make test-heterodyne` must pass before moving to the next task.

---

## File Structure

| File | Responsibility |
|---|---|
| `xpcsjax/optimization/nlsq/parameter_utils.py` | **Modify.** Gains `ResolvedPhysicalParameters` (with real value substitution), `resolve_optimized_physical_parameters()`, mask-based `strip_by_mask`/`restore_by_mask_numpy`/`restore_by_mask_jax`. Also gains the relocated `strip_fixed_parameters`/`restore_fixed_parameters` (moved from `strategies/sequential.py`, **unchanged** semantics). |
| `xpcsjax/optimization/nlsq/strategies/sequential.py` | **Modify.** `strip_fixed_parameters`/`restore_fixed_parameters` definitions removed, replaced with a re-export import. No behavior change. |
| `xpcsjax/config/parameter_manager.py` | **Modify.** Fix `active_parameters: []` truthiness bug; fix a misleading docstring example. |
| `xpcsjax/optimization/nlsq/wrapper.py` | **Modify.** `NLSQWrapper.fit()` accepts a new optional `resolved_physical` parameter; strips once right before solver dispatch (with an `isinstance` guard on `x_scale_value`), restores once right after. |
| `xpcsjax/optimization/nlsq/adapter.py` | **Modify.** `NLSQAdapter.fit()` accepts the same new parameter; strips the compact vector before `curve_fit()`, restores the normalized `popt`/`pcov` locals after. |
| `xpcsjax/optimization/nlsq/core.py` | **Modify.** `fit_nlsq_jax` computes the resolved-parameters descriptor and threads it into `adapter.fit()`/`wrapper.fit()`. `fit_nlsq_cmaes` strips/restores directly around its own `model_for_cmaes` closure and two solver-phase calls (dict access for warm-start, attribute access for the `CMAESResult`). |
| `xpcsjax/config/heterodyne_parameter_space.py` | **Modify.** New standalone `_apply_fixed_parameters` step invoked last in `ParameterSpace.from_config()`; `_apply_tied_parameters` gains a fixed-child conflict check; `_apply_initial_parameters`'s early-return precondition loosened. |
| `xpcsjax/config/heterodyne_parameter_manager.py` | **Modify.** New zero-varying-parameters guard before `from_config()` returns. |
| `tests/optimization/test_parameter_utils_resolve.py` | **Create.** Unit tests for the resolver and mask-based strip/restore, including the value-substitution invariant. |
| `tests/optimization/test_fixed_parameters_integration.py` | **Create.** Real-fit integration tests: homodyne (wrapper + adapter, static + laminar_flow, `x_scale_map` array branch), CMA-ES, multistart, heterodyne. |
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
  def restore_by_mask_numpy(free_values, full_values, mask) -> np.ndarray: ...
  def restore_by_mask_jax(free_values, full_values, mask): ...
  ```
  Plus the relocated, **semantically unchanged** `strip_fixed_parameters`/`restore_fixed_parameters` (bounds-equality based — used only by `sequential.py`).

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/optimization/nlsq/strategies/sequential.py:780-880` and `xpcsjax/optimization/nlsq/parameter_utils.py` in full to confirm current structure (imports, existing `__all__`, no naming collisions with the new symbols) before writing anything.

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
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": {"D_offset": 12.5}}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()  # values[2] (D_offset) == 50.0, NOT 12.5
    resolved = resolve_optimized_physical_parameters(pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper)
    d_offset_idx = PHYSICAL_NAMES_LAMINAR.index("D_offset")
    assert resolved.free_mask[d_offset_idx] == False  # noqa: E712
    assert resolved.free_mask.sum() == 6
    assert resolved.values_full[d_offset_idx] == 12.5  # NOT 50.0 -- this is the v1 bug
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
    free_result = np.array([10.0, 30.0])
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

Copy the exact existing bodies byte-for-byte (docstrings included) into `parameter_utils.py`, after its existing imports. Add `"strip_fixed_parameters"`, `"restore_fixed_parameters"` to `__all__`. Then modify `strategies/sequential.py`: delete the two function bodies, add to its import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    restore_fixed_parameters,
    strip_fixed_parameters,
)
```

Leave every call site inside `optimize_per_angle_sequential` completely unchanged.

- [ ] **Step 4: Run the relocation regression test**

Run: `uv run pytest tests/optimization/test_laminar_streaming_diag.py -v`
Expected: PASS unchanged.

- [ ] **Step 5: Add the resolver and mask-based primitives**

Add to `xpcsjax/optimization/nlsq/parameter_utils.py`:

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xpcsjax.config.types import SCALING_PARAM_NAMES

if TYPE_CHECKING:
    from xpcsjax.config.parameter_manager import ParameterManager


@dataclass
class ResolvedPhysicalParameters:
    """Which physical parameters are free vs. fixed for one NLSQ solve.

    ``free_mask`` comes from ``ParameterManager.get_optimizable_parameters()``
    (active-minus-fixed, physics-only by contract). ``values_full`` has the
    CONFIGURED fixed value substituted in at every fixed position -- it is
    NOT simply the caller's original values array.

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

    Physical-parameter-only by design (grilling round 1 Q1) -- a scaling name
    in ``fixed_parameters`` is a hard fit-time error (grilling round 1 Q5, Q8).

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


def restore_by_mask_numpy(free_values: np.ndarray, full_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Re-insert solved free values into a full-length array (post-solve only).

    NumPy indexed assignment -- safe only on already-concrete arrays. Do NOT
    call this inside a JAX-traced closure; use :func:`restore_by_mask_jax`.
    """
    result = np.array(full_values, dtype=np.float64)
    result[mask] = np.asarray(free_values)
    return result


def restore_by_mask_jax(free_values, full_values: np.ndarray, mask: np.ndarray):
    """JAX-traceable equivalent of :func:`restore_by_mask_numpy`.

    Uses immutable ``.at[].set()`` -- safe inside a function JAX traces for
    JIT/autodiff (model/residual closures passed to ``curve_fit``/CMA-ES).
    """
    import jax.numpy as jnp

    full_jnp = jnp.asarray(full_values)
    free_idx = jnp.asarray(np.where(mask)[0])
    return full_jnp.at[free_idx].set(jnp.asarray(free_values))
```

Add `"ResolvedPhysicalParameters"`, `"resolve_optimized_physical_parameters"`, `"strip_by_mask"`, `"restore_by_mask_numpy"`, `"restore_by_mask_jax"` to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_parameter_utils_resolve.py -v`
Expected: PASS, all 9 tests.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git add xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git commit -m "feat(optimization): add resolve_optimized_physical_parameters with value substitution, mask-based strip/restore"
```

---

### Task 2: Fix `active_parameters: []` truthiness bug and misleading docstring in `parameter_manager.py`

**Files:**
- Modify: `xpcsjax/config/parameter_manager.py:517` (truthiness check), `:771` (docstring example)
- Test: `tests/config/test_active_parameters_empty_list.py`

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/config/parameter_manager.py:505-525` and `:759-790` live. This task's line references were CONFIRMED by Codex across both review rounds, but confirm once more before writing the diff — a stale confirmation is still a confirmation of the past, not the present.

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

- [ ] **Step 3: Fix the truthiness check**

```python
# Before:
            if active_params_config and isinstance(active_params_config, list):
# After:
            if active_params_config is not None and isinstance(active_params_config, list):
```

- [ ] **Step 4: Fix the misleading docstring example**

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

- [ ] **Step 6: Run the existing `ParameterManager` suite for regressions**

Run: `uv run pytest tests/config/ -v -k "parameter_manager or active_parameters or fixed_parameters"`

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/config/parameter_manager.py tests/config/test_active_parameters_empty_list.py
git commit -m "fix(config): active_parameters: [] now means none-active, not absent"
```

---

### Task 3: Wire strip/restore into `wrapper.py` (implementation only — proven by Task 5)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/wrapper.py` — `NLSQWrapper.fit()` signature; a single strip point after `x_scale`/label construction and before solver dispatch; a single restore point immediately after dispatch returns.

**Interfaces:**
- Consumes: `ResolvedPhysicalParameters`, `strip_by_mask`, `restore_by_mask_numpy`, `restore_by_mask_jax` (Task 1).
- Produces: `NLSQWrapper.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)` — `None` (every caller not updated by this plan) is a complete no-op.

**This task has no standalone passing/failing test.** Per plan review round 2 (grilling), a hand-rolled direct `.fit()` call would need to replicate a meaningful slice of `core.py`'s own preprocessing — new, untested scaffolding that risks failing for reasons unrelated to the fix. The real proof that this task's code works is Task 5's `test_fixed_parameter_survives_real_fit`, parametrized over `use_adapter=[False, True]` and `analysis_mode=["static_isotropic", "laminar_flow"]`. This task's own verification is lint + an import smoke check (Step 5).

**Why the insertion points are exactly here (confirmed via source read):** `wrapper.py:1974-1983` applies forward shear transforms to `validated_params`/`nlsq_bounds` (value-only, does not change vector length; a no-op for static modes) — must run *before* reduction, since it indexes by `physical_index_map` computed for the full vector. `wrapper.py:1998-2022` then builds `param_labels` (mode-dependent) and `x_scale_value` (default the **string** `"jac"`; an array only when `optimization.nlsq.x_scale_map` config is set) — reduction must happen *after* this. `wrapper.py:2155-2172` dispatches into `_execute_optimization_with_fallback(..., wrapped_residual_fn=..., validated_params=..., nlsq_bounds=..., x_scale_value=..., ...)`, returning `popt, pcov, info, recovery_actions, convergence_status` (confirmed exact unpack). On the way out, `:2344-2359` inverse-transforms `popt`/`pcov` (full-length expected), `:2382-2393` computes residuals from `popt`, `:2524-2541` may expand a compressed scaling mode — restoring `popt`/`pcov` to full length immediately when dispatch returns means all of this runs unaware anything was reduced.

- [ ] **Step 0/1: Re-verify before editing**

Read `xpcsjax/optimization/nlsq/wrapper.py` lines 1953-2200 (strip region) and 2340-2545 (restore region) in full to confirm the exact current variable names (`validated_params`, `nlsq_bounds`, `param_labels`, `x_scale_value`, `wrapped_residual_fn`, `popt`, `pcov`) and the exact 5-value unpack of `_execute_optimization_with_fallback(...)` immediately before writing the diff.

- [ ] **Step 2: Add `resolved_physical` to `NLSQWrapper.fit()`'s signature and import**

```python
    def fit(
        self,
        data: Any, config: Any, initial_params: "np.ndarray | None" = None,
        bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
        analysis_mode: AnalysisMode = AnalysisMode.STATIC_ISOTROPIC,
        per_angle_scaling: bool = True, diagnostics_enabled: bool = False,
        shear_transforms: "dict[str, Any] | None" = None,
        per_angle_scaling_initial: "dict[str, list[float]] | None" = None,
        *, on_iteration: "Callable[[int, float], None] | None" = None,
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

(Signature confirmed exact via source read — every existing parameter name/order/default preserved; `resolved_physical` added as the new final keyword-only parameter.)

Add to `wrapper.py`'s import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    restore_by_mask_jax,
    restore_by_mask_numpy,
    strip_by_mask,
)
```

- [ ] **Step 3: Strip once, immediately before dispatch — with the `x_scale_value` string guard**

Insert immediately after the `x_scale_value`/`param_labels` construction block (~line 2022) and before the dispatch call (~line 2155):

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
            # x_scale_value defaults to the STRING "jac" -- only a numeric
            # sequence (array OR a manual list/tuple from config x_scale,
            # round 3 Codex finding #1) needs reducing.
            if isinstance(x_scale_value, (np.ndarray, list, tuple)):
                x_scale_value = np.asarray(x_scale_value, dtype=np.float64)
                x_scale_value = np.concatenate(
                    [x_scale_value[:-n_physical], strip_by_mask(x_scale_value[-n_physical:], _phys_free_mask)]
                )
            _fixed_physical_full = resolved_physical.values_full
            _base_wrapped_residual_fn = wrapped_residual_fn

            # CRITICAL (round 3 Codex finding #3): the real solver-facing
            # closure signature is f(xdata, *params) -- xdata FIRST, physical
            # params UNPACKED as individual scalar args, NOT one array.
            # wrapper.py:3862's own docstring and the call site at
            # wrapper.py:2388 (`base_residual_fn(xdata, *popt)`) both confirm
            # this. A closure that takes the params array first (as v3's
            # first draft did) silently swaps xdata and params -- wrong
            # physics, not a crash, so it would NOT be caught by lint or an
            # import smoke check. Re-verify this signature live (Step 0)
            # before writing -- do not trust this quote in isolation.
            def wrapped_residual_fn(xdata, *params):
                n_prefix = len(params) - int(_phys_free_mask.sum())
                params_array = jnp.stack(params)
                full_physical = restore_by_mask_jax(
                    params_array[n_prefix:], _fixed_physical_full, _phys_free_mask,
                )
                full_params = (*params[:n_prefix], *[full_physical[i] for i in range(n_physical)])
                return _base_wrapped_residual_fn(xdata, *full_params)
```

- [ ] **Step 4: Restore once, immediately after dispatch returns**

Immediately after `popt, pcov, info, recovery_actions, convergence_status = self._execute_optimization_with_fallback(...)` returns, insert, *before* any inverse-transform code runs:

```python
        if resolved_physical is not None and _phys_free_mask is not None and not _phys_free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            n_prefix = len(popt) - int(_phys_free_mask.sum())
            full_physical = restore_by_mask_numpy(popt[n_prefix:], resolved_physical.values_full, _phys_free_mask)
            popt = np.concatenate([popt[:n_prefix], full_physical])
            if pcov is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [n_prefix + i for i, free in enumerate(_phys_free_mask) if free]
                full_cov[np.ix_(free_idx, free_idx)] = pcov
                pcov = full_cov
```

- [ ] **Step 5: Import smoke check and lint (no dedicated test — see task header)**

```bash
uv run python -c "from xpcsjax.optimization.nlsq.wrapper import NLSQWrapper; print('ok')"
uv run ruff check xpcsjax/optimization/nlsq/wrapper.py
```
Expected: `ok`, no lint errors. Real verification happens in Task 5.

- [ ] **Step 6: Run the wrapper regression suite**

Run: `uv run pytest tests/optimization/test_phase5_model_function_modes.py -v` (and `grep -rl "NLSQWrapper(" tests/optimization/` for the full list)
Expected: PASS unchanged — `resolved_physical=None` default preserves the old path.

- [ ] **Step 7: Verify large-dataset memory-routing tiers inherit the strip**

Read `xpcsjax/optimization/nlsq/strategies/executors.py`'s dispatch logic and `xpcsjax/optimization/nlsq/fallback_chain.py:321-424` to confirm they receive the already-stripped `validated_params`/`nlsq_bounds` with no separate vector reconstruction. Then:

```bash
ls tests/optimization/ | grep -iE "hybrid_streaming|stratified_ls|out_of_core|chunking"
uv run pytest tests/optimization/test_strategy_chunking.py -v  # adjust filenames to whatever exists
```
If this read finds an independent vector construction anywhere in this chain, stop and add a sub-task before proceeding.

- [ ] **Step 8: Commit**

```bash
git add xpcsjax/optimization/nlsq/wrapper.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQWrapper.fit()"
```

---

### Task 4: Wire strip/restore into `adapter.py` (implementation only — proven by Task 5)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/adapter.py` — `NLSQAdapter.fit()` signature; strip before `curve_fit()`; restore after the existing tuple-vs-object result-normalization block.

**Interfaces:**
- Consumes: same as Task 3.
- Produces: `NLSQAdapter.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)`.

**No standalone test — same reasoning as Task 3.** Proven by Task 5's `use_adapter=True` parametrization.

**Confirmed (three times now, via source read): adapter.py does NOT expand compact→per-angle between model construction and `curve_fit()`.** `initial_params`/`bounds` are passed to `curve_fit()` exactly as received — strip operates directly on the compact vector.

- [ ] **Step 0/1: Re-verify before editing**

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

Restore must run *after* this entire block, operating on the local `popt`/`pcov` — never mutating `result`.

- [ ] **Step 2: Add `resolved_physical` to `NLSQAdapter.fit()`'s signature and import**

```python
    def fit(
        self,
        data: Any, config: Any, initial_params: "np.ndarray | None" = None,
        bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
        analysis_mode: AnalysisMode = AnalysisMode.STATIC_ISOTROPIC,
        per_angle_scaling: bool = True, diagnostics_enabled: bool = False,
        shear_transforms: "dict[str, Any] | None" = None,
        per_angle_scaling_initial: "dict[str, list[float]] | None" = None,
        anti_degeneracy_controller: "Any | None" = None,
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

(Confirmed exact via source read — no `*` keyword-only marker on this method, unlike `NLSQWrapper.fit()`; `resolved_physical` added as the new final parameter.) Add the same import as Task 3, Step 2.

- [ ] **Step 3: Strip before `curve_fit()`, JAX-safe restore inside `model_func`**

Immediately after `model_func, cache_hit, jit_compiled = self._build_model_function(...)` (~line 1342) and before building `fit_kwargs`:

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
                full_physical = restore_by_mask_jax(jnp.asarray(params[n_prefix:]), _fixed_physical_full, _phys_free_mask)
                full_params = (*params[:n_prefix], *[full_physical[i] for i in range(n_physical)])
                return _base_model_func(x, *full_params)
```

- [ ] **Step 4: Restore after result normalization**

Immediately after the tuple-vs-object extraction block quoted in Step 1 (runs regardless of which branch fired), before `NLSQAdapter.fit()` constructs its `OptimizationResult`:

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

- [ ] **Step 5: Import smoke check and lint**

```bash
uv run python -c "from xpcsjax.optimization.nlsq.adapter import NLSQAdapter; print('ok')"
uv run ruff check xpcsjax/optimization/nlsq/adapter.py
```

- [ ] **Step 6: Run the adapter regression suite**

Run: `uv run pytest tests/optimization/test_adapter_info_extraction.py tests/optimization/test_adapter_cost_default.py tests/optimization/test_adapter_flatten_phi_order.py -v`
Expected: PASS unchanged.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/optimization/nlsq/adapter.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQAdapter.fit()"
```

---

### Task 5: Wire the resolver into `core.py::fit_nlsq_jax` — the real end-to-end proof for Tasks 3 and 4

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_jax`, after the bounds-construction block, and at both dispatch call sites).

**Interfaces:**
- Consumes: `resolve_optimized_physical_parameters` (Task 1); `NLSQAdapter.fit`/`NLSQWrapper.fit` now accepting `resolved_physical` (Tasks 3-4 — **must both be committed first**).

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/optimization/nlsq/core.py` lines 404-465 (bounds construction) and the two dispatch call sites (`adapter.fit(...)` ~line 527, `NLSQWrapper(...).fit(...)` further down) live before editing.

- [ ] **Step 1: Compute the resolved descriptor**

Insert immediately after the bounds-construction block (ending ~`bounds = (lower_bounds, upper_bounds)`), guarded for the `HAS_PARAMETER_MANAGER=False` fallback branch:

```python
    resolved_physical = None
    if HAS_PARAMETER_MANAGER:
        from xpcsjax.optimization.nlsq.parameter_utils import resolve_optimized_physical_parameters

        physical_names = _get_physical_param_names(analysis_mode)
        full_names = _get_param_names(analysis_mode)
        physical_idx = [full_names.index(name) for name in physical_names]
        resolved_physical = resolve_optimized_physical_parameters(
            param_manager, physical_names,
            values_full=np.asarray(x0)[physical_idx],
            lower_full=lower_bounds[physical_idx],
            upper_full=upper_bounds[physical_idx],
        )
```

- [ ] **Step 2: Thread `resolved_physical` into both dispatch call sites**

Add `resolved_physical=resolved_physical` as a new keyword to both the `adapter.fit(...)` call and the `NLSQWrapper(...).fit(...)` call.

- [ ] **Step 3: Write the end-to-end integration tests (mode-parametrized)**

Create `tests/optimization/test_fixed_parameters_integration.py`:

```python
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
    "D0": 8000.0, "alpha": -1.2, "D_offset": 50.0,
    "gamma_dot_t0": 0.01, "beta": 0.1, "gamma_dot_t_offset": 0.0, "phi0": 0.0,
}
TRUE_PHYSICAL_STATIC = {"D0": 8000.0, "alpha": -1.2, "D_offset": 50.0}
CONTRAST, OFFSET, Q, L, DT = 0.3, 0.8, 0.005, 2_000_000.0, 0.001

_PHYSICAL_BY_MODE = {
    "laminar_flow": TRUE_PHYSICAL_LAMINAR,
    "static_isotropic": TRUE_PHYSICAL_STATIC,
    "static_anisotropic": TRUE_PHYSICAL_STATIC,  # same 3-name physical set as static_isotropic (round 3 Codex finding #11 -- spec names both)
}
_ALL_PHYSICAL_NAMES = list(TRUE_PHYSICAL_LAMINAR.keys())  # static forward-sim still uses the full 7-slot kernel


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
    phi = np.array([0.0, 45.0, 90.0])[:n_phi]
    params_vec = jnp.array([full_physical[name] for name in _ALL_PHYSICAL_NAMES])
    g2 = np.stack(
        [
            np.asarray(
                compute_g2_scaled(
                    params_vec, jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(p), Q, L, CONTRAST, OFFSET, DT,
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


def _config(analysis_mode="laminar_flow", fixed_parameters=None, active_parameters=None, extra_initial=None, extra_top=None):
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


@pytest.mark.parametrize("use_adapter", [False, True])
@pytest.mark.parametrize("analysis_mode", ["static_isotropic", "static_anisotropic", "laminar_flow"])
def test_fixed_parameter_survives_real_fit(analysis_mode, use_adapter):
    data = _synthetic_data(analysis_mode)
    fixed_value = 37.5  # different from the true simulated value (50.0)
    cm = ConfigManager(config_override=_config(analysis_mode, fixed_parameters={"D_offset": fixed_value}))
    result = fit_nlsq_jax(data, cm, use_adapter=use_adapter)
    names = ["contrast", "offset", *_PHYSICAL_BY_MODE[analysis_mode].keys()]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = names.index("D_offset")
    assert abs(params[d_offset_idx] - fixed_value) < 1e-9
    if result.uncertainties is not None:
        assert np.asarray(result.uncertainties).ravel()[d_offset_idx] == 0.0


def test_fixed_scaling_parameter_raises():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow", fixed_parameters={"contrast": 0.5}))
    with pytest.raises(ValueError, match="contrast"):
        fit_nlsq_jax(data, cm, use_adapter=False)


def test_unset_fixed_parameters_is_a_noop():
    data = _synthetic_data("laminar_flow")
    cm = ConfigManager(config_override=_config("laminar_flow"))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    assert np.asarray(result.parameters).size == 9


def test_restricted_active_parameters_real_fit():
    """A physical parameter excluded via active_parameters must not move from
    its initial value -- distinct mechanism entry point from fixed_parameters,
    same underlying resolver."""
    data = _synthetic_data("laminar_flow")
    active = ["D0", "alpha", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]  # excludes D_offset
    cm = ConfigManager(config_override=_config("laminar_flow", active_parameters=active))
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL_LAMINAR.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 50.0) < 1e-9  # unchanged from its initial value


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
    names = ["contrast", "offset", *TRUE_PHYSICAL_LAMINAR.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 37.5) < 1e-9
```

Adjust the `optimization.nlsq.x_scale_map` config shape in the last test once Step 0's re-read of `wrapper.py:1998-2022`/`build_per_parameter_x_scale`'s config-reading code confirms the exact expected keys/format (a `{param_name: scale}` dict was confirmed at the config-template level, `xpcsjax_laminar_flow.yaml:362-365`, but the exact nesting under `optimization.nlsq` and any name-mapping needs a live check against `build_per_parameter_x_scale`'s implementation before finalizing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -v`
Expected: PASS, all parametrizations (6 mode×adapter combinations plus the 4 single tests = 10 total).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): compute and thread resolved physical-parameter descriptor in fit_nlsq_jax"
```

---

### Task 6: Wire strip/restore into `core.py::fit_nlsq_cmaes`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_cmaes`).

**Interfaces:**
- Consumes: same as Task 3. `fit_nlsq_cmaes` calls neither `NLSQAdapter` nor `NLSQWrapper` — it builds its own `model_for_cmaes` JAX closure and calls a distinct CMA-ES-family `wrapper` object's `_run_nlsq_refinement`/`fit` methods.

**Confirmed two-phase sequencing and result shapes (source-verified, corrected in v3):**
- Phase 1: `wrapper._run_nlsq_refinement(model_func=model_for_cmaes, ..., p0=x0, bounds=bounds, ...)` returns a **plain `dict`** (`cmaes_wrapper.py:628-661`, keys `popt`/`pcov`/`infodict`/`mesg`/`ier`) — `warmstart_result["popt"]`/`["pcov"]` dict access is correct.
- `x0 = np.asarray(nlsq_warmstart_params)` reassigns `x0` for phase 2; **`bounds` is never reassigned across phases.**
- Phase 2: `cmaes_result = wrapper.fit(model_func=model_for_cmaes, ..., p0=x0, bounds=bounds, ...)` returns a **`CMAESResult` dataclass** (`cmaes_wrapper.py:461-492` — `.parameters`/`.covariance` **attributes**, not dict keys; not frozen, so attribute assignment works). `cmaes_result["popt"]` would raise `TypeError: 'CMAESResult' object is not subscriptable` — v2's plan had this wrong.

Strip `x0`/`bounds` **once**, before phase 1; do **not** restore between phases; restore only the final `cmaes_result.parameters`/`.covariance` once, after phase 2.

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/optimization/nlsq/core.py` lines 1900-2500 live, and `xpcsjax/optimization/nlsq/cmaes_wrapper.py:461-492,628-661,830-860` to reconfirm `CMAESResult`'s exact field names and `_run_nlsq_refinement`'s exact return dict keys, before writing the diff.

- [ ] **Step 1: Write the failing test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_cmaes_fit():
    data = _synthetic_data("laminar_flow")
    config = _config("laminar_flow", fixed_parameters={"D_offset": 37.5}, extra_top={"optimization": {"nlsq": {"cmaes": {"enable": True}}}})
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL_LAMINAR.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 37.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`

- [ ] **Step 3: Compute `physical_names` explicitly and strip once before phase 1**

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

- [ ] **Step 4: Wrap `model_for_cmaes` with a JAX-safe restore (used by both phases)**

Immediately after `model_for_cmaes` is defined (~line 2364):

**Confirmed live (round 3 Codex finding #4): `model_for_cmaes`'s real signature is `model_for_cmaes(xdata_unused, *params)`** — same `f(xdata, *params)` convention as Task 3's `wrapped_residual_fn`, not `(params_array, *args)`. Re-verify this at Step 0 before writing; a swapped signature here is a silent wrong-physics bug, not a crash.

```python
    if _cmaes_phys_free_mask is not None:
        _base_model_for_cmaes = model_for_cmaes
        _fixed_physical_full = resolved_physical.values_full
        _n_physical_cmaes = n_physical

        def model_for_cmaes(xdata_unused, *params):
            n_prefix = len(params) - int(_cmaes_phys_free_mask.sum())
            params_array = jnp.stack(params)
            full_physical = restore_by_mask_jax(
                params_array[n_prefix:], _fixed_physical_full, _cmaes_phys_free_mask,
            )
            full_params = (*params[:n_prefix], *[full_physical[i] for i in range(_n_physical_cmaes)])
            return _base_model_for_cmaes(xdata_unused, *full_params)
```

- [ ] **Step 5: Restore only the final result, after phase 2 — `CMAESResult` uses attribute access, and after the COMPLETE skip/no-skip branch**

**Confirmed (round 3 Codex finding #5): CMA-ES can be auto-skipped** — when skipped, `core.py` (~line 2440-2470) constructs a `CMAESResult` **directly**, without ever calling `wrapper.fit(...)`. Placing the restore only immediately after `wrapper.fit(...)` misses this branch entirely, leaving a reduced-length result whenever CMA-ES auto-skips. Re-read `core.py:2430-2490` at Step 0 to find the exact point after **both** branches (`if <skip condition>: cmaes_result = CMAESResult(...)` / `else: cmaes_result = wrapper.fit(...)`) have converged back to a single `cmaes_result` variable, and insert the restore there — not immediately after the `wrapper.fit(...)` call site alone:

```python
        if _cmaes_phys_free_mask is not None:
            n_prefix = len(cmaes_result.parameters) - int(_cmaes_phys_free_mask.sum())
            full_popt = restore_by_mask_numpy(
                cmaes_result.parameters[n_prefix:], resolved_physical.values_full, _cmaes_phys_free_mask,
            )
            cmaes_result.parameters = np.concatenate([cmaes_result.parameters[:n_prefix], full_popt])
            if cmaes_result.covariance is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [
                    n_prefix + i for i, free in enumerate(_cmaes_phys_free_mask) if free
                ]
                full_cov[np.ix_(free_idx, free_idx)] = cmaes_result.covariance
                cmaes_result.covariance = full_cov
```

Do **not** insert a restore step between phase 1 (`warmstart_result`, dict-keyed) and phase 2 (`cmaes_result`, attribute-based dataclass) — `nlsq_warmstart_params` stays reduced-length and feeds directly into phase 2's still-reduced `bounds`, which is correct.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`

- [ ] **Step 7: Run the CMA-ES regression suite**

Run: `uv run pytest tests/optimization/test_cmaes_trigger.py tests/optimization/test_heterodyne_cmaes_seed.py -v` (and `grep -rl "fit_nlsq_cmaes" tests/optimization/` for the full list)

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): honor fixed physical parameters in fit_nlsq_cmaes"
```

---

### Task 7: Narrow `_SingleFitWorker`'s sampling to the free physical subset, and prove multistart honors `fixed_parameters`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py::_SingleFitWorker`
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append)

**Interfaces:**
- Consumes: `resolve_optimized_physical_parameters` (Task 1); `fit_nlsq_jax` (Task 5).

**Round 3 revision — this is now a required implementation step, not a contingent fallback.** v3 treated the worker's sampling-scope narrowing as optional (a "Task 7b" to run only if a correctness test failed), reasoning that `_SingleFitWorker.__call__`'s recursive `fit_nlsq_jax(..., _skip_global_selection=True)` call already re-derives `resolved_physical` and overwrites any "fixed" slot's value regardless of what the worker sampled there — a correctness argument that no review round has refuted. But the spec's Component 2 explicitly names this as required behavior (not merely a correctness safety net — narrowing avoids wasting LHS sampling budget exploring dimensions that are locked anyway, and keeps multistart's own diversity metric meaningful). Implement it directly; the test below still exists to prove the end-to-end result, but the worker change is not conditioned on it failing first.

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/optimization/nlsq/core.py:1330-1436` (`_SingleFitWorker`) live to confirm the current `__init__`/`__call__` shape and the recursive `fit_nlsq_jax(..., _skip_global_selection=True)` call site, before writing the diff.

- [ ] **Step 1: Narrow the worker's sampling to the free physical subset**

Modify `_SingleFitWorker.__call__` so the sampled `start_params` array (currently zipped against the full `_get_param_names(self.analysis_mode)` list) is instead interpreted against the *free* subset: reconstruct a `ParameterManager` from `self.config_dict`/`self.analysis_mode` (mirroring the reconstruction the worker already does for `ConfigManager`), call `resolve_optimized_physical_parameters` to get `free_mask`/`values_full`, accept only the free-position entries from `start_params`, and fill fixed positions from `values_full` before constructing `params_dict` — mirroring Task 5, Step 1's pattern locally within the worker. This is defense-in-depth on top of the recursive call's own correctness, not a replacement for it.

- [ ] **Step 2: Write the proof test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_multistart_fit():
    data = _synthetic_data("laminar_flow")
    config = _config("laminar_flow", fixed_parameters={"D_offset": 37.5}, extra_top={"optimization": {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}})
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    names = ["contrast", "offset", *TRUE_PHYSICAL_LAMINAR.keys()]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 37.5) < 1e-9
```

- [ ] **Step 3: Run the test**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_multistart_fit -v`
Expected: PASS.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): narrow _SingleFitWorker sampling to free physical subset for fixed_parameters"
```

---

### Task 8: Heterodyne — loosen `_apply_initial_parameters`'s precondition, extract `_apply_fixed_parameters` as its own function

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::_apply_initial_parameters` and its `active_parameters` block.
- Test: `tests/config/test_heterodyne_fixed_parameters.py`

**Interfaces:**
- Consumes: `space.vary`, `space.values`, `_INBOUND_NAME_ALIAS`, `PARAMETER_NAME_MAPPING`, `coerce_finite_float` (all existing).
- Produces: `_apply_fixed_parameters(space, config)` — a **new, standalone** function, invoked separately by Task 9's orchestration change, so it can run *after* every overlay.

**Confirmed exact config-application order in `ParameterSpace.from_config()`:**
```python
_apply_initial_parameters(space, config)       # :342
_apply_parameter_space_bounds(space, config)   # :351
# inline grouped parameters.{group}.{param} overlay  :353-465
_apply_tied_parameters(space, config)          # :470
```
Any of these can set `space.vary[name] = True` after `_apply_initial_parameters` runs — why `fixed_parameters` must be its own step invoked *last* (Task 9).

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/config/heterodyne_parameter_space.py` lines 294-360 (`from_config`'s orchestration) and 481-557 (`_apply_initial_parameters`) live before editing.

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_heterodyne_fixed_parameters.py`:

```python
"""Heterodyne fixed_parameters: vary=False + value write, wins over EVERY
overlay including parameter_space.bounds and grouped parameters, including
scaling names."""

import pytest

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager


def _config(fixed_parameters, **extra_initial):
    initial = {"fixed_parameters": fixed_parameters, **extra_initial}
    return {"analysis_mode": "two_component", "initial_parameters": initial}


def test_fixed_parameter_value_is_honored_not_flat_list_value():
    config = _config(fixed_parameters={"D0_ref": 999.0}, parameter_names=["D0_ref"], values=[10000.0])
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
    """v2-review-identified gap: fixed_parameters must win even against
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

(Re-read `_apply_parameter_space_bounds` in Step 0 to confirm the exact `parameter_space.bounds` entry shape for setting `vary=True` on a named parameter — adjust the last test's config block if it differs, keeping the assertion intent unchanged.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: FAIL on all 6.

- [ ] **Step 3: Loosen `_apply_initial_parameters`'s early-return precondition**

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

Wrap the existing flat-value-application loop in `if has_flat_values:`. The `active_parameters` block below must run unconditionally.

- [ ] **Step 4: Extract `_apply_fixed_parameters` as a standalone function**

Add as a new top-level function, after `_apply_initial_parameters`:

```python
def _apply_fixed_parameters(space: ParameterSpace, config: dict[str, Any]) -> None:
    """Apply ``initial_parameters.fixed_parameters`` to *space*.

    Sets BOTH the vary flag and the value -- expand_varying_to_full() fills
    non-varying positions from space.values, so the value write is required.

    MUST run LAST in ParameterSpace.from_config()'s call sequence -- after
    _apply_initial_parameters, _apply_parameter_space_bounds, the grouped
    parameters.* overlay, and _apply_tied_parameters -- so a fixed parameter
    always wins regardless of what any other overlay sets. Not scoped to
    physical-only (grilling round 1 Q7 -- heterodyne's fixed_parameters
    mirrors active_parameters' existing scope, which already includes
    contrast/offset via ALL_PARAM_NAMES_WITH_SCALING).
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
        space.values[canonical] = coerce_finite_float(value, context=f"initial_parameters.fixed_parameters[{canonical!r}]")
        space.vary[canonical] = False
        logger.debug("fixed_parameters: fixed %s = %.6g", canonical, value)
```

- [ ] **Step 5: Run the value/scaling/no-flat-values tests**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -k "not LATER" -v`
Expected: PASS for the first 5. `test_fixed_wins_over_LATER_parameter_space_bounds_overlay` still FAILS — expected, not wired in yet (Task 9).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_space.py
git add xpcsjax/config/heterodyne_parameter_space.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): extract standalone _apply_fixed_parameters for heterodyne, loosen precondition"
```

---

### Task 9: Heterodyne — invoke `_apply_fixed_parameters` last, and validate the tied-child-fixed conflict

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::from_config` (call-sequence) and `_apply_tied_parameters`.
- Test: `tests/config/test_heterodyne_fixed_parameters.py` (append)

**Interfaces:**
- Consumes: `_apply_fixed_parameters` (Task 8).
- Produces: `fixed_parameters` wins against every overlay; `ValueError` when a tied child also appears in `fixed_parameters`.

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/config/heterodyne_parameter_space.py:341-470` (call sequence) and `:559-680` (`_apply_tied_parameters`'s validation chain) live before editing.

- [ ] **Step 1: Wire `_apply_fixed_parameters` to run last**

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

- [ ] **Step 3: Write the tied-child-fixed conflict test**

Append to `tests/config/test_heterodyne_fixed_parameters.py`:

```python
def test_tied_child_also_fixed_raises():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {"tied_parameters": {"D0_ref": "D0_sample"}, "fixed_parameters": {"D0_ref": 5.0}},
    }
    with pytest.raises(ValueError, match="tied_parameters.*D0_ref.*fixed_parameters"):
        ParameterManager.from_config(config)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_tied_child_also_fixed_raises -v`

- [ ] **Step 5: Add the conflict check to `_apply_tied_parameters`**

Confirmed exact validation order and message style:

```python
if child not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{child}'. Valid names: {list(ALL_PARAM_NAMES)}")
if parent not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{parent}'. Valid names: {list(ALL_PARAM_NAMES)}")
if child == parent: raise ValueError(f"tied_parameters: '{child}' cannot be tied to itself")
if parent in children: raise ValueError(f"tied_parameters: '{parent}' is itself a tied child (tied to '{tied_translated[parent]}') -- chained ties are not supported. Tie '{child}' directly to '{tied_translated[parent]}' instead.")
if not space.vary.get(parent, False): raise ValueError(f"tied_parameters: parent '{parent}' is not varying (fixed via active_parameters or vary: false) -- tying '{child}' to a fixed parent is not supported; fix '{child}' directly instead via active_parameters.")
```

Add, immediately after the `parent in children` check, inside the same per-pair validation loop — reads the **raw config dict** directly:

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

(Compute `fixed_raw`/`fixed_names` once before the per-pair loop begins.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 7.

- [ ] **Step 7: Run the existing tied-parameters regression suite**

Run: `uv run pytest tests/config/test_heterodyne_parameter_manager_tied.py -v`

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
- Consumes: `ParameterSpace.n_varying`/`.varying_names` (existing, scaling-inclusive; do **not** use `ParameterManager.varying_indices`, physics-only).
- Produces: `ValueError` before `ParameterManager.from_config()` returns, when `space.n_varying == 0`.

**Confirmed:** `ParameterManager.from_config()` returns `cls(space=space)` directly — no intermediate `instance` variable.

- [ ] **Step 0: Re-verify before editing**

Read `xpcsjax/config/heterodyne_parameter_manager.py:560-580` live to reconfirm the exact `return cls(space=space)` statement.

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

- [ ] **Step 3: Add the guard immediately before the return**

```python
        if space.n_varying == 0:
            raise ValueError(
                "Nothing left to optimize: active_parameters/fixed_parameters "
                "combine to leave zero varying parameters (physics and "
                "scaling combined). Free at least one parameter."
            )
        return cls(space=space)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 9.

- [ ] **Step 5: Run the full heterodyne config regression suite**

Run: `uv run pytest tests/config/ -k heterodyne -v`

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_manager.py
git add xpcsjax/config/heterodyne_parameter_manager.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): guard against zero varying parameters in heterodyne ParameterManager"
```

---

### Task 11: Heterodyne real-fit integration test (built around the real `_make_synthetic_heterodyne()` return shape) + full regression sweep

**Files:**
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append heterodyne case) — no production code changes.

**Interfaces:**
- Consumes: everything from Tasks 1-10.

**Confirmed real shape:** `_make_synthetic_heterodyne(n_phi=2, n_t=8, seed=0)` (`tests/optimization/test_heterodyne_hybrid_streaming.py:9-41`) returns **`(model, c2, phi)`**, built from the real `xpcsjax_two_component.yaml` template via `HeterodyneModel.from_config(...)`, **not** a name-keyed `(data_dict, true_physical_dict)` pair. `c2` is **noise-free**.

**Two round-3 corrections to the naive approach:**

1. **The public `fit_nlsq()` dispatcher ignores `t1`/`t2`/`q`/`dt`/`sigma` in the data dict entirely.** Confirmed via source read: `xpcsjax/optimization/nlsq/__init__.py::_fit_nlsq_heterodyne` builds a **fresh** `HeterodyneModel.from_config(config.config)` from the config alone, and reads only `c2`/`c2_exp`, `phi`/`phi_angles[_list]`, and optional `weights` from the data dict (`__init__.py:727-776`). A hand-built minimal `initial_parameters` config would give this fresh model *different* geometry (`t`, `q`, `dt`) than the one `_make_synthetic_heterodyne()` used to generate `c2` — a physics mismatch, not a config typo. **Fix: build the test's config by loading the same `xpcsjax_two_component.yaml` template `_make_synthetic_heterodyne()` uses (find its exact loading mechanism at Step 0), and override only `initial_parameters.fixed_parameters` plus `parameter_names`/`values` on top of it — leave every geometry/`t`/`q`/`dt` section from the template untouched, so the fitted model and the model that generated `c2` are provably the same.**
2. **`result.parameters` is not a full 14-name `ALL_PARAM_NAMES`-ordered vector.** The heterodyne solver returns a *reduced* varying-parameter vector whose layout depends on the per-angle mode; production code explicitly expands it via `expand_varying_to_full` before it means anything positionally (`heterodyne_core.py:4719`). **Fix: read `result.nlsq_diagnostics.get("parameter_names")` (the same mechanism `cli/optimization_runner.py::_warn_nlsq_bound_saturation` already uses to interpret `result.parameters` positionally) to build the name→index map, instead of assuming `ALL_PARAM_NAMES` order — confirm this key/shape is still correct at Step 0.**

- [ ] **Step 0: Re-verify before editing**

Read `tests/optimization/test_heterodyne_hybrid_streaming.py:1-60` live to reconfirm `_make_synthetic_heterodyne()`'s exact signature/return and how it loads the template config (file path, `ConfigManager`/`yaml.safe_load` call — whatever it is, reuse that same loading path in the test rather than hand-building a dict). Read `xpcsjax/optimization/nlsq/__init__.py:698-780` (`_fit_nlsq_heterodyne`) live to reconfirm the data-dict contract. Read `xpcsjax/cli/optimization_runner.py::_warn_nlsq_bound_saturation` and `OptimizationResult`'s `nlsq_diagnostics` field live to reconfirm `parameter_names` is really there and in what shape for the heterodyne result path specifically (it was previously confirmed for homodyne; verify it's the same for `two_component`).

- [ ] **Step 2: Write the heterodyne real-fit tests**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def _heterodyne_config(fixed_parameters):
    """Load the SAME template _make_synthetic_heterodyne() uses, so the
    fitted model shares the exact geometry (t/q/dt) that generated c2 --
    do not hand-build a minimal config (round 3 Codex finding #8)."""
    # Replace this body with whatever _make_synthetic_heterodyne() itself
    # does to load xpcsjax_two_component.yaml, confirmed at Step 0 -- e.g.
    # ConfigManager(config_file=<template path>).config, deep-copied.
    import copy

    from xpcsjax.config import ConfigManager

    base_config = copy.deepcopy(ConfigManager(config_file=_TWO_COMPONENT_TEMPLATE_PATH).config)
    base_config["initial_parameters"]["fixed_parameters"] = fixed_parameters
    return base_config


def _fixed_value_survives(result, name):
    """Read the fitted value for `name` using the SAME parameter_names
    metadata the codebase itself uses to interpret result.parameters
    positionally (round 3 Codex finding #10 -- not full ALL_PARAM_NAMES order)."""
    diagnostics = result.nlsq_diagnostics or {}
    param_names = diagnostics.get("parameter_names")
    assert param_names is not None, "result.nlsq_diagnostics['parameter_names'] missing -- re-check Step 0"
    params = np.asarray(result.parameters).ravel()
    return params[list(param_names).index(name)]


def test_heterodyne_fixed_parameter_survives_real_fit():
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES
    from xpcsjax.optimization.nlsq import fit_nlsq

    model, c2, phi = _make_synthetic_heterodyne()
    rng = np.random.default_rng(0)
    c2_noisy = c2 + rng.normal(scale=1e-4, size=c2.shape)  # inject noise -- helper's data is noise-free
    fixed_name = "D_offset_sample"
    physical_values = model.param_manager.get_full_values()
    fixed_value = float(physical_values[list(ALL_PARAM_NAMES).index(fixed_name)])

    data = {"c2": c2_noisy, "phi": phi}  # ONLY what _fit_nlsq_heterodyne actually reads -- see Step 0/1 above
    cm = ConfigManager(config_override=_heterodyne_config({fixed_name: fixed_value}))
    result = fit_nlsq(data, cm)
    assert abs(_fixed_value_survives(result, fixed_name) - fixed_value) < 1e-6


def test_heterodyne_fixed_scaling_parameter_survives_real_fit():
    """grilling round 1 Q7 end-to-end: heterodyne can fix a scaling name, unlike homodyne."""
    from tests.optimization.test_heterodyne_hybrid_streaming import _make_synthetic_heterodyne
    from xpcsjax.optimization.nlsq import fit_nlsq

    model, c2, phi = _make_synthetic_heterodyne()
    rng = np.random.default_rng(0)
    c2_noisy = c2 + rng.normal(scale=1e-4, size=c2.shape)
    contrast, _offset = model.scaling.get_for_angle(0)

    data = {"c2": c2_noisy, "phi": phi}
    cm = ConfigManager(config_override=_heterodyne_config({"contrast": contrast}))
    result = fit_nlsq(data, cm)  # must not raise -- homodyne would raise ValueError for this, heterodyne must not
    assert result.convergence_status is not None
```

`_TWO_COMPONENT_TEMPLATE_PATH` and `_heterodyne_config`'s exact loading body must be filled in from Step 0's live read of how `_make_synthetic_heterodyne()` itself loads the template — do not guess a path or a `ConfigManager` call shape here; copy the helper's own mechanism exactly so the two configs are provably identical apart from the `fixed_parameters` override.

- [ ] **Step 2b: Best-effort fixed-parameter coverage for the inherited strategy tiers**

The spec calls for an explicit fixed-parameter-survives assertion per strategy tier (hybrid-streaming, stratified-LS, sequential, out-of-core), not just the regression-suite runs Tasks 3/9 already do. Forcing these size-based branches requires either a large enough synthetic dataset or a config-level threshold override (e.g. `optimization.stratification.target_chunk_size` set very low) — read `xpcsjax/optimization/nlsq/memory.py`/`select_nlsq_strategy` live to find the smallest reliable way to force each branch with a tiny dataset, and add one `fixed_parameters`-set test per reachable tier following the same `_fixed_value_survives`-style assertion pattern above. If a tier's branch genuinely cannot be forced without a multi-GB dataset, document that explicitly in this task rather than silently skipping it — do not claim coverage the tests don't have.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -k heterodyne -v`

- [ ] **Step 4: Run the full optimization + config + heterodyne test suites**

```bash
make test-optimization
make test-heterodyne
uv run pytest tests/config/ -v
```

- [ ] **Step 5: Full local verification (forces the CPU-microarch-gated synthetic parity tests)**

```bash
make test-full-local
```
Expected: PASS, `rtol=1e-10` unchanged on `tests/parity/test_homodyne_engine_preservation.py` and `tests/parity/test_engine_heterodyne_fit_parity.py`, and `tests/parity/test_phase5_default_no_worse.py::test_synthetic_default_averaged_no_worse_than_individual` unaffected — every golden/no-worse config has `fixed_parameters`/`active_parameters` unset.

- [ ] **Step 6: Full suite + lint**

```bash
make test
make lint
uv run mypy xpcsjax
```

- [ ] **Step 7: Final commit**

```bash
git add tests/optimization/test_fixed_parameters_integration.py
git commit -m "test(optimization): heterodyne real-fit integration tests for fixed_parameters, including scaling names"
```

---

## Deferred / explicitly out of scope

Per the spec's Out-of-scope section: the separate CLI-override bug in `cli/config_handling.py` (`--initial-*` re-introducing a fixed parameter); dead-code cleanup of `xpcsjax/core/fitting.py::ParameterSpace` and `xpcsjax/config/parameter_space.py::ParameterSpace`; the `config/manager.py::_calculate_midpoint_defaults` misleading comment. File these as separate follow-up issues — do not fold them into this plan's tasks or commits.
