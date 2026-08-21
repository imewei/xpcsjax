# NLSQ `fixed_parameters` / `active_parameters` Correctness Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `initial_parameters.fixed_parameters` actually fix a parameter during the real NLSQ solve (currently a silent no-op in all three analysis-mode families), and make `initial_parameters.active_parameters` actually restrict the optimized parameter set in `static`/`laminar_flow` modes (currently a silent no-op there too, though already correct in `two_component`).

**Architecture:** Homodyne (`static_anisotropic`/`static_isotropic`/`laminar_flow`): a new shared resolver in `parameter_utils.py` computes which *physical* parameters are free vs. fixed; `core.py`'s three entry points (`fit_nlsq_jax`, `fit_nlsq_cmaes`) thread that descriptor into `adapter.py`/`wrapper.py`/CMA-ES's own model-building code, which strip the trailing physical-parameter slice from whatever vector (compact or per-angle-expanded) actually reaches their own solver call, and restore it afterward — reusing the `strip_fixed_parameters`/`restore_fixed_parameters` primitives already proven in `strategies/sequential.py`. `fit_nlsq_multistart` needs **no code change**: it recurses into `fit_nlsq_jax` per sampled start, so correctness is inherited automatically once `fit_nlsq_jax` is fixed (verified during plan research — see Task 7). Heterodyne (`two_component`): `fixed_parameters` sets `space.vary[name] = False` **and** `space.values[name] = value` in `heterodyne_parameter_space.py::_apply_initial_parameters`, mirroring the already-correct `active_parameters` code path — every heterodyne strategy tier already shares one `ParameterManager` instance and reads `varying_names`/`expand_varying_to_full`, so this one function-level change propagates everywhere automatically.

**Tech Stack:** Python 3.12+, JAX (`JAX_ENABLE_X64=1`), upstream `nlsq>=0.6.10` (`CurveFit`/`curve_fit`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-nlsq-fixed-active-parameters-design.md` (revised twice — Codex review, then a 3-round `/grilling` session; read both revision-note sections before starting, they record *why* each design choice was made, not just what it is).

## Global Constraints

- `JAX_ENABLE_X64=1` is set by `xpcsjax/__init__.py` before any JAX import — never set it elsewhere.
- No `from module import *` (ruff `F` rule).
- `tests/parity/_golden/` goldens are pinned at `rtol=1e-10` — every code path this plan touches must be a **provable no-op** when `fixed_parameters`/`active_parameters` are unset (the template default). If any golden test's numbers move, that is a plan bug, not a golden to regenerate.
- Homodyne `fixed_parameters`/`active_parameters` are scoped to **physical parameters only** (never `contrast`/`offset`) — permanent, not a v1 cut (spec, grilling Q1).
- Heterodyne `fixed_parameters` is **not** scoped that way — it mirrors `active_parameters`'s existing scope, which already includes `contrast`/`offset` (spec, grilling Q7). Do not port the homodyne restriction to heterodyne.
- A scaling parameter named in homodyne's `fixed_parameters` is a hard `ValueError` at fit time, not a warning, not config-load time (spec, grilling Q5, Q8).
- `strategies/sequential.py` keeps its existing, tested zero-length-covariance convention when every physical parameter is fixed — every *other* new call site raises `ValueError` instead (spec, grilling Q3). Multistart's own pre-existing `check_zero_volume_bounds` → single-start-fallback convention (in `multistart.py`, unrelated to this plan's mechanism) is untouched — see Task 7.
- Run `make lint` and `uv run mypy xpcsjax` (advisory) before each commit; `make test-optimization`/`make test-heterodyne` must pass before moving to the next task.

---

## File Structure

| File | Responsibility |
|---|---|
| `xpcsjax/optimization/nlsq/parameter_utils.py` | **Modify.** Gains `ResolvedPhysicalParameters` dataclass, `resolve_optimized_physical_parameters()`, and the relocated `strip_fixed_parameters`/`restore_fixed_parameters` (moved from `strategies/sequential.py`). |
| `xpcsjax/optimization/nlsq/strategies/sequential.py` | **Modify.** `strip_fixed_parameters`/`restore_fixed_parameters` definitions removed, replaced with a re-export import. No behavior change. |
| `xpcsjax/config/parameter_manager.py` | **Modify.** Fix `active_parameters: []` truthiness bug; fix a misleading docstring example. |
| `xpcsjax/optimization/nlsq/core.py` | **Modify.** `fit_nlsq_jax` computes the resolved-parameters descriptor and passes it to `NLSQAdapter.fit()`/`NLSQWrapper.fit()`. `fit_nlsq_cmaes` applies strip/restore directly around its own model closure and solver calls (it doesn't use adapter/wrapper). |
| `xpcsjax/optimization/nlsq/adapter.py` | **Modify.** `NLSQAdapter.fit()` accepts the new optional descriptor, strips/restores around its `curve_fit()` call. |
| `xpcsjax/optimization/nlsq/wrapper.py` | **Modify.** `NLSQWrapper.fit()` accepts the new optional descriptor, strips/restores right after its Step 6.6 per-angle expansion, before handoff to `strategies/executors.py`. |
| `xpcsjax/config/heterodyne_parameter_space.py` | **Modify.** `_apply_initial_parameters` gains a `fixed_parameters` block; `_apply_tied_parameters` gains a fixed-child conflict check; early-return precondition loosened. |
| `xpcsjax/config/heterodyne_parameter_manager.py` | **Modify.** New zero-varying-parameters guard. |
| `tests/optimization/test_parameter_utils_resolve.py` | **Create.** Unit tests for the new resolver. |
| `tests/optimization/test_fixed_parameters_integration.py` | **Create.** Real-fit integration tests, homodyne (all paths) + heterodyne. |
| `tests/config/test_active_parameters_empty_list.py` | **Create.** Regression test for the truthiness fix. |
| `tests/config/test_heterodyne_fixed_parameters.py` | **Create.** Heterodyne-specific: value-write, tied conflict, zero-varying guard, scaling-parameter fixing. |

---

### Task 1: Relocate strip/restore primitives and add the resolver to `parameter_utils.py`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/parameter_utils.py`
- Modify: `xpcsjax/optimization/nlsq/strategies/sequential.py:804-880`
- Test: `tests/optimization/test_parameter_utils_resolve.py`

**Interfaces:**
- Consumes: `xpcsjax.config.parameter_manager.ParameterManager.get_optimizable_parameters() -> list[str]` (existing, unmodified — physics-only, active-minus-fixed).
- Produces (used by Tasks 3-6):
  ```python
  @dataclass
  class ResolvedPhysicalParameters:
      physical_names: list[str]
      values_full: np.ndarray
      lower_full: np.ndarray
      upper_full: np.ndarray
      free_mask: np.ndarray

  def resolve_optimized_physical_parameters(
      param_manager: ParameterManager,
      physical_names: list[str],
      values_full: np.ndarray,
      lower_full: np.ndarray,
      upper_full: np.ndarray,
      *,
      allow_all_fixed: bool = False,
  ) -> ResolvedPhysicalParameters: ...

  def strip_fixed_parameters(
      initial_params: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

  def restore_fixed_parameters(
      free_result: np.ndarray, fixed_values: np.ndarray, free_mask: np.ndarray,
  ) -> np.ndarray: ...
  ```

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/test_parameter_utils_resolve.py`:

```python
"""Tests for resolve_optimized_physical_parameters (fixed/active parameter resolution)."""

import numpy as np
import pytest

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    resolve_optimized_physical_parameters,
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
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper,
    )
    assert isinstance(resolved, ResolvedPhysicalParameters)
    np.testing.assert_array_equal(resolved.free_mask, np.ones(7, dtype=bool))
    np.testing.assert_array_equal(resolved.values_full, values)
    np.testing.assert_array_equal(resolved.lower_full, lower)
    np.testing.assert_array_equal(resolved.upper_full, upper)


def test_fixed_parameter_excluded_from_free_mask():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"D_offset": 0.0}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper,
    )
    d_offset_idx = PHYSICAL_NAMES_LAMINAR.index("D_offset")
    assert resolved.free_mask[d_offset_idx] == False  # noqa: E712
    assert resolved.free_mask.sum() == 6


def test_scaling_name_in_fixed_parameters_raises():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"fixed_parameters": {"contrast": 0.5}},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="contrast"):
        resolve_optimized_physical_parameters(
            pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper,
        )


def test_all_physical_fixed_raises_by_default():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|all.*fixed"):
        resolve_optimized_physical_parameters(
            pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper,
        )


def test_all_physical_fixed_tolerated_when_allowed():
    fixed = {name: 0.0 for name in PHYSICAL_NAMES_LAMINAR}
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {"fixed_parameters": fixed}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    values, lower, upper = _base_arrays()
    resolved = resolve_optimized_physical_parameters(
        pm, PHYSICAL_NAMES_LAMINAR, values, lower, upper, allow_all_fixed=True,
    )
    assert resolved.free_mask.sum() == 0


def test_strip_and_restore_round_trip():
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

- [ ] **Step 3: Relocate `strip_fixed_parameters`/`restore_fixed_parameters` into `parameter_utils.py`**

Read the current `strategies/sequential.py:804-852` to copy the exact existing bodies (do not paraphrase — the docstrings and the `lower < upper` mask logic must be byte-identical to preserve `sequential.py`'s existing tested behavior). Add near the top of `xpcsjax/optimization/nlsq/parameter_utils.py` (after the existing imports; add `from dataclasses import dataclass` to the import block — it is not currently imported there):

```python
def strip_fixed_parameters(
    initial_params: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove fixed parameters (lower == upper) from the optimizer inputs.

    The TRF solver used by sequential optimization requires strict
    lower < upper for every parameter.  Fixed parameters (equality
    constraints encoded as lower == upper) must be stripped before the
    call and their known values re-inserted into the result.

    Parameters
    ----------
    initial_params : np.ndarray
        Full parameter vector including fixed parameters.
    lower_bounds : np.ndarray
        Lower bounds array (same length as initial_params).
    upper_bounds : np.ndarray
        Upper bounds array (same length as initial_params).

    Returns
    -------
    free_params : np.ndarray
        Subset of initial_params where lower < upper.
    free_lower : np.ndarray
        Lower bounds for free parameters.
    free_upper : np.ndarray
        Upper bounds for free parameters.
    free_mask : np.ndarray
        Boolean mask (length == len(initial_params)), True where free.

    Examples
    --------
    >>> p = np.array([1.0, 2.0, 3.0])
    >>> lo = np.array([0.0, 2.0, 0.0])
    >>> hi = np.array([5.0, 2.0, 5.0])
    >>> free, fl, fu, mask = strip_fixed_parameters(p, lo, hi)
    >>> free       # array([1.0, 3.0])
    >>> mask       # array([True, False, True])
    """
    free_mask = lower_bounds < upper_bounds
    return (
        initial_params[free_mask],
        lower_bounds[free_mask],
        upper_bounds[free_mask],
        free_mask,
    )


def restore_fixed_parameters(
    free_result: np.ndarray,
    fixed_values: np.ndarray,
    free_mask: np.ndarray,
) -> np.ndarray:
    """Re-insert fixed parameter values into the optimized result.

    Inverse of :func:`strip_fixed_parameters`.

    Parameters
    ----------
    free_result : np.ndarray
        Optimized values for the free parameters.
    fixed_values : np.ndarray
        Full reference parameter vector (fixed positions taken from here).
    free_mask : np.ndarray
        Boolean mask returned by :func:`strip_fixed_parameters`.

    Returns
    -------
    np.ndarray
        Full parameter vector with fixed values restored.
    """
    result = np.array(fixed_values, dtype=np.float64)
    result[free_mask] = free_result
    return result
```

Add `"strip_fixed_parameters"` and `"restore_fixed_parameters"` to the `__all__` list at the bottom of `parameter_utils.py`.

- [ ] **Step 4: Update `strategies/sequential.py` to import instead of define**

Modify `xpcsjax/optimization/nlsq/strategies/sequential.py`: delete the `strip_fixed_parameters`/`restore_fixed_parameters` function bodies (lines 804-880 in the pre-change file), and add near the top of the file's import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    restore_fixed_parameters,
    strip_fixed_parameters,
)
```

Leave every call site inside `optimize_per_angle_sequential` (the `strip_fixed_parameters(...)`/`restore_fixed_parameters(...)` calls around what was line 976 and 999/1014) completely unchanged — they call the same two names, now imported instead of locally defined.

- [ ] **Step 5: Run the relocation regression test**

Run: `uv run pytest tests/optimization/test_laminar_streaming_diag.py -v`
Expected: PASS (unchanged — this file exercises `optimize_per_angle_sequential`, proving the relocation didn't change behavior).

- [ ] **Step 6: Add `ResolvedPhysicalParameters` and `resolve_optimized_physical_parameters` to `parameter_utils.py`**

Add below the relocated functions:

```python
from dataclasses import dataclass

from xpcsjax.config.types import SCALING_PARAM_NAMES

if TYPE_CHECKING:
    from xpcsjax.config.parameter_manager import ParameterManager


@dataclass
class ResolvedPhysicalParameters:
    """Which physical parameters are free vs. fixed for one NLSQ solve.

    ``free_mask`` is computed from ``ParameterManager.get_optimizable_parameters()``
    (active-minus-fixed, physics-only by contract). When neither
    ``active_parameters`` nor ``fixed_parameters`` is set (the template
    default), ``free_mask`` is all-``True`` and every array here is
    byte-identical to the corresponding ``*_full`` input — a provable no-op.
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
    """Resolve the free/fixed split for homodyne's physical parameters.

    Physical-parameter-only by design: homodyne's ``active_parameters``/
    ``fixed_parameters`` never apply to ``contrast``/``offset`` — that scope
    cut is permanent (see the design spec, grilling Q1). A scaling name in
    ``fixed_parameters`` is a hard error here, not a silent ignore.

    Parameters
    ----------
    param_manager : ParameterManager
        Already constructed for this fit's config and analysis mode.
    physical_names : list[str]
        Physical parameter names in solver-vector order
        (``_get_physical_param_names(analysis_mode)``).
    values_full, lower_full, upper_full : np.ndarray
        Length ``len(physical_names)`` initial values / bounds, in the same
        order as ``physical_names``.
    allow_all_fixed : bool, default False
        If True, don't raise when every physical parameter is fixed —
        used only by callers (``strategies/sequential.py``) that have their
        own pre-existing, tested tolerance for that case.

    Returns
    -------
    ResolvedPhysicalParameters

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
            "which is not supported for this analysis mode — fixed_parameters "
            "only constrains physical parameters here. Use "
            "'per_angle_scaling' initial values to control contrast/offset "
            "instead."
        )

    optimizable = set(param_manager.get_optimizable_parameters())
    free_mask = np.array([name in optimizable for name in physical_names], dtype=bool)

    if not allow_all_fixed and not free_mask.any():
        raise ValueError(
            "fixed_parameters/active_parameters leave nothing left to "
            f"optimize: every physical parameter in {physical_names!r} is "
            "fixed or excluded. Free at least one parameter."
        )

    return ResolvedPhysicalParameters(
        physical_names=list(physical_names),
        values_full=np.asarray(values_full, dtype=np.float64),
        lower_full=np.asarray(lower_full, dtype=np.float64),
        upper_full=np.asarray(upper_full, dtype=np.float64),
        free_mask=free_mask,
    )
```

Add `TYPE_CHECKING` to the existing `typing` import if not already present, and add `"ResolvedPhysicalParameters"`, `"resolve_optimized_physical_parameters"` to `__all__`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_parameter_utils_resolve.py -v`
Expected: PASS, all 6 tests.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git add xpcsjax/optimization/nlsq/parameter_utils.py xpcsjax/optimization/nlsq/strategies/sequential.py tests/optimization/test_parameter_utils_resolve.py
git commit -m "feat(optimization): add resolve_optimized_physical_parameters, relocate strip/restore helpers"
```

---

### Task 2: Fix `active_parameters: []` truthiness bug and misleading docstring in `parameter_manager.py`

**Files:**
- Modify: `xpcsjax/config/parameter_manager.py:517` (truthiness check), `:771` (docstring example)
- Test: `tests/config/test_active_parameters_empty_list.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ParameterManager.get_active_parameters()` now treats an explicit `active_parameters: []` as "no active physical parameters" instead of falling back to defaults.

- [ ] **Step 1: Write the failing test**

Create `tests/config/test_active_parameters_empty_list.py`:

```python
"""Regression: initial_parameters.active_parameters: [] must mean 'fix everything',
not 'absent -> fall back to mode defaults'."""

from xpcsjax.config.parameter_manager import ParameterManager
from xpcsjax.config.parameter_registry import AnalysisMode


def test_empty_active_parameters_list_means_none_active():
    config = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {"active_parameters": []},
    }
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert pm.get_active_parameters() == []


def test_missing_active_parameters_key_falls_back_to_defaults():
    config = {"analysis_mode": "laminar_flow", "initial_parameters": {}}
    pm = ParameterManager(config, analysis_mode=AnalysisMode.LAMINAR_FLOW)
    assert len(pm.get_active_parameters()) == 7  # unchanged: absent key still uses mode defaults
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_active_parameters_empty_list.py::test_empty_active_parameters_list_means_none_active -v`
Expected: FAIL — `assert [...7 default names...] == []`.

- [ ] **Step 3: Fix the truthiness check**

Modify `xpcsjax/config/parameter_manager.py` at line 517 (inside `get_active_parameters`):

```python
# Before:
            if active_params_config and isinstance(active_params_config, list):
# After:
            if active_params_config is not None and isinstance(active_params_config, list):
```

- [ ] **Step 4: Fix the misleading docstring example**

Modify `xpcsjax/config/parameter_manager.py`'s `get_fixed_parameters()` docstring (around line 771 — the `Examples` section):

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

(`contrast`/`offset` in `fixed_parameters` now raises `ValueError` once Task 3 lands — the old example would be actively wrong if left as-is.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_active_parameters_empty_list.py -v`
Expected: PASS, both tests.

- [ ] **Step 6: Run the existing `ParameterManager` suite for regressions**

Run: `uv run pytest tests/config/ -v -k parameter_manager or active_parameters or fixed_parameters`
Expected: PASS — no existing test relies on the old truthiness bug (verified during spec research: no test asserts `active_parameters: []` falls back to defaults).

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/config/parameter_manager.py tests/config/test_active_parameters_empty_list.py
git commit -m "fix(config): active_parameters: [] now means none-active, not absent"
```

---

### Task 3: Wire the resolver into `core.py::fit_nlsq_jax`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_jax`, after the existing bounds-construction block ending around line 462)

**Interfaces:**
- Consumes: `resolve_optimized_physical_parameters` (Task 1); `_get_physical_param_names(analysis_mode)` (existing, `core.py:1245`).
- Produces: a local `resolved: ResolvedPhysicalParameters` variable, threaded into `adapter.fit(...)`/`wrapper.fit(...)` calls (Tasks 4-5) as a new `resolved_physical=resolved` keyword argument.

- [ ] **Step 1: Write the failing test**

Create (or append to) `tests/optimization/test_fixed_parameters_integration.py` — this is the first test in that file, more are added in later tasks:

```python
"""Integration tests: fixed_parameters/active_parameters actually constrain
the real NLSQ solve (not just ParameterManager in isolation)."""

import numpy as np
import pytest

from xpcsjax.config import ConfigManager
from xpcsjax.core.jax_backend import compute_g2_scaled
from xpcsjax.optimization.nlsq.core import fit_nlsq_jax

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
    data = _synthetic_laminar_data()
    cm = ConfigManager(config_override=_laminar_config(fixed_parameters={"D_offset": 50.0}))
    result = fit_nlsq_jax(data, cm, use_adapter=use_adapter)
    names = ["contrast", "offset", "D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    params = np.asarray(result.parameters).ravel()
    d_offset_idx = names.index("D_offset")
    assert abs(params[d_offset_idx] - 50.0) < 1e-9
    if result.uncertainties is not None:
        unc = np.asarray(result.uncertainties).ravel()
        assert unc[d_offset_idx] == 0.0


def test_fixed_scaling_parameter_raises():
    data = _synthetic_laminar_data()
    cm = ConfigManager(config_override=_laminar_config(fixed_parameters={"contrast": 0.5}))
    with pytest.raises(ValueError, match="contrast"):
        fit_nlsq_jax(data, cm, use_adapter=False)


def test_unset_fixed_parameters_is_a_noop():
    """The template default (fixed_parameters unset) must reach the identical
    code path as before this plan -- same x0/bounds, same result shape."""
    data = _synthetic_laminar_data()
    cm = ConfigManager(config_override=_laminar_config())
    result = fit_nlsq_jax(data, cm, use_adapter=False)
    assert np.asarray(result.parameters).size == 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py -v`
Expected: FAIL — `test_fixed_parameter_survives_real_fit` fails both parametrizations (`D_offset` moves away from 50.0); `test_fixed_scaling_parameter_raises` fails (no exception raised — currently a no-op); `test_unset_fixed_parameters_is_a_noop` passes already (baseline).

- [ ] **Step 3: Compute the resolved descriptor in `fit_nlsq_jax`**

Modify `xpcsjax/optimization/nlsq/core.py`. Immediately after the existing bounds-construction block (the code ending with `lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)` / `bounds = (lower_bounds, upper_bounds)`, currently around line 462-463), insert:

```python
    # Resolve which physical parameters are free vs. fixed for this fit
    # (fixed_parameters/active_parameters — see design spec, grilling Q1/Q5).
    from xpcsjax.optimization.nlsq.parameter_utils import resolve_optimized_physical_parameters

    physical_names = _get_physical_param_names(analysis_mode)
    physical_idx = [_get_param_names(analysis_mode).index(name) for name in physical_names]
    resolved_physical = resolve_optimized_physical_parameters(
        param_manager,
        physical_names,
        values_full=np.asarray(x0)[physical_idx],
        lower_full=lower_bounds[physical_idx],
        upper_full=upper_bounds[physical_idx],
    )
```

Note: `param_manager` is already constructed a few lines above this insertion point (the existing `param_manager = ParameterManager(config_dict=config_dict_for_pm, analysis_mode=analysis_mode)` in the `HAS_PARAMETER_MANAGER` branch) — reuse it, don't construct a second instance.

- [ ] **Step 4: Thread `resolved_physical` into the adapter/wrapper calls**

Modify the two dispatch call sites later in `fit_nlsq_jax`:

```python
# adapter.fit(...) call (around line 527) — add one kwarg:
            result = adapter.fit(
                data=data,
                config=config,
                initial_params=x0,
                bounds=bounds,
                analysis_mode=analysis_mode,
                per_angle_scaling=per_angle_scaling,
                diagnostics_enabled=diagnostics_enabled,
                shear_transforms=shear_transform_cfg,
                per_angle_scaling_initial=per_angle_scaling_initial,
                resolved_physical=resolved_physical,
            )
```

Find the corresponding `NLSQWrapper(...).fit(...)` call further down in the same function (the `not _use_adapter or fallback_occurred` branch) and add the same `resolved_physical=resolved_physical` keyword.

- [ ] **Step 5: Run tests to verify the scaling-name-raises case now passes (adapter/wrapper changes land in Tasks 4-5)**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_scaling_parameter_raises tests/optimization/test_fixed_parameters_integration.py::test_unset_fixed_parameters_is_a_noop -v`
Expected: PASS for both — the scaling-name check happens in `resolve_optimized_physical_parameters` itself, before any adapter/wrapper code runs, and the no-op test doesn't depend on Tasks 4-5.
`test_fixed_parameter_survives_real_fit` still FAILS at this point — expected, since `adapter.py`/`wrapper.py` don't consume `resolved_physical` yet. That's fixed in Tasks 4-5; do not treat this as a regression here.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): compute resolved physical-parameter descriptor in fit_nlsq_jax"
```

---

### Task 4: Wire strip/restore into `wrapper.py` (default path, `use_adapter=False`)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/wrapper.py` — `NLSQWrapper.fit()` signature, the Step 6.6 region (per-angle expansion), and its handoff to `strategies/executors.py`.

**Interfaces:**
- Consumes: `ResolvedPhysicalParameters` (Task 1); `strip_fixed_parameters`/`restore_fixed_parameters` (Task 1).
- Produces: `NLSQWrapper.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)` — when `None` (every existing caller not updated by this plan, e.g. `heterodyne_core.py`, tests), behavior is completely unchanged.

- [ ] **Step 1: Read the current Step 6.6 region before editing**

Read `xpcsjax/optimization/nlsq/wrapper.py` lines 1780-1970 in full (the region covering `resolved_per_angle_mode` resolution, Step 6's `residual_fn = self._create_residual_function(...)`, and Step 6.6's per-angle expansion producing `validated_params`/expanded bounds) so the exact current variable names at the handoff point to `strategies/executors.py` (confirmed via research: `curve_fit_fn=curve_fit` / `curve_fit_large_fn=curve_fit_large`, imported `from nlsq import curve_fit, curve_fit_large` at `wrapper.py:99`) are visible before writing the diff.

- [ ] **Step 2: Add the `resolved_physical` parameter to `NLSQWrapper.fit()`'s signature**

Modify the `fit()` method signature to add (keyword-only, default `None`, so every unmodified caller is unaffected):

```python
    def fit(
        self,
        # ... existing parameters unchanged ...
        *,
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

Add the import near the top of `wrapper.py`'s existing import block:

```python
from xpcsjax.optimization.nlsq.parameter_utils import (
    ResolvedPhysicalParameters,
    restore_fixed_parameters,
    strip_fixed_parameters,
)
```

- [ ] **Step 3: Strip the physical slice and wrap the residual function immediately after Step 6.6**

Immediately after the existing Step 6.6 block finishes assembling `validated_params` (the `np.concatenate([scaling_head, physical_params])` line) and its matching expanded bounds arrays, insert:

```python
        # Strip fixed physical parameters (grilling Q1/Q5) before handoff to
        # the solver. n_physical is invariant across every per-angle scaling
        # mode (constant/averaged/individual) -- only the scaling-head prefix
        # length varies -- so this always operates on the trailing slice.
        if resolved_physical is not None and not resolved_physical.free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            phys_lower = nlsq_bounds[0][-n_physical:]
            phys_upper = nlsq_bounds[1][-n_physical:]
            _, free_phys_lower, free_phys_upper, phys_free_mask = strip_fixed_parameters(
                validated_params[-n_physical:], phys_lower, phys_upper,
            )
            validated_params = np.concatenate(
                [validated_params[:-n_physical], validated_params[-n_physical:][phys_free_mask]]
            )
            nlsq_bounds = (
                np.concatenate([nlsq_bounds[0][:-n_physical], free_phys_lower]),
                np.concatenate([nlsq_bounds[1][:-n_physical], free_phys_upper]),
            )
            _fixed_physical_full = resolved_physical.values_full
            _base_residual_fn = base_residual_fn

            def residual_fn(params, *args, **kwargs):
                n_prefix = len(params) - int(phys_free_mask.sum())
                full_physical = restore_fixed_parameters(
                    params[n_prefix:], _fixed_physical_full, phys_free_mask,
                )
                full_params = np.concatenate([params[:n_prefix], full_physical])
                return _base_residual_fn(full_params, *args, **kwargs)
        else:
            phys_free_mask = None
```

(Variable names `nlsq_bounds`/`base_residual_fn` are placeholders for whatever the existing code at that exact point calls the expanded-bounds tuple and the Step-6 residual function — confirm the exact names from Step 1's read and use those; the logic above is what must execute regardless of the exact local names.)

- [ ] **Step 4: Restore fixed values and zero-pad covariance/uncertainty in the result**

Find where `NLSQWrapper.fit()` builds its final `OptimizationResult` (after the solver call in `strategies/executors.py` returns). Before constructing the result, insert:

```python
        if resolved_physical is not None and phys_free_mask is not None and not phys_free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            n_prefix = len(fitted_params) - int(phys_free_mask.sum())
            full_physical = restore_fixed_parameters(
                fitted_params[n_prefix:], resolved_physical.values_full, phys_free_mask,
            )
            fitted_params = np.concatenate([fitted_params[:n_prefix], full_physical])
            if covariance is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [
                    n_prefix + i for i, free in enumerate(phys_free_mask) if free
                ]
                full_cov[np.ix_(free_idx, free_idx)] = covariance
                covariance = full_cov
```

(`fitted_params`/`covariance` are placeholders for whatever the existing pre-result-construction code calls the solver's returned parameter vector and covariance matrix — use the exact existing names.)

- [ ] **Step 5: Run the integration test for the wrapper path**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_real_fit -k "False" -v`
Expected: PASS (the `use_adapter=False` parametrization).

- [ ] **Step 6: Run the wrapper regression suite**

Run: `uv run pytest tests/optimization/test_phase5_model_function_modes.py tests/optimization/test_adapter_flatten_phi_order.py -v` (and any other `wrapper.py`-exercising test files discovered via `grep -rl "NLSQWrapper(" tests/optimization/`)
Expected: PASS — `resolved_physical=None` default means every test not updated by this plan takes the exact old path.

- [ ] **Step 7: Verify large-dataset memory-routing tiers inherit the strip — do not assume**

The plan's architecture claim (design spec, Component "Architecture") is that `strategies/hybrid_streaming.py`, `strategies/stratified_ls.py`, `strategies/out_of_core.py`, and `strategies/chunking.py`/`executors.py`'s internal dispatch all receive the **same** `validated_params`/`nlsq_bounds` this task already strips at Step 3-4, since they're internal size-based dispatch branches inside `NLSQWrapper.fit()`'s call graph, not independent entry points that reconstruct their own parameter vector. This was **not independently confirmed** during plan research (research focused on the in-memory/standard path). Before closing this task:

1. Read `xpcsjax/optimization/nlsq/strategies/executors.py`'s dispatch logic (the size-based branch selecting between `curve_fit`/`curve_fit_large`/hybrid-streaming/stratified-LS/out-of-core) and confirm it receives the already-stripped `validated_params`/`nlsq_bounds` from this task's Step 3-4 changes, with no separate parameter-vector construction of its own.
2. If confirmed: run `uv run pytest tests/optimization/test_strategy_hybrid_streaming.py tests/optimization/test_strategy_chunking.py -v` (adjust filenames to whatever exists — `ls tests/optimization/ | grep -iE "hybrid_streaming|stratified_ls|out_of_core|chunking"`) as a regression check (should pass unchanged, `resolved_physical=None` default) — no new fixed-parameter-specific test needed for these tiers, since Task 4's own integration test already exercises the shared code path they inherit from.
3. If **not** confirmed — i.e., any of these strategies build their own x0/bounds/residual function independent of Step 6.6's output — stop and add a new sub-task applying the same strip/restore pattern at that independent construction point before proceeding to Task 5. Do not close this task with an unverified assumption.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/wrapper.py
git add xpcsjax/optimization/nlsq/wrapper.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQWrapper.fit()"
```

---

### Task 5: Wire strip/restore into `adapter.py` (experimental path, `use_adapter=True`)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/adapter.py` — `NLSQAdapter.fit()` signature and the region between `_build_model_function()`/`get_or_create_model()` and `self._fitter.curve_fit()`.

**Interfaces:**
- Consumes: same as Task 4.
- Produces: `NLSQAdapter.fit(..., resolved_physical: "ResolvedPhysicalParameters | None" = None)`.

- [ ] **Step 1: Read the current `fit()` method before editing**

Read `xpcsjax/optimization/nlsq/adapter.py` lines 1256-1420 in full to confirm the exact current variable names for `p0`, `bounds`, and the `model_func` returned by `_build_model_function`/`get_or_create_model`, and the exact `self._fitter.curve_fit(...)` call's kwargs, before writing the diff. Physical parameters are confirmed (via plan research) to occupy the trailing slice of `params_array` in **both** the cached (`get_or_create_model`, `adapter.py:542-554`) and uncached (`_build_model_function`'s inline closure, `adapter.py:961-1024`) model-function code paths — the strip/restore approach below is correct regardless of which branch is active.

- [ ] **Step 2: Add `resolved_physical` to `NLSQAdapter.fit()`'s signature and import**

```python
    def fit(
        self,
        # ... existing parameters unchanged ...
        resolved_physical: "ResolvedPhysicalParameters | None" = None,
    ) -> OptimizationResult:
```

Add the same import as Task 4, Step 2, to `adapter.py`'s import block.

- [ ] **Step 3: Strip before `curve_fit()`, wrap `model_func` to restore**

Immediately after `model_func, cache_hit, jit_compiled = self._build_model_function(...)` (around line 1342) and before the `curve_fit(...)` call (around line 1393), insert:

```python
        _phys_free_mask = None
        if resolved_physical is not None and not resolved_physical.free_mask.all():
            from xpcsjax.optimization.nlsq.parameter_utils import (
                restore_fixed_parameters,
                strip_fixed_parameters,
            )

            n_physical = len(resolved_physical.physical_names)
            _, free_lower, free_upper, _phys_free_mask = strip_fixed_parameters(
                np.asarray(initial_params)[-n_physical:],
                np.asarray(bounds[0])[-n_physical:],
                np.asarray(bounds[1])[-n_physical:],
            )
            initial_params = np.concatenate(
                [np.asarray(initial_params)[:-n_physical], np.asarray(initial_params)[-n_physical:][_phys_free_mask]]
            )
            bounds = (
                np.concatenate([np.asarray(bounds[0])[:-n_physical], free_lower]),
                np.concatenate([np.asarray(bounds[1])[:-n_physical], free_upper]),
            )
            _base_model_func = model_func
            _fixed_physical_full = resolved_physical.values_full

            def model_func(x, *params):
                n_prefix = len(params) - int(_phys_free_mask.sum())
                full_physical = restore_fixed_parameters(
                    np.asarray(params[n_prefix:]), _fixed_physical_full, _phys_free_mask,
                )
                full_params = (*params[:n_prefix], *full_physical.tolist())
                return _base_model_func(x, *full_params)
```

- [ ] **Step 4: Restore fixed values and zero-pad covariance/uncertainty after `curve_fit()` returns**

Immediately after the `result = self._fitter.curve_fit(...)` call (around line 1393) and before `NLSQAdapter.fit()` constructs its `OptimizationResult`, insert:

```python
        if resolved_physical is not None and _phys_free_mask is not None and not _phys_free_mask.all():
            n_physical = len(resolved_physical.physical_names)
            n_prefix = len(result.popt) - int(_phys_free_mask.sum())
            full_physical = restore_fixed_parameters(
                np.asarray(result.popt[n_prefix:]), resolved_physical.values_full, _phys_free_mask,
            )
            result.popt = np.concatenate([result.popt[:n_prefix], full_physical])
            if getattr(result, "pcov", None) is not None:
                n_full = n_prefix + n_physical
                full_cov = np.zeros((n_full, n_full))
                free_idx = list(range(n_prefix)) + [
                    n_prefix + i for i, free in enumerate(_phys_free_mask) if free
                ]
                full_cov[np.ix_(free_idx, free_idx)] = result.pcov
                result.pcov = full_cov
```

(`result.popt`/`result.pcov` are placeholders for whatever attribute names the upstream `nlsq` library's `curve_fit()` return object actually uses — confirm the exact attribute names from Step 1's read of the surrounding code, which already accesses them to build `OptimizationResult`.)

- [ ] **Step 5: Run the integration test for the adapter path**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_real_fit -k "True" -v`
Expected: PASS (the `use_adapter=True` parametrization).

- [ ] **Step 6: Run the adapter regression suite**

Run: `uv run pytest tests/optimization/test_adapter_info_extraction.py tests/optimization/test_adapter_cost_default.py tests/optimization/test_adapter_flatten_phi_order.py -v`
Expected: PASS unchanged — `resolved_physical=None` default preserves the old path for every unmodified caller.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/adapter.py
git add xpcsjax/optimization/nlsq/adapter.py
git commit -m "feat(optimization): honor fixed physical parameters in NLSQAdapter.fit()"
```

---

### Task 6: Wire strip/restore into `core.py::fit_nlsq_cmaes`

**Files:**
- Modify: `xpcsjax/optimization/nlsq/core.py` (inside `fit_nlsq_cmaes`, around its `model_for_cmaes` closure at `:2301-2364` and its two solver calls at `:2385-2392` and `:2481-2489`).

**Interfaces:**
- Consumes: same as Tasks 4-5. `fit_nlsq_cmaes` does **not** call `NLSQAdapter`/`NLSQWrapper` — confirmed via plan research it builds its own `model_for_cmaes` JAX closure and calls a CMA-ES-family `wrapper` object's `_run_nlsq_refinement`/`fit` methods directly with `p0=x0, bounds=bounds`.

- [ ] **Step 1: Write the failing test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_cmaes_fit():
    data = _synthetic_laminar_data()
    config = _laminar_config(fixed_parameters={"D_offset": 50.0})
    config["optimization"] = {"nlsq": {"cmaes": {"enable": True}}}
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)  # dispatches to fit_nlsq_cmaes internally
    names = ["contrast", "offset", "D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 50.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`
Expected: FAIL — `D_offset` moves during the CMA-ES solve.

- [ ] **Step 3: Read the current `fit_nlsq_cmaes` region before editing**

Read `xpcsjax/optimization/nlsq/core.py` lines 1900-2500 in full to see the exact current variable names for `x0`/`bounds` at the point the scaling-mode branch (constant/averaged/individual, `:1982-2154`) finishes producing them, and the exact `wrapper._run_nlsq_refinement(...)`/`wrapper.fit(...)` call sites — the docstring comment already there at `:2261-2264` (*"The physics block is ALWAYS the last n_physical entries regardless of the scaling-head layout"*) confirms the trailing-slice invariant this task relies on.

- [ ] **Step 4: Compute the resolved descriptor and strip before both solver calls**

Immediately after the scaling-mode branch produces its final `x0`/`bounds` (around line 2154, before `model_for_cmaes` is built at `:2301`), insert:

```python
    resolved_physical = resolve_optimized_physical_parameters(
        param_manager,
        physical_names,
        values_full=x0[-len(physical_names):],
        lower_full=bounds[0][-len(physical_names):],
        upper_full=bounds[1][-len(physical_names):],
    )
    _cmaes_phys_free_mask = None
    if not resolved_physical.free_mask.all():
        n_physical = len(physical_names)
        _, free_lower, free_upper, _cmaes_phys_free_mask = strip_fixed_parameters(
            x0[-n_physical:], bounds[0][-n_physical:], bounds[1][-n_physical:],
        )
        x0 = np.concatenate([x0[:-n_physical], x0[-n_physical:][_cmaes_phys_free_mask]])
        bounds = (
            np.concatenate([bounds[0][:-n_physical], free_lower]),
            np.concatenate([bounds[1][:-n_physical], free_upper]),
        )
```

(Import `resolve_optimized_physical_parameters`, `strip_fixed_parameters`, `restore_fixed_parameters` from `xpcsjax.optimization.nlsq.parameter_utils` at the top of `core.py` if not already imported by Task 3's changes — `core.py` is one file, so Task 3's import already covers this if placed at module level rather than function-local; prefer a module-level import here since both `fit_nlsq_jax` and `fit_nlsq_cmaes` need it.)

- [ ] **Step 5: Wrap `model_for_cmaes` to restore fixed values before physics evaluation**

Immediately after `model_for_cmaes` is defined (around line 2364), insert:

```python
    if _cmaes_phys_free_mask is not None:
        _base_model_for_cmaes = model_for_cmaes
        _fixed_physical_full = resolved_physical.values_full

        def model_for_cmaes(params_array, *args, **kwargs):
            n_physical = len(physical_names)
            n_prefix = len(params_array) - int(_cmaes_phys_free_mask.sum())
            full_physical = restore_fixed_parameters(
                params_array[n_prefix:], _fixed_physical_full, _cmaes_phys_free_mask,
            )
            full_params = jnp.concatenate([params_array[:n_prefix], full_physical])
            return _base_model_for_cmaes(full_params, *args, **kwargs)
```

- [ ] **Step 6: Restore fixed values in both result paths (warm-start and CMA-ES)**

After each of the two solver calls (`warmstart_result = wrapper._run_nlsq_refinement(...)` at `:2385-2392` and `cmaes_result = wrapper.fit(...)` at `:2481-2489`) returns its own fitted-parameter vector, apply the same restore-and-zero-pad-covariance pattern as Task 4 Step 4 / Task 5 Step 4, using `_cmaes_phys_free_mask` and `resolved_physical.values_full`. Read the exact attribute names each result object exposes (from the surrounding existing code that already consumes them to build the final `OptimizationResult`) before writing this diff.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_cmaes_fit -v`
Expected: PASS.

- [ ] **Step 8: Run the CMA-ES regression suite**

Run: `uv run pytest tests/optimization/test_cmaes_trigger.py tests/optimization/test_heterodyne_cmaes_seed.py -v` (and any other `fit_nlsq_cmaes`-exercising homodyne CMA-ES test files — `grep -rl "fit_nlsq_cmaes" tests/optimization/`)
Expected: PASS unchanged.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check xpcsjax/optimization/nlsq/core.py
git add xpcsjax/optimization/nlsq/core.py tests/optimization/test_fixed_parameters_integration.py
git commit -m "feat(optimization): honor fixed physical parameters in fit_nlsq_cmaes"
```

---

### Task 7: Prove `fit_nlsq_multistart` needs no code change

**Files:**
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append only — no production code changes in this task)

**Interfaces:**
- Consumes: `fit_nlsq_jax` (Task 3), already-fixed by this point in the plan.
- Produces: nothing new — this task exists to make an architectural fact from plan research **provable**, not assumed: `core.py::_SingleFitWorker.__call__` (`:1371-1414`) recurses into `fit_nlsq_jax(..., _skip_global_selection=True)` per sampled start, and `fit_nlsq_multistart` (`:1439-1616`) never calls `NLSQAdapter`/`NLSQWrapper` directly — it delegates entirely through that recursion. Since `fit_nlsq_jax` now resolves and honors `fixed_parameters` internally (Task 3), every multistart worker inherits correctness automatically regardless of what value Latin-hypercube sampling assigned to a "fixed" dimension — that sampled value is simply discarded and overridden by the resolved fixed value once `fit_nlsq_jax` runs. This also means multistart's own pre-existing `run_multistart_nlsq`/`check_zero_volume_bounds` → single-start-fallback convention (`multistart.py:944-964`, unrelated code path — it triggers on literal degenerate `parameter_space.bounds`, never on `fixed_parameters`, since this plan never produces degenerate bounds at the multistart LHS-sampling stage) is untouched by this plan and needs no reconciliation with Task 3's `ValueError`-on-all-fixed behavior.

- [ ] **Step 1: Write the test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_fixed_parameter_survives_multistart_fit():
    data = _synthetic_laminar_data()
    config = _laminar_config(fixed_parameters={"D_offset": 50.0})
    config["optimization"] = {"nlsq": {"multi_start": {"enable": True, "n_starts": 3}}}
    cm = ConfigManager(config_override=config)
    result = fit_nlsq_jax(data, cm, use_adapter=False)  # dispatches to fit_nlsq_multistart internally
    names = ["contrast", "offset", "D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    params = np.asarray(result.parameters).ravel()
    assert abs(params[names.index("D_offset")] - 50.0) < 1e-9
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_fixed_parameter_survives_multistart_fit -v`
Expected: PASS **without any production code change** — this is the expected/desired outcome given Task 3's fix and the recursion architecture described above. If this test fails, that falsifies the "no code change needed" architectural claim from plan research — stop and re-investigate `_SingleFitWorker`/`fit_nlsq_multistart` before proceeding to Task 8 (do not skip or weaken this test to force a pass).

- [ ] **Step 3: Commit**

```bash
git add tests/optimization/test_fixed_parameters_integration.py
git commit -m "test(optimization): prove fixed_parameters propagates through multistart via recursion, no code change needed"
```

---

### Task 8: Heterodyne — `fixed_parameters` block in `_apply_initial_parameters`, loosen precondition

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::_apply_initial_parameters` (currently lines 481-556)
- Test: `tests/config/test_heterodyne_fixed_parameters.py`

**Interfaces:**
- Consumes: `space.vary: dict[str, bool]`, `space.values: dict[str, float]`, `_INBOUND_NAME_ALIAS`, `PARAMETER_NAME_MAPPING`, `ALL_PARAM_NAMES_WITH_SCALING` (all existing).
- Produces: `fixed_parameters` in heterodyne config now sets both `vary=False` and the actual value — including for `contrast`/`offset` (grilling Q7, no physical-only restriction here).

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_heterodyne_fixed_parameters.py`:

```python
"""Heterodyne fixed_parameters: vary=False + value write, including scaling names."""

import pytest

from xpcsjax.config.heterodyne_parameter_manager import ParameterManager


def _config(fixed_parameters, parameter_names=None, values=None):
    initial = {"fixed_parameters": fixed_parameters}
    if parameter_names is not None:
        initial["parameter_names"] = parameter_names
        initial["values"] = values
    return {"analysis_mode": "two_component", "initial_parameters": initial}


def test_fixed_parameter_value_is_honored_not_flat_list_value():
    config = _config(
        fixed_parameters={"D0_ref": 999.0},
        parameter_names=["D0_ref"],
        values=[10000.0],  # flat list says 10000.0; fixed_parameters must win
    )
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 999.0
    assert pm.space.vary["D0_ref"] is False
    assert "D0_ref" not in pm.varying_names


def test_fixed_scaling_parameter_is_honored():
    config = _config(fixed_parameters={"contrast": 0.42})
    pm = ParameterManager.from_config(config)
    assert pm.space.values["contrast"] == 0.42
    assert pm.space.vary["contrast"] is False


def test_fixed_parameters_applies_without_flat_parameter_names():
    """A config setting ONLY fixed_parameters (no flat parameter_names/values
    pair) must still apply -- the coupling to flat values was accidental."""
    config = {"analysis_mode": "two_component", "initial_parameters": {"fixed_parameters": {"D0_ref": 7.0}}}
    pm = ParameterManager.from_config(config)
    assert pm.space.values["D0_ref"] == 7.0
    assert pm.space.vary["D0_ref"] is False


def test_fixed_wins_over_active_on_conflict():
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "active_parameters": ["D0_ref"],
            "fixed_parameters": {"D0_ref": 3.0},
        },
    }
    pm = ParameterManager.from_config(config)
    assert pm.space.vary["D0_ref"] is False
    assert pm.space.values["D0_ref"] == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: FAIL on all 4 — `fixed_parameters` currently has no effect at all.

- [ ] **Step 3: Read the current `_apply_initial_parameters` before editing**

Read `xpcsjax/config/heterodyne_parameter_space.py` lines 481-557 to confirm the exact current early-return condition and the exact `active_parameters` block's name-mapping chain (`_INBOUND_NAME_ALIAS.get(m, m)` composed with `PARAMETER_NAME_MAPPING.get(str(n), str(n))`) to reuse identically for `fixed_parameters`.

- [ ] **Step 4: Loosen the early-return precondition**

Modify the function's opening guard:

```python
# Before:
    param_names_raw = initial.get("parameter_names")
    param_values = initial.get("values")

    if (
        not param_names_raw
        or not isinstance(param_names_raw, list)
        or param_values is None
        or not isinstance(param_values, list)
    ):
        return
# After:
    param_names_raw = initial.get("parameter_names")
    param_values = initial.get("values")

    has_flat_values = (
        param_names_raw
        and isinstance(param_names_raw, list)
        and param_values is not None
        and isinstance(param_values, list)
    )
```

Then wrap the existing flat-value-application loop (the `for name, value in zip(param_names, param_values, ...)` block and its length-mismatch check) in `if has_flat_values:` instead of returning early — everything below that (the `active_parameters` block) must run **unconditionally**, not gated on `has_flat_values`.

- [ ] **Step 5: Add the `fixed_parameters` block after the existing `active_parameters` block**

Append, using the exact same name-mapping chain as the `active_parameters` block immediately above it:

```python
    # fixed_parameters: sets both the vary flag AND the value -- unlike
    # active_parameters, expand_varying_to_full() fills non-varying positions
    # from space.values, so the value write is required, not optional
    # (design spec, Codex review finding #11). Applied AFTER active_parameters
    # so fixed wins on conflict (grilling Q7 -- no physical-only restriction
    # here, mirrors active_parameters' existing scope including
    # contrast/offset).
    from xpcsjax.config.types import coerce_finite_float

    fixed_raw = initial.get("fixed_parameters")
    if fixed_raw is not None and isinstance(fixed_raw, dict):
        for name, value in fixed_raw.items():
            canonical = _INBOUND_NAME_ALIAS.get(
                PARAMETER_NAME_MAPPING.get(str(name), str(name)),
                PARAMETER_NAME_MAPPING.get(str(name), str(name)),
            )
            if canonical not in space.values:
                logger.warning("fixed_parameters: unknown parameter '%s', skipping", name)
                continue
            space.values[canonical] = coerce_finite_float(
                value, context=f"initial_parameters.fixed_parameters[{canonical!r}]"
            )
            space.vary[canonical] = False
            logger.debug("fixed_parameters: fixed %s = %.6g", canonical, value)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 4 tests.

- [ ] **Step 7: Run the existing heterodyne parameter test suite for regressions**

Run: `uv run pytest tests/config/test_heterodyne_parameter_manager_tied.py tests/config/test_parameter_manager_active_scaling.py -v`
Expected: PASS unchanged.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_space.py
git add xpcsjax/config/heterodyne_parameter_space.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): heterodyne fixed_parameters sets vary=False AND the value"
```

---

### Task 9: Heterodyne — tied-child-fixed conflict validation

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_space.py::_apply_tied_parameters` (currently lines 559-733)
- Test: `tests/config/test_heterodyne_fixed_parameters.py` (append)

**Interfaces:**
- Consumes: `fixed_raw` (the same dict `_apply_initial_parameters` reads, threaded into `_apply_tied_parameters` as a new parameter).
- Produces: `ValueError` when a tied child also appears in `fixed_parameters`.

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_tied_child_also_fixed_raises -v`
Expected: FAIL — no exception raised (silently accepted today).

- [ ] **Step 3: Thread `fixed_parameters` names into `_apply_tied_parameters`**

Read `xpcsjax/config/heterodyne_parameter_space.py` lines 559-680 (confirmed exact validation order and error-message style from plan research — quoted verbatim below) before editing:

```python
if child not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{child}'. Valid names: {list(ALL_PARAM_NAMES)}")
if parent not in ALL_PARAM_NAMES: raise ValueError(f"tied_parameters: unknown physics parameter '{parent}'. Valid names: {list(ALL_PARAM_NAMES)}")
if child == parent: raise ValueError(f"tied_parameters: '{child}' cannot be tied to itself")
if parent in children: raise ValueError(f"tied_parameters: '{parent}' is itself a tied child (tied to '{tied_translated[parent]}') -- chained ties are not supported. Tie '{child}' directly to '{tied_translated[parent]}' instead.")
if not space.vary.get(parent, False): raise ValueError(f"tied_parameters: parent '{parent}' is not varying (fixed via active_parameters or vary: false) -- tying '{child}' to a fixed parent is not supported; fix '{child}' directly instead via active_parameters.")
```

Add the new check immediately after the `parent in children` (chained-tie) check, in the same per-pair validation loop, matching the exact error-message style:

```python
        fixed_raw = initial.get("fixed_parameters")
        fixed_names: set[str] = set()
        if fixed_raw is not None and isinstance(fixed_raw, dict):
            fixed_names = {
                _INBOUND_NAME_ALIAS.get(
                    PARAMETER_NAME_MAPPING.get(str(n), str(n)),
                    PARAMETER_NAME_MAPPING.get(str(n), str(n)),
                )
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

(Place the `fixed_raw`/`fixed_names` computation once, before the per-pair loop begins, not inside the loop — read the loop's exact structure from Step 3's file read to place it correctly.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 5 tests (4 from Task 8 + this one).

- [ ] **Step 5: Run the existing tied-parameters regression suite**

Run: `uv run pytest tests/config/test_heterodyne_parameter_manager_tied.py -v`
Expected: PASS unchanged — the new check only fires on the new fixed-child condition, never on existing tied/active conflict paths.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_space.py
git add xpcsjax/config/heterodyne_parameter_space.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): reject a tied child also listed in heterodyne fixed_parameters"
```

---

### Task 10: Heterodyne — zero-varying-parameters guard

**Files:**
- Modify: `xpcsjax/config/heterodyne_parameter_manager.py`
- Test: `tests/config/test_heterodyne_fixed_parameters.py` (append)

**Interfaces:**
- Consumes: `ParameterSpace.n_varying` / `ParameterSpace.varying_names` (existing, `heterodyne_parameter_space.py:86-94` — **scaling-inclusive**, confirmed via plan research; do **not** use `ParameterManager.varying_indices`, which is physics-only and would miss the case where physics is free but all scaling is fixed, or vice versa).
- Produces: `ValueError` at `ParameterManager` construction time when `space.n_varying == 0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_heterodyne_fixed_parameters.py`:

```python
from xpcsjax.config.heterodyne_parameter_names import ALL_PARAM_NAMES_WITH_SCALING


def test_zero_varying_parameters_raises():
    fixed = {name: 0.0 for name in ALL_PARAM_NAMES_WITH_SCALING}
    config = {"analysis_mode": "two_component", "initial_parameters": {"fixed_parameters": fixed}}
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|no varying"):
        ParameterManager.from_config(config)


def test_active_parameters_empty_list_also_raises():
    """Pre-existing gap (grilling Q9): active_parameters: [] already reached
    zero-varying with no guard before this plan; must be caught now too."""
    config = {"analysis_mode": "two_component", "initial_parameters": {"active_parameters": []}}
    with pytest.raises(ValueError, match="[Nn]othing left to optimize|no varying"):
        ParameterManager.from_config(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py::test_zero_varying_parameters_raises tests/config/test_heterodyne_fixed_parameters.py::test_active_parameters_empty_list_also_raises -v`
Expected: FAIL — no exception raised (unguarded today, confirmed via plan research grep).

- [ ] **Step 3: Read `ParameterManager.from_config` / `__post_init__` before editing**

Read `xpcsjax/config/heterodyne_parameter_manager.py` lines 1-140 and 560-580 (`from_config`, `__post_init__`, `_sync_bounds_from_space`) to find the exact point after `_apply_initial_parameters`/`_apply_tied_parameters` have both run, where `self.space` is fully resolved — the guard must run there, once per config resolution, not per solver call.

- [ ] **Step 4: Add the guard**

At the point identified in Step 3 (end of `from_config`, after tied/fixed/active application, before returning the constructed `ParameterManager`):

```python
        if instance.space.n_varying == 0:
            raise ValueError(
                "Nothing left to optimize: active_parameters/fixed_parameters "
                "combine to leave zero varying parameters (physics and "
                "scaling combined). Free at least one parameter."
            )
```

(`instance` is a placeholder for whatever local variable name `from_config` currently uses for the `ParameterManager` it's about to return — confirm from Step 3's read.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/config/test_heterodyne_fixed_parameters.py -v`
Expected: PASS, all 7 tests.

- [ ] **Step 6: Run the full heterodyne config regression suite**

Run: `uv run pytest tests/config/ -k heterodyne -v`
Expected: PASS — no existing test config leaves zero varying parameters (this is a new guard against a previously-unreachable-in-practice, now-more-reachable condition, not a behavior change for any passing config).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check xpcsjax/config/heterodyne_parameter_manager.py
git add xpcsjax/config/heterodyne_parameter_manager.py tests/config/test_heterodyne_fixed_parameters.py
git commit -m "feat(config): guard against zero varying parameters in heterodyne ParameterManager"
```

---

### Task 11: Heterodyne fit-path integration test + full regression sweep

**Files:**
- Test: `tests/optimization/test_fixed_parameters_integration.py` (append heterodyne case)
- No production code changes — this task verifies the whole plan end-to-end and closes it out.

**Interfaces:**
- Consumes: everything from Tasks 1-10.

- [ ] **Step 1: Write the heterodyne real-fit test**

Append to `tests/optimization/test_fixed_parameters_integration.py`:

```python
def test_heterodyne_fixed_parameter_survives_real_fit():
    import jax.numpy as jnp

    from xpcsjax.core.heterodyne_jax_backend import compute_c2_heterodyne
    from xpcsjax.optimization.nlsq import fit_nlsq

    n_t, n_phi = 10, 3
    t = np.arange(1, n_t + 1) * DT
    t1, t2 = np.meshgrid(t, t, indexing="ij")
    phi = np.array([0.0, 45.0, 90.0])
    true_physical = [10000.0, 0.0, 0.0, 10000.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    c2 = np.asarray(
        compute_c2_heterodyne(
            jnp.array(true_physical), jnp.asarray(t1), jnp.asarray(t2), jnp.asarray(phi),
            Q, L, DT,
        )
    )
    rng = np.random.default_rng(1)
    c2_noisy = c2 + rng.normal(scale=1e-4, size=c2.shape)
    data = {
        "phi": phi, "g2": c2_noisy, "t1": t1, "t2": t2,
        "q": Q, "L": L, "dt": DT, "sigma": 1e-4 * np.ones_like(c2_noisy),
    }
    config = {
        "analysis_mode": "two_component",
        "initial_parameters": {
            "parameter_names": [
                "D0_ref", "alpha_ref", "D_offset_ref", "D0_sample", "alpha_sample",
                "D_offset_sample", "v0", "v_beta", "v_offset", "f0", "f1", "f2", "f3", "phi0_het",
            ],
            "values": [9000.0, 0.0, 0.0, 9000.0, 0.0, 0.0, 900.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            "fixed_parameters": {"D_offset_sample": 0.0},
        },
    }
    cm = ConfigManager(config_override=config)
    result = fit_nlsq(data, cm)
    idx = 5  # D_offset_sample position in the 14-name list above
    params = np.asarray(result.parameters).ravel()
    assert abs(params[idx] - 0.0) < 1e-6
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/optimization/test_fixed_parameters_integration.py::test_heterodyne_fixed_parameter_survives_real_fit -v`
Expected: PASS — Task 8's fix in `_apply_initial_parameters` is consumed automatically by every heterodyne strategy tier (`varying_names`/`expand_varying_to_full`), no additional wiring needed in `heterodyne_core.py`.

- [ ] **Step 3: Run the full optimization + config + heterodyne test suites**

```bash
make test-optimization
make test-heterodyne
uv run pytest tests/config/ -v
```
Expected: PASS, zero failures, zero new skips.

- [ ] **Step 4: Re-run golden parity (forced locally)**

```bash
XPCSJAX_RUN_CHARACTERIZATION=1 XPCSJAX_RUN_ENGINE_PARITY=1 uv run pytest tests/parity/ -v
```
Expected: PASS, `rtol=1e-10` unchanged — every golden config has `fixed_parameters`/`active_parameters` unset, which this plan guarantees is a byte-identical no-op path (`ResolvedPhysicalParameters.free_mask` all-`True`, every strip/restore branch short-circuited via the `not resolved_physical.free_mask.all()` / `not free_mask.all()` guards added in Tasks 4-6).

- [ ] **Step 5: Full suite + lint**

```bash
make test
make lint
uv run mypy xpcsjax
```
Expected: `make test` and `make lint` pass; `mypy` advisory (per root `CLAUDE.md`, non-blocking).

- [ ] **Step 6: Final commit**

```bash
git add tests/optimization/test_fixed_parameters_integration.py
git commit -m "test(optimization): heterodyne real-fit integration test for fixed_parameters"
```

---

## Deferred / explicitly out of scope (do not implement as part of this plan)

Per the spec's Out-of-scope section: the separate CLI-override bug in `cli/config_handling.py` (`--initial-*` re-introducing a fixed parameter); dead-code cleanup of `xpcsjax/core/fitting.py::ParameterSpace` and `xpcsjax/config/parameter_space.py::ParameterSpace`; the `config/manager.py::_calculate_midpoint_defaults` misleading comment. File these as separate follow-up issues if desired — do not fold them into this plan's tasks or commits.
