# Fix Remaining Debug-Audit Bugs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 6 bugs held back from PR #14's second-pass debug audit (each needed a design decision, now settled and codex/agy-reviewed in `docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md`).

**Architecture:** Each fix is independent — different files, no shared state between tasks except Task 4a/4b sharing the "sigma-weighted L2 hierarchical loss" pattern across two otherwise-separate modules (laminar_flow vs two_component). Every task follows TDD: write a test that exercises the real function (not a reimplemented closure), watch it fail for the right reason, implement, watch it pass.

**Tech Stack:** Python 3.12+, JAX (`jax.numpy`), pytest, `uv run`.

## Global Constraints

- Every fix must be a minimal, targeted diff — no reformatting, no unrelated refactors.
- Every regression test must call the real function/method, not restate its logic in a local closure (this was the exact anti-pattern PR #14's review caught and required fixing).
- After all 7 tasks: run `make verify` (ruff + advisory mypy + parallel smoke) and the domain-scoped targets `make test-optimization` / `make test-heterodyne`, since Tasks 3 and 4 touch actual solve paths with no `rtol` golden gate.
- `git add -f` is required for anything under `docs/superpowers/` (gitignored).
- Working directory for all commands: `/home/wei/Documents/GitHub/xpcsjax/.claude/worktrees/fix-remaining-debug-audit-bugs` (already an isolated git worktree, branch `worktree-fix-remaining-debug-audit-bugs`).

---

### Task 1: NaN-tolerant `wavevector_q_list` + guarded scalar-extraction sites

**Files:**
- Modify: `xpcsjax/data/xpcs_loader.py:322-349` (`_validate_loaded_arrays`)
- Modify: `xpcsjax/optimization/nlsq/core.py:679-726` (`_normalize_data_to_object`)
- Modify: `xpcsjax/optimization/nlsq/core.py:1645-1810` (`fit_nlsq_cmaes`, the `q = float(...)` extraction around line 1802)
- Modify: `xpcsjax/optimization/nlsq/adapter.py:837-880` (`NLSQAdapter._build_model_function`)
- Test: `tests/data/test_loaded_array_validation.py` (extend existing file)
- Test: `tests/optimization/test_debug_audit_2026_07_23_nan_q.py` (new file, extraction-site guards)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing tests for the validator change**

Add to `tests/data/test_loaded_array_validation.py` (after `test_accepts_non_monotonic_q_list`):

```python
def test_accepts_nan_in_q_list():
    # Bad-pixel NaN in wavevector_q_list is legitimate and must not raise.
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.01, np.nan, 0.02])
    assert _validate_loaded_arrays(data, source="ok.h5") is None


def test_accepts_all_nan_q_list():
    # An all-NaN q-list must also not raise the hard-fail gate (downstream
    # nan-safe consumers degrade gracefully; this validator's job is only
    # to reject non-NaN corruption like inf).
    data = _good_data()
    data["wavevector_q_list"] = np.full(3, np.nan)
    assert _validate_loaded_arrays(data, source="ok.h5") is None


def test_still_rejects_negative_inf_in_q_list():
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.01, -np.inf, 0.02])
    with pytest.raises(XPCSDataFormatError, match="inf"):
        _validate_loaded_arrays(data, source="evil.h5")
```

Note: `test_rejects_inf_in_q_list` already exists in this file (asserts `+inf` still raises) — do not remove it; it must keep passing after this change. **However its assertion's `match="NaN/inf"` will NOT survive Step 3 as-is**: Step 3 gives `wavevector_q_list` its own branch whose message is `"...contains inf values..."` (no literal "NaN" substring), so `re.search("NaN/inf", msg)` would fail post-fix. Update that pre-existing test's `pytest.raises` call in this same step, from:

```python
def test_rejects_inf_in_q_list():
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.01, np.inf, 0.02])
    with pytest.raises(XPCSDataFormatError, match="NaN/inf"):
        _validate_loaded_arrays(data, source="evil.h5")
```

to:

```python
def test_rejects_inf_in_q_list():
    data = _good_data()
    data["wavevector_q_list"] = np.array([0.01, np.inf, 0.02])
    with pytest.raises(XPCSDataFormatError, match="inf"):
        _validate_loaded_arrays(data, source="evil.h5")
```

`match="inf"` is a substring of both the pre-fix message ("...contains NaN/inf values...") and the post-fix message ("...contains inf values..."), so this edit is safe to make now, before Step 3's implementation exists.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_loaded_array_validation.py -v`
Expected: `test_accepts_nan_in_q_list` and `test_accepts_all_nan_q_list` FAIL with `XPCSDataFormatError: wavevector_q_list from 'ok.h5' contains NaN/inf values...` (the current blanket check rejects NaN too). `test_still_rejects_negative_inf_in_q_list` and the pre-existing `test_rejects_inf_in_q_list` (now asserting `match="inf"` per the edit above) should already PASS (nothing to verify-fail there, just confirming no regression yet).

- [ ] **Step 3: Implement the validator change**

In `xpcsjax/data/xpcs_loader.py`, replace the loop body of `_validate_loaded_arrays` (currently):

```python
    for key in ("c2_exp", "t1", "t2", "wavevector_q_list", "phi_angles_list"):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size and not np.all(np.isfinite(arr)):
            raise XPCSDataFormatError(
                f"{key} from {source!r} contains NaN/inf values; refusing to "
                "proceed with corrupt correlation data."
            )
```

with:

```python
    for key in ("c2_exp", "t1", "t2", "phi_angles_list"):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size and not np.all(np.isfinite(arr)):
            raise XPCSDataFormatError(
                f"{key} from {source!r} contains NaN/inf values; refusing to "
                "proceed with corrupt correlation data."
            )

    # wavevector_q_list gets its own, NaN-tolerant check: NaN there is
    # legitimate (one entry per (q, phi) pair; a bad/masked detector pixel
    # legitimately produces NaN at that pair's q-value), but inf/-inf still
    # indicates corrupt data and must keep hard-failing.
    if "wavevector_q_list" in data:
        q_arr = np.asarray(data["wavevector_q_list"])
        if q_arr.size and np.isinf(q_arr).any():
            raise XPCSDataFormatError(
                f"wavevector_q_list from {source!r} contains inf values; "
                "refusing to proceed with corrupt correlation data."
            )
```

Also update the two docstrings that describe this function/module as rejecting NaN/inf in *every* loaded array — after this change `wavevector_q_list` is the one exception (NaN-tolerant, inf-rejecting). In this same file's module docstring, change:

```
Runtime validation runs unconditionally at the I/O boundary: loaded arrays are
checked for finite values (no NaN/inf), square 2-D correlation matrices, bounded
allocation size, and monotonic time axes. Any violation raises
:class:`XPCSDataFormatError`.
```

to:

```
Runtime validation runs unconditionally at the I/O boundary: loaded arrays are
checked for finite values (no NaN/inf, except ``wavevector_q_list`` which
tolerates NaN — see below), square 2-D correlation matrices, bounded
allocation size, and monotonic time axes. Any violation raises
:class:`XPCSDataFormatError`.
```

And in `_validate_loaded_arrays`'s own docstring, change:

```
    * **Finite values** — no NaN/inf in any loaded array. Corrupt data must stop
      the run, not silently drive a numerically wrong fit.
```

to:

```
    * **Finite values** — no NaN/inf in any loaded array, EXCEPT
      ``wavevector_q_list`` which tolerates NaN (legitimate bad-pixel masking,
      one entry per (q, phi) pair) but still hard-rejects inf. Corrupt data
      must stop the run, not silently drive a numerically wrong fit.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_loaded_array_validation.py -v`
Expected: all PASS, including the pre-existing `test_rejects_inf_in_q_list` (with its `match="inf"` update from Step 1) and `test_rejects_nan_in_c2_exp`.

- [ ] **Step 5: Write the failing tests for the extraction-site guards**

Create `tests/optimization/test_debug_audit_2026_07_23_nan_q.py`:

```python
"""Regression: scalar q-extraction sites must reject NaN, not silently poison
the JAX physics model.

Finding #1 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md:
xpcs_loader.py's _validate_loaded_arrays now tolerates NaN in
wavevector_q_list (bad-pixel masking is legitimate). But three call sites
extract wavevector_q_list[0] as a bare scalar q and feed it straight into the
JAX physics model with no guard. A bad-pixel NaN landing at index 0 (or being
the only value reaching these extraction sites) must raise, not silently
produce a NaN-poisoned fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from xpcsjax.optimization.nlsq.adapter import NLSQAdapter
from xpcsjax.optimization.nlsq.core import _normalize_data_to_object


def test_normalize_data_to_object_rejects_nan_q():
    data = {
        "phi_angles_list": np.array([0.0, 45.0]),
        "c2_exp": np.ones((2, 4, 4)),
        "wavevector_q_list": np.array([np.nan, 0.02]),
    }
    with pytest.raises(ValueError, match="wavevector_q_list"):
        _normalize_data_to_object(data, config=object(), logger=logging.getLogger("t"))


def test_normalize_data_to_object_accepts_finite_q():
    data = {
        "phi_angles_list": np.array([0.0, 45.0]),
        "c2_exp": np.ones((2, 4, 4)),
        "wavevector_q_list": np.array([0.02, np.nan]),  # NaN elsewhere is fine
    }
    obj = _normalize_data_to_object(data, config=object(), logger=logging.getLogger("t"))
    assert obj.q == pytest.approx(0.02)


def test_build_model_function_rejects_nan_q(monkeypatch):
    adapter = NLSQAdapter.__new__(NLSQAdapter)  # bypass __init__, only need _build_model_function
    data = {
        "wavevector_q_list": np.array([np.nan]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }
    with pytest.raises(ValueError, match="wavevector_q_list"):
        adapter._build_model_function(
            data,
            config=object(),
            analysis_mode=None,
            per_angle_scaling=False,
            n_phi=2,
        )
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_nan_q.py -v`
Expected: `test_normalize_data_to_object_rejects_nan_q` and `test_build_model_function_rejects_nan_q` FAIL — no `ValueError` is currently raised (the extraction sites are unguarded; `test_normalize_data_to_object_rejects_nan_q` will actually get past construction with `data_obj.q = nan` and fail later or not raise at all — confirm the actual failure mode when you run it, since `_normalize_data_to_object` may raise a *different* error (e.g. from `_ensure_positive_sigma`) before reaching completion; if so note the real pre-fix failure in the test docstring). `test_normalize_data_to_object_accepts_finite_q` should already PASS.

- [ ] **Step 7: Implement the three extraction-site guards**

In `xpcsjax/optimization/nlsq/core.py`'s `_normalize_data_to_object`, change:

```python
        # Extract scalar q from wavevector_q_list if present
        if hasattr(data_obj, "wavevector_q_list"):
            q_list = np.atleast_1d(np.asarray(data_obj.wavevector_q_list))
            if q_list.size > 0:
                data_obj.q = float(q_list[0])
                logger.debug(f"Extracted q = {data_obj.q:.6f} from wavevector_q_list")
```

to:

```python
        # Extract scalar q from wavevector_q_list if present
        if hasattr(data_obj, "wavevector_q_list"):
            q_list = np.atleast_1d(np.asarray(data_obj.wavevector_q_list))
            if q_list.size > 0:
                extracted_q = float(q_list[0])
                if not np.isfinite(extracted_q):
                    raise ValueError(
                        "wavevector_q_list[0] is not finite (NaN/inf); a "
                        "bad-pixel q-value must not silently reach the fit"
                    )
                data_obj.q = extracted_q
                logger.debug(f"Extracted q = {data_obj.q:.6f} from wavevector_q_list")
```

In `xpcsjax/optimization/nlsq/core.py`'s `fit_nlsq_cmaes` (around line 1802), change:

```python
        # Get q value
        if "wavevector_q_list" in data:
            q = float(np.asarray(data["wavevector_q_list"])[0])
        else:
            q = float(data.get("q", 0.01))
```

to:

```python
        # Get q value
        if "wavevector_q_list" in data:
            q = float(np.asarray(data["wavevector_q_list"])[0])
            if not np.isfinite(q):
                raise ValueError(
                    "wavevector_q_list[0] is not finite (NaN/inf); a "
                    "bad-pixel q-value must not silently reach the fit"
                )
        else:
            q = float(data.get("q", 0.01))
```

In `xpcsjax/optimization/nlsq/adapter.py`'s `NLSQAdapter._build_model_function`, change:

```python
        # Extract wavevector q
        q = self._get_attr(data, "q")
        if q is None:
            q = self._get_attr(data, "wavevector_q_list", [1.0])
        if isinstance(q, (list, np.ndarray)):
            q = q[0]
```

to:

```python
        # Extract wavevector q
        q = self._get_attr(data, "q")
        if q is None:
            q = self._get_attr(data, "wavevector_q_list", [1.0])
        if isinstance(q, (list, np.ndarray)):
            q = q[0]
        if not np.isfinite(float(q)):
            raise ValueError(
                "wavevector_q_list[0] is not finite (NaN/inf); a bad-pixel "
                "q-value must not silently reach the fit"
            )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_nan_q.py tests/data/test_loaded_array_validation.py -v`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add xpcsjax/data/xpcs_loader.py xpcsjax/optimization/nlsq/core.py xpcsjax/optimization/nlsq/adapter.py tests/data/test_loaded_array_validation.py tests/optimization/test_debug_audit_2026_07_23_nan_q.py
git commit -m "fix: tolerate NaN (not inf) in wavevector_q_list, guard scalar q extraction

Finding #1 of the 2026-07-23 debug-audit-fixes spec. NaN in
wavevector_q_list is legitimate bad-pixel masking; inf still indicates
corrupt data. The three call sites that extract wavevector_q_list[0] as a
scalar q now raise on non-finite values instead of silently feeding a NaN
q into the JAX physics model."
```

---

### Task 2: Skip mandatory diagonal correction when preprocessing already ran it

**Files:**
- Modify: `xpcsjax/data/preprocessing.py:483-538` (`PreprocessingPipeline._execute_stage`, `CORRECT_DIAGONAL` branch)
- Modify: `xpcsjax/data/xpcs_loader.py:1033-1040` (mandatory diagonal-correction call site)
- Test: `tests/data/test_debug_audit_2026_07_23_diagonal_skip.py` (new file)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_debug_audit_2026_07_23_diagonal_skip.py`:

```python
"""Regression: the loader's mandatory diagonal correction must not silently
overwrite a diagonal correction preprocessing already applied.

Finding #2 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.data.preprocessing import PreprocessingPipeline


def _synthetic_data(n_mat: int = 2, n_t: int = 6) -> dict:
    rng = np.random.default_rng(0)
    c2 = np.ones((n_mat, n_t, n_t)) + 0.1 * rng.standard_normal((n_mat, n_t, n_t))
    # Make the diagonal deliberately wrong so a real correction changes it.
    for i in range(n_mat):
        np.fill_diagonal(c2[i], 5.0)
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01] * n_mat),
        "phi_angles_list": np.linspace(0.0, 90.0, n_mat),
    }


def test_correct_diagonal_stage_marks_data_as_corrected():
    config = {
        "preprocessing": {
            "enabled": True,
            "stages": {"correct_diagonal": {"method": "statistical"}},
        },
    }
    pipeline = PreprocessingPipeline(config)
    result = pipeline.process(_synthetic_data())
    assert result.success
    assert result.data.get("_diagonal_corrected") is True
    # The 'statistical' method must actually have changed the diagonal away
    # from the deliberately-wrong 5.0 seed value.
    assert not np.allclose(np.diagonal(result.data["c2_exp"][0]), 5.0)


def test_disabled_preprocessing_does_not_set_marker():
    config = {"preprocessing": {"enabled": False}}
    pipeline = PreprocessingPipeline(config)
    result = pipeline.process(_synthetic_data())
    assert "_diagonal_corrected" not in result.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_diagonal_skip.py -v`
Expected: `test_correct_diagonal_stage_marks_data_as_corrected` FAILS with `assert None is True` (or `AssertionError` on the marker check) — the marker doesn't exist yet. `test_disabled_preprocessing_does_not_set_marker` should already PASS.

- [ ] **Step 3: Implement the marker in `_execute_stage`**

In `xpcsjax/data/preprocessing.py`'s `_execute_stage`, change:

```python
        if stage == PreprocessingStage.CORRECT_DIAGONAL:
            processed_data = self._correct_diagonal_enhanced(data, stage_config)
            method = stage_config.get("method", "statistical")
```

to:

```python
        if stage == PreprocessingStage.CORRECT_DIAGONAL:
            processed_data = self._correct_diagonal_enhanced(data, stage_config)
            processed_data["_diagonal_corrected"] = True
            method = stage_config.get("method", "statistical")
```

- [ ] **Step 4: Run test to verify it passes (marker half)**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_diagonal_skip.py -v`
Expected: both tests PASS now (the marker is set; the loader-skip half isn't tested yet by this file, that requires the xpcs_loader.py change in Step 5, which is a separate end-to-end concern covered by the existing PR #14 test for finding #2 in the spec's Testing section — add it now).

- [ ] **Step 5: Write the failing end-to-end test for the loader skip**

Add to `tests/data/test_debug_audit_2026_07_23_diagonal_skip.py`:

```python
def test_loader_skips_mandatory_correction_when_already_corrected(monkeypatch):
    """End-to-end: apply_diagonal_correction_batch must NOT be called again
    when the preprocessing pipeline already set _diagonal_corrected."""
    import xpcsjax.data.xpcs_loader as xl

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    data = _synthetic_data()
    data["_diagonal_corrected"] = True
    result = xl._maybe_apply_mandatory_diagonal_correction(data)
    assert calls == [], "mandatory correction must be skipped when already corrected"
    assert result is data


def test_loader_applies_mandatory_correction_when_not_preprocessed(monkeypatch):
    import xpcsjax.data.xpcs_loader as xl

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    data = _synthetic_data()
    xl._maybe_apply_mandatory_diagonal_correction(data)
    assert len(calls) == 1, "mandatory correction must still run by default"


def test_loader_applies_configured_diagonal_correction_end_to_end(tmp_path, monkeypatch):
    """End-to-end (design spec Testing item 2): drive the REAL
    load_experimental_data with preprocessing enabled and a non-'basic'
    correct_diagonal.method, and confirm both that the mandatory post-load
    'basic' pass is skipped AND that the final c2_exp diagonal reflects the
    configured method's own correction -- not just the isolated helper's
    unit behavior. A wiring bug between _apply_preprocessing_pipeline and
    _maybe_apply_mandatory_diagonal_correction inside load_experimental_data
    itself would not be caught by the helper-level tests above; this is."""
    import xpcsjax.data.xpcs_loader as xl

    hdf_path = tmp_path / "fake.h5"
    hdf_path.write_bytes(b"")  # existence is all load_experimental_data checks
    # before handing off to _load_from_hdf, which is replaced below.

    def fake_load_from_hdf(self, path):
        return _synthetic_data()

    monkeypatch.setattr(xl.XPCSDataLoader, "_load_from_hdf", fake_load_from_hdf)

    calls = []
    original = xl.apply_diagonal_correction_batch

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(xl, "apply_diagonal_correction_batch", spy)

    config = {
        "experimental_data": {
            "data_folder_path": str(tmp_path),
            "data_file_name": "fake.h5",
        },
        "v2_features": {"cache_strategy": "none"},
        "preprocessing": {
            "enabled": True,
            "stages": {"correct_diagonal": {"method": "statistical"}},
        },
    }
    loader = xl.XPCSDataLoader(config_dict=config, configure_logging=False)
    result = loader.load_experimental_data()

    assert calls == [], (
        "the mandatory post-load 'basic' correction must be skipped end-to-end "
        "-- preprocessing's 'statistical' correction already ran"
    )
    c2 = np.asarray(result["c2_exp"])
    assert not np.allclose(np.diagonal(c2[0]), 5.0), (
        "preprocessing's configured 'statistical' correction must actually "
        "have changed the deliberately-wrong seeded diagonal, end-to-end"
    )
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_diagonal_skip.py -v`
Expected: `test_loader_skips_mandatory_correction_when_already_corrected` and `test_loader_applies_mandatory_correction_when_not_preprocessed` FAIL with `AttributeError: module 'xpcsjax.data.xpcs_loader' has no attribute '_maybe_apply_mandatory_diagonal_correction'` — this helper doesn't exist yet; Step 7 extracts the mandatory-correction call site into it (a small, independently-testable unit, per this plan's Task Right-Sizing convention) rather than testing it only in the middle of the much larger `load_experimental_data`. `test_loader_applies_configured_diagonal_correction_end_to_end` FAILS with `assert calls == []` failing (`len(calls) == 1` today, since the mandatory correction always runs pre-fix) — this is the real end-to-end bug the whole task fixes.

- [ ] **Step 7: Extract the call site into a testable helper and wire the skip**

In `xpcsjax/data/xpcs_loader.py`, find:

```python
        # Apply mandatory diagonal correction (post-load for consistent behavior)
        # Uses unified diagonal_correction module
        logger.debug("Applying mandatory diagonal correction to correlation matrices")
        if HAS_DIAGONAL_CORRECTION:
            data["c2_exp"] = apply_diagonal_correction_batch(data["c2_exp"])
        else:
            # Fallback to local implementation if unified module not available
            data["c2_exp"] = self._correct_diagonal_batch(data["c2_exp"])
```

Replace it with a call to a new module-level helper:

```python
        data = _maybe_apply_mandatory_diagonal_correction(data, self._correct_diagonal_batch)
```

And add the helper function at module level (near `apply_diagonal_correction_batch`'s import, before the class that contains `load_experimental_data`):

```python
def _maybe_apply_mandatory_diagonal_correction(
    data: dict[str, Any],
    fallback_correct_diagonal_batch: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Apply the mandatory post-load diagonal correction, unless the
    preprocessing pipeline already corrected the diagonal.

    Preprocessing's CORRECT_DIAGONAL stage sets ``data["_diagonal_corrected"]
    = True`` on success (xpcsjax/data/preprocessing.py's ``_execute_stage``).
    Re-applying the mandatory 'basic' correction on top of that would
    silently discard whatever method the user configured
    (statistical/interpolation) — see Finding #2 of the 2026-07-23
    debug-audit-fixes spec.
    """
    if data.get("_diagonal_corrected", False):
        logger.debug(
            "Skipping mandatory diagonal correction: preprocessing already "
            "corrected the diagonal (_diagonal_corrected=True)"
        )
        return data

    logger.debug("Applying mandatory diagonal correction to correlation matrices")
    if HAS_DIAGONAL_CORRECTION:
        data["c2_exp"] = apply_diagonal_correction_batch(data["c2_exp"])
    elif fallback_correct_diagonal_batch is not None:
        data["c2_exp"] = fallback_correct_diagonal_batch(data["c2_exp"])
    return data
```

(`Callable` and `Any` are already imported in this module per the existing type hints elsewhere in the file — verify at the top of `xpcs_loader.py` and add `from collections.abc import Callable` if it's missing.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_diagonal_skip.py -v`
Expected: all 5 tests PASS (the 2 marker/skip-helper tests from Step 1/5, the skip/apply helper tests, and the new end-to-end test).

- [ ] **Step 9: Run the full existing xpcs_loader test suite to confirm no regression**

Run: `uv run pytest tests/data/test_loaded_array_validation.py tests/data/ -k "diagonal or loader" -v`
Expected: all PASS (the extraction preserves identical behavior for the default/no-marker case).

- [ ] **Step 10: Commit**

```bash
git add xpcsjax/data/preprocessing.py xpcsjax/data/xpcs_loader.py tests/data/test_debug_audit_2026_07_23_diagonal_skip.py
git commit -m "fix: skip mandatory diagonal correction when preprocessing already corrected it

Finding #2 of the 2026-07-23 debug-audit-fixes spec. The loader
previously always re-applied a 'basic' diagonal correction after
preprocessing's CORRECT_DIAGONAL stage ran, silently overwriting the
user's configured method (default 'statistical'). Extracted the mandatory
correction into a testable helper that skips when
data['_diagonal_corrected'] is already True."
```

---

### Task 3: STREAMING + recovery soft-failure escalation

**Files:**
- Modify: `xpcsjax/optimization/nlsq/fallback_chain.py:218-458` (`execute_optimization_with_fallback`)
- Test: `tests/test_debug_audit_regressions.py` (extend existing file, alongside the sibling `test_fallback_no_recovery_reports_failed_on_stagnation` test)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_debug_audit_regressions.py` (after `test_fallback_no_recovery_reports_failed_on_stagnation`):

```python
def test_streaming_soft_failure_escalates_to_chunked() -> None:
    """Audit [2026-07-23]: a STREAMING soft-failure (success=False, no
    exception) must escalate to the next fallback strategy (CHUNKED), not
    terminate immediately with convergence_status='partial'. Finding #3 of
    the 2026-07-23 debug-audit-fixes spec."""
    import logging
    import time

    from xpcsjax.optimization.nlsq.fallback_chain import (
        OptimizationStrategy,
        execute_optimization_with_fallback,
    )

    p0 = np.array([1.0, 2.0])
    attempted_strategies = []

    def fake_streaming(**kwargs):
        attempted_strategies.append("streaming")
        return p0, np.eye(2), {"success": False}

    def fake_curve_fit_large(_resid, _x, _y, p0, **_kw):
        attempted_strategies.append("chunked")
        popt = np.asarray(p0, dtype=float) + 1.0  # visibly different -> "converged"
        return popt, np.eye(len(popt)), {}

    popt, pcov, info, recovery_actions, status = execute_optimization_with_fallback(
        strategy=OptimizationStrategy.STREAMING,
        wrapped_residual_fn=lambda p, x: np.zeros_like(x),
        xdata=np.arange(5.0),
        ydata=np.zeros(5),
        validated_params=p0,
        nlsq_bounds=None,
        loss_name="linear",
        x_scale_value=1.0,
        config=object(),
        start_time=time.time(),
        log=logging.getLogger("test_streaming_escalation"),
        enable_recovery=False,
        execute_with_recovery_fn=lambda **_k: None,  # not reached
        fit_with_hybrid_streaming_fn=fake_streaming,
        streaming_available=True,
        curve_fit_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("STANDARD not reached")),
        curve_fit_large_fn=fake_curve_fit_large,
    )

    assert attempted_strategies == ["streaming", "chunked"], (
        f"expected escalation streaming->chunked, got {attempted_strategies!r}"
    )
    assert status == "converged"


def test_recovery_soft_failure_escalates_to_next_strategy() -> None:
    """Audit [2026-07-23]: enable_recovery=True's execute_with_recovery
    returning convergence_status='failed' (a plain return, not a raise)
    must also escalate to the next fallback strategy — the default
    (enable_recovery=True) flow, not an edge case. Finding #3."""
    import logging
    import time

    from xpcsjax.optimization.nlsq.fallback_chain import (
        OptimizationStrategy,
        execute_optimization_with_fallback,
    )

    p0 = np.array([1.0, 2.0])
    attempted_strategies = []

    def fake_recovery(**kwargs):
        attempted_strategies.append("recovery_standard")
        return p0, np.eye(2), {}, [], "failed"

    def fake_curve_fit(_resid, _x, _y, p0, **_kw):
        # STANDARD is the base of the chain; get_fallback_strategy(STANDARD)
        # returns None, so this scenario must be entered via a strategy that
        # HAS a fallback -- use CHUNKED so the next attempt is STANDARD via
        # the enable_recovery branch too (both attempts are "recovery"-shaped
        # because enable_recovery=True routes every non-STREAMING strategy
        # through execute_with_recovery_fn).
        raise AssertionError("plain curve_fit path not reached under enable_recovery=True")

    popt, pcov, info, recovery_actions, status = execute_optimization_with_fallback(
        strategy=OptimizationStrategy.CHUNKED,
        wrapped_residual_fn=lambda p, x: np.zeros_like(x),
        xdata=np.arange(5.0),
        ydata=np.zeros(5),
        validated_params=p0,
        nlsq_bounds=None,
        loss_name="linear",
        x_scale_value=1.0,
        config=object(),
        start_time=time.time(),
        log=logging.getLogger("test_recovery_escalation"),
        enable_recovery=True,
        execute_with_recovery_fn=fake_recovery,
        fit_with_hybrid_streaming_fn=lambda **_k: None,  # not reached
        streaming_available=False,
        curve_fit_fn=fake_curve_fit,
        curve_fit_large_fn=fake_curve_fit,
    )

    assert len(attempted_strategies) >= 2, (
        f"expected recovery-soft-failure to escalate past the first strategy, "
        f"got {attempted_strategies!r}"
    )
```

Note: `fake_recovery` always returns `"failed"` regardless of which strategy it's called for — this is deliberate, so the test observes the fallback chain actually iterating (CHUNKED → LARGE → STANDARD) via `attempted_strategies` growing, rather than converging. If your implementation of the shared success predicate causes this to raise `RuntimeError` once STANDARD is also exhausted (expected, per the spec's corrected Guardrail), wrap the call in `pytest.raises(RuntimeError)` instead of unpacking a return tuple — check which actually happens when you run Step 2 and adjust the test to match the real (spec-documented) behavior; do not adjust the source to avoid the raise.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_debug_audit_regressions.py -k "escalat" -v`
Expected: `test_streaming_soft_failure_escalates_to_chunked` FAILS with `attempted_strategies == ["streaming"]` (only one attempt, no escalation to chunked) or an `UnboundLocalError`/wrong-`popt` assertion. `test_recovery_soft_failure_escalates_to_next_strategy` FAILS similarly (only one call to `fake_recovery`). Confirm the exact failure text and, if the second test's control flow doesn't match reality (e.g. `RuntimeError` is raised instead of returning), rewrite the test's final assertion per the note in Step 1 before moving on — do not proceed to Step 3 until you've confirmed exactly why each test fails.

- [ ] **Step 3: Implement the shared success predicate and route both branches through escalation**

In `xpcsjax/optimization/nlsq/fallback_chain.py`, add a module-level helper near the top (after imports, before `get_fallback_strategy`):

```python
def _is_soft_failure(convergence_status: str) -> bool:
    """True when a strategy attempt completed (no exception) but did not
    actually succeed, per the shared escalation predicate: Finding #3 of the
    2026-07-23 debug-audit-fixes spec. Both the STREAMING branch's
    success=False->"partial" and the recovery branch's "failed" (a plain
    return from execute_with_recovery, not a raise) must escalate to the
    next fallback strategy exactly like a caught exception does, per this
    function's own docstring ("degrades... until one succeeds or all are
    exhausted")."""
    return convergence_status in ("partial", "failed")
```

Then change the STREAMING branch from:

```python
                popt, pcov, info = fit_with_hybrid_streaming_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=log,
                    nlsq_config=config,
                )
                recovery_actions = info.get("recovery_actions", [])
                convergence_status = "converged" if info.get("success", False) else "partial"
```

to:

```python
                popt, pcov, info = fit_with_hybrid_streaming_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=log,
                    nlsq_config=config,
                )
                recovery_actions = info.get("recovery_actions", [])
                convergence_status = "converged" if info.get("success", False) else "partial"
                if _is_soft_failure(convergence_status):
                    raise RuntimeError(
                        f"STREAMING strategy completed without converging "
                        f"(convergence_status={convergence_status!r}); escalating"
                    )
```

and the recovery branch from:

```python
            elif enable_recovery:
                popt, pcov, info, recovery_actions, convergence_status = execute_with_recovery_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    strategy=current_strategy,
                    logger=log,
                    loss_name=loss_name,
                    x_scale_value=x_scale_value,
                    callback=callback,
                )
```

to:

```python
            elif enable_recovery:
                popt, pcov, info, recovery_actions, convergence_status = execute_with_recovery_fn(
                    residual_fn=wrapped_residual_fn,
                    xdata=xdata,
                    ydata=ydata,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    strategy=current_strategy,
                    logger=log,
                    loss_name=loss_name,
                    x_scale_value=x_scale_value,
                    callback=callback,
                )
                if _is_soft_failure(convergence_status):
                    raise RuntimeError(
                        f"{current_strategy.value} strategy (recovery path) completed "
                        f"without converging (convergence_status={convergence_status!r}); "
                        "escalating"
                    )
```

Raising `RuntimeError` here routes both cases through the existing `except (ValueError, RuntimeError, TypeError, AttributeError, OSError, MemoryError)` block below, which already implements the CHUNKED→LARGE→STANDARD escalation via `get_fallback_strategy`. Do not modify the `except` block itself — this reuses it as-is, per the spec's stated preference for the smaller diff.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_debug_audit_regressions.py -v`
Expected: all tests in the file PASS, including the two new ones and the pre-existing `test_fallback_no_recovery_reports_failed_on_stagnation` (unaffected — that test uses `enable_recovery=False` and the plain `else` branch, which this task does not touch).

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/optimization/nlsq/fallback_chain.py tests/test_debug_audit_regressions.py
git commit -m "fix: STREAMING and recovery soft-failures now escalate to the next fallback strategy

Finding #3 of the 2026-07-23 debug-audit-fixes spec. Both
execute_optimization_with_fallback's STREAMING branch (success=False) and
its enable_recovery=True branch (execute_with_recovery returning
convergence_status='failed') previously fell through to an unconditional
break, never reaching the escalation logic despite the function's
docstring promising to degrade 'until one succeeds or all are exhausted'.
A shared _is_soft_failure predicate now routes both through the existing
exception-based escalation path. Per the spec's corrected Guardrail: a
fully-exhausted fallback chain now raises RuntimeError (as it always did
for hard exceptions) rather than silently returning a partial result --
this is the intended consequence of honoring the docstring's contract."
```

---

### Task 4a: L2 hierarchical sigma-weighting — `two_component` (heterodyne)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:795-828` (`_hier_loss`, `_loss_jax`)
- Test: `tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py` (new file — shared by Task 4a and 4b)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing Task 4b depends on (independent edits to a different file), but shares one test file — Task 4b appends to it.

- [ ] **Step 1: Write the failing test**

Create `tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py`:

```python
"""Regression: the L2 hierarchical loss must honor per-point sigma weighting,
matching the sibling plain-path branch in the same function.

Finding #4 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
Covers both heterodyne_hybrid_streaming.py (two_component) and
hybrid_streaming.py (laminar_flow).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_heterodyne_hier_loss_honors_nonuniform_sigma():
    """_sigma_weighted_mse (the arithmetic building block _hier_loss/_loss_jax
    call) must divide residuals by sigma when sigma is non-uniform, not just
    compute an unweighted mean(residuals**2). This is a pure arithmetic check
    of the helper in isolation -- see
    test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse below for
    the separate proof that _hier_loss/_loss_jax actually CALL this helper
    (an implementation could add the helper without wiring it in and this
    test alone would not catch that)."""
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hhs

    y_data = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.5, 3.5, 3.5])  # residuals = [-0.5, 0.5, -0.5, 0.5]
    residuals = y_data - pred
    uniform_sigma = np.ones(4)
    nonuniform_sigma = np.array([0.1, 1.0, 0.1, 1.0])

    loss_uniform = hhs._sigma_weighted_mse(residuals, uniform_sigma) * y_data.shape[0]
    loss_nonuniform = hhs._sigma_weighted_mse(residuals, nonuniform_sigma) * y_data.shape[0]
    loss_unweighted = np.mean(residuals**2) * y_data.shape[0]

    assert loss_uniform == pytest.approx(loss_unweighted)
    assert loss_nonuniform != pytest.approx(loss_unweighted)


def test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse(monkeypatch):
    """Wiring check: drive the REAL fit_with_stratified_hybrid_streaming_heterodyne
    through its L2 hierarchical branch (per_angle_mode='individual') with
    non-uniform per-point weights, and spy on the module-level
    _sigma_weighted_mse to prove _hier_loss/_loss_jax actually call it. Reuses
    the proven synthetic-heterodyne fixture from
    tests/optimization/test_heterodyne_hybrid_streaming.py's own
    test_l2_individual_runs_and_beats_frozen_baseline (n_phi=2 -> auto
    resolves to individual, exercising the L2 branch)."""
    from xpcsjax.optimization.nlsq.heterodyne_stratified_data import (
        build_heterodyne_stratified_data,
    )
    from xpcsjax.optimization.nlsq.strategies import heterodyne_hybrid_streaming as hs

    from tests.optimization.test_heterodyne_hybrid_streaming import (
        _make_synthetic_heterodyne,
    )

    model, c2, phi = _make_synthetic_heterodyne(n_phi=2, n_t=8)
    rng = np.random.default_rng(0)
    weights = 1.0 + rng.random(c2.shape)  # non-uniform -> non-uniform sigma
    strat = build_heterodyne_stratified_data(model, c2, phi, weights=weights)
    lo, hi = model.param_manager.get_bounds()

    call_count = [0]
    original = hs._sigma_weighted_mse

    def spy(residuals, sigma):
        call_count[0] += 1
        return original(residuals, sigma)

    monkeypatch.setattr(hs, "_sigma_weighted_mse", spy)

    hs.fit_with_stratified_hybrid_streaming_heterodyne(
        stratified_data=strat,
        model=model,
        physical_param_names=list(model.param_manager.varying_names),
        initial_params=np.asarray(model.param_manager.get_initial_values(), dtype=np.float64),
        bounds=(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)),
        hybrid_config={"verbose": 0},
        anti_degeneracy_config={
            "per_angle_mode": "individual",
            "hierarchical": {"max_outer_iterations": 3},
        },
    )

    assert call_count[0] > 0, (
        "_hier_loss/_loss_jax never called _sigma_weighted_mse -- the helper "
        "exists but is not wired into the hierarchical loss closures"
    )
```

Note: if the L2 branch isn't entered with this fixture (`call_count[0]` stays 0 for a reason unrelated to wiring), inspect the same gating `test_l2_individual_runs_and_beats_frozen_baseline` already relies on (`per_angle_mode="individual"` forces the L2 branch regardless of `n_phi`) before concluding the wiring is broken.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py::test_heterodyne_hier_loss_honors_nonuniform_sigma tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py::test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse -v`
Expected: both FAIL with `AttributeError: module '...heterodyne_hybrid_streaming' has no attribute '_sigma_weighted_mse'` — this helper doesn't exist yet.

- [ ] **Step 3: Implement `_sigma_weighted_mse` and use it in both closures**

In `xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py`, add a module-level helper near the top of the file (after imports):

```python
def _sigma_weighted_mse(residuals: Any, sigma: Any | None) -> Any:
    """Mean-squared residual, optionally weighted by per-point sigma.

    Mirrors the safe_sigma/valid_sigma EPS-guard convention used in
    strategies/residual_jit.py EXACTLY: points with sigma <= EPS are excluded
    from the loss entirely (residual treated as zero for that point, matching
    residual_jit.py's `jnp.where(valid_sigma, (obs-theory)/safe_sigma, 0.0)`)
    rather than falling back to an unweighted raw residual, which would be a
    materially different (and uncited) aggregation behavior. When sigma is
    None, this is exactly the pre-existing unweighted jnp.mean(residuals**2).
    """
    if sigma is None:
        return jnp.mean(residuals**2)
    EPS = 1e-10
    sigma_jax = jnp.asarray(sigma)
    valid_sigma = sigma_jax > EPS
    safe_sigma = jnp.where(valid_sigma, sigma_jax, 1.0)
    weighted_sq = jnp.where(valid_sigma, (residuals / safe_sigma) ** 2, 0.0)
    return jnp.mean(weighted_sq)
```

Then change `_hier_loss` from:

```python
        def _hier_loss(params: np.ndarray) -> float:
            """Loss in the native scaling-first param space [scaling | physics].

            Includes L3 adaptive regularization when active.
            """
            params_jax = jnp.asarray(params)
            pred = model_fn(x_data_jax, *params_jax)
            residuals = y_data_jax - pred
            wl = jnp.mean(residuals**2) * y_data.shape[0]
```

to:

```python
        def _hier_loss(params: np.ndarray) -> float:
            """Loss in the native scaling-first param space [scaling | physics].

            Includes L3 adaptive regularization when active. Honors sigma
            weighting (Finding #4, 2026-07-23) matching the plain-path
            branch's optimizer.fit(sigma=sigma, ...) below.
            """
            params_jax = jnp.asarray(params)
            pred = model_fn(x_data_jax, *params_jax)
            residuals = y_data_jax - pred
            wl = _sigma_weighted_mse(residuals, sigma) * y_data.shape[0]
```

and `_loss_jax` from:

```python
        def _loss_jax(ph: jnp.ndarray) -> jnp.ndarray:
            """Loss in the native scaling-first param space [scaling | physics] (JAX)."""
            pred = model_fn(x_data_jax, *ph)
            residuals = y_data_jax - pred
            wl = jnp.mean(residuals**2) * y_data.shape[0]
```

to:

```python
        def _loss_jax(ph: jnp.ndarray) -> jnp.ndarray:
            """Loss in the native scaling-first param space [scaling | physics] (JAX)."""
            pred = model_fn(x_data_jax, *ph)
            residuals = y_data_jax - pred
            wl = _sigma_weighted_mse(residuals, sigma) * y_data.shape[0]
```

`sigma` is already in scope at both closure definitions (built earlier in `fit_with_stratified_hybrid_streaming_heterodyne` from `meta["sigma"]` — verify this is still true by reading the function before editing; if the local variable has a different name at your checkout, use that name instead).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py::test_heterodyne_hier_loss_honors_nonuniform_sigma tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py::test_heterodyne_hier_loss_actually_wired_to_sigma_weighted_mse -v`
Expected: both PASS.

- [ ] **Step 5: Run the existing heterodyne hybrid-streaming test suite to confirm no regression**

Run: `uv run pytest tests/optimization/test_heterodyne_hybrid_streaming.py -v`
Expected: all PASS (uniform-sigma and sigma=None cases are numerically unchanged by construction — `_sigma_weighted_mse` reduces to the old `jnp.mean(residuals**2)` in both cases).

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py
git commit -m "fix: heterodyne L2 hierarchical loss honors sigma weighting

Finding #4 (heterodyne half) of the 2026-07-23 debug-audit-fixes spec.
_hier_loss/_loss_jax computed an unweighted mean(residuals**2), ignoring
the sigma the sibling plain-path branch already threads into
optimizer.fit(sigma=sigma, ...). sigma was already in scope here (unlike
laminar_flow, which needs new plumbing -- see the next commit); this is
the divide-by-sigma edit only."
```

---

### Task 4b: L2 hierarchical sigma-weighting — `laminar_flow` (prerequisite plumbing + closure edit)

**Files:**
- Modify: `xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:1276-1299` (x_data/y_data/mask construction — add sigma alignment)
- Modify: `xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:1465-1524` (`loss_fn`, `grad_fn`)
- Test: `tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py` (extend from Task 4a)

**Interfaces:**
- Consumes: `_sigma_weighted_mse`-style helper pattern established in Task 4a (a new, separate copy in this file — `hybrid_streaming.py` and `heterodyne_hybrid_streaming.py` are different modules with no shared import between them for this helper; do not try to import across files, just mirror the same small function here).
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing test**

Add to `tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py`:

```python
class _FakeStratifiedDataWithSigma:
    def __init__(self, phi_flat, t1_flat, t2_flat, g2_flat, sigma):
        self.phi_flat = phi_flat
        self.t1_flat = t1_flat
        self.t2_flat = t2_flat
        self.g2_flat = g2_flat
        self.sigma = sigma
        self.q = 0.0237
        self.L = 2_000_000.0
        self.dt = 0.1


class _HierarchicalOptimizerSpy:
    """Stand-in for HierarchicalOptimizer: captures loss_fn/grad_fn and
    returns a stub result without running real optimization."""

    captured: dict = {}

    def __init__(self, config, n_phi, n_physical):
        self.n_phi = n_phi
        self.n_physical = n_physical

    def fit(self, loss_fn, grad_fn, p0, bounds, outer_iteration_callback=None):
        type(self).captured["loss_fn"] = loss_fn
        type(self).captured["grad_fn"] = grad_fn

        class _Result:
            x = p0
            fun = 0.0
            success = True
            n_outer_iterations = 0
            message = "stub"
            history = []

        return _Result()


def test_laminar_loss_fn_honors_nonuniform_sigma(monkeypatch):
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    n_phi = 2
    n_t = 4
    phi = np.repeat([0.0, 45.0], n_t * n_t)
    t1 = np.tile(np.repeat(np.arange(n_t, dtype=float), n_t), n_phi)
    t2 = np.tile(np.tile(np.arange(n_t, dtype=float), n_t), n_phi)
    g2 = np.ones_like(phi) + 0.01 * np.arange(phi.size)
    # sigma must be the raw (n_phi, n_t, n_t) GRID, matching production
    # StratifiedData.sigma (wrapper.py copies it verbatim from
    # original_data.sigma, never flattened) -- the plumbing this task adds
    # in Step 3 indexes it as sigma_3d[phi_idx_arr, t1_idx_arr, t2_idx_arr],
    # which requires 3 real grid axes, not a flat per-point array.
    sigma_3d = np.ones((n_phi, n_t, n_t))
    sigma_3d[0] = 0.1  # non-uniform: first phi angle tightly weighted

    stratified_data = _FakeStratifiedDataWithSigma(phi, t1, t2, g2, sigma_3d)

    monkeypatch.setattr(hs, "HierarchicalOptimizer", _HierarchicalOptimizerSpy)
    _HierarchicalOptimizerSpy.captured = {}

    n_physical = 7
    initial_params = np.concatenate([np.ones(2 * n_phi), np.ones(n_physical)])
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params) * 10)

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=[
            "D0", "alpha", "D_offset", "gamma_dot_t0", "beta",
            "gamma_dot_t_offset", "phi0",
        ],
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma"),
    )

    loss_fn = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    assert loss_fn is not None, "hierarchical path was not entered -- check gating config"

    p0 = initial_params
    loss_with_sigma = float(loss_fn(p0))

    # Now re-run with uniform sigma and confirm the loss differs -- proving
    # sigma is actually consulted, not just present but unused.
    stratified_data_uniform = _FakeStratifiedDataWithSigma(
        phi, t1, t2, g2, np.ones((n_phi, n_t, n_t))
    )
    _HierarchicalOptimizerSpy.captured = {}
    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data_uniform,
        per_angle_scaling=True,
        physical_param_names=[
            "D0", "alpha", "D_offset", "gamma_dot_t0", "beta",
            "gamma_dot_t_offset", "phi0",
        ],
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma"),
    )
    loss_fn_uniform = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    loss_uniform = float(loss_fn_uniform(p0))

    assert loss_with_sigma != loss_uniform, (
        "loss must change when sigma is non-uniform -- sigma is not being consulted"
    )


def test_laminar_loss_fn_combines_sigma_with_shear_weighting(monkeypatch):
    """Audit follow-up (2026-07-23): the plan's initial fix only edited the
    `else` branch of loss_fn (no shear weighter), leaving sigma silently
    ignored whenever L5 shear weighting is ALSO active -- the common
    laminar_flow case: shear_weighter is constructed whenever
    `is_laminar_flow and shear_weighting_enabled(default True) and n_phi > 3`
    (hybrid_streaming.py). n_phi=4 here (>3) activates shear; explicit
    per_angle_mode='individual' keeps L2 hierarchical active too (n_phi>=3
    would otherwise auto-resolve to 'averaged', which disables hierarchical
    -- see use_constant/per_angle_mode_actual gating). Both layers active
    simultaneously is the scenario Step 4's combined fix must cover."""
    from xpcsjax.optimization.nlsq.strategies import hybrid_streaming as hs

    n_phi = 4
    n_t = 4
    phi = np.repeat([0.0, 30.0, 60.0, 90.0], n_t * n_t)
    t1 = np.tile(np.repeat(np.arange(n_t, dtype=float), n_t), n_phi)
    t2 = np.tile(np.tile(np.arange(n_t, dtype=float), n_t), n_phi)
    g2 = np.ones_like(phi) + 0.01 * np.arange(phi.size)
    sigma_3d = np.ones((n_phi, n_t, n_t))
    sigma_3d[0] = 0.1  # non-uniform: first phi angle tightly weighted

    stratified_data = _FakeStratifiedDataWithSigma(phi, t1, t2, g2, sigma_3d)

    monkeypatch.setattr(hs, "HierarchicalOptimizer", _HierarchicalOptimizerSpy)
    _HierarchicalOptimizerSpy.captured = {}

    n_physical = 7
    initial_params = np.concatenate([np.ones(2 * n_phi), np.ones(n_physical)])
    bounds = (np.zeros_like(initial_params), np.ones_like(initial_params) * 10)
    physical_param_names = [
        "D0", "alpha", "D_offset", "gamma_dot_t0", "beta",
        "gamma_dot_t_offset", "phi0",
    ]

    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=physical_param_names,
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma_shear"),
        anti_degeneracy_config={"per_angle_mode": "individual"},
    )
    loss_fn = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    assert loss_fn is not None, "hierarchical path was not entered -- check gating config"
    loss_with_sigma = float(loss_fn(initial_params))

    stratified_data_uniform = _FakeStratifiedDataWithSigma(
        phi, t1, t2, g2, np.ones((n_phi, n_t, n_t))
    )
    _HierarchicalOptimizerSpy.captured = {}
    hs.fit_with_stratified_hybrid_streaming(
        stratified_data=stratified_data_uniform,
        per_angle_scaling=True,
        physical_param_names=physical_param_names,
        initial_params=initial_params,
        bounds=bounds,
        logger=__import__("logging").getLogger("test_laminar_sigma_shear"),
        anti_degeneracy_config={"per_angle_mode": "individual"},
    )
    loss_fn_uniform = _HierarchicalOptimizerSpy.captured.get("loss_fn")
    loss_uniform = float(loss_fn_uniform(initial_params))

    assert loss_with_sigma != loss_uniform, (
        "loss must change when sigma is non-uniform even with shear weighting "
        "active -- the shear_weighter_local branch must also honor sigma"
    )
```

Note: if `fit_with_stratified_hybrid_streaming`'s hierarchical path is not entered with this minimal fixture (the `loss_fn is not None` assertion fails), inspect the function's gating conditions (`enable_hierarchical`, `per_angle_scaling`, `use_constant` — derived from `per_angle_mode_actual`, which defaults to resolving "auto" based on `n_phi` vs `constant_scaling_threshold`) and adjust `n_phi`/config until the hierarchical branch is reached, per the plan's research (small `n_phi`, default config, `per_angle_scaling=True` should resolve to `per_angle_mode_actual="individual"` and enter the hierarchical path by default). Do not weaken the assertion — fix the fixture. For the shear-combination test above, confirm `shear_weighter_local is not None` is actually reached (e.g. via a temporary print/log of `anti_degeneracy_components.get("shear_weighter")`) before concluding a fixture problem — the `n_phi > 3` and `per_angle_mode="individual"` gates are independent per the source read during planning, but re-verify against the checkout being implemented against.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py -k laminar -v`
Expected: `test_laminar_loss_fn_honors_nonuniform_sigma` FAILS — either `loss_with_sigma == loss_uniform` (sigma not consulted, the actual bug) or a `NameError`/`AttributeError` if `sigma` isn't in scope in `loss_fn` at all yet (also expected pre-fix, per the spec's finding that laminar has no `sigma` name in scope). `test_laminar_loss_fn_combines_sigma_with_shear_weighting` FAILS the same way (sigma not consulted at all, pre-fix, regardless of shear). Confirm which failure mode you see before proceeding.

- [ ] **Step 3: Implement the prerequisite sigma plumbing**

In `xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py`, find the diagonal-masking block:

```python
    n_points_before = len(y_data)
    non_diagonal_mask = t1_idx_arr != t2_idx_arr
    x_data = x_data[non_diagonal_mask]
    y_data = y_data[non_diagonal_mask]
    n_diagonal_removed = n_points_before - len(y_data)
```

Add sigma alignment immediately after it (same indentation level):

```python
    # Sigma plumbing (Finding #4, 2026-07-23): stratified_data.sigma exists
    # (wrapper.py's StratifiedData copies it from original_data.sigma) but was
    # never threaded into this function before. Align it to x_data/y_data's
    # flattened, non-diagonal-filtered order the same way heterodyne's
    # build_heterodyne_pointwise_model does: index the raw (n_phi, n_t, n_t)
    # sigma grid by the pre-filter (phi_idx, t1_idx, t2_idx) triples, then
    # apply the same non_diagonal_mask.
    sigma: np.ndarray | None = None
    if getattr(stratified_data, "sigma", None) is not None:
        sigma_3d = np.asarray(stratified_data.sigma, dtype=np.float64)
        sigma_sel = sigma_3d[phi_idx_arr, t1_idx_arr, t2_idx_arr]
        sigma = sigma_sel[non_diagonal_mask]
```

- [ ] **Step 4: Implement the closure edit**

In the same file, add the module-level helper near the top (mirroring Task 4a's, in this separate module):

```python
def _sigma_weighted_residuals(residuals: Any, sigma: Any | None) -> Any:
    """Per-point residuals divided by sigma, EXCLUDING invalid points.

    Mirrors the safe_sigma/valid_sigma EPS-guard convention used in
    strategies/residual_jit.py EXACTLY: points with sigma <= EPS are excluded
    from the loss entirely (residual set to zero for that point, matching
    residual_jit.py's `jnp.where(valid_sigma, (obs-theory)/safe_sigma, 0.0)`)
    rather than falling back to the raw unweighted residual, which would be a
    materially different (and uncited) aggregation behavior. When sigma is
    None, this is the identity (residuals unchanged) -- both call sites below
    then reduce to their pre-existing unweighted form.
    """
    if sigma is None:
        return residuals
    EPS = 1e-10
    sigma_jax = jnp.asarray(sigma)
    valid_sigma = sigma_jax > EPS
    safe_sigma = jnp.where(valid_sigma, sigma_jax, 1.0)
    return jnp.where(valid_sigma, residuals / safe_sigma, 0.0)


def _sigma_weighted_mse(residuals: Any, sigma: Any | None) -> Any:
    """Mean-squared residual, optionally weighted by per-point sigma.

    Built on :func:`_sigma_weighted_residuals` so the same EPS-guard
    convention applies here as in the shear-combined branch below. When
    sigma is None, this is exactly the pre-existing unweighted
    jnp.mean(residuals**2).
    """
    return jnp.mean(_sigma_weighted_residuals(residuals, sigma) ** 2)
```

Then in `loss_fn` (hierarchical branch), change:

```python
            if shear_weighter_local is not None:
                # Use shear-weighted loss instead of uniform MSE
                weighted_loss = shear_weighter_local.apply_weights_to_loss(
                    residuals, phi_indices_jax
                )
            else:
                # CRITICAL: Use jnp.mean, NOT np.mean!
                # np.mean breaks JAX autodiff and causes zero gradients
                weighted_loss = jnp.mean(residuals**2) * len(y_data)
```

to:

```python
            # sigma weighting (Finding #4, 2026-07-23) is orthogonal to shear
            # weighting and must apply in BOTH branches below -- pre-divide
            # residuals by sigma once (identity when sigma is None) so shear
            # weighting, when also active, combines with it rather than
            # silently overriding it.
            residuals_sw = _sigma_weighted_residuals(residuals, sigma)
            if shear_weighter_local is not None:
                # Use shear-weighted loss instead of uniform MSE. Passing the
                # already sigma-divided residuals means apply_weights_to_loss's
                # sum(w * r**2) becomes sum(w * (r/sigma)**2) -- both layers
                # combined, matching the sibling plain-path branch's
                # optimizer.fit(sigma=sigma, ...).
                weighted_loss = shear_weighter_local.apply_weights_to_loss(
                    residuals_sw, phi_indices_jax
                )
            else:
                # CRITICAL: Use jnp operations, NOT np -- np.mean breaks JAX
                # autodiff and causes zero gradients.
                weighted_loss = jnp.mean(residuals_sw**2) * len(y_data)
```

Both branches now honor sigma weighting — the earlier plan draft left the `shear_weighter_local is not None` branch untouched, which silently no-op'd this fix whenever L5 shear weighting was also active (the common `laminar_flow` + `n_phi > 3` case, since `shear_weighting_enabled` defaults to `True`). Pre-dividing residuals once, before the branch, closes that gap with a single shared edit instead of duplicating the sigma logic into both branches.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py -v`
Expected: all PASS, including `test_laminar_loss_fn_combines_sigma_with_shear_weighting` (the shear-active combination case).

- [ ] **Step 6: Run the existing laminar hybrid-streaming test suites to confirm no regression**

Run: `uv run pytest tests/optimization/test_hybrid_streaming_retry.py tests/optimization/test_hybrid_streaming_constant_quantile_fallback.py -v`
Expected: all PASS (sigma=None or uniform-sigma cases are numerically unchanged).

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py tests/optimization/test_debug_audit_2026_07_23_sigma_weighting.py
git commit -m "fix: laminar_flow L2 hierarchical loss now threads and honors sigma

Finding #4 (laminar half) of the 2026-07-23 debug-audit-fixes spec.
Unlike heterodyne, laminar_flow's fit_with_stratified_hybrid_streaming
had no sigma name in scope in its loss_fn/grad_fn closures at all --
stratified_data.sigma was populated upstream but never threaded into this
function. Added the prerequisite alignment step (mirroring
build_heterodyne_pointwise_model's approach), then applied a
_sigma_weighted_residuals pre-division shared by BOTH the shear-weighted
and plain-mean branches of loss_fn -- an earlier draft of this fix only
edited the plain branch, silently no-op'ing on every laminar_flow run
with n_phi > 3 (L5 shear weighting active by default), which is the
common case, not an edge case."
```

---

### Task 5: Wire `cmaes_seed` end-to-end

**Files:**
- Modify: `xpcsjax/optimization/nlsq/config.py` (`NLSQConfig` dataclass, `from_dict`, `to_dict`)
- Test: `tests/optimization/test_debug_audit_2026_07_23_cmaes_seed.py` (new file)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks depend on. (Do not touch `cmaes_wrapper.py` — its existing `getattr(config, "cmaes_seed", None)` at `from_nlsq_config` line 325 already resolves correctly once the field exists; per the spec, switching it to a bare attribute access would be wrong.)

- [ ] **Step 1: Write the failing test**

Create `tests/optimization/test_debug_audit_2026_07_23_cmaes_seed.py`:

```python
"""Regression: cmaes.seed in YAML config must actually reach
CMAESWrapperConfig.seed, not silently no-op.

Finding #5 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig
from xpcsjax.optimization.nlsq.config import NLSQConfig


def test_cmaes_seed_field_exists_with_none_default():
    config = NLSQConfig()
    assert config.cmaes_seed is None


def test_cmaes_seed_parsed_from_dict():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    assert config.cmaes_seed == 123


def test_cmaes_seed_round_trips_through_to_dict():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    d = config.to_dict()
    assert d["cmaes"]["seed"] == 123


def test_cmaes_seed_reaches_wrapper_config():
    config = NLSQConfig.from_dict({"cmaes": {"seed": 123}})
    wrapper_config = CMAESWrapperConfig.from_nlsq_config(config)
    assert wrapper_config.seed == 123


def test_cmaes_seed_defaults_to_none_when_unset():
    config = NLSQConfig.from_dict({})
    wrapper_config = CMAESWrapperConfig.from_nlsq_config(config)
    assert wrapper_config.seed is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_cmaes_seed.py -v`
Expected: `test_cmaes_seed_field_exists_with_none_default`, `test_cmaes_seed_parsed_from_dict`, `test_cmaes_seed_round_trips_through_to_dict` FAIL with `AttributeError: 'NLSQConfig' object has no attribute 'cmaes_seed'`. `test_cmaes_seed_reaches_wrapper_config` FAILS the same way (the `from_dict` call fails first). `test_cmaes_seed_defaults_to_none_when_unset` currently PASSES only by coincidence (the field doesn't exist so `getattr(..., None)` in `from_nlsq_config` already returns `None`) — confirm it passes today, then re-verify after Step 3 that it still passes for the right reason (field exists, defaults to `None`).

- [ ] **Step 3: Implement the field, parsing, and serialization**

In `xpcsjax/optimization/nlsq/config.py`'s `NLSQConfig` dataclass, add the new field near the other `cmaes_*` fields:

```python
    cmaes_max_generations: int | None = None  # None = use preset + adaptive scaling
    cmaes_popsize: int | None = None  # Population size (None = auto from 4+3*ln(n))
    cmaes_seed: int | None = None  # Deterministic seed; None = nondeterministic (default)
```

In `from_dict()`'s `cmaes` block, add the parse line next to `cmaes_popsize`:

```python
            cmaes_max_generations=cmaes.get("max_generations"),  # None = adaptive
            cmaes_popsize=cmaes.get("popsize"),  # None = auto
            cmaes_seed=cmaes.get("seed"),  # None = nondeterministic (default)
```

In `to_dict()`'s `"cmaes"` block, add the emit line next to `"popsize"`:

```python
                "max_generations": self.cmaes_max_generations,
                "popsize": self.cmaes_popsize,
                "seed": self.cmaes_seed,
```

Do NOT modify `cmaes_wrapper.py` — its existing `getattr(config, "cmaes_seed", None)` at `from_nlsq_config` (line 325) already resolves to the real value the moment the field exists on `NLSQConfig`; per the spec, keeping `getattr` here is correct and matches every other field in that method.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimization/test_debug_audit_2026_07_23_cmaes_seed.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Run the existing cmaes config test suite to confirm no regression**

Run: `uv run pytest tests/optimization/test_heterodyne_cmaes_seed.py tests/optimization/test_cmaes_trigger.py -v`
Expected: all PASS (these test the separate `heterodyne_config.NLSQConfig`/`heterodyne_core.py` path, unaffected by this change — confirms the two config classes really are independent, per this plan's research).

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/optimization/nlsq/config.py tests/optimization/test_debug_audit_2026_07_23_cmaes_seed.py
git commit -m "fix: wire cmaes_seed end-to-end on NLSQConfig (laminar_flow path)

Finding #5 of the 2026-07-23 debug-audit-fixes spec.
CMAESWrapperConfig.from_nlsq_config already reads
getattr(config, 'cmaes_seed', None), but NLSQConfig never declared,
parsed, or serialized the field -- a user-supplied cmaes.seed in YAML had
no effect. Added the field (default None, preserving current
nondeterministic-unless-set behavior), parsing, and serialization;
from_nlsq_config's getattr is left as-is, matching its own uniform
20+-field getattr convention. This is the laminar_flow config path only
-- heterodyne_core.py builds CMAESWrapperConfig by hand and already has
its own, separately-fixed seed wiring (test_heterodyne_cmaes_seed.py)."
```

---

### Task 6: Gate negative-correlation repair on per-matrix normalization state

**Files:**
- Modify: `xpcsjax/data/preprocessing.py:740-838` (`_normalize_data`, STATISTICAL/ROBUST branches)
- Modify: `xpcsjax/data/quality_controller.py:1412-1450` (`_repair_negative_correlations`)
- Test: `tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py` (new file)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py`:

```python
"""Regression: negative-correlation repair must not clamp matrices that were
genuinely normalized (legitimate negatives), but must still clamp matrices
that hit the zero-variance/zero-IQR skip branch (never actually
transformed).

Finding #6 of docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md.
"""

from __future__ import annotations

import numpy as np

from xpcsjax.data.preprocessing import PreprocessingPipeline
from xpcsjax.data.quality_controller import DataQualityController


def _data_with_one_skipped_matrix(n_t: int = 6) -> dict:
    rng = np.random.default_rng(1)
    normal_matrix = 1.0 + 0.3 * rng.standard_normal((n_t, n_t))
    constant_matrix = np.full((n_t, n_t), 2.0)  # zero variance -> skip branch
    c2 = np.stack([normal_matrix, constant_matrix])
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01, 0.01]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }


def test_normalize_data_tracks_per_matrix_mask():
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "statistical"}}}}
    )
    data = _data_with_one_skipped_matrix()
    result = pipeline._normalize_data(data, {"method": "statistical"})
    mask = result.get("_normalized_mask")
    assert mask is not None
    assert list(mask) == [True, False], (
        "matrix 0 (real variance) must be marked normalized, "
        "matrix 1 (zero variance, skip branch) must not be"
    )


def test_repair_negative_correlations_respects_per_matrix_mask():
    controller = DataQualityController.__new__(DataQualityController)
    data = _data_with_one_skipped_matrix()
    # Manufacture negatives: one in the "normalized" matrix (legitimate,
    # e.g. z-score), one in the "skipped" matrix (never transformed, must
    # still be repaired).
    data["c2_exp"] = data["c2_exp"].astype(np.float64)
    data["c2_exp"][0, 0, 0] = -1.5
    data["c2_exp"][1, 0, 0] = -2.5
    data["_normalized_mask"] = [True, False]

    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(data, repairs_applied)

    assert modified is True
    assert data["c2_exp"][0, 0, 0] == -1.5, "normalized matrix must NOT be clamped"
    assert data["c2_exp"][1, 0, 0] == 1e-6, "skipped (never-normalized) matrix must still be clamped"


def test_repair_negative_correlations_clamps_everything_when_unmarked():
    controller = DataQualityController.__new__(DataQualityController)
    data = _data_with_one_skipped_matrix()
    data["c2_exp"] = data["c2_exp"].astype(np.float64)
    data["c2_exp"][0, 0, 0] = -1.5
    data["c2_exp"][1, 0, 0] = -2.5
    # No _normalized_mask key at all -- must behave exactly as before this fix.

    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(data, repairs_applied)

    assert modified is True
    assert data["c2_exp"][0, 0, 0] == 1e-6
    assert data["c2_exp"][1, 0, 0] == 1e-6


def _data_with_one_skipped_matrix_robust(n_t: int = 6) -> dict:
    rng = np.random.default_rng(2)
    normal_matrix = 1.0 + 0.3 * rng.standard_normal((n_t, n_t))
    constant_matrix = np.full((n_t, n_t), 3.0)  # zero IQR -> ROBUST skip branch
    c2 = np.stack([normal_matrix, constant_matrix])
    return {
        "c2_exp": c2,
        "t1": np.arange(n_t, dtype=np.float64),
        "t2": np.arange(n_t, dtype=np.float64),
        "wavevector_q_list": np.array([0.01, 0.01]),
        "phi_angles_list": np.array([0.0, 45.0]),
    }


def test_normalize_data_tracks_per_matrix_mask_robust():
    """ROBUST method's zero-IQR skip branch must also be tracked per-matrix,
    mirroring the STATISTICAL zero-variance skip branch tested above. The
    design spec covers both normalization methods symmetrically (both gate on
    the same np.finfo-eps skip-guard shape); the plan's first draft only
    exercised STATISTICAL, leaving ROBUST's skip path completely untested."""
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "robust"}}}}
    )
    data = _data_with_one_skipped_matrix_robust()
    result = pipeline._normalize_data(data, {"method": "robust"})
    mask = result.get("_normalized_mask")
    assert mask is not None
    assert list(mask) == [True, False], (
        "matrix 0 (real IQR) must be marked normalized, "
        "matrix 1 (zero IQR, ROBUST skip branch) must not be"
    )


def test_repair_negative_correlations_respects_real_pipeline_mask():
    """End-to-end (design spec Testing item 6): thread the REAL
    _normalize_data output into DataQualityController._repair_negative_correlations,
    not a hand-injected _normalized_mask. The hand-injected-mask test above
    (test_repair_negative_correlations_respects_per_matrix_mask) proves the
    controller reads the mask correctly, but never proves the mask-producing
    pipeline and the mask-consuming controller actually agree on what the
    mask means -- a real seam bug between the two would not be caught there."""
    pipeline = PreprocessingPipeline(
        {"preprocessing": {"stages": {"normalize_data": {"method": "statistical"}}}}
    )
    data = _data_with_one_skipped_matrix()
    normalized = pipeline._normalize_data(data, {"method": "statistical"})
    normalized["c2_exp"] = np.asarray(normalized["c2_exp"], dtype=np.float64)

    # Z-scoring matrix 0 legitimately produces negatives (it's mean-centered);
    # confirm the fixture assumption, then inject a definite negative into
    # matrix 1 (the skipped, never-transformed matrix), which would
    # otherwise have no reason to go negative on its own.
    assert np.any(normalized["c2_exp"][0] < 0), (
        "z-score normalization of matrix 0 should legitimately produce a "
        "negative somewhere -- fixture assumption broken"
    )
    normalized["c2_exp"][1, 0, 0] = -2.5

    controller = DataQualityController.__new__(DataQualityController)
    repairs_applied: list[str] = []
    modified = controller._repair_negative_correlations(normalized, repairs_applied)

    assert modified is True
    assert np.any(normalized["c2_exp"][0] < 0), (
        "matrix 0's real normalization-produced negatives must survive repair"
    )
    assert normalized["c2_exp"][1, 0, 0] == 1e-6, (
        "matrix 1 (never normalized) must still be clamped"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py -v`
Expected: `test_normalize_data_tracks_per_matrix_mask` FAILS (`mask is None`). `test_normalize_data_tracks_per_matrix_mask_robust` FAILS the same way for the ROBUST branch. `test_repair_negative_correlations_respects_per_matrix_mask` and `test_repair_negative_correlations_respects_real_pipeline_mask` FAIL (`data["c2_exp"][...] == 1e-6`, not the expected surviving negative — currently everything gets clamped regardless of the mask). `test_repair_negative_correlations_clamps_everything_when_unmarked` should already PASS (this pins today's existing behavior for the no-marker case).

- [ ] **Step 3: Implement the per-matrix mask in `_normalize_data`**

In `xpcsjax/data/preprocessing.py`'s `_normalize_data`, first add mask initialization right after `normalized_data` is built:

```python
        normalized_data = {
            k: (np.array(v) if hasattr(v, "shape") else copy.deepcopy(v)) for k, v in data.items()
        }
```

add immediately after:

```python
        # Per-matrix normalization-applied tracking (Finding #6, 2026-07-23):
        # STATISTICAL/ROBUST skip individual near-zero-variance matrices
        # without failing the stage, so a dataset-level flag would be wrong
        # -- it would suppress negative-correlation repair even on matrices
        # that were never actually transformed. Only set for the methods
        # whose "skip" branches produce legitimate negatives when they DO
        # run; other methods (BASELINE/MINMAX/PHYSICS_BASED) don't produce
        # negatives by design and don't need this tracking.
        normalized_mask: list[bool] | None = None
        if method in (NormalizationMethod.STATISTICAL, NormalizationMethod.ROBUST):
            normalized_mask = [False] * len(c2_exp)
```

Then in the `STATISTICAL` branch, change:

```python
        elif method == NormalizationMethod.STATISTICAL:
            # Z-score normalization
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                mean_val = np.nanmean(c2_matrix)
                std_val = np.nanstd(c2_matrix)
                # Use epsilon guard instead of exact float equality: std_val near
                # zero (subnormal) would pass != 0 but cause overflow on division.
                if abs(std_val) > np.finfo(np.float64).eps:
                    normalized_data["c2_exp"][i] = (c2_matrix - mean_val) / std_val
                else:
                    logger.warning(
                        f"Zero standard deviation at matrix {i}, skipping normalization",
                    )
```

to:

```python
        elif method == NormalizationMethod.STATISTICAL:
            # Z-score normalization
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                mean_val = np.nanmean(c2_matrix)
                std_val = np.nanstd(c2_matrix)
                # Use epsilon guard instead of exact float equality: std_val near
                # zero (subnormal) would pass != 0 but cause overflow on division.
                if abs(std_val) > np.finfo(np.float64).eps:
                    normalized_data["c2_exp"][i] = (c2_matrix - mean_val) / std_val
                    if normalized_mask is not None:
                        normalized_mask[i] = True
                else:
                    logger.warning(
                        f"Zero standard deviation at matrix {i}, skipping normalization",
                    )
```

And in the `ROBUST` branch, change:

```python
        elif method == NormalizationMethod.ROBUST:
            # Robust scaling using percentiles
            percentile_range = config.get("percentile_range", [25, 75])
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                q25, q75 = np.nanpercentile(c2_matrix, percentile_range)
                if q75 - q25 > np.finfo(float).eps * max(abs(q75), 1.0):
                    median_val = np.nanmedian(c2_matrix)
                    normalized_data["c2_exp"][i] = (c2_matrix - median_val) / (q75 - q25)
                else:
                    logger.warning(
                        f"No variance in percentile range at matrix {i}, skipping normalization",
                    )
```

to:

```python
        elif method == NormalizationMethod.ROBUST:
            # Robust scaling using percentiles
            percentile_range = config.get("percentile_range", [25, 75])
            for i in range(len(c2_exp)):
                c2_matrix = c2_exp[i]
                q25, q75 = np.nanpercentile(c2_matrix, percentile_range)
                if q75 - q25 > np.finfo(float).eps * max(abs(q75), 1.0):
                    median_val = np.nanmedian(c2_matrix)
                    normalized_data["c2_exp"][i] = (c2_matrix - median_val) / (q75 - q25)
                    if normalized_mask is not None:
                        normalized_mask[i] = True
                else:
                    logger.warning(
                        f"No variance in percentile range at matrix {i}, skipping normalization",
                    )
```

Finally, at the end of `_normalize_data`, change:

```python
        return normalized_data
```

to:

```python
        if normalized_mask is not None:
            normalized_data["_normalized_mask"] = normalized_mask

        return normalized_data
```

- [ ] **Step 4: Run the `_normalize_data` tests to verify they pass**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py::test_normalize_data_tracks_per_matrix_mask tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py::test_normalize_data_tracks_per_matrix_mask_robust -v`
Expected: both PASS (STATISTICAL and ROBUST skip-tracking).

- [ ] **Step 5: Implement the per-matrix gating in `_repair_negative_correlations`**

In `xpcsjax/data/quality_controller.py`'s `_repair_negative_correlations`, change:

```python
        data_modified = False

        c2_exp = data.get("c2_exp")
        if c2_exp is not None:
            try:
                arr = np.asarray(c2_exp)
                negative_mask = arr < 0

                if np.any(negative_mask):
                    # Simple approach: set negatives to small positive value
                    arr[negative_mask] = 1e-6
                    data["c2_exp"] = arr
                    data_modified = True
                    repairs_applied.append("Repaired negative correlation values")
            except (AttributeError, TypeError, IndexError):
                pass

        return data_modified
```

to:

```python
        data_modified = False

        c2_exp = data.get("c2_exp")
        if c2_exp is not None:
            try:
                arr = np.asarray(c2_exp)
                negative_mask = arr < 0
                normalized_mask = data.get("_normalized_mask")

                # Finding #6 (2026-07-23): a matrix that was genuinely
                # normalized (STATISTICAL/ROBUST actually ran, not skipped
                # for near-zero variance/IQR) legitimately holds negative
                # values by design -- don't clamp those. Matrices with no
                # tracked mask (no preprocessing, or a non-normalizing
                # method) keep today's behavior: clamp unconditionally.
                if normalized_mask is not None and arr.ndim >= 1:
                    skip_repair = np.zeros_like(negative_mask, dtype=bool)
                    n_marked = min(len(normalized_mask), arr.shape[0])
                    for i in range(n_marked):
                        if normalized_mask[i]:
                            skip_repair[i] = True
                    effective_mask = negative_mask & ~skip_repair
                else:
                    effective_mask = negative_mask

                if np.any(effective_mask):
                    arr[effective_mask] = 1e-6
                    data["c2_exp"] = arr
                    data_modified = True
                    if normalized_mask is not None:
                        n_skipped_matrices = int(np.sum(normalized_mask[:n_marked]))
                        repairs_applied.append(
                            f"Repaired negative correlation values "
                            f"({n_skipped_matrices} normalized matrix/matrices exempted)"
                        )
                    else:
                        repairs_applied.append("Repaired negative correlation values")
            except (AttributeError, TypeError, IndexError):
                pass

        return data_modified
```

- [ ] **Step 6: Run all 5 tests to verify they pass**

Run: `uv run pytest tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py -v`
Expected: all 5 PASS (`test_normalize_data_tracks_per_matrix_mask`, `test_normalize_data_tracks_per_matrix_mask_robust`, `test_repair_negative_correlations_respects_per_matrix_mask`, `test_repair_negative_correlations_respects_real_pipeline_mask`, `test_repair_negative_correlations_clamps_everything_when_unmarked`).

- [ ] **Step 7: Run the existing quality controller smoke test to confirm no regression**

Run: `uv run pytest tests/data/test_quality_controller_smoke.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add xpcsjax/data/preprocessing.py xpcsjax/data/quality_controller.py tests/data/test_debug_audit_2026_07_23_negative_correlation_repair.py
git commit -m "fix: gate negative-correlation repair on per-matrix normalization state

Finding #6 of the 2026-07-23 debug-audit-fixes spec.
_repair_negative_correlations previously clamped every negative c2_exp
value whenever aggressive auto-repair fired, even when
STATISTICAL/ROBUST normalization legitimately produced them. A
dataset-level flag would have been wrong too: those methods skip
individual near-zero-variance/IQR matrices without failing the stage, so
a coarse flag would suppress repair on matrices that were never actually
transformed. Track per-matrix instead (_normalize_data now returns
_normalized_mask), and only exempt matrices where normalization genuinely
ran."
```

---

### Task 7: Full verification

**Files:** none (verification only).

**Interfaces:**
- Consumes: all fixes from Tasks 1-6.
- Produces: nothing (terminal task).

- [ ] **Step 1: Run the full domain-scoped test suites**

Run: `uv run pytest tests/optimization/ tests/heterodyne/ tests/data/ tests/test_debug_audit_regressions.py -v`
Expected: all PASS, 0 failures.

- [ ] **Step 2: Run `make verify`**

Run: `make verify`
Expected: `ALL CHECKS PASSED - SAFE TO PUSH` (ruff clean, mypy clean, parallel smoke suite green).

- [ ] **Step 3: If a real (non-synthetic) XPCS dataset is available, sanity-check Tasks 3 and 4**

Per the spec's Testing section: Tasks 3 (fallback escalation) and 4 (sigma-weighting) touch actual solve paths with no `rtol` golden gate. If a real dataset is available in this environment, run a laminar_flow and a two_component fit before and after this branch's changes and confirm the fitted parameters are still physically sensible (not wildly different) — this is a manual sanity check, not an automated test, per the project's stated preference for real-data verification on solve-path changes. If no real dataset is available in this environment, note that explicitly rather than skipping silently.

- [ ] **Step 4: Update the design spec's status**

Edit `docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md`'s header line `Status: Approved` to `Status: Implemented`.

- [ ] **Step 5: Commit the status update**

```bash
git add -f docs/superpowers/specs/2026-07-23-fix-remaining-debug-audit-bugs-design.md
git commit -m "docs: mark the 2026-07-23 debug-audit-fixes spec as implemented"
```

- [ ] **Step 6: Push and report**

```bash
git push origin worktree-fix-remaining-debug-audit-bugs
```

Report to the user: all 6 findings fixed and verified, ready for PR review (mirroring the codex+agy+Claude review pattern already used on the spec, if requested) and merge.
