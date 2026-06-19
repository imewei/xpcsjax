# GUI Phase 1A — JAX-Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `xpcsjax.config` importable without pulling in JAX, and add a JAX-free `FitEvent` schema module — so a future GUI process can import config-validation and event types while honoring the "GUI process never imports JAX" boundary.

**Architecture:** Two small, behavior-preserving changes plus regression guards. (1) A new dependency-light `xpcsjax/service/events.py` defining the `FitEvent` dataclass family (no `jax`, `core`, Qt, or `h5py`). (2) A targeted lazy-import refactor of `xpcsjax/config/parameter_manager.py`, whose eager top-level `from xpcsjax.core.physics import …` (line 21) drags JAX into `import xpcsjax.config` (and likely into the sibling `manager`/`physics_validators`/`parameter_space` imports `config/__init__` makes — if any has an *independent* edge, `config/__init__` is lazy-ified instead; see Task 2 Step 7). Subprocess import-graph tests lock the boundary in, asserting `xpcsjax.config.parameter_registry` + `types` (which Plan I needs) are JAX-free.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, standard-library `dataclasses`/`enum`/`subprocess`. No new runtime dependencies.

## Global Constraints

- **Python ≥ 3.12** (per `pyproject.toml`).
- **uv-first.** Run everything through `uv run …`; never bare `pip`/`pytest`.
- **JAX never imported by the GUI/event layer.** `xpcsjax/service/events.py` must import **no** `jax`, `jaxlib`, `xpcsjax.core`, `PySide6`/Qt, or `h5py`. After this plan, `import xpcsjax.config` must not load `jax` into `sys.modules`.
- **Behavior-preserving.** No numeric or validation behavior changes. The `rtol=1e-10` homodyne parity oracles and all existing config/parameter tests must stay green. This is a refactor, not a feature.
- **Lint/style.** ruff line-length 100, rules `E,F,W,I,B,UP,N`; NumPy-style docstrings enforced by the ruff `D` gate on `xpcsjax/**` (tests are exempt).
- **Gate.** `make verify` (lint + advisory mypy + smoke) must pass before the final commit.

---

### Task 1: JAX-free `FitEvent` schema module

Greenfield. Defines the structured event types the worker will later stream to the GUI. Built now (ahead of the worker) because it is the JAX-free type contract that Plan B's `run_fit(on_event=...)` and the GUI both depend on, and because its JAX-freedom must be guarded from day one.

**Files:**
- Create: `xpcsjax/service/__init__.py`
- Create: `xpcsjax/service/events.py`
- Test: `tests/service/test_events.py`
- Create: `tests/service/__init__.py` (empty — makes the test dir a package, matching the repo layout)

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces (relied on by Plan B + GUI phases):
  - `class FitEvent` — frozen dataclass base with fields `run_id: str`, `seq: int`.
  - Subclasses (all frozen dataclasses, all inherit `run_id`, `seq`):
    - `Started(mode: str, settings_summary: str)`
    - `Iteration(n: int, ssr: float, chi2: float)`
    - `LayerStatus(layers: dict[str, bool], mode: str)`
    - `Banner(text: str, kind: BannerKind)`
    - `LogLine(level: str, msg: str)`
    - `Finished(result_path: str)`
    - `Failed(traceback: str)`
    - `Died(exit_code: int | None, signal: int | None)`
  - `class BannerKind(str, Enum)` with members `CMAES_ESCAPE`, `GRADIENT_COLLAPSE`, `INFO`.
  - `TERMINAL_EVENTS: tuple[type[FitEvent], ...]` = `(Finished, Failed, Died)`.

- [ ] **Step 1: Write the failing test**

Create `tests/service/__init__.py` (empty file), then `tests/service/test_events.py`:

```python
"""Tests for the JAX-free FitEvent schema (xpcsjax/service/events.py)."""

import pickle
import subprocess
import sys
import textwrap


_IMPORT_CLEAN = 0  # module imported, "jax" NOT in sys.modules
_IMPORT_LOADS_JAX = 1  # module imported, but "jax" IS in sys.modules (the leak)
_IMPORT_ERROR = 2  # the module itself failed to import


def _probe_import(module: str) -> int:
    """Import ``module`` in a fresh interpreter; return 0/1/2 (see contract above)."""
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    return completed.returncode


def _imports_jax(module: str) -> bool:
    """Return True iff importing ``module`` cleanly loads jax.

    Raises AssertionError if the module fails to import, so a missing module or
    unrelated import failure is reported as a real test failure rather than
    being silently conflated with "jax was loaded" (both used to return rc=1).
    """
    rc = _probe_import(module)
    assert rc != _IMPORT_ERROR, f"{module!r} failed to import in a fresh interpreter"
    assert rc in (_IMPORT_CLEAN, _IMPORT_LOADS_JAX), (
        f"unexpected probe exit code {rc} for {module!r}"
    )
    return rc == _IMPORT_LOADS_JAX


def test_events_module_does_not_import_jax():
    assert not _imports_jax("xpcsjax.service.events")


def test_event_subclasses_carry_run_id_and_seq():
    from xpcsjax.service.events import Finished, Iteration

    it = Iteration(run_id="r1", seq=3, n=10, ssr=1.5, chi2=0.9)
    assert (it.run_id, it.seq, it.n, it.ssr, it.chi2) == ("r1", 3, 10, 1.5, 0.9)

    fin = Finished(run_id="r1", seq=99, result_path="/tmp/out.npz")
    assert fin.result_path == "/tmp/out.npz"


def test_events_are_picklable_round_trip():
    # Spawn-based IPC requires every event to survive pickle.
    from xpcsjax.service.events import Banner, BannerKind, Died

    b = Banner(run_id="r1", seq=1, text="CMA-ES escape", kind=BannerKind.CMAES_ESCAPE)
    assert pickle.loads(pickle.dumps(b)) == b

    d = Died(run_id="r1", seq=42, exit_code=None, signal=9)
    assert pickle.loads(pickle.dumps(d)) == d


def test_terminal_events_set():
    from xpcsjax.service.events import Died, Failed, Finished, TERMINAL_EVENTS

    assert set(TERMINAL_EVENTS) == {Finished, Failed, Died}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/service/test_events.py -q`
Expected: FAIL — the import-graph test (`test_events_module_does_not_import_jax`) fails with `AssertionError: 'xpcsjax.service.events' failed to import in a fresh interpreter` (probe rc=2, module not created yet); the other three tests fail with `ModuleNotFoundError: No module named 'xpcsjax.service'` on their `from xpcsjax.service.events import …` lines. After Task 1 lands, the module imports cleanly with no jax (probe rc=0) and all pass.

- [ ] **Step 3: Create the `service` package init (JAX-free)**

Create `xpcsjax/service/__init__.py`:

```python
"""Headless core-service layer for xpcsjax.

This package is the argparse-free, Qt-free orchestration seam shared by the CLI
and (in later phases) the GUI worker. **Import discipline:** this ``__init__``
must stay free of eager imports that pull in ``jax`` / ``xpcsjax.core`` so that
JAX-free consumers (e.g. ``xpcsjax.service.events``) can be imported without
loading JAX. Heavier submodules (``fit``, ``data``, ``plots``) are added in
Plan B and must be imported directly by callers, not re-exported here.
"""

from __future__ import annotations
```

- [ ] **Step 4: Implement the event schema**

Create `xpcsjax/service/events.py`:

```python
"""Structured fit-progress events streamed from a worker to the GUI.

This module is deliberately dependency-light: it imports only the standard
library so it can be imported by the JAX-free GUI process *and* pickled across
a ``multiprocessing`` (spawn) boundary. It must never import ``jax``,
``xpcsjax.core``, Qt, or ``h5py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BannerKind(str, Enum):
    """Category of a diagnostic banner emitted during a fit."""

    CMAES_ESCAPE = "cmaes_escape"
    GRADIENT_COLLAPSE = "gradient_collapse"
    INFO = "info"


@dataclass(frozen=True)
class FitEvent:
    """Base class for all fit events.

    Every event carries the originating ``run_id`` and a monotonic per-run
    sequence number ``seq`` so a shared reader can demultiplex multiple
    concurrent runs and order events within a single run.
    """

    # NOTE: keep every field non-default. Frozen-dataclass inheritance + pickle
    # rely on it — giving run_id/seq a default would force every subclass field
    # to have a default too (dataclass field-ordering / MRO rule).
    run_id: str
    seq: int


@dataclass(frozen=True)
class Started(FitEvent):
    """A fit has begun."""

    mode: str
    settings_summary: str


@dataclass(frozen=True)
class Iteration(FitEvent):
    """One solver iteration's convergence sample."""

    n: int
    ssr: float
    chi2: float


@dataclass(frozen=True)
class LayerStatus(FitEvent):
    """Anti-degeneracy layer activation snapshot (L1-L5)."""

    # `layers` is a dict: the dataclass is only top-level frozen, contents stay
    # mutable. For Phase 1A the IPC contract is pickle round-trip + equality only
    # (both hold); deep immutability (tuple-of-pairs / MappingProxyType) is
    # deferred and NOT required.
    layers: dict[str, bool]
    mode: str


@dataclass(frozen=True)
class Banner(FitEvent):
    """A diagnostic banner (e.g. CMA-ES escape, gradient collapse)."""

    text: str
    kind: BannerKind


@dataclass(frozen=True)
class LogLine(FitEvent):
    """A forwarded log record (level + message strings only)."""

    level: str
    msg: str


@dataclass(frozen=True)
class Finished(FitEvent):
    """Terminal: the fit completed; result written to ``result_path``."""

    result_path: str


@dataclass(frozen=True)
class Failed(FitEvent):
    """Terminal: the fit raised; ``traceback`` is the formatted exception."""

    traceback: str


@dataclass(frozen=True)
class Died(FitEvent):
    """Terminal (synthetic, parent-emitted): the worker exited abnormally."""

    exit_code: int | None
    signal: int | None


#: Event types that end a run. A bounded queue must never drop these.
TERMINAL_EVENTS: tuple[type[FitEvent], ...] = (Finished, Failed, Died)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/service/test_events.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Lint the new files**

Run: `uv run ruff check xpcsjax/service/`
Expected: no errors. (If `D`-gate flags a missing docstring, every public class above already has one — fix any genuine finding.)

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/service/__init__.py xpcsjax/service/events.py tests/service/__init__.py tests/service/test_events.py
git commit -m "feat(service): add JAX-free FitEvent schema module"
```

---

### Task 2: Make `xpcsjax.config` importable without JAX

The empirically-confirmed F1 leak: `xpcsjax/config/parameter_manager.py:21` eagerly does `from xpcsjax.core.physics import ValidationResult, validate_parameters_detailed`, and `core.physics` pulls in `jax`. The fix mirrors the lazy pattern already used at `xpcsjax/config/manager.py:467` (which defers `from xpcsjax.core.models import make_model` with the comment "lazy because xpcsjax.core.models pulls in JAX").

**Files:**
- Modify: `xpcsjax/config/parameter_manager.py` (line 21 import; method bodies at ~252 and ~595; module-import header)
- Test: `tests/config/test_config_jax_free.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no API change. `ParameterManager.validate_physical_constraints(...) -> ValidationResult` and `ParameterManager.validate_parameters(...) -> ValidationResult` keep identical signatures and behavior; only *when* `xpcsjax.core.physics` is imported changes (lazily, at first call, instead of at module import).

- [ ] **Step 1: Write the failing import-graph test**

Create `tests/config/test_config_jax_free.py`:

```python
"""Regression guard: importing xpcsjax.config must not load JAX (F1)."""

import subprocess
import sys
import textwrap


_IMPORT_CLEAN = 0  # module imported, "jax" NOT in sys.modules
_IMPORT_LOADS_JAX = 1  # module imported, but "jax" IS in sys.modules (the leak)
_IMPORT_ERROR = 2  # the module itself failed to import


def _probe_import(module: str) -> int:
    """Import ``module`` in a fresh interpreter; return 0/1/2 (see contract above)."""
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        try:
            importlib.import_module({module!r})
        except BaseException:
            sys.exit(2)
        sys.exit(1 if "jax" in sys.modules else 0)
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    return completed.returncode


def _imports_jax(module: str) -> bool:
    """Return True iff importing ``module`` cleanly loads jax.

    Raises AssertionError if the module fails to import, so a missing module or
    unrelated import failure is reported as a real test failure rather than
    being silently conflated with "jax was loaded" (both used to return rc=1).
    """
    rc = _probe_import(module)
    assert rc != _IMPORT_ERROR, f"{module!r} failed to import in a fresh interpreter"
    assert rc in (_IMPORT_CLEAN, _IMPORT_LOADS_JAX), (
        f"unexpected probe exit code {rc} for {module!r}"
    )
    return rc == _IMPORT_LOADS_JAX


def test_importing_config_does_not_load_jax():
    # The GUI process imports config-validation directly; it must stay JAX-free.
    assert not _imports_jax("xpcsjax.config")


def test_importing_parameter_manager_does_not_load_jax():
    assert not _imports_jax("xpcsjax.config.parameter_manager")


def test_importing_registry_and_types_does_not_load_jax():
    # Plan I's in-process config validation imports exactly these two directly;
    # importing a submodule runs config/__init__, so these pin the whole package.
    assert not _imports_jax("xpcsjax.config.parameter_registry")
    assert not _imports_jax("xpcsjax.config.types")


def test_parameter_validation_still_works_after_lazy_import():
    # Behavior preservation: the lazily-imported validators must still run.
    import numpy as np

    from xpcsjax.config.parameter_manager import ParameterManager

    pm = ParameterManager()
    names = pm.get_all_parameter_names()
    params = np.array([pm.get_parameter_bounds([n])[0]["min"] for n in names])
    result = pm.validate_parameters(params, names)
    assert result.valid  # at the lower bound, all parameters are in range

    phys = pm.validate_physical_constraints({names[0]: float(params[0])})
    assert hasattr(phys, "valid")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_config_jax_free.py -q`
Expected: the three import-graph tests FAIL — jax IS loaded today, so the probe returns rc=1 and `_imports_jax` returns True (`assert not _imports_jax(...)` fails). `test_parameter_validation_still_works_after_lazy_import` PASSES (behavior is already correct — it's the pin we must not break).

- [ ] **Step 3: Add `from __future__ import annotations` + TYPE_CHECKING import**

In `xpcsjax/config/parameter_manager.py`, the module currently begins (after the docstring) with `import re`. Insert the future import as the **first** statement after the docstring, and add `TYPE_CHECKING` to the existing `typing` import.

Change the top of the import block from:

```python
import re
from typing import Any, cast
```

to:

```python
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast
```

- [ ] **Step 4: Replace the eager core import with a TYPE_CHECKING-only import**

Replace line 21:

```python
from xpcsjax.core.physics import ValidationResult, validate_parameters_detailed
```

with:

```python
if TYPE_CHECKING:
    # Type-only import: xpcsjax.core.physics pulls in JAX, so it must not load at
    # import time. Under `from __future__ import annotations` the `-> ValidationResult`
    # return annotations are strings, so this TYPE_CHECKING import satisfies type
    # checkers without loading JAX. `validate_parameters_detailed` is NOT imported
    # here — it is used only at runtime, via the lazy local import added in Step 6
    # (importing it here too would be an unused-import F401).
    from xpcsjax.core.physics import ValidationResult
```

- [ ] **Step 5: Add the lazy runtime import in `validate_physical_constraints`**

This method (currently `def validate_physical_constraints(...)` ending with `return ValidationResult(...)` at ~line 269) constructs `ValidationResult` at runtime. Add a local import as the first line of the method body, immediately after its docstring (before the `if HAS_PHYSICS_VALIDATORS:` line):

```python
        from xpcsjax.core.physics import ValidationResult

        if HAS_PHYSICS_VALIDATORS:
```

- [ ] **Step 6: Add the lazy runtime import in `validate_parameters`**

This method (currently `def validate_parameters(...)`, calling `validate_parameters_detailed(...)` at ~line 605) calls the validator at runtime. Add a local import as the first line of the method body, immediately after its docstring (before the `if param_names is None:` line):

```python
        from xpcsjax.core.physics import validate_parameters_detailed

        if param_names is None:
```

- [ ] **Step 7: Run the import-graph test to verify it passes**

Run: `uv run pytest tests/config/test_config_jax_free.py -q`
Expected: PASS (4 passed) — `xpcsjax.config`, `xpcsjax.config.parameter_manager`, and `xpcsjax.config.parameter_registry`/`types` no longer load JAX, and validation still works. (If the registry/types test still fails, an independent eager edge survives in `manager`/`physics_validators`/`parameter_space` — see the Step-7 note; lazy-ify `config/__init__.py`.)

> **Expect a likely second pass here.** A deeper 2026-06-18 re-probe (done for Plan I, which needs `xpcsjax.config.parameter_registry` + `xpcsjax.config.types` JAX-free) found that `xpcsjax/config/__init__.py` eagerly imports **four** submodules that *all* currently load JAX — `manager` (line 33), `parameter_manager` (34), `physics_validators` (47), `parameter_space` (44). The `parameter_manager.py:21` edge fixed above may be the common transitive root (the others import through it) — in which case this single fix clears them all — **or** one of `manager`/`physics_validators`/`parameter_space` has an *independent* eager `core`/`jax` edge that survives. Find any survivor with:
> `uv run python -X importtime -c "import xpcsjax.config.parameter_registry" 2>&1 | grep -i "core\|jax" | head`
> and apply the same TYPE_CHECKING / lazy-import treatment to it. **If multiple independent edges exist, prefer lazy-ifying `config/__init__.py`** (a module-level `__getattr__` mirroring the top-level `xpcsjax/__init__.py`) so submodule imports don't drag the whole package in. **The test below must assert `xpcsjax.config.parameter_registry` and `xpcsjax.config.types` are JAX-free, not just `xpcsjax.config`** — Plan I's in-process config validation depends on exactly those two.
>
> **Resolved as built (2026-06-19 audit):** `parameter_manager.py:21` *was* the common transitive root — deferring it cleared all four eager edges, so no independent survivor existed and the `config/__init__.py` lazy-ify fallback was **not** triggered. `config/__init__.py` re-exports remain eager but JAX-free; `tests/config/test_config_jax_free.py` confirms `xpcsjax.config.parameter_registry` and `xpcsjax.config.types` import without `jax`.

- [ ] **Step 8: Prove no behavior regression in the config layer**

Run: `uv run pytest tests/ -k "config or parameter" -q`
Expected: PASS (same set as before this task — no new failures, no changed counts).

- [ ] **Step 9: Type-check the modified module**

Run: `uv run mypy xpcsjax/config/parameter_manager.py`
Expected: no *new* errors versus baseline. (`make type-check` is non-strict; the `from __future__ import annotations` + `TYPE_CHECKING` import keeps `ValidationResult` resolvable for the return annotations.)

- [ ] **Step 10: Commit**

```bash
git add xpcsjax/config/parameter_manager.py tests/config/test_config_jax_free.py
git commit -m "fix(config): defer core.physics import so xpcsjax.config is JAX-free (F1)"
```

---

### Task 3: Lock the boundary in the gate + record the data-path audit

Consolidation/verification task: confirm the new guards run in the normal suite, the full pre-push gate is green, and document the explicit decision that the **data-loading** path is *not* made JAX-free in 1A (it legitimately needs JAX for array output and runs worker-side in later phases). No production code changes here beyond a short documentation note.

**Files:**
- Modify: `xpcsjax/service/__init__.py` (append an "Import boundary" note documenting the audit outcome)
- Test: `tests/service/test_events.py` (already covers events JAX-freedom), `tests/config/test_config_jax_free.py` (covers config) — no new test code; this task runs the gate.

**Interfaces:**
- Consumes: Task 1 (`xpcsjax.service`), Task 2 (JAX-free `xpcsjax.config`).
- Produces: nothing new; documents scope.

- [ ] **Step 1: Record the data-path audit decision**

Append to the docstring in `xpcsjax/service/__init__.py` (after the existing text, before the `from __future__` line):

```python
"""... (existing docstring text above) ...

Import boundary (audited 2026-06-18)
------------------------------------
JAX-free in-process (GUI-importable): ``xpcsjax.service.events`` and
``xpcsjax.config`` (after the parameter_manager lazy-import fix). NOT JAX-free,
by design: ``xpcsjax.data`` (the loader emits JAX arrays via
``xpcsjax.data.xpcs_loader``). Data loading therefore runs worker-side in later
phases; a JAX-free HDF5 *metadata-only* reader for the GUI preview is deferred
to Phase 2, not Phase 1A.
"""
```

(Place the heading inside the existing module docstring; keep `from __future__ import annotations` as the first executable statement.)

- [ ] **Step 2: Verify the import-graph guards run in the default collection**

Run: `uv run pytest tests/service/test_events.py tests/config/test_config_jax_free.py -q`
Expected: PASS (8 passed total — 4 events + 4 config).

- [ ] **Step 3: Run the domain-scoped suites touched by this change**

Run: `uv run pytest tests/ -k "config or parameter or service" -q`
Expected: PASS, no regressions.

- [ ] **Step 4: Run the full pre-push gate**

Run: `make verify`
Expected: lint clean, advisory mypy non-blocking, smoke tests pass. Green gate.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/service/__init__.py
git commit -m "docs(service): record JAX import-boundary audit for the GUI layer"
```

---

## Self-Review

**1. Spec coverage (Phase 1A slice of the design spec):**
- F1 "GUI imports JAX via config/event modules" → Task 1 (events JAX-free) + Task 2 (config JAX-free) + the subprocess import-graph regression tests in both. ✔
- Spec §3 "Event schema is JAX-free (`xpcsjax/service/events.py`)" → Task 1. ✔
- Spec §3/§10 "Phase-1 JAX-boundary fix: defer/relocate `parameter_manager.py:21`; audit `xpcsjax.data`" → Task 2 (fix) + Task 3 Step 1 (audit recorded). ✔
- Spec §9 "import-graph regression test (subprocess; `'jax' not in sys.modules`)" → Tasks 1 & 2 tests. ✔
- Spec §3/§4 pickling requirement (events survive spawn) → Task 1 `test_events_are_picklable_round_trip`. ✔
- Out of 1A scope (correctly deferred, no task): the full `service/` extraction (`data.py`/`fit.py`/`persist.py`/`plots.py`/`config.py`), CLI rewiring, input-equivalence tests — these are **Plan B**. The IPC/queue/cancellation robustness (F2–F7, F9–F13) belongs to Phase 2's worker, not 1A.

**2. Placeholder scan:** No TBD/TODO/"add error handling"/"similar to" placeholders. Every code step shows complete code; every run step shows the exact command and expected outcome. The one conditional (Task 2 Step 7 note) is a labelled safety-net diagnostic, not a placeholder — the expected path is fully specified.

**3. Type consistency:** `FitEvent`/`BannerKind`/`TERMINAL_EVENTS` names and field names in Task 1's interface block match the implementation in Step 4 and the assertions in Step 1 (`Iteration(n, ssr, chi2)`, `Finished(result_path)`, `Died(exit_code, signal)`, `Banner(text, kind)`). `ValidationResult` / `validate_parameters_detailed` names in Task 2 match the existing symbols in `xpcsjax.core.physics` and their usage sites (lines 269, 605). No drift.

---

## Review provenance (2026-06-18)

Reviewed by two external agents (`codex` → SOUND-WITH-MINOR-FIXES; `agy` →
SOUND-AS-IS) + a Claude verify/adversarial pass. **Source-verified clarification
(2026-06-18):** `parameter_manager.py:21` is the only *direct*
`from xpcsjax.core.physics import …` edge among the `config` submodules — but
`config/__init__.py` eagerly re-exports six submodules, and the siblings
(`manager`/`physics_validators`/`parameter_space`) pull JAX *transitively*. So
deferring `:21` alone may or may not close the boundary on a submodule-direct
import; the Task 2 Step 7 note + the submodule-level import-graph test
(`xpcsjax.config.parameter_registry`/`types`) are what **prove** it is closed,
and lazy-ifying `config/__init__.py` is the fallback if a survivor remains.
(Earlier wording here — "`:21` is the only eager JAX edge" — was imprecise and is
corrected; it contradicted the Task 2 Step 7 analysis at line 449.) The reviewers
validated the `__future__`/`TYPE_CHECKING`/lazy-import pattern and the
frozen-dataclass/pickle assumptions. One consensus fix was applied: the
import-graph helper was hardened to a 0/1/2 exit-code scheme (clean / jax-loaded /
import-error) so an import failure can't masquerade as a JAX leak (the old
`returncode == 1` conflated them — empirically reproduced). The adversarial pass
then caught and removed an `F401` (unused parent `importlib`) and wrapped a long
assert, keeping the helper lint-clean under `make verify`. Two clarifying comments
were added to `events.py` (no-default-fields rule; `LayerStatus` dict-mutability).
All other findings were confirmed no-action.

## Follow-on plans (not in this plan)

- **Plan B — Core-service extraction:** `xpcsjax/service/{data,fit,persist,plots,config}.py` lifting the argparse-free orchestration out of `xpcsjax/cli/`, CLI rewired as a thin adapter, `run_fit(..., on_event=…)` seam typed against `xpcsjax.service.events`, and input-equivalence tests (F8). Depends on this plan (Tasks 1–2).
- **Plans C–G:** GUI skeleton + worker IPC (Phase 2), rich live diagnostics (Phase 3), project model (Phase 4), interactive plots (Phase 5), distributable hardening (Phase 6) — one plan per phase per the spec's §10 phasing.
