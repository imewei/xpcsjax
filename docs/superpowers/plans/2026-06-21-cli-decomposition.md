# CLI Decomposition & Error Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the two CLI god-modules (`plot_dispatch.py` 683, `config_generator.py` 564) into focused units and harden 3 config-loading error paths — strictly behavior-preserving, all public contracts intact.

**Architecture:** `plot_dispatch.py` → `plot_backend.py` (backend/dir/shared-counter) + `plot_families/{experimental,simulated,postfit}.py` + a thin `dispatch_plots` orchestrator. `config_generator.py` → `config_template.py` (template logic) with `config_generator.py` kept as a facade re-exporting `generate_config` and retaining `build_parser`/`main`. `config_handling.py` error paths made actionable/non-silent.

**Tech Stack:** Python 3.12+, argparse, pytest. `uv` (`uv run pytest|ruff|mypy`). Bare `python` is NOT on PATH — always `uv run python`.

## Global Constraints

- **Strictly behavior-preserving** for all moves: functions move *verbatim* (no logic edits); only their home module and the import sites change. The existing test suite is the safety net — moved-function tests must stay green.
- **Public contracts unchanged:** `plot_dispatch.dispatch_plots` (lazy export in `cli/__init__.py`, imported by `commands.py`) stays in `plot_dispatch.py`. `config_generator.generate_config` / `.build_parser` / `.main` stay importable from `config_generator` (facade). `completion_spec.py` accesses `config_generator.build_parser` as a module attribute.
- **NLSQ-only package** — no Bayesian/MCMC anything. Plotting/config are NOT part of the `rtol=1e-10` homodyne parity contract.
- Ruff: line-length 100, rules `E,F,W,I,B,UP,N`, + numpydoc `D` gate on `xpcsjax/` (tests exempt). New modules + public functions need NumPy docstrings; **moved functions keep their existing docstrings**.
- `uv run mypy xpcsjax` is the HARD CI gate — must stay clean.
- After moving any function, **distribute its imports** to the new home (e.g. `import jax.numpy`, `numpy`, logging utils) and remove now-unused imports from the source (ruff `F401`).
- One commit per task (move + rewire + test updates together — a half-moved symbol does not pass tests).
- `make verify` green before the branch is done.

## File Structure

**Create:**
- `xpcsjax/cli/plot_backend.py` — `_current_run_id`, `resolve_plots_dir`, `should_use_datashader`, `_PLOT_DISPATCH_CALL_COUNTER` (shared across plot families).
- `xpcsjax/cli/plot_families/__init__.py` — empty package marker.
- `xpcsjax/cli/plot_families/experimental.py` — `_plot_experimental_data`.
- `xpcsjax/cli/plot_families/simulated.py` — `resolve_phi_angles_for_sim`, `_plot_simulated_from_config`, `_evaluate_model_c2` (the only direct `import jax.numpy`, currently `plot_dispatch.py:381`).
- `xpcsjax/cli/plot_families/postfit.py` — `_generate_post_fit_plots`, `_save_fit_comparison_only` (delegates to `service/plots.py`; uses the shared counter).
- `xpcsjax/cli/config_template.py` — `_MODE_TO_TEMPLATE`, `get_template_path`, `generate_config`, `show_template`, `validate_config`, `_prompt`, `interactive_builder`.

**Modify:**
- `xpcsjax/cli/plot_dispatch.py` — reduce to the `dispatch_plots` orchestrator (+ its inner `_record`), importing the moved symbols; update `__all__`.
- `xpcsjax/cli/config_generator.py` — keep `_build_parser`/`build_parser`/`main`; re-export the template API from `config_template`.
- `xpcsjax/cli/config_handling.py` — harden L108/L149/L158; fix the L37 docstring cross-ref.
- `xpcsjax/cli/main.py` — fix the stale "eagerly imports JAX" comment (~L26).
- Tests: `tests/cli/test_output_resolution.py`, `tests/cli/test_simulated_data_grid.py`, `tests/cli/test_plot_dispatch_logging.py` (import/patch-target updates).

**Current symbol → destination (verbatim moves):**

| Symbol (current `plot_dispatch.py` line) | Destination |
|---|---|
| `_PLOT_DISPATCH_CALL_COUNTER` (43) | `plot_backend.py` |
| `_current_run_id` (46) | `plot_backend.py` |
| `resolve_plots_dir` (70) | `plot_backend.py` |
| `should_use_datashader` (94) | `plot_backend.py` |
| `_plot_experimental_data` (129) | `plot_families/experimental.py` |
| `resolve_phi_angles_for_sim` (106) | `plot_families/simulated.py` |
| `_plot_simulated_from_config` (184) | `plot_families/simulated.py` |
| `_evaluate_model_c2` (345) | `plot_families/simulated.py` |
| `_generate_post_fit_plots` (412) | `plot_families/postfit.py` |
| `_save_fit_comparison_only` (432) | `plot_families/postfit.py` |
| `dispatch_plots` (560) + inner `_record` | stays in `plot_dispatch.py` |

---

## Task 1: `plot_backend.py` — backend/dir resolution + shared counter

**Files:**
- Create: `xpcsjax/cli/plot_backend.py`
- Modify: `xpcsjax/cli/plot_dispatch.py` (remove the 4 symbols; import them back)
- Modify: `xpcsjax/cli/config_handling.py:37` (docstring cross-ref)
- Test: `tests/cli/test_output_resolution.py` (repoint `resolve_plots_dir` import)

**Interfaces:**
- Produces (signatures unchanged from current `plot_dispatch.py`):
  - `_PLOT_DISPATCH_CALL_COUNTER` (an `itertools.count()`)
  - `_current_run_id() -> str | None`
  - `resolve_plots_dir(args: Any, config_manager: ConfigManager | None) -> Path`
  - `should_use_datashader(backend: str | None) -> bool`

- [ ] **Step 1: Create `plot_backend.py` with the four symbols moved verbatim**

Cut `_PLOT_DISPATCH_CALL_COUNTER` (L43), `_current_run_id` (L46-…), `resolve_plots_dir` (L70-…), `should_use_datashader` (L94-…) from `plot_dispatch.py` into a new `xpcsjax/cli/plot_backend.py`. Add a module docstring and the imports those functions need: `itertools`, `from pathlib import Path`, `from typing import Any`, `from xpcsjax.utils.logging import _LOG_CONTEXT, get_logger` (`_current_run_id` uses `_LOG_CONTEXT`), and the `TYPE_CHECKING` import of `ConfigManager` used in the `resolve_plots_dir` annotation. Keep each function's existing docstring.

- [ ] **Step 2: Rewire `plot_dispatch.py` to import them**

In `plot_dispatch.py`, replace the removed definitions with:
```python
from xpcsjax.cli.plot_backend import (
    _PLOT_DISPATCH_CALL_COUNTER,
    _current_run_id,
    resolve_plots_dir,
    should_use_datashader,
)
```
(Do not yet prune other now-unused imports — later plot tasks revisit `plot_dispatch.py`; Task 5 does the final import sweep.)

- [ ] **Step 3: Update the test import**

In `tests/cli/test_output_resolution.py:18`, change `from xpcsjax.cli.plot_dispatch import resolve_plots_dir` → `from xpcsjax.cli.plot_backend import resolve_plots_dir`. (The `_generate_post_fit_plots` references at `:110,144` stay for now — Task 4 moves that.)

- [ ] **Step 4: Fix the stale docstring cross-ref**

In `xpcsjax/cli/config_handling.py:37`, change the docstring mention `plot_dispatch.resolve_plots_dir` → `plot_backend.resolve_plots_dir`.

- [ ] **Step 5: Run the covering tests**

Run: `uv run pytest tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py tests/cli/test_simulated_data_grid.py -q`
Expected: PASS (the move is behavior-preserving; `plot_dispatch` re-imports the symbols so its internal callers still resolve).

- [ ] **Step 6: Lint + type-check the touched files**

Run: `uv run ruff check xpcsjax/cli/plot_backend.py xpcsjax/cli/plot_dispatch.py xpcsjax/cli/config_handling.py tests/cli/test_output_resolution.py`
Run: `uv run mypy xpcsjax/cli/plot_backend.py xpcsjax/cli/plot_dispatch.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/cli/plot_backend.py xpcsjax/cli/plot_dispatch.py xpcsjax/cli/config_handling.py tests/cli/test_output_resolution.py
git commit -m "refactor(cli): extract plot_backend (dir/backend resolution + call counter)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `plot_families/experimental.py`

**Files:**
- Create: `xpcsjax/cli/plot_families/__init__.py` (empty), `xpcsjax/cli/plot_families/experimental.py`
- Modify: `xpcsjax/cli/plot_dispatch.py` (remove `_plot_experimental_data`; import it back)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `_plot_experimental_data(data: dict[str, Any], plots_dir: Path) -> Path | None` (signature unchanged).

- [ ] **Step 1: Create the package marker**

Create `xpcsjax/cli/plot_families/__init__.py` containing only a one-line module docstring: `"""CLI plot families (experimental / simulated / post-fit)."""`

- [ ] **Step 2: Move `_plot_experimental_data` verbatim**

Cut `_plot_experimental_data` (L129-…) into `xpcsjax/cli/plot_families/experimental.py`. Add its needed imports (`from pathlib import Path`, `from typing import Any`, the `xpcsjax.viz` plotting helpers it calls, and the logging utils it uses — copy only the imports it references). Keep its docstring.

- [ ] **Step 3: Rewire `plot_dispatch.py`**

Add `from xpcsjax.cli.plot_families.experimental import _plot_experimental_data` to `plot_dispatch.py` (the orchestrator `dispatch_plots` calls it).

- [ ] **Step 4: Run the covering tests**

Run: `uv run pytest tests/cli/test_plot_dispatch_logging.py tests/cli/test_output_resolution.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/plot_families/ xpcsjax/cli/plot_dispatch.py`
Run: `uv run mypy xpcsjax/cli/plot_families/experimental.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/cli/plot_families/ xpcsjax/cli/plot_dispatch.py
git commit -m "refactor(cli): extract plot_families/experimental

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `plot_families/simulated.py` — JAX-isolated family

**Files:**
- Create: `xpcsjax/cli/plot_families/simulated.py`
- Modify: `xpcsjax/cli/plot_dispatch.py` (remove 3 symbols; import back)
- Test: `tests/cli/test_simulated_data_grid.py` (string-form monkeypatch + call repoint)

**Interfaces:**
- Consumes: `_PLOT_DISPATCH_CALL_COUNTER` from `plot_backend` (Task 1).
- Produces: `resolve_phi_angles_for_sim(...)`, `_plot_simulated_from_config(...)`, `_evaluate_model_c2(...)` (signatures unchanged). `_evaluate_model_c2` contains the only direct `import jax.numpy as jnp`.

- [ ] **Step 1: Move the three functions verbatim**

Cut `resolve_phi_angles_for_sim` (L106-…), `_plot_simulated_from_config` (L184-…), and `_evaluate_model_c2` (L345-…, including its inner `import jax.numpy as jnp`) into `xpcsjax/cli/plot_families/simulated.py`. Add the imports they reference: `numpy as np`, `from pathlib import Path`, `from typing import Any`, `from xpcsjax.cli.plot_backend import _PLOT_DISPATCH_CALL_COUNTER`, the `xpcsjax.viz` helpers, logging utils, and the `ConfigManager` `TYPE_CHECKING` import. Keep docstrings. `_plot_simulated_from_config` calls `_evaluate_model_c2` by bare name — keep that call so a module-level monkeypatch on `simulated._evaluate_model_c2` intercepts it.

- [ ] **Step 2: Rewire `plot_dispatch.py`**

Add to `plot_dispatch.py`:
```python
from xpcsjax.cli.plot_families.simulated import (
    _plot_simulated_from_config,
    resolve_phi_angles_for_sim,
)
```
(`_evaluate_model_c2` need not be imported into `plot_dispatch` — only `simulated` and the test reference it.)

- [ ] **Step 3: Update `test_simulated_data_grid.py` to the new module**

The test currently (`:53`) does `monkeypatch.setattr(plot_dispatch, "_evaluate_model_c2", _capture)` and (`:57`) calls `plot_dispatch._plot_simulated_from_config(...)`. Change the import to `from xpcsjax.cli.plot_families import simulated` and rewrite:
```python
monkeypatch.setattr(simulated, "_evaluate_model_c2", _capture)
...
simulated._plot_simulated_from_config(...)
```
(Object-form `setattr(simulated, …)` is correct here because the patched name and the calling function now live in the same module.)

- [ ] **Step 4: Run the covering test**

Run: `uv run pytest tests/cli/test_simulated_data_grid.py -q`
Expected: PASS — the monkeypatch intercepts `_evaluate_model_c2` (verify the captured-call assertion still fires; if the grid is empty, the patch missed the right module).

- [ ] **Step 5: Confirm JAX still lazy on cheap paths**

Run: `uv run python -c "import sys; import xpcsjax.cli.plot_dispatch; print('jax after plot_dispatch import:', 'jax' in sys.modules)"`
Expected: depends on existing behavior — record it. The goal is that importing `simulated` is where the direct `import jax` lives; if `plot_dispatch` no longer imports `_evaluate_model_c2`, importing `plot_dispatch` should not eagerly import jax via the simulated module unless it imports `simulated` at top. (If `plot_dispatch` importing `simulated` pulls jax, that matches pre-refactor behavior where the function was in-file — acceptable, not a regression.)

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/plot_families/simulated.py xpcsjax/cli/plot_dispatch.py tests/cli/test_simulated_data_grid.py`
Run: `uv run mypy xpcsjax/cli/plot_families/simulated.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/cli/plot_families/simulated.py xpcsjax/cli/plot_dispatch.py tests/cli/test_simulated_data_grid.py
git commit -m "refactor(cli): extract plot_families/simulated (isolates direct jax import)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `plot_families/postfit.py`

**Files:**
- Create: `xpcsjax/cli/plot_families/postfit.py`
- Modify: `xpcsjax/cli/plot_dispatch.py` (remove 2 symbols; import back)
- Test: `tests/cli/test_output_resolution.py` (`_generate_post_fit_plots` ref), `tests/cli/test_plot_dispatch_logging.py` (`_save_fit_comparison_only` calls)

**Interfaces:**
- Consumes: `_PLOT_DISPATCH_CALL_COUNTER` from `plot_backend`; `xpcsjax.service.plots.generate_plots`.
- Produces: `_generate_post_fit_plots(...)`, `_save_fit_comparison_only(...)` (signatures unchanged).

- [ ] **Step 1: Move the two functions verbatim**

Cut `_generate_post_fit_plots` (L412-…) and `_save_fit_comparison_only` (L432-…) into `xpcsjax/cli/plot_families/postfit.py`. Add imports they reference: `from pathlib import Path`, `from typing import Any`, `from xpcsjax.cli.plot_backend import _PLOT_DISPATCH_CALL_COUNTER`, `from xpcsjax.service.plots import generate_plots` (currently `plot_dispatch.py:420`), the `_evaluate_c2_per_angle` import from `xpcsjax.viz.nlsq_plots` (currently `plot_dispatch.py:494` — this is why postfit transitively uses JAX, by design), logging utils, and `ConfigManager` `TYPE_CHECKING`. Keep docstrings.

- [ ] **Step 2: Rewire `plot_dispatch.py`**

Add `from xpcsjax.cli.plot_families.postfit import _generate_post_fit_plots, _save_fit_comparison_only` to `plot_dispatch.py` (the orchestrator calls both, e.g. `:655`).

- [ ] **Step 3: Update the tests**

- `tests/cli/test_output_resolution.py:110,144`: these reference `plot_dispatch._generate_post_fit_plots(...)` via `from xpcsjax.cli import plot_dispatch` (`:138`). Change to `from xpcsjax.cli.plot_families import postfit` and call `postfit._generate_post_fit_plots(...)`.
- `tests/cli/test_plot_dispatch_logging.py:117,163,171`: these call `pd._save_fit_comparison_only(...)`. Add `from xpcsjax.cli.plot_families import postfit` and change those calls to `postfit._save_fit_comparison_only(...)`. (The `xpcsjax.viz.*` string-form patches at `:111-112` stay unchanged.)

- [ ] **Step 4: Run the covering tests**

Run: `uv run pytest tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py`
Run: `uv run mypy xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py
git commit -m "refactor(cli): extract plot_families/postfit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Thin `plot_dispatch.py` orchestrator — final sweep

**Files:**
- Modify: `xpcsjax/cli/plot_dispatch.py` (`__all__`, import sweep, confirm thinness)

**Interfaces:**
- Produces: `dispatch_plots(...)` (signature unchanged) — the only public symbol.

- [ ] **Step 1: Sweep imports and update `__all__`**

In `plot_dispatch.py`: the body should now be the `from xpcsjax.cli.plot_backend …` / `…plot_families.* import …` lines plus `dispatch_plots` and its inner `_record`. Remove any import now unused by the remaining code (e.g. `itertools` if the counter moved; `numpy` if no longer referenced; unused logging utils). Update `__all__` (currently `["dispatch_plots", "resolve_plots_dir", "resolve_phi_angles_for_sim", "should_use_datashader"]`) to `["dispatch_plots"]` — the other three moved out and have no external importer requiring re-export.

- [ ] **Step 2: Verify the orchestrator is thin and dispatch_plots is intact**

Run: `uv run python -c "import inspect, xpcsjax.cli.plot_dispatch as pd; print('dispatch_plots' in dir(pd)); print('LOC:', len(inspect.getsource(pd).splitlines()))"`
Expected: `True` and a small LOC count (≈100–150). `dispatch_plots` must still be importable.

- [ ] **Step 3: Run the full CLI plot test surface**

Run: `uv run pytest tests/cli -q`
Expected: PASS (all plot/output/simulated/logging tests green after the split).

- [ ] **Step 4: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/plot_dispatch.py`
Run: `uv run mypy xpcsjax/cli/plot_dispatch.py`
Expected: clean (no `F401` unused imports).

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/cli/plot_dispatch.py
git commit -m "refactor(cli): reduce plot_dispatch to thin dispatch_plots orchestrator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `config_generator.py` → `config_template.py` + facade

**Files:**
- Create: `xpcsjax/cli/config_template.py`
- Modify: `xpcsjax/cli/config_generator.py` (facade)
- Test: covered by `tests/cli/test_config_generator_yaml.py`, `tests/cli/test_build_parser_factories.py`, `tests/cli/test_completion_parity.py`

**Interfaces:**
- Produces in `config_template.py`: `_MODE_TO_TEMPLATE`, `get_template_path(mode)`, `generate_config(...)`, `show_template(mode)`, `validate_config(config_path)`, `_prompt(...)`, `interactive_builder(mode)` (signatures unchanged).
- `config_generator.py` keeps `_build_parser`, `build_parser`, `main` and re-exports `generate_config`, `show_template`, `validate_config`, `interactive_builder`, `get_template_path`.

- [ ] **Step 1: Move template logic verbatim into `config_template.py`**

Cut `_MODE_TO_TEMPLATE` (L43), `get_template_path` (L53), `generate_config` (L90), `show_template` (L197), `validate_config` (L217), `_prompt` (L276), `interactive_builder` (L322) from `config_generator.py` into a new `xpcsjax/cli/config_template.py`. Add the imports they reference (e.g. `from pathlib import Path`, `from typing import Any`, `from xpcsjax.config import ConfigManager`, yaml, logging). Keep docstrings.

- [ ] **Step 2: Make `config_generator.py` a facade**

In `config_generator.py`, add a re-export so the GUI (`main_window.py:459`) and tests keep importing from `config_generator`:
```python
from xpcsjax.cli.config_template import (
    generate_config,
    get_template_path,
    interactive_builder,
    show_template,
    validate_config,
)
```
Keep `_build_parser`, `build_parser`, `main`. Update `main` and any internal references to call the (now re-exported / imported) template functions. Update `config_generator.__all__` to include the re-exported names plus `build_parser`/`main`.

- [ ] **Step 3: Confirm contracts intact**

Run: `uv run python -c "from xpcsjax.cli.config_generator import generate_config, build_parser, main; print('facade OK')"`
Run: `uv run python -c "import sys; import xpcsjax.cli.config_generator; print('config_generator jax-free:', 'jax' not in sys.modules)"`
Expected: `facade OK` and `config_generator jax-free: True`.

- [ ] **Step 4: Run covering tests (incl. completion parity + build_parser + GUI import)**

Run: `uv run pytest tests/cli/test_config_generator_yaml.py tests/cli/test_build_parser_factories.py tests/cli/test_completion_parity.py -q`
Run: `uv run python -c "from xpcsjax.cli.config_generator import generate_config; print('gui import path OK')"`
Expected: PASS / `gui import path OK`.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/config_template.py xpcsjax/cli/config_generator.py`
Run: `uv run mypy xpcsjax/cli/config_template.py xpcsjax/cli/config_generator.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/cli/config_template.py xpcsjax/cli/config_generator.py
git commit -m "refactor(cli): split config_template out of config_generator (facade keeps generate_config)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Harden `config_handling.py` error paths + fix `main.py` comment

**Files:**
- Modify: `xpcsjax/cli/config_handling.py` (L108, L149-152, L158-160)
- Modify: `xpcsjax/cli/main.py` (~L26 comment)
- Test: `tests/cli/test_config_handling_errors.py` (create)

**Interfaces:** none (internal hardening).

- [ ] **Step 1: Write failing regression tests**

Create `tests/cli/test_config_handling_errors.py`:
```python
import logging
from pathlib import Path

import pytest

from xpcsjax.cli import config_handling


def test_load_failure_names_the_file(tmp_path):
    bad = tmp_path / "missing.yaml"
    with pytest.raises(Exception) as exc:
        config_handling.load_and_merge_config(bad)  # adjust to the real loader entry
    assert str(bad) in str(exc.value)  # error names which config failed


def test_normalize_gate_tolerates_object_without_method():
    class _NoNormalize:
        pass

    # apply_cli_overrides must not crash when the config-manager-shaped object
    # lacks _normalize_analysis_mode (defensive gate, was `except AttributeError`).
    # Build the minimal args/config the function needs; assert no exception.
    # (Fill args/config per apply_cli_overrides' real signature.)


def test_non_dict_output_block_is_logged(caplog, tmp_path):
    # When config['output'] is not a dict, the reset is logged (not silent).
    with caplog.at_level(logging.WARNING):
        ...  # invoke the override path with output set to a string
    assert any("output" in r.message for r in caplog.records)
```
Adjust the three test bodies to the real function signatures (`load_and_merge_config`, `apply_cli_overrides`) — read `config_handling.py` for the exact params. Each test pins one hardening behavior.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/cli/test_config_handling_errors.py -q`
Expected: FAIL (load error lacks file context; non-dict reset is silent).

- [ ] **Step 3: Harden L108 (load failure context)**

Wrap the bare `ConfigManager(yaml_path)` (L108) load in try/except and re-raise with context:
```python
try:
    config_manager = ConfigManager(yaml_path)
except Exception as e:
    raise type(e)(f"Failed to load config from {yaml_path}: {e}") from e
```
(Use a form that preserves the exception class if simple; otherwise raise `ValueError` with the message — choose to keep `FileNotFoundError` semantics if a test depends on the type.)

- [ ] **Step 4: Harden L149-152 (defensive normalize gate)**

Replace the broad `try: config_manager._normalize_analysis_mode() except AttributeError: pass` with an explicit guard (verified: the method exists on real `ConfigManager`; this branch is `# pragma: no cover` defensive for lightweight doubles):
```python
# Defensive: lightweight config-manager doubles may omit this private method.
_normalize = getattr(config_manager, "_normalize_analysis_mode", None)
if callable(_normalize):
    _normalize()
```
Do NOT add a raise.

- [ ] **Step 5: Harden L158-160 (non-dict output reset)**

Keep the defensive reset but log it:
```python
if not isinstance(out, dict):
    logger.warning("config 'output' expected a mapping, got %s; resetting to {}", type(out).__name__)
    out = {}
    config["output"] = out
```

- [ ] **Step 6: Fix the stale `main.py` comment**

In `xpcsjax/cli/main.py` (~L26), correct the comment that says `xpcsjax/__init__.py` "eagerly imports JAX" to reflect reality: `__init__.py` only sets JAX env vars; JAX is imported lazily via `__getattr__` on first use of a JAX-backed export (verified: `xpcsjax --help` is JAX-free).

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_config_handling_errors.py -q`
Expected: PASS.

- [ ] **Step 8: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/config_handling.py xpcsjax/cli/main.py tests/cli/test_config_handling_errors.py`
Run: `uv run mypy xpcsjax/cli/config_handling.py xpcsjax/cli/main.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add xpcsjax/cli/config_handling.py xpcsjax/cli/main.py tests/cli/test_config_handling_errors.py
git commit -m "fix(cli): actionable config-load errors; non-silent guards; correct main.py JAX comment

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Final verification

**Files:** none (gate run).

- [ ] **Step 1: Full CLI + runtime + completion suites**

Run: `uv run pytest tests/cli tests/runtime -q`
Expected: PASS (decomposition + hardening green; completion parity intact).

- [ ] **Step 2: Hard gates**

Run: `uv run mypy xpcsjax`  (HARD CI gate)
Run: `make verify`
Expected: both green.

- [ ] **Step 3: Confirm cheap paths still JAX-free**

Run: `uv run python -c "import sys; import xpcsjax.cli.config_generator; print('config jax-free:', 'jax' not in sys.modules)"`
Run: `uv run python -c "import sys; import xpcsjax.runtime.shell.generate_completion as g; g.generate(); print('completion jax-free:', 'jax' not in sys.modules)"`
Expected: both `True`.

---

## Self-Review (completed by plan author)

**Spec coverage:** plot_dispatch split (Tasks 1–5), config_generator facade split (Task 6), config_handling 3 error paths + main.py comment + L37 docstring (Tasks 1 & 7), JAX-isolation isolated to simulated (Task 3) with postfit's expected transitive JAX acknowledged (Task 4), `_PLOT_DISPATCH_CALL_COUNTER` in plot_backend shared by both families (Tasks 1/3/4), `__all__` update (Task 5), all contract-table test repoints (Tasks 1/3/4), facade + completion + GUI contracts verified (Task 6), final gates (Task 8). All acceptance criteria mapped.

**Placeholder scan:** Task 7's test bodies intentionally say "adjust to real signature" — this is because the exact `load_and_merge_config`/`apply_cli_overrides` parameters must be read from the file; the behavior each test pins is fully specified. No other placeholders; move tasks specify exact symbol→destination and the import rewire.

**Type consistency:** moved functions keep verbatim signatures (listed once in the symbol→destination table and per-task Interfaces); `_PLOT_DISPATCH_CALL_COUNTER` single home (plot_backend) referenced consistently; facade re-export names match between Task 6's `config_template` Produces and `config_generator` re-export list.

**Known judgment call:** Task 3 Step 5 records (not asserts) whether importing `plot_dispatch` pulls JAX — pre-refactor it did (the function was in-file), so matching that is acceptable; the isolation goal is about the *direct* import living in `simulated.py`, not about making `plot_dispatch` import-time JAX-free.
