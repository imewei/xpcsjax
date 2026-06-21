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
- After moving any function, **distribute its imports** to the new home (`numpy`, logging utils, etc.) and remove now-unused imports from the source (ruff `F401`).
- **Keep JAX / matplotlib imports exactly where they are — function-local — do NOT hoist them to a module top.** `import jax.numpy` (in `_evaluate_model_c2`) and the matplotlib/viz imports (in the experimental/simulated/postfit functions) are currently function-local; verbatim moves preserve that. `plot_dispatch` imports the family modules at module top, so hoisting any of these into a family module's top level would make `import plot_dispatch` eagerly load JAX/matplotlib and regress the verified JAX-free `--help`/config/completion startup. After Tasks 3–4, verify: `uv run python -c "import sys; import xpcsjax.cli.plot_dispatch; print('jax' in sys.modules)"` (compare to the pre-refactor baseline recorded in Task 3 Step 5 — must not newly become `True`).
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
**Prune the imports THIS task orphans, now** (each move task must leave `plot_dispatch.py` ruff-clean at its own commit, since Step 6 runs `ruff check` on it): moving `_PLOT_DISPATCH_CALL_COUNTER` orphans `import itertools` (L18); moving `_current_run_id` orphans `_LOG_CONTEXT` from the `xpcsjax.utils.logging` import (L25) **unless** another still-present function uses it — grep `_LOG_CONTEXT` in `plot_dispatch.py` after the move and drop it from the import if unused. Do not defer pruning to Task 5; F401 would fail this task's ruff gate.

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

Cut `_plot_experimental_data` (L129-…) into `xpcsjax/cli/plot_families/experimental.py`. Add EVERY name it references: `import numpy as np` (uses `np.asarray` at L144-145), `import logging` (uses `logging.WARNING` at L173 — this is the **stdlib** `logging`, distinct from the `xpcsjax.utils.logging` helpers), `from pathlib import Path`, `from typing import Any`, the `xpcsjax.viz` plotting helpers, and the `xpcsjax.utils.logging` utils it calls (e.g. `log_once`). Keep its docstring. **Verification (not enumeration) is the safety net:** after the move run `ruff check` (F821 undefined-name / F401 unused) + `mypy` on the new file and add/drop imports until clean — do not rely on this list being exhaustive.

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
- Test: `tests/cli/test_simulated_data_grid.py` (object-form monkeypatch on the `simulated` module + call repoint)

**Interfaces:**
- Consumes: `_PLOT_DISPATCH_CALL_COUNTER` from `plot_backend` (Task 1).
- Produces: `resolve_phi_angles_for_sim(...)`, `_plot_simulated_from_config(...)`, `_evaluate_model_c2(...)` (signatures unchanged). `_evaluate_model_c2` contains the only direct `import jax.numpy as jnp`.

- [ ] **Step 1: Move the three functions verbatim**

Cut `resolve_phi_angles_for_sim` (L106-…), `_plot_simulated_from_config` (L184-…), and `_evaluate_model_c2` (L345-…, including its inner `import jax.numpy as jnp`) into `xpcsjax/cli/plot_families/simulated.py`. Add the imports they reference: `import numpy as np`, `import logging` (**stdlib**, for `logging.WARNING` at L222/233/282/304/334 — distinct from the `xpcsjax.utils.logging` helpers), `from pathlib import Path`, `from typing import Any`, **`from xpcsjax.cli.plot_backend import _PLOT_DISPATCH_CALL_COUNTER, _current_run_id`** (`_plot_simulated_from_config` calls `_current_run_id()` at the rate-limited-warning sites, currently `plot_dispatch.py:301,331` — omitting it raises `NameError` at runtime), the `xpcsjax.viz` helpers, the `xpcsjax.utils.logging` utils, and the `ConfigManager` `TYPE_CHECKING` import. After the move, `ruff`+`mypy` on the new file must be clean (catches any missed/spurious import). Keep docstrings. `_plot_simulated_from_config` calls `_evaluate_model_c2` by bare name — keep that call so a module-level monkeypatch on `simulated._evaluate_model_c2` intercepts it.

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

- [ ] **Step 5: Assert JAX stays unloaded on `import plot_dispatch` (STRICT — regression gate)**

Run: `uv run python -c "import sys; import xpcsjax.cli.plot_dispatch; print('jax:', 'jax' in sys.modules, 'matplotlib:', 'matplotlib' in sys.modules)"`
Expected (STRICT): `jax: False matplotlib: False`. Verified baseline: importing `plot_dispatch` today loads neither (the `import jax.numpy` is function-local in `_evaluate_model_c2`). Because `plot_dispatch` imports `simulated` at module top, this stays `False` ONLY IF `simulated.py` keeps `import jax.numpy` function-local inside `_evaluate_model_c2` (per the Global Constraint). If this prints `jax: True`, you hoisted the import — move it back inside the function. This is a regression gate, not a "record it" — a `True` here fails the task.

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

Cut `_generate_post_fit_plots` (L412-…) and `_save_fit_comparison_only` (L432-…) into `xpcsjax/cli/plot_families/postfit.py`. Add imports they reference: `from pathlib import Path`, `from typing import Any`, **`from xpcsjax.cli.plot_backend import _PLOT_DISPATCH_CALL_COUNTER, _current_run_id, should_use_datashader`** (`_generate_post_fit_plots` calls `should_use_datashader` at L427; both functions call `_current_run_id()` at L498,522,542 — omitting either raises `NameError`), `from xpcsjax.service.plots import generate_plots` (currently `plot_dispatch.py:420`), `import logging` (**stdlib**, for `logging.WARNING` at L461/501/525/545), the `xpcsjax.utils.logging` utils, and under `TYPE_CHECKING` both `ConfigManager` and `OptimizationResult` (from `xpcsjax.optimization.nlsq.results`, used in the `result:` parameter annotation). Run `ruff`+`mypy` on the new file after the move and add/drop imports until clean. Note: the `_evaluate_c2_per_angle` import from `xpcsjax.viz.nlsq_plots` is a **function-local** import inside `_save_fit_comparison_only` (currently `plot_dispatch.py:494`) — it moves *verbatim with the function body* (do not hoist it to module top; keeping it local preserves the string-path monkeypatch `"xpcsjax.viz.nlsq_plots._evaluate_c2_per_angle"` used by the test, and is why postfit transitively uses JAX by design). Keep docstrings.

- [ ] **Step 2: Rewire `plot_dispatch.py`**

Add `from xpcsjax.cli.plot_families.postfit import _generate_post_fit_plots, _save_fit_comparison_only` to `plot_dispatch.py` (the orchestrator calls both, e.g. `:655`).

- [ ] **Step 3: Update the tests**

- `tests/cli/test_output_resolution.py:110,144`: these reference `plot_dispatch._generate_post_fit_plots(...)` via `from xpcsjax.cli import plot_dispatch` (`:138`). Change to `from xpcsjax.cli.plot_families import postfit` and call `postfit._generate_post_fit_plots(...)`.
- `tests/cli/test_plot_dispatch_logging.py:117,163,171`: these call `pd._save_fit_comparison_only(...)`. Add `from xpcsjax.cli.plot_families import postfit` and change those calls to `postfit._save_fit_comparison_only(...)`. (The `xpcsjax.viz.*` string-form patches at `:111-112` stay unchanged.) Note: `plot_dispatch` re-imports `_save_fit_comparison_only` (Step 2), so `pd._save_fit_comparison_only` would technically still resolve — repointing to `postfit` is the clearer home and is preferred.
- `xpcsjax/service/plots.py:3` docstring: update the cross-reference `cli.plot_dispatch._generate_post_fit_plots` → `cli.plot_families.postfit._generate_post_fit_plots`.

- [ ] **Step 4: Run the covering tests**

Run: `uv run pytest tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py`
Run: `uv run mypy xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/cli/plot_families/postfit.py xpcsjax/cli/plot_dispatch.py xpcsjax/service/plots.py tests/cli/test_output_resolution.py tests/cli/test_plot_dispatch_logging.py
git commit -m "refactor(cli): extract plot_families/postfit

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Thin `plot_dispatch.py` orchestrator — final sweep

**Files:**
- Modify: `xpcsjax/cli/plot_dispatch.py` (`__all__`, import sweep, confirm thinness)

**Interfaces:**
- Produces: `dispatch_plots(...)` (signature unchanged) — the primary public entry point; `resolve_plots_dir`, `resolve_phi_angles_for_sim`, `should_use_datashader` remain public **re-exports** via the unchanged `__all__`.

- [ ] **Step 1: Sweep imports and update `__all__`**

In `plot_dispatch.py`: the body should now be the import lines plus `dispatch_plots` and its inner `_record`. Remove imports now unused by the remaining code (e.g. `itertools` if the counter moved; unused logging utils) — but see the next sentence before removing the re-exported helpers. **Keep `__all__` UNCHANGED** (`["dispatch_plots", "resolve_plots_dir", "resolve_phi_angles_for_sim", "should_use_datashader"]`) to preserve the public API: ensure `plot_dispatch` re-exports the three moved helpers by importing them back — `from xpcsjax.cli.plot_backend import resolve_plots_dir, should_use_datashader` and `from xpcsjax.cli.plot_families.simulated import resolve_phi_angles_for_sim` (some are already imported for `dispatch_plots`' own use; add the rest purely as re-exports). Do NOT reduce `__all__` to `["dispatch_plots"]` — that would drop three names currently declared public (a contract break). Because these names are intentionally re-exported, ruff will not flag them as unused (`__all__` membership counts as use).

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

Cut `_MODE_TO_TEMPLATE` (L43), `_VALID_MODES` (L50, `= tuple(_MODE_TO_TEMPLATE.keys())`), `get_template_path` (L53), `generate_config` (L90), `show_template` (L197), `validate_config` (L217), `_prompt` (L276), `interactive_builder` (L322) from `config_generator.py` into a new `xpcsjax/cli/config_template.py`. Add the imports they reference: `import sys` (`show_template` writes via `sys.stdout.write`, `config_generator.py:214`), `from importlib.resources import files` (`get_template_path` calls `files("xpcsjax.config")`, `config_generator.py:82`), `from pathlib import Path`, `from typing import Any`, `from xpcsjax.config import ConfigManager` (used by `validate_config`; stays JAX-free), plus yaml/logging as referenced. Run `ruff`+`mypy` on the new file until clean. Keep docstrings.

- [ ] **Step 2: Make `config_generator.py` a facade**

In `config_generator.py`, add a re-export so the GUI (`main_window.py:459`) and tests keep importing from `config_generator`:
```python
from xpcsjax.cli.config_template import (
    _MODE_TO_TEMPLATE,
    _VALID_MODES,
    generate_config,
    get_template_path,
    interactive_builder,
    show_template,
    validate_config,
)
```
`_VALID_MODES` is **required** by `_build_parser` (`choices=list(_VALID_MODES)`,
currently L419) which stays in `config_generator.py` — omitting it raises
`NameError` at parser construction. Keep `_build_parser`, `build_parser`, `main`.
Update `main` and any internal references to call the (now imported) template
functions. Update `config_generator.__all__` to include the public re-exported
names plus `build_parser`/`main` (leave the underscore-prefixed `_VALID_MODES` /
`_MODE_TO_TEMPLATE` out of `__all__` — they are imported for internal use).

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
import argparse
import logging

import pytest

from xpcsjax.cli import config_handling


def test_load_failure_names_the_file(tmp_path):
    # Signatures (verified): load_and_merge_config(yaml_path, cli_args);
    # ConfigManager(str(yaml_path)) raises on a bad file. The wrap must name the
    # path for load errors that don't already (e.g. malformed YAML). Type is NOT
    # contracted (no existing test pins it).
    bad = tmp_path / "broken.yaml"
    bad.write_text("not: [valid: yaml", encoding="utf-8")  # malformed
    with pytest.raises(Exception) as exc:
        config_handling.load_and_merge_config(bad, argparse.Namespace())
    assert str(bad) in str(exc.value)  # error names which config failed


def test_normalize_gate_tolerates_object_without_method():
    # apply_cli_overrides(config_manager, args) reads config_manager.config and,
    # when args.mode is set, calls config_manager._normalize_analysis_mode().
    # A config-manager-shaped double WITHOUT that method must not crash the
    # override (the defensive gate, formerly `except AttributeError: pass`).
    class _NoNormalize:
        config = {"analysis_mode": "static_anisotropic"}

    config_handling.apply_cli_overrides(
        _NoNormalize(), argparse.Namespace(mode="static_isotropic", output=None)
    )  # no exception == gate works


def test_non_dict_output_block_is_logged(caplog):
    # When config['output'] is not a mapping, the reset must be logged (not silent).
    class _BadConfig:
        config = {"output": "not_a_dict"}

    with caplog.at_level(logging.WARNING):
        config_handling.apply_cli_overrides(
            _BadConfig(), argparse.Namespace(mode=None, output="/tmp/out")
        )
    assert any("output" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/cli/test_config_handling_errors.py -q`
Expected: `test_non_dict_output_block_is_logged` FAILS (the reset is currently silent — no warning). `test_normalize_gate_tolerates_object_without_method` already passes (the current `try/except AttributeError` tolerates the missing method — the Step 4 getattr rewrite keeps it green; it's a guard against regressing that). `test_load_failure_names_the_file` FAILS iff `ConfigManager` does not already name the file on a YAML parse error — if it passes pre-fix, that is acceptable (it documents the contract; the Step 3 wrap guarantees it). The genuine RED→GREEN driver is the non-dict-output test.

- [ ] **Step 3: Harden L108 (load failure context)**

Wrap the bare `ConfigManager(str(yaml_path))` (L108) load in try/except and
re-raise with file context:
```python
try:
    config_manager = ConfigManager(str(yaml_path))
except Exception as e:
    raise ValueError(f"Failed to load config from {yaml_path}: {e}") from e
```
Use `ValueError` (message-based), NOT `raise type(e)(...)`: verified no existing
test pins the exception *type* on this entry, and reconstructing
`FileNotFoundError(msg)` would silently set `.filename=None` (the one-arg `OSError`
constructor). `from e` preserves the original traceback/cause. The regression
test asserts the path appears in the message, not a specific type.

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

Also update these stale doc/comment cross-references to relocated symbols (cosmetic; `config_handling.py:37` is already covered in Task 1, `service/plots.py:3` in Task 4):
- `xpcsjax/service/plots.py:68`: `_generate_post_fit_plots` mention → `cli.plot_families.postfit._generate_post_fit_plots`.
- `xpcsjax/viz/nlsq_plots.py:504`: comment `plot_dispatch._evaluate_model_c2` → `plot_families.simulated._evaluate_model_c2`.
- `xpcsjax/gui/views/config_dialogs.py:34`: comment `config_generator._MODE_TO_TEMPLATE` → `config_template._MODE_TO_TEMPLATE` (comment-only; the values are duplicated as a literal there).
Add these files to this task's `git add`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_config_handling_errors.py -q`
Expected: PASS.

- [ ] **Step 8: Lint + type-check**

Run: `uv run ruff check xpcsjax/cli/config_handling.py xpcsjax/cli/main.py tests/cli/test_config_handling_errors.py`
Run: `uv run mypy xpcsjax/cli/config_handling.py xpcsjax/cli/main.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add xpcsjax/cli/config_handling.py xpcsjax/cli/main.py xpcsjax/service/plots.py xpcsjax/viz/nlsq_plots.py xpcsjax/gui/views/config_dialogs.py tests/cli/test_config_handling_errors.py
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

**JAX-free invariant (verified):** importing `plot_dispatch` today loads neither `jax` nor `matplotlib` (the `import jax.numpy` is function-local in `_evaluate_model_c2`). Task 3 Step 5 **strictly asserts** this stays `False` post-refactor; the Global Constraint forbids hoisting JAX/matplotlib imports to module top in the family modules, which is what preserves it. (An earlier draft wrongly called an eager import "acceptable" — corrected.)

## Review & validation (2026-06-21, round 2)

Second plan-review pass by **codex**, **agy**, and **Claude** (all three completed;
codex run with stdin `< /dev/null`; codex also ran the named CLI tests → 27 passed).
They re-confirmed the round-1 cross-module-import fixes and the `__all__`/`_VALID_MODES`
decisions are correct. Round-2 fixes applied (each verified against code):

- **MAJOR (codex):** per-task ruff gate would fail — Task 1 deferred pruning but
  runs `ruff check`; moving the counter/`_current_run_id` orphans `itertools` +
  `_LOG_CONTEXT`. Now: **prune per task** before the gate.
- **MAJOR (codex):** importing `plot_dispatch` is JAX/matplotlib-free **today**
  (verified). Task 3 Step 5 is now a **strict** `jax: False` regression gate (was
  "record it / acceptable"); self-review note corrected.
- **MAJOR (agy):** family modules' import lists omitted stdlib imports the moved
  functions use — added `numpy`/`logging` (experimental, simulated), `logging` +
  `OptimizationResult` typing (postfit), `sys` + `importlib.resources.files`
  (config_template); plus a "verify with ruff F821/mypy, don't rely on the list"
  procedure.
- **MINOR (codex):** Task 5 "only public symbol" wording fixed (re-exports stay).
- **MINOR (Claude/agy):** Task 7 test bodies fleshed out against the verified
  `apply_cli_overrides(config_manager, args)` signature; Step 2 RED expectation
  corrected (the non-dict-output test is the genuine driver; the missing-file case
  is already named by `ConfigManager`).

Optional follow-up (NIT, pre-existing, out of scope): `service/config.py:24-25`
comment claims importing `config_generator` pulls JAX — `ConfigManager` is
JAX-free, so the comment is already stale independent of this refactor.

## Review & validation (2026-06-21, round 1)

Reviewed by **agy** and a **Claude** agent (both completed, converging); **codex**
failed this round (hung on stdin, produced no findings). Every fix-driving fact
was independently verified against the code before editing:

- **MAJOR:** `_plot_simulated_from_config` calls `_current_run_id()` (`:301,331`)
  → added to Task 3's `simulated.py` import list (was missing → `NameError`).
- **MAJOR:** `_generate_post_fit_plots` calls `should_use_datashader` (`:427`) and
  both postfit fns call `_current_run_id()` (`:498,522,542`) → both added to
  Task 4's `postfit.py` import list.
- **MAJOR:** `_build_parser` uses `choices=list(_VALID_MODES)` (`:419`) → Task 6
  now moves `_VALID_MODES` (`:50`) to `config_template` and imports it back into
  the facade (else `NameError` at parser build).
- **MINOR:** Task 7 test skeleton now passes the required 2nd arg
  (`load_and_merge_config(bad, argparse.Namespace())`); the L108 wrap uses a
  message-based `ValueError` (no test pins the type; avoids
  `FileNotFoundError.filename=None`).
- **MINOR/NIT:** monkeypatch wording aligned to object-form; `service/plots.py:3`
  docstring cross-ref added to Task 4; the postfit `_evaluate_c2_per_angle`
  function-local import is kept local (preserves the string-path patch).

Both reviewers confirmed: clean module DAG (no circular import), facade +
`build_parser`/completion contracts intact, `config_template` stays JAX-free,
task ordering sound (re-imports keep old attribute paths alive through the
migration), and all symbol→destination line numbers correct.
