# GUI Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `main_window.py` (728-LOC god-class) into a `MainWindow` facade + 4 `QObject` collaborators, and split `plots_view.py` into a `plots/` subpackage behind a re-export facade — strictly behavior-preserving, public + test-imported contracts unchanged, GUI stays JAX-free.

**Architecture:** `plots_view.py` → `views/plots/{helpers,squares,maps,residuals,grid}.py` + a re-export facade. `main_window.py` keeps `MainWindow` (UI construction, signal wiring, public API, ALL `_on_*` slot shims, `_show_result_with_bundle`, `_expand_path`); the *bodies* of four domains move into `views/main_window_support/{status_manager,result_presenter,project_dialog_handler,run_controller}.py` — each a `QObject` parented to `MainWindow` holding a back-ref (`self._mw`). Slot shims delegate to collaborators, so signal connections, direct test calls, and Qt lifetime are preserved.

**Tech Stack:** Python 3.12+, PySide6 (QtWidgets/QtCore), pyqtgraph, pytest (headless Qt via `QT_QPA_PLATFORM=offscreen`). `uv` (`uv run pytest|ruff|mypy`). Bare `python` is NOT on PATH — always `uv run python`.

## Global Constraints

- **Strictly behavior-preserving.** Plotting/window behavior, signal semantics, and the public API are unchanged. The existing GUI suite is the safety net.
- **Public + test contracts unchanged.** `MainWindow` (imported by `app.py` + 4 test files) and module-level `_expand_path` stay in `main_window.py`. ALL `_on_*` slots + `_show_result_with_bundle` stay as methods on `MainWindow` (signal wiring connects `self._on_*`; tests call `win._on_run()`/`win._on_create_config()`/`win._on_runs_selected()`/`win._show_result_with_bundle()`). `plots_view.py` re-exports all 10 test/src-imported names.
- **GUI process stays JAX-free** — no collaborator/plots module imports JAX. `test_gui_jax_free.py` must pass (and gains probes for the new modules).
- **Collaborators are `QObject` subclasses** parented to the `MainWindow` instance (`super().__init__(main_window)`), holding `self._mw = main_window`. Constructed in `MainWindow.__init__` and stored on `self` BEFORE signal wiring.
- **Body-move transform:** moving a method body into `Collaborator.method` means copying it verbatim EXCEPT every `self.<x>` reference becomes `self._mw.<x>` (the collaborator reaches `MainWindow` state/methods through the back-ref). `MainWindow.<method>` then becomes a one-line shim `return self._<collab>.<method>(...)` preserving its name/signature.
- Ruff: line-length 100, `E,F,W,I,B,UP,N` + numpydoc `D` gate (`xpcsjax/` only; tests exempt). New modules/classes/public methods need NumPy docstrings; moved methods keep theirs.
- `uv run mypy xpcsjax` is the HARD CI gate.
- One commit per task. After moving any function, run `ruff` (F401/F821) + `mypy` on the touched files and fix until clean — the import lists below are a guide, the gates are the safety net.

## File Structure

**Create:**
- `xpcsjax/gui/views/plots/__init__.py`, `plots/helpers.py`, `plots/squares.py`, `plots/maps.py`, `plots/residuals.py`, `plots/grid.py`
- `xpcsjax/gui/views/main_window_support/__init__.py`, `status_manager.py`, `result_presenter.py`, `project_dialog_handler.py`, `run_controller.py`

**Modify:**
- `xpcsjax/gui/views/plots_view.py` — becomes a re-export facade.
- `xpcsjax/gui/views/main_window.py` — collaborators constructed + method bodies → shims.
- `tests/gui/test_gui_jax_free.py` — add probes for the new modules.
- `tests/gui/test_gui_redesign.py` — repoint the `_on_create_config` monkeypatch targets (Task 4).

**plots_view.py symbol → destination (verbatim moves):**

| Symbol (current line) | Destination |
|---|---|
| `_leading_dim_matches` (27), `_apply_colormap` (44), `_time_rect` (78), `_c2_levels` (100), `_residual_levels` (116); constants `_C2_COLORMAP` (40), `_RESIDUAL_COLORMAP` (41), `_SCATTER_MAX_POINTS` (290) | `plots/helpers.py` |
| `_SquareBase`, `_SquareAspectMixin` (136), `_fit_square_view` (56) | `plots/squares.py` |
| `TwoTimeMapView` (153), `ResidualMapView` (220) | `plots/maps.py` |
| `ResidualHistogramView` (293), `DiagonalResidualView` (323), `ResidualsVsFittedView` (348) | `plots/residuals.py` |
| `_PhiSection` (378), `PhiResultsGrid` (487) | `plots/grid.py` |

**main_window.py method body → collaborator (shim stays on MainWindow):**

| Method (line) | Collaborator method |
|---|---|
| `set_status` (233), `append_log` (246) | `StatusManager` |
| `show_result` (250), `show_error` (296), `show_inspector` (380), `_show_result_with_bundle` (270, rendering body) | `ResultPresenter` |
| `_on_create_project` (573), `_on_create_config` (578), `_on_edit_config` (614), `_on_load_config` (622), `_on_save_project` (630), `_on_open_project` (637), `_on_close_project` (644) | `ProjectDialogHandler` |
| `_on_run` (664), `_on_cancel` (679), `_on_export_figure` (686) | `RunController` |
| `_on_run_status` (300), `_on_log` (320), `_on_run_failed` (324), `_on_run_finished` (332), `_on_runs_selected` (355) | **stay on MainWindow** (cross-cutting; not extracted) |

---

## Task 1: `plots/` subpackage + `plots_view.py` re-export facade

**Files:**
- Create: `xpcsjax/gui/views/plots/__init__.py` + `helpers.py`, `squares.py`, `maps.py`, `residuals.py`, `grid.py`
- Modify: `xpcsjax/gui/views/plots_view.py` (→ facade), `tests/gui/test_gui_jax_free.py`

**Interfaces:**
- Produces: every name in the plots_view move table, importable from BOTH its new module and (re-exported) from `xpcsjax.gui.views.plots_view`.

- [ ] **Step 1: Create the package + move helpers/squares verbatim**

Create `plots/__init__.py` (one-line docstring). Move into `plots/helpers.py`: `_leading_dim_matches`, `_apply_colormap`, `_time_rect`, `_c2_levels`, `_residual_levels` + the constants `_C2_COLORMAP`, `_RESIDUAL_COLORMAP`, `_SCATTER_MAX_POINTS` (with their imports: `numpy as np`, `pyqtgraph as pg`, `QRectF` from `PySide6.QtCore`, as referenced). Move into `plots/squares.py`: `_SquareBase`, `_SquareAspectMixin`, `_fit_square_view` (with `pyqtgraph as pg` etc.). `squares.py` imports nothing from the plot widgets.

- [ ] **Step 2: Move the widget classes verbatim**

`plots/maps.py` ← `TwoTimeMapView`, `ResidualMapView`; `plots/residuals.py` ← `ResidualHistogramView`, `DiagonalResidualView`, `ResidualsVsFittedView`; `plots/grid.py` ← `_PhiSection`, `PhiResultsGrid`. Each imports what it uses from `..helpers` / `..squares` (e.g. maps/residuals use `_SquareAspectMixin`, the level/colormap helpers, `_SCATTER_MAX_POINTS`; grid imports the widget classes from `.maps`/`.residuals`). Keep all docstrings.

- [ ] **Step 3: Make `plots_view.py` a re-export facade**

Replace `plots_view.py` body with imports + `__all__`:
```python
"""Facade re-exporting the GUI plot widgets (now in the plots/ subpackage)."""
from xpcsjax.gui.views.plots.grid import PhiResultsGrid, _PhiSection
from xpcsjax.gui.views.plots.helpers import (
    _SCATTER_MAX_POINTS, _apply_colormap, _c2_levels, _leading_dim_matches,
    _residual_levels, _time_rect,
)
from xpcsjax.gui.views.plots.maps import ResidualMapView, TwoTimeMapView
from xpcsjax.gui.views.plots.residuals import (
    DiagonalResidualView, ResidualHistogramView, ResidualsVsFittedView,
)
from xpcsjax.gui.views.plots.squares import _SquareAspectMixin

__all__ = [
    "PhiResultsGrid", "TwoTimeMapView", "ResidualMapView", "ResidualHistogramView",
    "DiagonalResidualView", "ResidualsVsFittedView", "_SCATTER_MAX_POINTS",
    "_c2_levels", "_residual_levels", "_time_rect", "_SquareAspectMixin", "_PhiSection",
]
```
(`__all__` membership keeps ruff from flagging the re-exports as F401.) `main_window.py:42`'s `from …plots_view import PhiResultsGrid` is unchanged.

- [ ] **Step 4: Add JAX-free probes for the new plots modules**

In `tests/gui/test_gui_jax_free.py` add:
```python
def test_plots_subpackage_is_jax_free():
    for m in ("helpers", "squares", "maps", "residuals", "grid"):
        assert _probe_import(f"xpcsjax.gui.views.plots.{m}") == 0
```

- [ ] **Step 5: Run the covering tests**

Run: `uv run pytest tests/gui/test_plots_view.py tests/gui/test_gui_redesign.py tests/gui/test_gui_debug_fixes.py tests/gui/test_gui_debug_2026_06_21.py tests/gui/test_gui_jax_free.py -q`
Expected: PASS (facade keeps every imported name resolvable; plots modules JAX-free).

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check xpcsjax/gui/views/plots/ xpcsjax/gui/views/plots_view.py tests/gui/test_gui_jax_free.py`
Run: `uv run mypy xpcsjax/gui/views/plots xpcsjax/gui/views/plots_view.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add xpcsjax/gui/views/plots/ xpcsjax/gui/views/plots_view.py tests/gui/test_gui_jax_free.py
git commit -m "refactor(gui): split plots_view into plots/ subpackage behind a re-export facade

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `main_window_support/` package + `StatusManager`

**Files:**
- Create: `xpcsjax/gui/views/main_window_support/__init__.py`, `status_manager.py`
- Modify: `xpcsjax/gui/views/main_window.py`

**Interfaces:**
- Produces: `StatusManager(main_window)` (a `QObject`) with `set_status(status)` / `append_log(level, message)`. `MainWindow.set_status`/`.append_log` become shims delegating to it.

- [ ] **Step 1: Create the package + StatusManager**

Create `main_window_support/__init__.py` (one-line docstring). Create `status_manager.py`:
```python
"""Status-bar + log presentation collaborator for MainWindow."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

if TYPE_CHECKING:
    from xpcsjax.gui.views.main_window import MainWindow


class StatusManager(QObject):
    """Owns the set_status / append_log bodies (operates on MainWindow widgets)."""

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(main_window)
        self._mw = main_window

    def set_status(self, status: str) -> None:
        ...  # body of MainWindow.set_status (L233-245), self.X -> self._mw.X

    def append_log(self, level: str, message: str) -> None:
        ...  # body of MainWindow.append_log (L246-249), self.X -> self._mw.X
```
Fill the two method bodies from `main_window.py` (L233-249), applying the `self.`→`self._mw.` transform (per Global Constraints). Import any names the bodies reference.

- [ ] **Step 2: Wire into MainWindow + convert to shims**

In `MainWindow.__init__` (after the widgets exist, before signal wiring), add `self._status_manager = StatusManager(self)`. Replace the bodies of `MainWindow.set_status`/`.append_log` with shims:
```python
def set_status(self, status: str) -> None:
    self._status_manager.set_status(status)

def append_log(self, level: str, message: str) -> None:
    self._status_manager.append_log(level, message)
```
Keep the `status_text`/`log_text` properties on `MainWindow` (they read the widgets).

- [ ] **Step 3: Write a focused StatusManager unit test**

Create `tests/gui/test_status_manager.py` (uses the offscreen-Qt fixture pattern from the other GUI tests — see `tests/gui/conftest.py`):
```python
def test_status_manager_sets_status_and_appends_log(qtbot_or_app_fixture):
    win = MainWindow()
    win.set_status("ready")
    assert win.status_text == "ready"   # delegates through StatusManager
    win.append_log("INFO", "hello")
    assert "hello" in win.log_text
```
Adjust the fixture/assertions to the real `status_text`/`log_text` behavior (read `conftest.py` + `test_main_window.py` for the pattern).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/gui/test_status_manager.py tests/gui/test_main_window.py tests/gui/test_gui_redesign.py -q`
Expected: PASS (status/log behavior unchanged via the shims).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/gui/views/main_window_support/ xpcsjax/gui/views/main_window.py tests/gui/test_status_manager.py`
Run: `uv run mypy xpcsjax/gui/views/main_window_support xpcsjax/gui/views/main_window.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/gui/views/main_window_support/ xpcsjax/gui/views/main_window.py tests/gui/test_status_manager.py
git commit -m "refactor(gui): extract StatusManager collaborator (set_status/append_log)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `ResultPresenter`

**Files:**
- Create: `xpcsjax/gui/views/main_window_support/result_presenter.py`
- Modify: `xpcsjax/gui/views/main_window.py`

**Interfaces:**
- Consumes: nothing from Task 2 (independent).
- Produces: `ResultPresenter(main_window)` (`QObject`) with `show_result(summary)`, `show_error(message)`, `show_inspector(summary)`, and `show_result_with_bundle(summary, result_dir)`. `MainWindow.show_result`/`.show_error`/`.show_inspector`/`._show_result_with_bundle` become shims.

- [ ] **Step 1: Create ResultPresenter**

Create `result_presenter.py` with a `QObject` subclass (same skeleton as StatusManager: `__init__(self, main_window)` → `super().__init__(main_window); self._mw = main_window`). Move the bodies of `show_result` (L250-269), `show_error` (L296-…), `show_inspector` (L380-…), and `_show_result_with_bundle` (L270-295) into methods `show_result`/`show_error`/`show_inspector`/`show_result_with_bundle`, applying `self.`→`self._mw.` (they touch `_result_grid`, `_inspector`, `_results`, `_central_stack`, and call `self.show_result`/`set_status` → `self._mw.show_result`/`self._mw.set_status`). `show_error` must keep using `present_failure` from `xpcsjax.gui.error_presenter` exactly as today. Import what the bodies reference (`present_failure`, `ResultSummary` under `TYPE_CHECKING`, etc.).

- [ ] **Step 2: Wire + shims**

In `MainWindow.__init__` add `self._result_presenter = ResultPresenter(self)`. Replace the four method bodies with shims:
```python
def show_result(self, summary: Any) -> None:
    self._result_presenter.show_result(summary)
def show_error(self, message: str) -> None:
    self._result_presenter.show_error(message)
def show_inspector(self, summary: ResultSummary | None) -> None:
    self._result_presenter.show_inspector(summary)
def _show_result_with_bundle(self, summary: Any, result_dir: str | None) -> None:
    self._result_presenter.show_result_with_bundle(summary, result_dir)
```
The run-lifecycle slots that call `self._show_result_with_bundle`/`self.show_inspector` are unchanged (they hit the shims).

- [ ] **Step 3: Focused unit test**

Create `tests/gui/test_result_presenter.py` exercising `show_result` / `show_error` routing through a real `MainWindow` (offscreen Qt). Assert: after `win.show_result(summary)`, `win._central_stack.currentIndex()` reflects the grid (index 1) for a bundle-bearing summary; after `win.show_error("boom")`, `win.result_text`/status reflect the error. Read `test_gui_redesign.py:310` (`_show_result_with_bundle`) and `test_main_window.py` for the real summary shapes + assertions to mirror.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/gui/test_result_presenter.py tests/gui/test_main_window.py tests/gui/test_gui_redesign.py tests/gui/test_gui_debug_2026_06_21.py -q`
Expected: PASS (`test_gui_redesign.py:310` calls `win._show_result_with_bundle` → shim; `test_gui_debug_2026_06_21.py:63` calls `win._on_runs_selected` which routes through the shim).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/gui/views/main_window_support/result_presenter.py xpcsjax/gui/views/main_window.py tests/gui/test_result_presenter.py`
Run: `uv run mypy xpcsjax/gui/views/main_window_support/result_presenter.py xpcsjax/gui/views/main_window.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/gui/views/main_window_support/result_presenter.py xpcsjax/gui/views/main_window.py tests/gui/test_result_presenter.py
git commit -m "refactor(gui): extract ResultPresenter (show_result/error/inspector + bundle rendering)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `ProjectDialogHandler`

**Files:**
- Create: `xpcsjax/gui/views/main_window_support/project_dialog_handler.py`
- Modify: `xpcsjax/gui/views/main_window.py`, `tests/gui/test_gui_redesign.py`

**Interfaces:**
- Produces: `ProjectDialogHandler(main_window)` (`QObject`) with methods `on_create_project`, `on_create_config`, `on_edit_config`, `on_load_config`, `on_save_project`, `on_open_project`, `on_close_project`. The corresponding `MainWindow._on_*` slots become shims.

- [ ] **Step 1: Create ProjectDialogHandler**

Create `project_dialog_handler.py` (`QObject` skeleton). Move the bodies of the 7 dialog slots (`_on_create_project` L573, `_on_create_config` L578, `_on_edit_config` L614, `_on_load_config` L622, `_on_save_project` L630, `_on_open_project` L637, `_on_close_project` L644) into `on_*` methods, applying `self.`→`self._mw.` (they call `self.create_project`/`self.create_config`/`self.add_dataset`/`self.set_status` etc. → `self._mw.…`, and construct Qt dialogs with `self._mw` as the parent). **Import `CreateConfigDialog` and `QMessageBox` (and any other dialog classes the bodies use) into this module** — these names are now looked up here.

- [ ] **Step 2: Wire + shims**

In `MainWindow.__init__` add `self._dialog_handler = ProjectDialogHandler(self)`. Replace each `_on_*` dialog-slot body with a shim, e.g.:
```python
def _on_create_config(self) -> None:
    self._dialog_handler.on_create_config()
```
(same for the other six). The menu/button `connect(self._on_create_project)` wiring is unchanged. If `main_window.py` no longer references `CreateConfigDialog`/`QMessageBox` after the moves, remove those now-unused imports from `main_window.py` (ruff F401) — they live in `project_dialog_handler.py` now.

- [ ] **Step 3: Repoint the monkeypatch in test_gui_redesign.py**

`test_on_create_config_guards_overwrite_retry_failure` (around L401-451) does `import xpcsjax.gui.views.main_window as mw` then `monkeypatch.setattr(mw, "CreateConfigDialog", _FakeDialog)` and `monkeypatch.setattr(mw.QMessageBox, "question"/"warning", …)`. Since `_on_create_config`'s body now looks up those names in `project_dialog_handler`, change the patches to target that module:
```python
import xpcsjax.gui.views.main_window_support.project_dialog_handler as pdh
monkeypatch.setattr(pdh, "CreateConfigDialog", _FakeDialog)
monkeypatch.setattr(pdh.QMessageBox, "question", …)
monkeypatch.setattr(pdh.QMessageBox, "warning", …)
```
The `win._on_create_config()` call at the end is unchanged (it routes through the shim). Assertions unchanged.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/gui/test_gui_redesign.py tests/gui/test_main_window.py tests/gui/test_project_io_wiring.py -q`
Expected: PASS (the overwrite-retry-guard test now patches the handler namespace; project dialog flows unchanged).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check xpcsjax/gui/views/main_window_support/project_dialog_handler.py xpcsjax/gui/views/main_window.py tests/gui/test_gui_redesign.py`
Run: `uv run mypy xpcsjax/gui/views/main_window_support/project_dialog_handler.py xpcsjax/gui/views/main_window.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add xpcsjax/gui/views/main_window_support/project_dialog_handler.py xpcsjax/gui/views/main_window.py tests/gui/test_gui_redesign.py
git commit -m "refactor(gui): extract ProjectDialogHandler (project/config dialog slots)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `RunController`

**Files:**
- Create: `xpcsjax/gui/views/main_window_support/run_controller.py`
- Modify: `xpcsjax/gui/views/main_window.py`

**Interfaces:**
- Produces: `RunController(main_window)` (`QObject`) with `on_run`, `on_cancel`, `on_export_figure`. `MainWindow._on_run`/`._on_cancel`/`._on_export_figure` become shims.

- [ ] **Step 1: Create RunController**

Create `run_controller.py` (`QObject` skeleton). Move the bodies of `_on_run` (L664-678), `_on_cancel` (L679-685), `_on_export_figure` (L686-724) into `on_run`/`on_cancel`/`on_export_figure`, applying `self.`→`self._mw.` (they reach `_queue`, `_sidebar`, `_project`, `_active_dataset_id`, `_per_run_output_dir`, `set_status`; `on_export_figure` constructs `QFileDialog`/`QMessageBox` with `self._mw` as parent and calls the `export_figures` helper). Import `QFileDialog`/`QMessageBox`/`export_figures` (and whatever else the bodies use) into this module.

- [ ] **Step 2: Wire + shims**

In `MainWindow.__init__` add `self._run_controller = RunController(self)`. Replace the three slot bodies with shims (`def _on_run(self): self._run_controller.on_run()`, etc.). Wiring of these slots to buttons is unchanged.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/gui/test_main_window.py tests/gui/test_export_flow.py tests/gui/test_fit_queue.py -q`
Expected: PASS (`test_main_window.py:105-106` calls `win._on_run()` → shim; export flow unchanged).

- [ ] **Step 4: Lint + type-check**

Run: `uv run ruff check xpcsjax/gui/views/main_window_support/run_controller.py xpcsjax/gui/views/main_window.py`
Run: `uv run mypy xpcsjax/gui/views/main_window_support/run_controller.py xpcsjax/gui/views/main_window.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add xpcsjax/gui/views/main_window_support/run_controller.py xpcsjax/gui/views/main_window.py
git commit -m "refactor(gui): extract RunController (run/cancel/export-figure slots)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: JAX-free probes for support modules + final verification

**Files:**
- Modify: `tests/gui/test_gui_jax_free.py`

- [ ] **Step 1: Add support-module JAX-free probes**

In `tests/gui/test_gui_jax_free.py` add:
```python
def test_main_window_support_is_jax_free():
    for m in ("status_manager", "result_presenter", "project_dialog_handler", "run_controller"):
        assert _probe_import(f"xpcsjax.gui.views.main_window_support.{m}") == 0
```

- [ ] **Step 2: Confirm MainWindow shrank + shims/slots intact**

Run: `uv run python -c "import inspect, xpcsjax.gui.views.main_window as m; src=inspect.getsource(m); print('LOC:', len(src.splitlines())); w=m.MainWindow; print('slots present:', all(hasattr(w,n) for n in ['_on_run','_on_create_config','_on_runs_selected','_show_result_with_bundle','set_status','show_result']))"`
Expected: LOC ≈ 430–480; `slots present: True`.

- [ ] **Step 3: Full GUI suite + hard gates**

Run: `uv run pytest tests/gui -q`
Run: `uv run mypy xpcsjax`  (HARD CI gate)
Run: `make verify`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/gui/test_gui_jax_free.py
git commit -m "test(gui): JAX-free probes for the new main_window_support collaborators

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** plots_view facade split (Task 1), the 4 collaborators (Tasks 2-5) as QObject-parented back-ref objects with `_on_*`/`show_*`/`_show_result_with_bundle` shims retained on MainWindow, run-lifecycle slots kept on MainWindow (move table), the monkeypatch test repoint (Task 4 Step 3), JAX-free probes for both new package trees (Tasks 1 & 6), the real dialog slot names (`_on_load_config`/`_on_edit_config`/`_on_close_project`, no `_on_add_dataset`), `_central_stack`/`present_failure`/`export_figures` dependencies, and the final gates (Task 6). All spec acceptance criteria map to a task.

**Placeholder scan:** collaborator method bodies are specified as "move L-range body, apply self.→self._mw." (a verbatim-with-prefix transform, the standard way to specify a move without duplicating 40-line Qt bodies); the QObject skeletons + shims show real code. Unit-test bodies say "mirror the real summary shapes from test_X" because the exact ResultSummary fixtures live in those tests — the behavior each asserts is specified.

**Type consistency:** `self._status_manager`/`self._result_presenter`/`self._dialog_handler`/`self._run_controller` names used consistently between the wire step and the shims; collaborator method names (`set_status`, `show_result`, `on_create_config`, `on_run`, …) match between their "Produces" blocks and the shims that call them; `_mw` back-ref name uniform across all four collaborators.

**Known judgment call:** the `self.`→`self._mw.` transform is mechanical but pervasive; ruff F821 (undefined name) + mypy + the per-task GUI tests are the net that catches a missed reference. Each task lints/types only its touched files; Task 6 runs the whole suite + `make verify`.
