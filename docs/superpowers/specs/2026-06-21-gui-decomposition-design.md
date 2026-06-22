# GUI Decomposition — Design

**Date:** 2026-06-21
**Status:** Approved (brainstorming) — pending implementation plan
**Sub-project:** 3 of 3 (GUI / CLI / shell completion optimization effort)

## Problem

Two GUI view modules are large:

- `xpcsjax/gui/views/main_window.py` (728 LOC) — a single `MainWindow` class with
  **42 methods** spanning 8 unrelated concerns (UI construction, status/logging,
  result presentation, run-lifecycle events, project-sync API, dialog handlers,
  run control, lifecycle). A genuine god-class.
- `xpcsjax/gui/views/plots_view.py` (593 LOC) — 13 focused top-level units
  (helpers, an aspect mixin, 5 plot widgets, a per-φ section, the public grid).
  **Already modular** (large by sum-of-parts, not tangled), but the user wants it
  split into per-concern files for navigability.

## Goal

**Strictly behavior-preserving** structural decomposition of both modules. All
public + test-imported contracts unchanged. GUI process stays JAX-free. Verified
by the existing GUI test suite + `make verify` — **no UX, layout, responsiveness,
or behavior changes**.

## Non-goals (YAGNI)

- UX / responsiveness / layout changes (deferred; would need concrete goals + a
  different verification approach — there is no parity oracle for GUI behavior).
- Changes to the controller (`fit_queue`), project model, IPC, worker, or the
  JAX boundary. Collaborators introduced here are pure GUI-side.
- Renaming or relocating `MainWindow` or any public/test-imported symbol.

## Contract constraints (verified importers — must not break)

| Symbol | Importer(s) | Constraint |
|---|---|---|
| `main_window.MainWindow` | `gui/app.py:11`; `tests/gui/test_app_wiring.py`, `test_main_window.py`, `test_gui_debug_fixes.py`, `test_gui_redesign.py` | Stays the public `QMainWindow` in `main_window.py` |
| `main_window._expand_path` | `tests/gui/test_gui_debug_fixes.py:155` (+ internal use L515/519) | Stays a module-level function in `main_window.py` |
| `MainWindow` public methods/properties: `set_status`, `append_log`, `show_result`, `show_error`, `show_inspector`, `status_text`, `log_text`, `result_text`, `sidebar_dataset_count`, `create_project`, `create_config`, `add_dataset`, `save_project_to`, `open_project_from`, `close_project`, `closeEvent` | tests + `app.py` | Stay on `MainWindow` (as thin delegates where logic moves to a collaborator) |
| **Test-pinned PRIVATE slots called directly on a `MainWindow` instance**: `_on_run` (`test_main_window.py:105`), `_on_create_config` (`test_gui_redesign.py:449`), `_on_runs_selected` (`test_gui_debug_2026_06_21.py:63`), `_show_result_with_bundle` (`test_gui_redesign.py:310`) | tests | **MUST remain methods on `MainWindow`** as thin shims forwarding to the owning collaborator (removing/relocating → `AttributeError` in tests). All other `_on_*` slots also stay on `MainWindow` (signal wiring connects `self._on_*`). |
| **Monkeypatched module-level names** `CreateConfigDialog`, `QMessageBox` | `test_gui_redesign.py:429,442-446` patch them via `import …main_window as mw; monkeypatch.setattr(mw, "CreateConfigDialog", …)` / `mw.QMessageBox` | The lookup must occur in whichever module the test patches. Either keep these imported at `main_window.py` module level AND construct the dialogs there, **or** update that test to patch the collaborator's namespace (decision in Component 1). |
| `plots_view.PhiResultsGrid` | `main_window.py:42`; `test_gui_redesign.py:20`; `test_plots_view.py` | Re-exported from `plots_view.py` (facade) |
| `plots_view.{TwoTimeMapView, ResidualMapView, ResidualHistogramView, DiagonalResidualView, ResidualsVsFittedView}` | `test_plots_view.py`; `test_gui_debug_fixes.py:32`; `test_gui_debug_2026_06_21.py:189` | Re-exported from `plots_view.py` |
| `plots_view.{_SCATTER_MAX_POINTS, _c2_levels, _residual_levels, _time_rect}` | `test_plots_view.py:14`; `test_gui_debug_2026_06_21.py:189` (`_residual_levels`) | Re-exported from `plots_view.py` (test-imported "private" names) |

## Components

### 1. `main_window.py` (728) → `MainWindow` facade + 4 collaborators

`MainWindow` remains the public `QMainWindow` and **keeps**: UI construction
(incl. owning `_central_stack`, `_result_grid`, `_inspector`, `_status`, `_log`,
`_results`, `_sidebar` and state attrs `_active_dataset_id`/`_active_run_id`/
`_viewing_run_id`/`_project`/`_queue`), the signal/slot wiring block, the public
project-sync API, `closeEvent`, the module-level `_expand_path`, and — critically —
**ALL `_on_*` slot methods + `_show_result_with_bundle` as thin shims** so that
(a) the existing signal wiring (`self._queue.…connect(self._on_run_status)`,
menu `action.triggered.connect(self._on_create_project)`, etc.) is unchanged, and
(b) tests that call `win._on_run()`/`win._on_create_config()`/`win._on_runs_selected()`/
`win._show_result_with_bundle()` keep working. Each shim's *body* delegates to the
owning collaborator; the slot's identity on `MainWindow` is preserved.

**Run-lifecycle slots stay on `MainWindow` (not extracted):** `_on_run_status`
(L300), `_on_log` (L320), `_on_run_failed` (L324), `_on_run_finished` (L332),
`_on_runs_selected` (L355). They are cross-cutting (touch status, result
presentation, `_active_run_id`/`_viewing_run_id`, `_sidebar`, `_project`) and
calling `_show_result_with_bundle`/`show_inspector`; extracting them would smear
references across every collaborator. `_show_result_with_bundle` (L270, uses
`_central_stack`/`_result_grid`) likewise stays on `MainWindow` (it is the
result-routing helper the lifecycle slots call) but delegates its rendering body
to `ResultPresenter`.

**Collaborator pattern.** Collaborators live under a new
`xpcsjax/gui/views/main_window_support/` package. Each is a **`QObject` subclass
parented to the `MainWindow` instance** (`super().__init__(parent=main_window)`)
and holds a reference to it, so (i) it can read/mutate `MainWindow` state and call
its public API, and (ii) it joins Qt's C++ lifetime tree — avoiding the dangling
"C++ object already deleted" risk and the strong-refcycle that a plain object
owned-by-and-referencing `MainWindow` would create. `MainWindow.__init__`
constructs and stores each collaborator (`self._status_manager`, etc.) **before**
wiring signals.

- **`status_manager.py`** — `StatusManager(main_window)` with the bodies of
  `set_status` / `append_log` (operates on `main_window._status`/`._log`).
  `MainWindow.set_status`/`.append_log` delegate; `status_text`/`log_text`
  properties stay on `MainWindow`.
- **`result_presenter.py`** — `ResultPresenter(main_window)` with the rendering
  bodies of `show_result` / `show_error` / `show_inspector` and the rendering
  body of `_show_result_with_bundle` (operates on `_result_grid`, `_inspector`,
  `_results`, `_central_stack`). `MainWindow.show_*` + `_show_result_with_bundle`
  delegate. `show_error` must preserve the existing `present_failure`
  (`xpcsjax.gui.error_presenter`) behavior. (`test_error_presenter.py` tests that
  separate, untouched `error_presenter.present_failure` module — it is NOT a
  `ResultPresenter` contract; `ResultPresenter` needs its own new unit tests.)
- **`project_dialog_handler.py`** — the dialog slot bodies for the **actual**
  slots: `_on_create_project` (L573), `_on_create_config` (L578),
  `_on_edit_config` (L614), `_on_load_config` (L622), `_on_save_project` (L630),
  `_on_open_project` (L637), `_on_close_project` (L644). (There is **no**
  `_on_add_dataset` slot — the public `add_dataset` API stays on `MainWindow`.)
  These construct Qt dialogs with `main_window` as the Qt parent and call
  `MainWindow`'s public project API.
  **Monkeypatch resolution:** `_on_create_config` constructs `CreateConfigDialog`
  and calls `QMessageBox` which `test_gui_redesign.py` patches via the
  `main_window` module namespace. To keep that test green with minimal churn,
  the dialog handler imports `CreateConfigDialog`/`QMessageBox` and the test's
  patch targets are updated to the `project_dialog_handler` namespace (a
  behavior-preserving test-mechanics update, listed in Testing). `main_window.py`
  retains its `CreateConfigDialog` import only if still used by a retained shim.
- **`run_controller.py`** — `RunController(main_window)` with the bodies of
  `_on_run` / `_on_cancel` / `_on_export_figure`. Dependencies (via the
  `main_window` ref, verified against the code): `_queue`, `_sidebar`, `_project`,
  `_active_dataset_id`, `_per_run_output_dir`, `set_status`, the `export_figures`
  helper, and `main_window` as Qt dialog parent. (**Not** `result_grid` — none of
  these three methods use it.)

Target: `MainWindow` ≈ 430–480 LOC (the slot shims + lifecycle slots + project
API + UI construction stay; only the extracted method *bodies* leave).

**Behavior-preservation rule:** the extracted bodies are *verbatim* moves into
collaborator methods that reach `MainWindow` state through the back-ref. The slot
identities, signal connections, and Qt semantics are unchanged. No control-flow
changes.

### 2. `plots_view.py` (593) → `plots/` subpackage + re-export facade

New `xpcsjax/gui/views/plots/` package (verbatim moves, intra-package imports):
- `helpers.py` — `_leading_dim_matches`, `_apply_colormap`, `_time_rect`,
  `_c2_levels`, `_residual_levels`, and the constants `_SCATTER_MAX_POINTS`,
  `_C2_COLORMAP` (`"jet"`), `_RESIDUAL_COLORMAP` (`"RdBu_r"`) — the last two are
  used only by `_apply_colormap` and must move with it (else orphaned).
- `squares.py` — `_SquareBase`, `_SquareAspectMixin`, and the free function
  `_fit_square_view`.
- `maps.py` — `TwoTimeMapView`, `ResidualMapView`.
- `residuals.py` — `ResidualHistogramView`, `DiagonalResidualView`,
  `ResidualsVsFittedView`.
- `grid.py` — `_PhiSection`, `PhiResultsGrid`.

`plots_view.py` becomes a **re-export facade** that imports and re-exports every
name in the contract table (and lists them in `__all__`), so
`from xpcsjax.gui.views.plots_view import …` is unchanged for every importer.
`main_window.py:42`'s `from …plots_view import PhiResultsGrid` keeps working.

## Data flow / boundaries (unchanged)

`MainWindow` still owns `self._project` (model) + `self._queue`
(`FitQueueController`); the worker subprocess still runs the fit and emits Qt
signals back. Collaborators are GUI-side only — none import JAX, the model, the
worker, or IPC internals beyond what `MainWindow` passes them. The
view↔model↔worker↔IPC seam and the JAX-free guarantee are untouched.

## Error handling

No change to error behavior. `show_error` logic moves verbatim into
`ResultPresenter`; `MainWindow.show_error` delegates.

## Testing

- Behavior-preserving — the existing GUI suite is the safety net:
  `test_main_window`, `test_app_wiring`, `test_gui_redesign`, `test_gui_debug_fixes`,
  `test_gui_debug_2026_06_21`, `test_plots_view`, `test_error_presenter`, plus the
  JAX-free guard `test_gui_jax_free`. All must stay green after the moves +
  import-path/facade updates.
- New focused unit tests for the collaborators with isolatable logic
  (`StatusManager`, `ResultPresenter`) where they can be exercised without a full
  window.
- **Required test-mechanics updates** (behavior-preserving; same assertions):
  `test_gui_redesign.py::test_on_create_config_guards_overwrite_retry_failure`
  patches `mw.CreateConfigDialog` / `mw.QMessageBox` — repoint those
  `monkeypatch.setattr` targets to the `project_dialog_handler` module namespace
  (where `_on_create_config` now constructs them). All `win._on_*(...)` /
  `win._show_result_with_bundle(...)` direct calls keep working because the slot
  shims remain on `MainWindow`.
- **JAX-free guard:** extend `test_gui_jax_free.py`'s import-probe list to include
  `xpcsjax.gui.views.main_window_support.{status_manager,result_presenter,project_dialog_handler,run_controller}`
  and `xpcsjax.gui.views.plots.{helpers,squares,maps,residuals,grid}` (importing
  `main_window`/`plots_view` covers them transitively, but explicit probes catch
  a stray import even if the import graph later changes). It must still pass.
- `test_error_presenter.py` tests the separate, untouched
  `xpcsjax.gui.error_presenter.present_failure` — leave it alone.
- Before moving any symbol, confirm its importers are only those in the contract
  table (grep); the facades cover the listed ones.
- `make verify` + `mypy xpcsjax` (hard CI gate) green before merge.

## Acceptance criteria

1. `main_window.py` ≈ 430–480 LOC; `main_window_support/{status_manager,result_presenter,project_dialog_handler,run_controller}.py` created as `QObject` subclasses parented to `MainWindow`; the extracted method *bodies* live there. `MainWindow` retains `_expand_path`, all listed public methods/properties, AND all `_on_*` slot shims + `_show_result_with_bundle` (delegating bodies) so signal wiring + direct test calls are unchanged.
2. `plots_view.py` is a re-export facade; `plots/{helpers,squares,maps,residuals,grid}.py` created; all 10 contract-table names still importable from `plots_view`.
3. GUI process remains JAX-free (`test_gui_jax_free` passes).
4. Every pre-existing GUI test passes after import-path updates; new collaborator unit tests added.
5. `make verify` + `mypy xpcsjax` green.

## Review & validation (2026-06-21)

Reviewed by **codex**, **agy**, and a **Claude** agent (all three completed,
converging). The `plots_view` facade contract was confirmed complete. Fixes
applied to the `main_window` decomposition, each verified against code:

- **MAJOR (all 3):** test-pinned private slots (`_on_run`, `_on_create_config`,
  `_on_runs_selected`, `_show_result_with_bundle`) are called directly on
  `MainWindow` by tests → they MUST stay as slot shims on `MainWindow`. Added to
  the contract table; all `_on_*` slots now explicitly retained.
- **MAJOR (Claude/agy):** `test_gui_redesign` patches `mw.CreateConfigDialog`/
  `mw.QMessageBox` by module namespace → moving `_on_create_config` breaks the
  patch unless the test target is repointed. Resolution documented + listed as a
  required test update.
- **MAJOR (all 3):** collaborators are deeply coupled to `MainWindow` mutable
  state → they take a `MainWindow` back-ref; and (agy) to avoid C++-lifetime/
  refcycle hazards they are `QObject` subclasses parented to `MainWindow`.
- **MAJOR (all 3):** `run_controller` deps corrected (`_sidebar`/`_project`/
  `_active_dataset_id`/`_per_run_output_dir`/`set_status`/`export_figures`/Qt
  parent — NOT `result_grid`); run-lifecycle slots (`_on_run_status` etc.)
  explicitly kept on `MainWindow`; `_show_result_with_bundle`+`_central_stack`
  assigned.
- **MAJOR (codex):** spec named a nonexistent slot `_on_add_dataset` → replaced
  with the real dialog slots (`_on_load_config`/`_on_edit_config`/`_on_close_project`
  + create/open/save project).
- **MINOR/NIT:** `test_error_presenter` is a separate untouched module (not a
  `ResultPresenter` contract); `helpers.py` gains the colormap constants;
  `test_gui_jax_free` probes extended to the new submodules; LOC target revised
  to ≈430–480 (slot shims stay).

## Follow-ups

UX / responsiveness polish (the deferred half of the original SP3 description) can
be a separate effort once concrete UX goals are defined.
