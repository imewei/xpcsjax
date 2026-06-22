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
| `plots_view.PhiResultsGrid` | `main_window.py:42`; `test_gui_redesign.py:20`; `test_plots_view.py` | Re-exported from `plots_view.py` (facade) |
| `plots_view.{TwoTimeMapView, ResidualMapView, ResidualHistogramView, DiagonalResidualView, ResidualsVsFittedView}` | `test_plots_view.py`; `test_gui_debug_fixes.py:32`; `test_gui_debug_2026_06_21.py:189` | Re-exported from `plots_view.py` |
| `plots_view.{_SCATTER_MAX_POINTS, _c2_levels, _residual_levels, _time_rect}` | `test_plots_view.py:14`; `test_gui_debug_2026_06_21.py:189` (`_residual_levels`) | Re-exported from `plots_view.py` (test-imported "private" names) |

## Components

### 1. `main_window.py` (728) → `MainWindow` facade + 4 collaborators

`MainWindow` remains the public `QMainWindow` and **keeps**: UI construction, the
signal/slot wiring block, the run-lifecycle event handlers, the public
project-sync API (the methods/properties in the contract table), `closeEvent`,
and the module-level `_expand_path`. Logic for four domains moves into
collaborator objects under a new `xpcsjax/gui/views/main_window_support/` package
(keeps the views/ directory uncluttered). Each collaborator is constructed by
`MainWindow` with exactly the widgets/refs it needs; `MainWindow`'s public
methods become thin delegates, so signal wiring and behavior are preserved.

- **`status_manager.py`** — `StatusManager(status_widget, log_widget)` with the
  bodies of `set_status` / `append_log`. `MainWindow.set_status`/`.append_log`
  delegate; the `status_text`/`log_text` properties stay on `MainWindow` (they
  read the widgets it still owns).
- **`result_presenter.py`** — `ResultPresenter(result_grid, inspector, results_pane, …)`
  with the bodies of `show_result` / `show_error` / `show_inspector`.
  `MainWindow.show_result`/`.show_error`/`.show_inspector` delegate. (Note: a
  `test_error_presenter.py` already exists — confirm its expectations are met by
  the extracted presenter.)
- **`project_dialog_handler.py`** — the dialog slot handlers (`_on_create_project`,
  `_on_open_project`, `_on_add_dataset`, `_on_create_config`, `_on_save_project`,
  …) which call `MainWindow`'s public project API. Signal connections in
  `MainWindow` point at the handler's methods.
- **`run_controller.py`** — `_on_run` / `_on_cancel` / `_on_export_figure`,
  given the fit queue + result_grid.

Target: `MainWindow` ≈ 375 LOC.

**Behavior-preservation rule:** these are *verbatim* logic moves into collaborator
methods. The only added code is the wiring (instantiate collaborator, point the
delegate/signal at it). No control-flow or Qt-signal-semantics changes. If a slot
was connected to `self._on_run`, after the move it connects to
`self._run_controller.on_run` (or `MainWindow._on_run` stays as a thin delegate —
implementer picks the lower-risk form per slot, documented in the plan).

### 2. `plots_view.py` (593) → `plots/` subpackage + re-export facade

New `xpcsjax/gui/views/plots/` package (verbatim moves, intra-package imports):
- `helpers.py` — `_leading_dim_matches`, `_apply_colormap`, `_time_rect`,
  `_c2_levels`, `_residual_levels`, and the `_SCATTER_MAX_POINTS` constant.
- `squares.py` — `_SquareBase`, `_SquareAspectMixin`, `_fit_square_view`.
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
- `test_gui_jax_free` must still pass (collaborators add no JAX import).
- Before moving any symbol, confirm its importers are only those in the contract
  table (grep); the facades cover the listed ones.
- `make verify` + `mypy xpcsjax` (hard CI gate) green before merge.

## Acceptance criteria

1. `main_window.py` ≈ 375 LOC; `main_window_support/{status_manager,result_presenter,project_dialog_handler,run_controller}.py` created; `MainWindow` + `_expand_path` + all listed public methods/properties unchanged and importable.
2. `plots_view.py` is a re-export facade; `plots/{helpers,squares,maps,residuals,grid}.py` created; all 10 contract-table names still importable from `plots_view`.
3. GUI process remains JAX-free (`test_gui_jax_free` passes).
4. Every pre-existing GUI test passes after import-path updates; new collaborator unit tests added.
5. `make verify` + `mypy xpcsjax` green.

## Follow-ups

UX / responsiveness polish (the deferred half of the original SP3 description) can
be a separate effort once concrete UX goals are defined.
