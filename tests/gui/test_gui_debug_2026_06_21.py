"""Regression tests for the 2026-06-21 GUI debug-audit fixes.

Each test pins one bug found by the multi-agent (find + adversarial-verify)
audit of ``xpcsjax/gui/`` and its viz render path. Grouped by the file the fix
lives in. The viz uncertainty-slice fix (P1) lives in
``tests/viz/test_review_regressions.py`` alongside the sibling param-unpacking
regressions.
"""

import json

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import numpy as np  # noqa: E402

from xpcsjax.gui.result_loader import ResultSummary  # noqa: E402


def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _summary(tmp_path, marker: str) -> ResultSummary:
    """A minimal summary whose convergence_status carries a unique marker so the
    central panel's rendered text reveals which run is on screen."""
    return ResultSummary(
        result_dir=tmp_path,
        success=True,
        convergence_status=marker,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        quality_flag="good",
        parameters={"D0": 1234.5},
    )


# ----------------------------------------------------------------------------
# views/main_window.py
# ----------------------------------------------------------------------------
def test_finishing_active_run_does_not_clobber_selected_run(qtbot, tmp_path):
    """P2 (twin-path): while run A is active, the user clicks an earlier finished
    run C in the sidebar to inspect it. When A finishes, the queue-driven refresh
    must NOT yank the central panel away from the user's deliberate selection."""
    win = _window(qtbot)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    ds = win._project.add_dataset(str(cfg))
    run_c = win._project.add_run(ds.dataset_id)
    run_a = win._project.add_run(ds.dataset_id)
    run_c.summary = _summary(tmp_path, "VIEWEDCEE")
    run_a.summary = _summary(tmp_path, "ACTIVEAYE")

    # A is the active/running run (sets _active_run_id).
    win._queue.run_status_changed.emit(run_a.run_id, "running")
    # User clicks finished run C to inspect it.
    win._on_runs_selected([run_c.run_id])
    assert "VIEWEDCEE" in win.result_text()

    # A finishes — the panel must stay on C, not flip to A.
    win._queue.run_finished.emit(run_a.run_id, "", run_a.summary)
    assert "VIEWEDCEE" in win.result_text()
    assert "ACTIVEAYE" not in win.result_text()


def test_finishing_run_still_shown_when_nothing_else_selected(qtbot, tmp_path):
    """Guard against over-correction: with no competing selection, a finishing
    active run must still auto-populate the panel (the common single-run path)."""
    win = _window(qtbot)
    rid = "aabbccdd1234567890abcdef12345678"
    win._queue.run_status_changed.emit(rid, "running")
    win._queue.run_finished.emit(rid, "", _summary(tmp_path, "SHOWNRUN"))
    assert "SHOWNRUN" in win.result_text()


def test_fit_log_cleared_between_runs_and_on_close(qtbot):
    """P3 (stale-state): the append-only Fitting-Process log must be cleared when
    the active run changes and when the project is closed, so a new run's lines
    never interleave below a prior run's / prior project's lines."""
    win = _window(qtbot)
    run_a = "a" * 32
    run_b = "b" * 32

    win._queue.run_status_changed.emit(run_a, "running")
    win._queue.log_received.emit(run_a, "INFO", "ALPHA_LINE")
    assert "ALPHA_LINE" in win.log_text()

    # A new active run must start with a clean log.
    win._queue.run_status_changed.emit(run_b, "running")
    win._queue.log_received.emit(run_b, "INFO", "BETA_LINE")
    assert "ALPHA_LINE" not in win.log_text()
    assert "BETA_LINE" in win.log_text()

    win.close_project()
    assert win.log_text() == ""


# ----------------------------------------------------------------------------
# result_loader.py + views/inspector.py
# ----------------------------------------------------------------------------
def test_loader_coerces_null_diagnostics_to_dict(tmp_path):
    """P3: a partial/external nlsq_result.json whose nlsq_diagnostics is null must
    load as an empty dict (symmetric with metadata/parameters), never None."""
    (tmp_path / "nlsq_result.json").write_text(
        json.dumps(
            {
                "metadata": {"success": True, "nlsq_diagnostics": None},
                "parameters": {"D0": {"value": 1.0, "uncertainty": 0.1}},
            }
        ),
        encoding="utf-8",
    )
    from xpcsjax.gui.result_loader import load_result_summary

    summary = load_result_summary(tmp_path)
    assert summary is not None
    assert isinstance(summary.diagnostics, dict)
    assert summary.diagnostics == {}


def test_inspector_tolerates_non_dict_diagnostics(qtbot, tmp_path):
    """P3: the inspector must not raise when handed a non-dict diagnostics block
    (defense-in-depth for summaries built outside the loader)."""
    from xpcsjax.gui.views.inspector import InspectorDock

    dock = InspectorDock()
    qtbot.addWidget(dock)
    # ResultSummary is frozen; construct with a non-dict diagnostics directly.
    summary = ResultSummary(
        result_dir=tmp_path,
        success=True,
        convergence_status="x",
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        quality_flag="good",
        parameters={},
        diagnostics=None,  # type: ignore[arg-type]
    )
    dock.show_summary(summary)  # must not raise AttributeError


# ----------------------------------------------------------------------------
# controllers/fit_queue.py
# ----------------------------------------------------------------------------
def test_stale_non_terminal_event_for_freed_run_is_dropped(qtbot):
    """P3 (signal-wiring): a LogLine/Iteration/Banner for a run no longer in
    _handles (freed/cancelled) must NOT be re-emitted to the monitor widgets."""
    from PySide6.QtCore import QObject, Signal

    from xpcsjax.gui.controllers.fit_queue import FitQueueController
    from xpcsjax.service.events import Iteration

    class _FakeHandle(QObject):
        event = Signal(object)

        def __init__(self, job):
            super().__init__()
            self.job = job
            self._alive = False

        def start(self):
            self._alive = True

        def cancel(self):
            self._alive = False

    q = FitQueueController(handle_factory=lambda job: _FakeHandle(job))
    iters: list = []
    q.iteration_received.connect(lambda rid, n, ssr: iters.append((rid, n, ssr)))

    # No handle named "ghost" exists — a late event must be ignored.
    q._on_event("ghost", Iteration(run_id="ghost", seq=1, n=2, ssr=9.0, chi2=9.0))
    assert iters == []


# ----------------------------------------------------------------------------
# views/plots_view.py
# ----------------------------------------------------------------------------
def test_residual_map_levels_computed_from_full_resolution(qtbot):
    """P3 (twin-path): the residual heatmap color window must be computed from the
    FULL-resolution residuals, not the block-mean-decimated display array, so it
    agrees with the histogram/diagonal/scatter diagnostics fed the full surface."""
    from xpcsjax.gui.views.plots_view import ResidualMapView, _residual_levels
    from xpcsjax.gui.views.raster import rasterize

    # Checkerboard ±50 larger than max_dim: 2×2 block-mean cancels opposite signs
    # to exactly 0, so _residual_levels of the decimated array hits its 1.0 floor —
    # far below the true ±50 window. (Amplitude > the floor so the bug is visible.)
    amp = 50.0
    n = 1100
    arr = np.empty((n, n), dtype=float)
    arr[0::2, 0::2] = amp
    arr[1::2, 1::2] = amp
    arr[0::2, 1::2] = -amp
    arr[1::2, 0::2] = -amp

    view = ResidualMapView()
    qtbot.addWidget(view)
    view.show_map(arr)

    levels = tuple(view._image_item.getLevels())
    full_levels = _residual_levels(arr)
    decimated_levels = _residual_levels(rasterize(arr))

    assert levels == pytest.approx(full_levels)
    # The bug computed levels from the decimated array; those collapse to ~0.
    assert abs(decimated_levels[1]) < 0.5 * abs(full_levels[1])
