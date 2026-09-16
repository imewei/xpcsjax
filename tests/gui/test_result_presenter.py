"""Unit tests for ResultPresenter collaborator (show_result / show_error / show_inspector / show_result_with_bundle)."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.result_loader import ResultSummary
from xpcsjax.gui.views.main_window import MainWindow
from xpcsjax.gui.views.main_window_support.result_presenter import (
    ResultPresenter,
)


def _window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _summary(tmp_path, label: str = "converged") -> ResultSummary:
    return ResultSummary(
        result_dir=tmp_path,
        success=True,
        convergence_status=label,
        chi_squared=1.0,
        reduced_chi_squared=0.9,
        quality_flag="good",
        parameters={"D0": 42.0},
    )


# --- wiring / identity tests -------------------------------------------------


def test_result_presenter_is_qobject_parented_to_window(qtbot):
    """ResultPresenter is a QObject child of MainWindow."""
    win = _window(qtbot)
    assert isinstance(win._result_presenter, ResultPresenter)
    assert win._result_presenter.parent() is win


# --- show_result: text-only, does NOT switch to grid page --------------------


def test_show_result_text_only(qtbot, tmp_path):
    """show_result renders text and keeps the stack on page 0 (text-summary page)."""
    win = _window(qtbot)
    summary = _summary(tmp_path, "converged")
    win.show_result(summary)
    assert "converged" in win.result_text()
    assert win._central_stack.currentIndex() == 0


def test_show_result_contains_summary_fields(qtbot, tmp_path):
    """show_result writes chi^2, quality, and parameters into the text panel."""
    win = _window(qtbot)
    summary = _summary(tmp_path)
    win.show_result(summary)
    text = win.result_text()
    assert "42.0" in text or "D0" in text
    assert "good" in text


def test_show_result_none_summary(qtbot, tmp_path):
    """show_result with None writes the 'no result file' message."""
    win = _window(qtbot)
    win.show_result(None)
    assert "no result file" in win.result_text()
    assert win._central_stack.currentIndex() == 0


# --- show_error: text-only with FIT FAILED prefix ----------------------------


def test_show_error_renders_fit_failed(qtbot, tmp_path):
    """show_error writes 'FIT FAILED' and the message to the text panel."""
    win = _window(qtbot)
    win.show_error("boom")
    assert "FIT FAILED" in win.result_text()
    assert "boom" in win.result_text()


def test_show_error_does_not_switch_to_grid(qtbot, tmp_path):
    """show_error is text-only — stack stays on page 0."""
    win = _window(qtbot)
    win.show_error("some error")
    assert win._central_stack.currentIndex() == 0


# --- _show_result_with_bundle: grid routing via real bundle ------------------


def _write_bundle(tmp_path) -> None:
    """Write the artifact that load_viz_bundle reads: <result_dir>/plots/simulated_data/c2_fitted_data.npz."""
    import numpy as np

    sim = tmp_path / "plots" / "simulated_data"
    sim.mkdir(parents=True)
    exp = np.ones((2, 10, 10))
    model = np.ones((2, 10, 10))
    np.savez(
        sim / "c2_fitted_data.npz",
        c2_exp=exp,
        c2_fitted=model,
        residuals=exp - model,
        t1=np.arange(10.0),
        t2=np.arange(10.0),
        phi_angles=np.array([0.0, 45.0]),
    )


def test_show_result_with_bundle_uses_grid_when_bundle_found(qtbot, tmp_path):
    """_show_result_with_bundle switches to the grid page (index 1) when a bundle loads.

    The bundle load runs on a QThreadPool worker thread (D10 fix) and applies
    itself back to the UI via a queued signal, so this must wait for the
    event loop to process it rather than asserting immediately.
    """
    _write_bundle(tmp_path)

    win = _window(qtbot)
    summary = _summary(tmp_path)
    win._show_result_with_bundle(summary, str(tmp_path))
    qtbot.waitUntil(lambda: win._central_stack.currentIndex() == 1, timeout=5000)


def test_show_result_with_bundle_falls_back_to_text_when_no_bundle(qtbot, tmp_path):
    """_show_result_with_bundle falls back to text (page 0) when result_dir has no bundle."""
    # tmp_path exists but contains no bundle file — load_viz_bundle raises.
    win = _window(qtbot)
    summary = _summary(tmp_path, "no_bundle_converged")
    win._show_result_with_bundle(summary, str(tmp_path))
    # Falls back to text page (async load off the UI thread; wait for it).
    qtbot.waitUntil(lambda: "no_bundle_converged" in win.result_text(), timeout=5000)
    assert win._central_stack.currentIndex() == 0


def test_show_result_with_bundle_discards_stale_load(qtbot, tmp_path, monkeypatch):
    """A slower load for an earlier selection must not clobber a newer one.

    Regression for the "finished-run-clobber" bug class: if run A's bundle
    load is still in flight when the user (or a finishing run) switches the
    panel to run B, A's load completing later must be a no-op rather than
    overwriting B's already-applied result.

    Made deterministic (D10 review finding): the stale load used to just run
    on the QThreadPool and race the second ``_show_result_with_bundle`` call
    -- if the pool finished the stale load first, the discard guard
    (``result_dir != self._pending_result_dir``) was never exercised and the
    test passed vacuously (verified: with the guard defeated it failed 5/5
    at the time of writing, but only because the pool happened to be slower
    than the second call). Block the stale load on a ``threading.Event``
    until after the fresh call has completed, so the guard is always
    actually exercised regardless of scheduling speed.
    """
    import threading

    from xpcsjax.gui.views.main_window_support import result_presenter as rp_module

    stale_dir = tmp_path / "stale"
    fresh_dir = tmp_path / "fresh"
    _write_bundle(stale_dir)
    # fresh_dir has no bundle -> falls back to its own text summary.

    win = _window(qtbot)
    stale_summary = _summary(stale_dir, "stale_run")
    fresh_summary = _summary(fresh_dir, "fresh_run")

    release = threading.Event()
    loaded_stale = threading.Event()
    real_load_viz_bundle = rp_module.load_viz_bundle

    def _blocking_load(result_dir):
        is_stale = str(result_dir) == str(stale_dir)
        if is_stale:
            release.wait(timeout=5.0)
        result = real_load_viz_bundle(result_dir)
        if is_stale:
            loaded_stale.set()
        return result

    monkeypatch.setattr(rp_module, "load_viz_bundle", _blocking_load)

    # Simulate: stale load was requested (pending_result_dir set)...
    win._show_result_with_bundle(stale_summary, str(stale_dir))
    # ...then immediately superseded by a newer selection before it completes.
    win._show_result_with_bundle(fresh_summary, str(fresh_dir))

    qtbot.waitUntil(lambda: "fresh_run" in win.result_text(), timeout=5000)
    # Now let the stale load finish and hit the discard guard.
    release.set()
    qtbot.waitUntil(lambda: loaded_stale.is_set(), timeout=5000)
    # One more event-loop turn for the stale load's queued `finished` signal
    # to actually be delivered/processed on the main thread.
    qtbot.wait(50)

    # The stale grid must never have been applied over the fresh text result.
    assert win._central_stack.currentIndex() == 0
    assert "stale_run" not in win.result_text()


def test_show_result_with_bundle_none_result_dir_falls_back(qtbot, tmp_path):
    """_show_result_with_bundle with result_dir=None forces the text fallback."""
    win = _window(qtbot)
    summary = _summary(tmp_path, "null_dir")
    win._show_result_with_bundle(summary, None)
    assert win._central_stack.currentIndex() == 0
    assert "null_dir" in win.result_text()


# --- show_inspector: delegates to the inspector dock -------------------------


def test_show_inspector_populates_inspector(qtbot, tmp_path):
    """show_inspector passes the summary to the inspector dock without error."""
    win = _window(qtbot)
    summary = _summary(tmp_path)
    # Should not raise; inspector receives the summary.
    win.show_inspector(summary)


def test_show_inspector_none_clears(qtbot):
    """show_inspector(None) clears the inspector without error."""
    win = _window(qtbot)
    win.show_inspector(None)
