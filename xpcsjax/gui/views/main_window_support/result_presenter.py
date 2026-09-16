"""Result presentation collaborator for MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from xpcsjax.gui.theme import current_palette
from xpcsjax.gui.viz_bundle import VizBundle, load_viz_bundle
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from xpcsjax.gui.result_loader import ResultSummary
    from xpcsjax.gui.views.main_window import MainWindow


class _BundleLoadSignals(QObject):
    """Signal carrier for :class:`_BundleLoadTask`.

    A ``QRunnable`` cannot itself declare Qt signals, so the worker task owns
    one of these to report back to the main thread.
    """

    finished = Signal(object, str)  # (VizBundle | None, result_dir)


class _BundleLoadTask(QRunnable):
    """Loads a :class:`~xpcsjax.gui.viz_bundle.VizBundle` off the UI thread.

    Runs entirely in a ``QThreadPool`` worker thread: only ``load_viz_bundle``
    (file I/O + numpy array prep, no Qt) executes here. Widget construction
    (``ResultGrid.set_bundle``) must stay on the UI thread and happens in
    :meth:`ResultPresenter._on_bundle_loaded`, invoked via the queued
    ``finished`` signal once this task completes.
    """

    def __init__(self, result_dir: str) -> None:
        super().__init__()
        self._result_dir = result_dir
        self.signals = _BundleLoadSignals()

    def run(self) -> None:
        try:
            bundle = load_viz_bundle(self._result_dir)
        except Exception:  # pragma: no cover — defensive only
            logger.warning(
                "Failed to load viz bundle for %s; falling back to text summary.",
                self._result_dir,
                exc_info=True,
            )
            bundle = None
        self.signals.finished.emit(bundle, self._result_dir)


class ResultPresenter(QObject):
    """Owns the result/error/inspector presentation bodies (operates on MainWindow widgets).

    Parameters
    ----------
    main_window : MainWindow
        The owning MainWindow instance.  Passed as Qt parent so this object's
        lifetime is tied to the window.
    """

    def __init__(self, main_window: MainWindow) -> None:
        """Initialise and attach to *main_window* as Qt parent.

        Parameters
        ----------
        main_window : MainWindow
            The owning MainWindow instance.
        """
        super().__init__(main_window)
        self._mw = main_window
        # The most recently REQUESTED bundle load, used as a stale-result
        # guard: a slower load that finishes after a newer run/selection has
        # superseded it must not clobber the panel (memory: "finished-run-
        # clobber" bug class — run selection changed while loading).
        self._pending_summary: Any = None
        self._pending_result_dir: str | None = None

    def show_result(self, summary: Any) -> None:
        """Render the finished-fit summary (a ResultSummary or None) in the text panel.

        Parameters
        ----------
        summary : Any
            A ``ResultSummary`` (or ``None``) to render as plain text.
        """
        # A synchronous render supersedes any bundle still loading for an
        # earlier selection; otherwise its late arrival would replace this panel.
        self._pending_result_dir = None
        if summary is None:
            self._mw._results.setPlainText("Fit finished, but no result file was found.")
            return
        lines = [
            f"status:          {summary.convergence_status}",
            f"success:         {summary.success}",
            f"chi^2:           {summary.chi_squared}",
            f"reduced chi^2:   {summary.reduced_chi_squared}",
            f"quality:         {summary.quality_flag}",
            f"results dir:     {summary.result_dir}",
            "",
            "parameters:",
            # "NaN" (not the literal "None") for consistency with inspector.py's
            # and project_panel.py's rendering of a non-finite parameter.
            *[
                f"  {name} = {'NaN' if value is None else value}"
                for name, value in summary.parameters.items()
            ],
            "",
            f"Publication figures (Matplotlib) were written under {summary.result_dir}/plots.",
        ]
        self._mw._results.setPlainText("\n".join(lines))

    def show_result_with_bundle(self, summary: Any, result_dir: str | None) -> None:
        """Render the result: per-phi grid when a bundle exists, text otherwise.

        The bundle (file I/O + numpy prep, potentially 100s of MB, see
        io/json_utils.py) is loaded off the UI thread via ``QThreadPool``;
        this method returns immediately and the panel updates asynchronously
        once the load completes (see :meth:`_on_bundle_loaded`).

        Parameters
        ----------
        summary : Any
            A ``ResultSummary`` (or ``None``) to show in the text fallback.
        result_dir : str | None
            The run's result directory; used to locate the viz bundle.
            ``None`` forces the text-summary path (no background load).
        """
        self._pending_summary = summary
        self._pending_result_dir = result_dir
        if not result_dir:
            self._mw.show_result(summary)
            self._mw._central_stack.setCurrentIndex(0)
            return

        task = _BundleLoadTask(result_dir)
        task.signals.finished.connect(self._on_bundle_loaded)
        QThreadPool.globalInstance().start(task)

    def _on_bundle_loaded(self, bundle: VizBundle | None, result_dir: str) -> None:
        """Apply a background-loaded bundle to the UI (main-thread slot).

        Discards the result if a newer ``show_result_with_bundle`` call (a
        different run finishing, or the user selecting a different run) has
        superseded this one while it was loading.
        """
        if result_dir != self._pending_result_dir:
            return
        summary = self._pending_summary

        if bundle is not None:
            self._mw._result_grid.set_bundle(bundle)
            if self._mw._result_grid.section_count() > 0:
                self._mw._central_stack.setCurrentIndex(1)  # show per-phi grid
                return
            # set_bundle degraded a malformed exp_c2 (bad shape) to an empty
            # grid — fall through to the text summary below rather than
            # showing a blank per-phi page.

        # Fall back to (or keep) the text summary.
        # NOTE: calls self._mw.show_result (the MainWindow shim) deliberately —
        # NOT self.show_result() directly — so future overrides on MainWindow are
        # respected and to preserve the indirection contract.
        self._mw.show_result(summary)
        self._mw._central_stack.setCurrentIndex(0)

    def show_error(self, message: str) -> None:
        """Render a fit failure in the text panel, with a color-coded header.

        Parameters
        ----------
        message : str
            The error message text to display.
        """
        # A colored "FIT FAILED" header is a secondary signal (the status pill
        # and the modal show_failure() dialog already carry the primary one), but plain
        # text gave a scanning eye zero anchor between this and a normal result.
        # Invalidate any in-flight bundle load: a prior run's successful load
        # completing after this must not replace the failure panel.
        self._pending_result_dir = None
        color = current_palette().danger
        self._mw._results.clear()
        self._mw._results.appendHtml(f'<b style="color:{color};">FIT FAILED</b>')
        self._mw._results.appendPlainText("")
        self._mw._results.appendPlainText(message)

    def show_inspector(self, summary: ResultSummary | None) -> None:
        """Populate the inspector dock with *summary* (or clear on None).

        Parameters
        ----------
        summary : ResultSummary | None
            A :class:`~xpcsjax.gui.result_loader.ResultSummary` or ``None``.
        """
        self._mw._inspector.show_summary(summary)
