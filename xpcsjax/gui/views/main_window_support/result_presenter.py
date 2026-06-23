"""Result presentation collaborator for MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject

from xpcsjax.gui.viz_bundle import load_viz_bundle

if TYPE_CHECKING:
    from xpcsjax.gui.result_loader import ResultSummary
    from xpcsjax.gui.views.main_window import MainWindow


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

    def show_result(self, summary: Any) -> None:
        """Render the finished-fit summary (a ResultSummary or None) in the text panel.

        Parameters
        ----------
        summary : Any
            A ``ResultSummary`` (or ``None``) to render as plain text.
        """
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
            *[f"  {name} = {value}" for name, value in summary.parameters.items()],
            "",
            f"Publication figures (Matplotlib) were written under {summary.result_dir}/plots.",
        ]
        self._mw._results.setPlainText("\n".join(lines))

    def show_result_with_bundle(self, summary: Any, result_dir: str | None) -> None:
        """Render the result: per-phi grid when a bundle exists, text otherwise.

        Parameters
        ----------
        summary : Any
            A ``ResultSummary`` (or ``None``) to show in the text fallback.
        result_dir : str | None
            The run's result directory; used to locate the viz bundle.
            ``None`` forces the text-summary path.
        """
        bundle = None
        if result_dir:
            try:
                bundle = load_viz_bundle(result_dir)
            except Exception:  # pragma: no cover — defensive only
                bundle = None

        if bundle is not None:
            self._mw._result_grid.set_bundle(bundle)
            self._mw._central_stack.setCurrentIndex(1)  # show per-phi grid
        else:
            # Fall back to (or keep) the text summary.
            # NOTE: calls self._mw.show_result (the MainWindow shim) deliberately —
            # NOT self.show_result() directly — so future overrides on MainWindow are
            # respected and to preserve the indirection contract.
            self._mw.show_result(summary)
            self._mw._central_stack.setCurrentIndex(0)

    def show_error(self, message: str) -> None:
        """Render a fit failure in the text panel.

        Parameters
        ----------
        message : str
            The error message text to display.
        """
        self._mw._results.setPlainText(f"FIT FAILED\n\n{message}")

    def show_inspector(self, summary: ResultSummary | None) -> None:
        """Populate the inspector dock with *summary* (or clear on None).

        Parameters
        ----------
        summary : ResultSummary | None
            A :class:`~xpcsjax.gui.result_loader.ResultSummary` or ``None``.
        """
        self._mw._inspector.show_summary(summary)
