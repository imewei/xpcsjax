"""Status-bar + log presentation collaborator for MainWindow."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from xpcsjax.gui.theme import current_palette

if TYPE_CHECKING:
    from xpcsjax.gui.views.main_window import MainWindow

# Levels mapped to the active Palette's semantic tokens (danger/warning) — every
# other level (INFO, DEBUG, ...) inherits the widget's default text color, no
# entry needed. Without this every log line looked identical regardless of
# severity, so scanning a long fitting-process log for the one ERROR line had
# no visual anchor.
_LEVEL_TOKEN = {
    "ERROR": "danger",
    "CRITICAL": "danger",
    "WARNING": "warning",
}


class StatusManager(QObject):
    """Owns the set_status / append_log bodies (operates on MainWindow widgets).

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

    def set_status(self, status: str) -> None:
        """Render the current run status.

        Parameters
        ----------
        status : str
            Status string to display in the status label.
        """
        self._mw._status.setText(status)

    def append_log(self, level: str, message: str) -> None:
        """Append one forwarded log line to the tail.

        Parameters
        ----------
        level : str
            Log level string (e.g. ``"INFO"``, ``"WARNING"``).
        message : str
            The log message text.
        """
        token = _LEVEL_TOKEN.get(level.upper())
        tag = f"[{escape(level)}]"
        if token is not None:
            color = getattr(current_palette(), token)
            tag = f'<span style="color:{color}; font-weight:600;">{tag}</span>'
        self._mw._log.appendHtml(f"{tag} {escape(message)}")
