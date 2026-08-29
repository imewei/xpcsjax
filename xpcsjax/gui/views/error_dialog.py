"""Qt error dialog with a collapsible "Show details" pane for raw failure text."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


def show_failure(
    parent: QWidget | None,
    title: str,
    friendly: str,
    details: str,
) -> None:
    """Show a modal error dialog with a collapsible details pane.

    Parameters
    ----------
    parent:
        The parent widget (used for dialog centering).
    title:
        Text for the dialog window title bar.
    friendly:
        The user-facing message shown prominently in the dialog body.
    details:
        The raw failure text placed behind Qt's built-in "Show details"
        collapsible pane.
    """
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(friendly)
    box.setDetailedText(details)
    # Critical, not Warning — a fit run outright failed, not a soft caution.
    box.setIcon(QMessageBox.Icon.Critical)
    box.exec()
