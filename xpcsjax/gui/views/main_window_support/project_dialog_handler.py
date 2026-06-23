"""Project and config dialog slot collaborator for MainWindow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog, QMessageBox

from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog, CreateConfigDialog

if TYPE_CHECKING:
    from xpcsjax.gui.views.main_window import MainWindow


class ProjectDialogHandler(QObject):
    """Owns the 7 project/config dialog slot bodies (operates on MainWindow state).

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

    def on_create_project(self) -> None:
        """Open a directory chooser and create / switch to the chosen project directory.

        Delegates to ``MainWindow.create_project`` after the user selects a
        directory.
        """
        path = QFileDialog.getExistingDirectory(self._mw, "Create / choose project directory")
        if path:
            self._mw.create_project(path)

    def on_create_config(self) -> None:
        """Show the Create Config dialog and write the generated YAML template.

        Handles the overwrite-confirmation prompt and guards the overwrite
        retry against write errors (permission denied, disk full, etc.).
        """
        dialog = CreateConfigDialog(
            self._mw, default_dir=self._mw._project_dir or self._mw._output_dir
        )
        if dialog.exec() != int(CreateConfigDialog.DialogCode.Accepted):
            return
        output_path = dialog.output_path()
        if not output_path:
            QMessageBox.information(self._mw, "Create Config", "No output path was given.")
            return
        mode = dialog.selected_mode()
        try:
            kwargs = dialog.generation_kwargs()
        except ValueError as exc:
            QMessageBox.warning(self._mw, "Create Config", f"Invalid input:\n{exc}")
            return
        try:
            self._mw.create_config(mode, output_path, overwrite=False, **kwargs)
        except FileExistsError:
            resp = QMessageBox.question(
                self._mw,
                "Create Config",
                f"{output_path} exists. Overwrite?",
            )
            if resp == QMessageBox.StandardButton.Yes:
                # The overwrite retry can itself fail (permission denied, disk
                # full, read-only FS) — guard it so the error surfaces as a
                # warning instead of escaping the slot through the event loop.
                try:
                    self._mw.create_config(mode, output_path, overwrite=True, **kwargs)
                except (ValueError, FileNotFoundError, OSError) as exc:
                    QMessageBox.warning(
                        self._mw, "Create Config", f"Could not create config:\n{exc}"
                    )
        except (ValueError, FileNotFoundError, OSError) as exc:
            # OSError covers write failures on the initial create (FileExistsError,
            # an OSError subclass, is caught above first so its overwrite prompt
            # still runs).
            QMessageBox.warning(self._mw, "Create Config", f"Could not create config:\n{exc}")

    def on_edit_config(self) -> None:
        """Open a file-chooser and launch the YAML text-editor dialog.

        The start directory defaults to the active project directory when one
        is set.
        """
        start_dir = str(self._mw._project_dir) if self._mw._project_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            self._mw, "Edit config", start_dir, "YAML configs (*.yaml *.yml)"
        )
        if path:
            ConfigTextEditorDialog(path, self._mw).exec()

    def on_load_config(self) -> None:
        """Open a file-chooser and add the selected config as a new dataset.

        The start directory defaults to the active project directory when one
        is set.
        """
        start_dir = str(self._mw._project_dir) if self._mw._project_dir else ""
        path, _ = QFileDialog.getOpenFileName(
            self._mw, "Load config", start_dir, "YAML configs (*.yaml *.yml)"
        )
        if path:
            self._mw.add_dataset(path)

    def on_save_project(self) -> None:
        """Open a save-file dialog and persist the current project to disk."""
        path, _ = QFileDialog.getSaveFileName(
            self._mw,
            "Save Project",
            "",
            "xpcsjax project (*.xpcsproj);;All files (*)",
        )
        if path:
            self._mw.save_project_to(path)

    def on_open_project(self) -> None:
        """Open a file-chooser and load a previously saved project file."""
        path, _ = QFileDialog.getOpenFileName(
            self._mw,
            "Open Project",
            "",
            "xpcsjax project (*.xpcsproj);;All files (*)",
        )
        if path:
            self._mw.open_project_from(path)

    def on_close_project(self) -> None:
        """Prompt for confirmation and close the current project."""
        resp = QMessageBox.question(
            self._mw,
            "Close Project",
            "Close the current project? Unsaved results will be cleared.",
        )
        if resp == QMessageBox.StandardButton.Yes:
            self._mw.close_project()
