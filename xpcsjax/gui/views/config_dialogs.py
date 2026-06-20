"""File-menu config dialogs: Create Config (from templates) and Edit Config (text).

JAX-free, Qt-only views. ``CreateConfigDialog`` collects the inputs for
:func:`xpcsjax.cli.config_generator.generate_config` (the same function the
``xpcsjax-config`` console script uses) so the GUI and CLI stay behaviorally
identical. ``ConfigTextEditorDialog`` is a minimal in-app text editor that loads
a YAML config, lets the user edit it, and saves it back to disk.

Neither dialog imports JAX (nor ``generate_config``); the MainWindow performs the
actual generation via a deferred import, keeping the GUI process JAX-free.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# The four production analysis modes, mirroring config_generator._MODE_TO_TEMPLATE.
# Duplicated as a literal (not imported) so this view stays import-light and
# JAX-free; config_generator validates the mode again at generation time.
ANALYSIS_MODES: tuple[str, ...] = (
    "static_anisotropic",
    "static_isotropic",
    "laminar_flow",
    "two_component",
)


def _optional_float(text: str) -> float | None:
    """Parse a stripped line-edit value to ``float``, or ``None`` when blank/invalid."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_int(text: str) -> int | None:
    """Parse a stripped line-edit value to ``int``, or ``None`` when blank/invalid."""
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


class CreateConfigDialog(QDialog):
    """Collect mode + optional data/scattering/timing for a new config from a template.

    The dialog only gathers inputs; the caller passes them to
    :func:`xpcsjax.cli.config_generator.generate_config`. All numeric fields are
    optional — left blank, they leave the template's canonical placeholder values
    untouched (matching ``xpcsjax-config`` with no ``--q``/``--dt``/... flags).

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    default_dir : str or Path or None
        Directory pre-filled into the output-path field (the project working
        directory when one has been created).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Config")
        self.setObjectName("dialog_create_config")

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("create_config_mode")
        self._mode_combo.addItems(ANALYSIS_MODES)

        default_out = ""
        if default_dir is not None:
            default_out = str(Path(default_dir) / "xpcsjax_config.yaml")
        self._output_edit = QLineEdit(default_out)
        self._output_edit.setObjectName("create_config_output")
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._choose_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_edit)
        out_row.addWidget(browse_out)
        out_row_w = QWidget()
        out_row_w.setLayout(out_row)

        self._data_edit = QLineEdit()
        self._data_edit.setObjectName("create_config_data")
        browse_data = QPushButton("Browse…")
        browse_data.clicked.connect(self._choose_data)
        data_row = QHBoxLayout()
        data_row.addWidget(self._data_edit)
        data_row.addWidget(browse_data)
        data_row_w = QWidget()
        data_row_w.setLayout(data_row)

        self._q_edit = QLineEdit()
        self._dt_edit = QLineEdit()
        self._time_edit = QLineEdit()

        form = QFormLayout()
        form.addRow("Analysis mode", self._mode_combo)
        form.addRow("Output config", out_row_w)
        form.addRow("Data file (optional)", data_row_w)
        form.addRow("Wavevector q (optional)", self._q_edit)
        form.addRow("dt seconds (optional)", self._dt_edit)
        form.addRow("Frames / end_frame (optional)", self._time_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Copy a mode-specific template and (optionally) inject values:"))
        layout.addLayout(form)
        layout.addWidget(buttons)

    # --- input collection (testable without showing the dialog) ---------------
    def selected_mode(self) -> str:
        """Return the chosen analysis mode."""
        return self._mode_combo.currentText()

    def output_path(self) -> str:
        """Return the destination config path (may be empty if unset)."""
        return self._output_edit.text().strip()

    def generation_kwargs(self) -> dict[str, object]:
        """Return kwargs for ``generate_config`` (omitting blank optional fields)."""
        kwargs: dict[str, object] = {}
        data = self._data_edit.text().strip()
        if data:
            kwargs["data_path"] = data
        q = _optional_float(self._q_edit.text())
        if q is not None:
            kwargs["q"] = q
        dt = _optional_float(self._dt_edit.text())
        if dt is not None:
            kwargs["dt"] = dt
        time_length = _optional_int(self._time_edit.text())
        if time_length is not None:
            kwargs["time_length"] = time_length
        return kwargs

    # --- browse slots ---------------------------------------------------------
    def _choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Config output path", self._output_edit.text(), "YAML configs (*.yaml *.yml)"
        )
        if path:
            self._output_edit.setText(path)

    def _choose_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Data file", "", "All files (*)")
        if path:
            self._data_edit.setText(path)


class ConfigTextEditorDialog(QDialog):
    """A minimal in-app text editor for a YAML config file.

    Loads *path* into a monospace ``QPlainTextEdit`` on construction; the Save
    button writes the current text back to *path* and accepts the dialog. This is
    the raw-text replacement for the removed form-based config editor.

    Parameters
    ----------
    path : str or Path
        The config file to edit.
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self.setWindowTitle(f"Edit Config — {self._path.name}")
        self.setObjectName("dialog_edit_config")
        self.resize(680, 560)

        self._editor = QPlainTextEdit()
        self._editor.setObjectName("edit_config_text")
        # Fixed-pitch font so YAML indentation reads correctly.
        self._editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._load_error: str | None = None
        self.load()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(str(self._path)))
        layout.addWidget(self._editor)
        layout.addWidget(buttons)

    # --- text access (testable) -----------------------------------------------
    def text(self) -> str:
        """Return the current editor text."""
        return self._editor.toPlainText()

    def set_text(self, text: str) -> None:
        """Replace the editor text (test/inspection helper)."""
        self._editor.setPlainText(text)

    def load_error(self) -> str | None:
        """Return the read error message, or ``None`` if the file loaded cleanly."""
        return self._load_error

    def load(self) -> None:
        """Read *path* into the editor; record a recoverable error on failure."""
        try:
            self._editor.setPlainText(self._path.read_text(encoding="utf-8"))
            self._load_error = None
        except OSError as exc:
            self._load_error = str(exc)
            self._editor.setPlainText("")

    def save(self) -> None:
        """Write the editor text back to *path* (parent dirs created if needed)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._editor.toPlainText(), encoding="utf-8")

    # --- slot ------------------------------------------------------------------
    def _on_save(self) -> None:
        self.save()
        self.accept()
