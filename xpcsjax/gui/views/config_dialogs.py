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

import math
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

# The four production analysis modes, mirroring config_template._MODE_TO_TEMPLATE.
# Duplicated as a literal (not imported) so this view stays import-light and
# JAX-free; config_template validates the mode again at generation time.
ANALYSIS_MODES: tuple[str, ...] = (
    "static_anisotropic",
    "static_isotropic",
    "laminar_flow",
    "two_component",
)


def _parse_optional(text: str, caster: type, label: str) -> float | int | None:
    """Parse a stripped line-edit value, returning ``None`` only when BLANK.

    A blank field means "leave the template default" (the intended optional
    behavior). A NON-blank value that fails to cast raises ``ValueError`` naming
    the field — so a typo is surfaced to the user rather than silently discarded
    and written as the template placeholder (the project's no-silent-data-loss
    rule applies to user input too). ``float("nan")``/``float("inf")`` cast
    without error, so a non-finite result is rejected explicitly — otherwise it
    would be serialized (``.nan``/``.inf``) into the generated config and feed a
    NaN/Inf wavevector or dt into the fit (the reject-non-finite rule).
    """
    text = text.strip()
    if not text:
        return None
    try:
        result = caster(text)
    except ValueError:
        raise ValueError(f"{label}: {text!r} is not a valid {caster.__name__}.") from None
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError(f"{label}: {text!r} is not a finite {caster.__name__}.")
    return result


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
        """Return kwargs for ``generate_config`` (omitting blank optional fields).

        Raises
        ------
        ValueError
            If a non-blank q/dt/frames field does not parse — surfaced to the
            user by the caller rather than silently dropped.
        """
        kwargs: dict[str, object] = {}
        data = self._data_edit.text().strip()
        if data:
            kwargs["data_path"] = data
        q = _parse_optional(self._q_edit.text(), float, "Wavevector q")
        if q is not None:
            kwargs["q"] = q
        dt = _parse_optional(self._dt_edit.text(), float, "dt")
        if dt is not None:
            kwargs["dt"] = dt
        time_length = _parse_optional(self._time_edit.text(), int, "Frames / end_frame")
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

        self._error_label = QLabel()
        self._error_label.setObjectName("edit_config_error")
        self._error_label.setStyleSheet("color: #e06c75;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        self._save_btn.clicked.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(str(self._path)))
        layout.addWidget(self._error_label)
        layout.addWidget(self._editor)
        layout.addWidget(buttons)

        # Load AFTER the widgets exist so a read failure can disable Save and show
        # the error — never let a blank editor silently overwrite the real file.
        self.load()

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
        """Read *path* into the editor; record a recoverable error on failure.

        On a read failure the editor is left empty, Save is DISABLED, and the
        error is shown — so a failed read can never be silently saved back as an
        empty file, truncating the user's config.
        """
        try:
            self._editor.setPlainText(self._path.read_text(encoding="utf-8"))
            self._load_error = None
        except OSError as exc:
            self._load_error = str(exc)
            self._editor.setPlainText("")
        if self._load_error is None:
            self._error_label.hide()
            self._save_btn.setEnabled(True)
        else:
            self._error_label.setText(f"Could not read file — Save disabled:\n{self._load_error}")
            self._error_label.show()
            self._save_btn.setEnabled(False)

    def save(self) -> None:
        """Write the editor text back to *path* (parent dirs created if needed).

        A no-op when the file failed to load, so a blank editor never truncates
        the on-disk config.
        """
        if self._load_error is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._editor.toPlainText(), encoding="utf-8")

    # --- slot ------------------------------------------------------------------
    def _on_save(self) -> None:
        if self._load_error is not None:
            return  # guarded: never overwrite when the load failed
        try:
            self.save()
        except OSError as exc:
            # Mirror the read-path treatment: surface the failure and keep the
            # dialog open with the user's edits intact, rather than letting the
            # OSError escape the slot silently (the no-silent-loss contract
            # applies to the write the user actually cares about, not just read).
            self._error_label.setText(f"Could not save file:\n{exc}")
            self._error_label.show()
            return
        self.accept()
