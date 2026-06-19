"""Config-editor widget: form editor + raw-YAML toggle + live validation.

Logic-free view — all numeric/validation logic delegated to
:func:`xpcsjax.service.config.validate_config`. JAX-free: no xpcsjax.core or
optimization imports.
"""

from __future__ import annotations

import copy

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.config.parameter_registry import ParameterRegistry
from xpcsjax.service.config import (
    ValidationReport,
    available_modes,
    template_dict,
    validate_config,
)


class ConfigEditor(QWidget):
    """Form editor for an xpcsjax config dict.

    Provides:

    - A mode :class:`~PySide6.QtWidgets.QComboBox` populated from
      :func:`~xpcsjax.service.config.available_modes`.
    - One :class:`~PySide6.QtWidgets.QLineEdit` row per
      ``initial_parameters.parameter_names`` entry (seeded from template
      ``values``; bounds shown as tooltip only — never clamped).
    - A "Raw YAML" :class:`~PySide6.QtWidgets.QCheckBox` that swaps to a
      :class:`~PySide6.QtWidgets.QPlainTextEdit` and round-trips via
      ``yaml.safe_dump`` / ``yaml.safe_load``.
    - A "Validate" button and status :class:`~PySide6.QtWidgets.QLabel`.
    - :attr:`config_ready` signal emitted when validation passes.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    config_ready: Signal = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._registry = ParameterRegistry()
        # The full template dict for the current mode (kept intact so
        # current_config() can return a complete schema, not just the edited
        # subset).
        self._template: dict = {}
        # Ordered parameter names from initial_parameters.parameter_names.
        self._param_names: list[str] = []
        # name -> QLineEdit mapping (rebuilt on each set_mode call).
        self._fields: dict[str, QLineEdit] = {}

        self._build_ui()
        # Load the initial mode's template so a freshly-opened editor shows a
        # populated form. Without this, ``_template`` stays ``{}`` and a first
        # "Validate" would emit a config with empty parameter_names/values
        # (validate_config passes it) — launching a worker with no parameters.
        self.set_mode(self._mode_combo.currentText())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ---- mode selector ----
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo: QComboBox = QComboBox()
        for m in available_modes():
            self._mode_combo.addItem(m)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        root.addLayout(mode_row)

        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # ---- stacked widget: form pane (0) / raw YAML pane (1) ----
        self._stack = QStackedWidget()

        # Pane 0 — form
        form_pane = QWidget()
        form_outer = QVBoxLayout(form_pane)
        form_outer.setContentsMargins(0, 0, 0, 0)
        self._form_layout = QFormLayout()
        form_outer.addLayout(self._form_layout)
        form_outer.addStretch()
        self._stack.addWidget(form_pane)

        # Pane 1 — raw YAML
        self._raw_edit = QPlainTextEdit()
        self._raw_edit.setPlaceholderText("YAML source")
        self._stack.addWidget(self._raw_edit)

        root.addWidget(self._stack)

        # ---- toolbar row: raw toggle + validate button + status label ----
        ctrl_row = QHBoxLayout()
        self._raw_check = QCheckBox("Raw YAML")
        self._raw_check.toggled.connect(self.toggle_raw)
        ctrl_row.addWidget(self._raw_check)

        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self.validate)
        ctrl_row.addWidget(validate_btn)

        self._status_label = QLabel("")
        ctrl_row.addWidget(self._status_label)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_form(self) -> None:
        """Clear and repopulate the QFormLayout from ``self._template``."""
        # Remove all existing rows
        while self._form_layout.rowCount():
            self._form_layout.removeRow(0)
        self._fields.clear()

        ip = self._template.get("initial_parameters")
        if not isinstance(ip, dict):  # malformed raw YAML (e.g. a scalar) → treat as empty
            ip = {}
        names: list[str] = list(ip.get("parameter_names") or [])
        values = ip.get("values") or []
        self._param_names = names

        for i, name in enumerate(names):
            field = QLineEdit()
            # Seed from template values when present
            if isinstance(values, list) and i < len(values):
                field.setText(str(values[i]))
            # Bounds as tooltip only — never setRange/clamp
            try:
                lo, hi = self._registry.get_bounds(name)
                field.setToolTip(f"Bounds: [{lo}, {hi}]")
            except Exception:  # noqa: BLE001
                pass
            self._fields[name] = field
            self._form_layout.addRow(name, field)

    def _raw_in_sync(self) -> None:
        """Push the current form state into the raw editor."""
        self._raw_edit.setPlainText(
            yaml.safe_dump(self.current_config(), default_flow_style=False, sort_keys=False)
        )

    def _form_from_raw(self) -> None:
        """Parse raw YAML back into the form fields (best-effort)."""
        try:
            data = yaml.safe_load(self._raw_edit.toPlainText()) or {}
        except yaml.YAMLError:
            return
        # Update template to reflect raw edits
        self._template = data
        # Sync the mode combo to a raw-edited analysis_mode, otherwise
        # current_config() (form mode) would silently overwrite the user's raw
        # mode change with the stale combo value — a silent loss of their edit.
        if isinstance(data, dict):
            mode = data.get("analysis_mode")
            if mode is not None:
                idx = self._mode_combo.findText(str(mode))
                if idx >= 0:
                    self._mode_combo.blockSignals(True)
                    self._mode_combo.setCurrentIndex(idx)
                    self._mode_combo.blockSignals(False)
        self._rebuild_form()

    def _on_mode_changed(self, mode: str) -> None:
        """Slot: mode combo changed — reload template and rebuild form."""
        self.set_mode(mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Load the packaged template for *mode* and rebuild the form.

        Parameters
        ----------
        mode : str
            One of the four known analysis modes.
        """
        # Temporarily disconnect to avoid recursive signal during combo update
        self._mode_combo.blockSignals(True)
        idx = self._mode_combo.findText(mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_combo.blockSignals(False)

        self._template = template_dict(mode)
        self._rebuild_form()

        if self._stack.currentIndex() == 1:
            # Raw pane is visible — sync it too
            self._raw_in_sync()

    def current_config(self) -> dict:
        """Return the assembled config dict ready for the worker.

        When in raw mode, returns the parsed YAML directly.  In form mode,
        rebuilds the template with ``analysis_mode`` = the dropdown value and
        ``initial_parameters.values`` aligned to ``parameter_names`` order.

        Returns
        -------
        dict
            Complete config dict.
        """
        if self._stack.currentIndex() == 1:
            # Raw mode — parse and return
            try:
                return yaml.safe_load(self._raw_edit.toPlainText()) or {}
            except yaml.YAMLError:
                return {}

        # Form mode — clone the template, override analysis_mode + ip.values
        cfg = copy.deepcopy(self._template)
        cfg["analysis_mode"] = self._mode_combo.currentText()

        ip = cfg.setdefault("initial_parameters", {})
        if not isinstance(ip, dict):
            ip = {}
            cfg["initial_parameters"] = ip
        ip["parameter_names"] = list(self._param_names)

        field_values: list[float | str] = []
        for name in self._param_names:
            raw_text = self._fields[name].text().strip() if name in self._fields else ""
            try:
                field_values.append(float(raw_text))
            except ValueError:
                field_values.append(raw_text)
        ip["values"] = field_values

        return cfg

    def toggle_raw(self, on: bool) -> None:
        """Switch between form view and raw-YAML view.

        Parameters
        ----------
        on : bool
            ``True`` to show raw YAML; ``False`` to show the form.
        """
        # Keep checkbox in sync (in case called programmatically)
        self._raw_check.blockSignals(True)
        self._raw_check.setChecked(on)
        self._raw_check.blockSignals(False)

        if on:
            # Sync raw editor from current form state before showing it
            self._raw_in_sync()
            self._stack.setCurrentIndex(1)
        else:
            # Parse raw back into form before switching to form pane
            self._form_from_raw()
            self._stack.setCurrentIndex(0)

    def raw_text(self) -> str:
        """Return the current text in the raw-YAML editor.

        Returns
        -------
        str
            Plain text contents of the raw editor (may be stale if the form
            pane is active and ``toggle_raw`` was not yet called).
        """
        return self._raw_edit.toPlainText()

    def validate(self) -> ValidationReport:
        """Validate ``current_config()`` and render errors into the status label.

        Emits :attr:`config_ready` when the report is ``ok``.

        Returns
        -------
        ValidationReport
            Frozen report with ``ok``, ``errors``, and ``warnings``.
        """
        cfg = self.current_config()
        rep = validate_config(cfg)

        if rep.ok:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet("color: green;")
            self.config_ready.emit(cfg)
        else:
            msgs = "; ".join(rep.errors)
            if rep.warnings:
                msgs += " | " + "; ".join(rep.warnings)
            self._status_label.setText(f"Errors: {msgs}")
            self._status_label.setStyleSheet("color: red;")

        return rep

    def set_parameter(self, name: str, value: float | str) -> None:
        """Update the form field for *name* to *value*.

        Parameters
        ----------
        name : str
            Parameter name (must match one of ``initial_parameters.parameter_names``).
        value : float or str
            New value; converted via ``str(value)`` before setting the field.
        """
        if name in self._fields:
            self._fields[name].setText(str(value))
