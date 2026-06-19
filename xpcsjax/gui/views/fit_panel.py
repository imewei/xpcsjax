"""Fit-settings summary panel: read-only rendering of config + overrides.

JAX-free: display-only widget. No fitting logic; no numerical computation.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class FitPanel(QWidget):
    """Read-only summary of the fit configuration and any applied overrides.

    Parameters
    ----------
    parent:
        Optional parent widget.

    Notes
    -----
    - ``show_settings(config, overrides)`` renders the resolved settings as plain text.
    - No JAX imports; no numerical logic.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Fit Settings"))

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_settings(self, config: dict, overrides: dict | None = None) -> None:
        """Render *config* and any *overrides* as a human-readable summary.

        Parameters
        ----------
        config:
            The resolved fit configuration dict (e.g. from ConfigManager.to_dict()).
        overrides:
            Optional dict of key→value pairs applied on top of the base config.
        """
        lines: list[str] = []

        # Mode
        mode = config.get("analysis_mode") or config.get("mode", "—")
        lines.append(f"Analysis mode : {mode}")
        lines.append("")

        # Initial parameters
        params_block = config.get("parameters") or config.get("initial_parameters")
        if isinstance(params_block, dict):
            lines.append("Initial parameters:")
            for name, spec in params_block.items():
                if isinstance(spec, dict):
                    val = spec.get("value", spec.get("initial", "—"))
                    fixed = spec.get("fixed", False)
                    fixed_tag = "  [fixed]" if fixed else ""
                    lines.append(f"  {name} = {val}{fixed_tag}")
                else:
                    lines.append(f"  {name} = {spec}")
            lines.append("")

        # Applied overrides
        if overrides:
            lines.append("Applied overrides:")
            for key, val in overrides.items():
                lines.append(f"  {key} = {val}")
        else:
            lines.append("Applied overrides: none")

        self._text.setPlainText("\n".join(lines))

    def clear(self) -> None:
        """Clear the displayed settings."""
        self._text.clear()
