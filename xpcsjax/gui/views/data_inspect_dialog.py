"""Data-file inspector dialog: HDF5 dataset listing + a C₂ preview.

Wires the previously-orphaned ``xpcsjax.gui.data_inspect`` module (fully
implemented and tested, but never reachable from any view) into the GUI. The
config-first workflow this app settled on has no dedicated "Data" tab, so this
is a File-menu dialog like Create/Edit Config, not a permanent surface.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.data_inspect import read_c2_preview, read_hdf5_metadata
from xpcsjax.gui.views.plots.maps import TwoTimeMapView

_DATA_TYPES: tuple[tuple[str, str | None], ...] = (
    ("auto (best-effort)", None),
    ("aps_old", "aps_old"),
    ("aps_u", "aps_u"),
)


class DataInspectDialog(QDialog):
    """List every dataset in an HDF5 file and preview one angle's C₂ matrix.

    Parameters
    ----------
    path : str or Path
        The HDF5 file to inspect.
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self.setWindowTitle(f"Inspect Data — {self._path.name}")
        self.setObjectName("dialog_data_inspect")
        self.resize(760, 560)

        self._load_error: str | None = None

        self._error_label = QLabel()
        self._error_label.setObjectName("data_error")
        self._error_label.setWordWrap(True)
        self._error_label.hide()

        self._dataset_list = QListWidget()
        self._dataset_list.setObjectName("data_inspect_datasets")

        self._data_type_combo = QComboBox()
        self._data_type_combo.setObjectName("data_inspect_data_type")
        for label, _ in _DATA_TYPES:
            self._data_type_combo.addItem(label)

        self._phi_spin = QSpinBox()
        self._phi_spin.setObjectName("data_inspect_phi_index")
        self._phi_spin.setMinimum(0)
        self._phi_spin.setMaximum(9999)

        preview_btn = QPushButton("Preview C₂")
        preview_btn.clicked.connect(self._on_preview)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Data type:"))
        controls.addWidget(self._data_type_combo)
        controls.addWidget(QLabel("φ index:"))
        controls.addWidget(self._phi_spin)
        controls.addWidget(preview_btn)
        controls.addStretch(1)

        self._preview = TwoTimeMapView()
        self._preview.setMinimumHeight(280)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._error_label)
        layout.addWidget(QLabel("Datasets:"))
        layout.addWidget(self._dataset_list)
        layout.addLayout(controls)
        layout.addWidget(self._preview)
        layout.addWidget(buttons)

        self._load()

    # --- loading ----------------------------------------------------------
    def _load(self) -> None:
        """Populate the dataset list; show an error label on a read failure."""
        try:
            infos = read_hdf5_metadata(self._path)
        except OSError as exc:
            self._set_error(f"Could not read file:\n{exc}")
            return
        for info in infos:
            self._dataset_list.addItem(f"{info.name}   shape={info.shape}  dtype={info.dtype}")

    # --- preview ------------------------------------------------------------
    def _on_preview(self) -> None:
        """Read and display the selected φ index's C₂ matrix."""
        _, data_type = _DATA_TYPES[self._data_type_combo.currentIndex()]
        current = self._dataset_list.currentItem()
        dataset = current.text().split()[0] if current is not None else ""
        try:
            arr = read_c2_preview(
                self._path,
                dataset,
                data_type=data_type,
                phi_index=self._phi_spin.value(),
            )
        except OSError as exc:
            self._set_error(f"Could not read file:\n{exc}")
            return
        if arr is None:
            self._set_error("Preview unavailable for this dataset/φ index.")
            return
        self._load_error = None
        self._error_label.hide()
        self._preview.show_map(arr)

    def _set_error(self, message: str) -> None:
        """Record *message* as the current error and show it in the label.

        Tracked explicitly (not via ``QLabel.isVisible()``) since a dialog that
        was never shown reports every child widget as not-visible regardless of
        ``show()``/``hide()`` calls — the same pattern ConfigTextEditorDialog
        already uses for its own load-error tracking.
        """
        self._load_error = message
        self._error_label.setText(message)
        self._error_label.show()

    # --- inspection helpers (tests) -----------------------------------------
    def dataset_count(self) -> int:
        """Return the number of datasets listed (test/inspection helper)."""
        return self._dataset_list.count()

    def load_error(self) -> str | None:
        """Return the current error message, or ``None`` if loaded cleanly."""
        return self._load_error
