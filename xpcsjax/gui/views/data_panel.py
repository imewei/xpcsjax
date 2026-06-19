"""Data tab panel: JAX-free HDF5 metadata browser + two-time C₂ preview.

No xpcsjax.data, no JAX — h5py only for reading. The C₂ preview routes
through :func:`~xpcsjax.gui.data_inspect.read_c2_preview` which uses the
shared ``_C2_PREVIEW_LAYOUTS`` descriptor (spec §6) to mirror the real
loader's group layout without pulling in the JAX kernel stack.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xpcsjax.gui.data_inspect import DatasetInfo, read_c2_preview, read_hdf5_metadata
from xpcsjax.gui.views.plots_view import TwoTimeMapView


class DataPanel(QWidget):
    """File-path field + HDF5 metadata tree + dataset combo + C₂ preview.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        # --- Path bar ---
        self._path_field = QLineEdit()
        self._path_field.setReadOnly(True)
        self._path_field.setPlaceholderText("No file loaded")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File:"))
        path_row.addWidget(self._path_field)
        path_row.addWidget(browse_btn)

        # --- Metadata tree ---
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Dataset", "Shape", "Dtype"])
        self._tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # --- Dataset combo ---
        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._on_dataset_changed)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Preview dataset:"))
        combo_row.addWidget(self._combo)

        # --- Two-time preview ---
        self._preview = TwoTimeMapView()

        # Display-only downsampling notice (spec §5 / F12): the preview is
        # block-mean decimated for *rendering only* — categorically distinct from
        # the project's prohibited *analysis* downsampling. No fit ever consumes
        # this decimated preview; labelling it keeps the integrity rule visible.
        self._downsample_note = QLabel(
            "Preview is downsampled for display only — fits always use the full-resolution data."
        )
        self._downsample_note.setWordWrap(True)

        # Error notice: a non-HDF5 / unreadable file must surface as a recoverable
        # message, never an uncaught OSError out of the browse slot (PySide6 6.5+
        # aborts the process on an unhandled exception in a slot).
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c62828;")
        self._error_label.setVisible(False)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.addLayout(path_row)
        layout.addWidget(self._error_label)
        layout.addWidget(self._tree)
        layout.addLayout(combo_row)
        layout.addWidget(self._preview)
        layout.addWidget(self._downsample_note)
        self.setLayout(layout)

        self._current_path: str | None = None
        self._current_data_type: str | None = None
        self._infos: list[DatasetInfo] = []
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, path: str | Path, *, data_type: str | None = None) -> None:
        """Populate the panel from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file.
        data_type : str or None
            Known ``data_type`` (``"aps_old"`` / ``"aps_u"``) forwarded to
            :func:`~xpcsjax.gui.data_inspect.read_c2_preview`.  ``None``
            triggers the best-effort heuristic.
        """
        path = str(path)
        self._current_path = path
        self._current_data_type = data_type
        self._path_field.setText(path)

        # Populate metadata tree. A non-HDF5 / unreadable file raises OSError (or
        # ValueError) from h5py — catch it and clear the panel rather than letting
        # it escape the (browse) slot and abort the GUI process.
        try:
            self._infos = read_hdf5_metadata(path)
        except (OSError, ValueError) as exc:
            self._infos = []
            self._last_error = f"Could not read '{Path(path).name}': {exc}"
            self._tree.clear()
            self._combo.blockSignals(True)
            self._combo.clear()
            self._combo.blockSignals(False)
            self._preview.clear_map()
            self._error_label.setText(self._last_error)
            self._error_label.setVisible(True)
            return

        self._last_error = None
        self._error_label.setVisible(False)
        self._tree.clear()
        self._combo.blockSignals(True)
        self._combo.clear()
        for info in self._infos:
            item = QTreeWidgetItem([info.name, str(info.shape), info.dtype])
            self._tree.addTopLevelItem(item)
            self._combo.addItem(info.name)
        self._combo.blockSignals(False)

        # Trigger preview for the first item
        if self._infos:
            self._refresh_preview(self._infos[0].name)

    def metadata_tree(self) -> QTreeWidget:
        """Return the :class:`QTreeWidget` showing dataset metadata."""
        return self._tree

    def path_field(self) -> QLineEdit:
        """Return the read-only path :class:`QLineEdit`."""
        return self._path_field

    def dataset_combo(self) -> QComboBox:
        """Return the dataset-selector :class:`QComboBox`."""
        return self._combo

    def preview_view(self) -> TwoTimeMapView:
        """Return the embedded :class:`TwoTimeMapView`."""
        return self._preview

    def last_error(self) -> str | None:
        """Return the last load error message (or None if the last load succeeded)."""
        return self._last_error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        """Open a file dialog and load the selected HDF5 file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 file",
            "",
            "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)",
        )
        if path:
            # A manually-browsed file carries no config context, so don't reuse a
            # stale config-driven data_type (which could apply the wrong layout to a
            # different format and silently render the wrong angle). Fall back to the
            # heuristic raw read; a config-driven load() still passes data_type explicitly.
            self.load(path)

    def _on_dataset_changed(self, name: str) -> None:
        """Slot: re-run preview when the combo selection changes."""
        if self._current_path and name:
            self._refresh_preview(name)

    def _refresh_preview(self, dataset: str) -> None:
        """Read and display the C₂ preview for ``dataset``."""
        if not self._current_path:
            return
        try:
            arr = read_c2_preview(
                self._current_path,
                dataset,
                data_type=self._current_data_type,
                phi_index=0,
                max_dim=512,
            )
        except (OSError, ValueError):
            arr = None
        # Clear a stale preview when this dataset has no renderable C₂ — never
        # leave the previous dataset's image showing for a different selection.
        if arr is not None:
            self._preview.show_map(arr)
        else:
            self._preview.clear_map()
