"""Tests for DataPanel — JAX-free HDF5 metadata + C₂ preview widget."""

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")
pyqtgraph = pytest.importorskip("pyqtgraph")
h5py = pytest.importorskip("h5py")

from PySide6.QtWidgets import QApplication  # noqa: E402

from xpcsjax.gui.views.data_panel import DataPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists for the module."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture()
def h5_file(tmp_path):
    """Tiny HDF5 file with two datasets."""
    p = tmp_path / "sample.h5"
    with h5py.File(p, "w") as f:
        rng = np.random.default_rng(1)
        f.create_dataset("c2", data=rng.random((3, 64, 64)))
        f.create_dataset("phi", data=np.array([0.0, 45.0, 90.0]))
    return p


def test_data_panel_load_populates_tree(qapp, h5_file):
    """load(path) populates the metadata tree with one row per dataset."""
    panel = DataPanel()
    panel.load(str(h5_file))

    tree = panel.metadata_tree()
    assert tree.topLevelItemCount() >= 2
    # Both 'c2' and 'phi' should appear somewhere in the tree items
    names = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert any("c2" in n for n in names)
    assert any("phi" in n for n in names)


def test_data_panel_load_shows_path(qapp, h5_file):
    """load(path) updates the path field."""
    panel = DataPanel()
    panel.load(str(h5_file))
    assert str(h5_file) in panel.path_field().text()


def test_data_panel_load_populates_dataset_combo(qapp, h5_file):
    """load(path) fills the dataset combo with dataset names."""
    panel = DataPanel()
    panel.load(str(h5_file))
    combo = panel.dataset_combo()
    items = [combo.itemText(i) for i in range(combo.count())]
    assert len(items) >= 2
    assert any("c2" in it for it in items)


def test_data_panel_preview_image_shown(qapp, h5_file):
    """load(path) drives the TwoTimeMapView to show an image for a 3-D dataset."""
    panel = DataPanel()
    panel.load(str(h5_file))
    # The preview widget should have been given an image after load
    assert panel.preview_view().has_image()


def test_data_panel_labels_preview_as_display_only(qapp):
    """Spec §5 / F12: the C₂ preview carries a visible 'display-only downsampling'
    label distinguishing it from the project's prohibited analysis downsampling."""
    panel = DataPanel()
    note = panel._downsample_note.text().lower()
    assert "downsampled" in note
    assert "display" in note
    assert panel._downsample_note.isVisibleTo(panel)


def test_data_panel_load_aps_u_group(qapp, tmp_path):
    """load(path, data_type='aps_u') uses the group layout and reconstructs the matrix."""
    p = tmp_path / "u.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("xpcs/twotime/correlation_map")
        for k in ("c2_00001", "c2_00002"):
            g.create_dataset(k, data=np.tril(np.ones((48, 48))))
        f.create_dataset("phi", data=np.array([0.0, 45.0]))

    panel = DataPanel()
    panel.load(str(p), data_type="aps_u")
    assert panel.preview_view().has_image()


def test_data_panel_missing_group_no_crash(qapp, h5_file):
    """load(path, data_type='aps_old') with no exchange/C2T_all group → no crash."""
    panel = DataPanel()
    # Should not raise even when the C2 group is absent
    panel.load(str(h5_file), data_type="aps_old")
    # Preview unavailable — has_image may be False (graceful)
    # Just ensure no exception was raised
