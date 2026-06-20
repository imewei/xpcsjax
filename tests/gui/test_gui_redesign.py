"""Tests for the 2026-06-20 GUI redesign.

Covers the new surfaces that replaced the Data/Config/Fit tabs and the Fit
Monitor SSR view:
  - ``PhiResultsGrid`` (one section per phi angle, graceful on missing data).
  - ``find_diagnostics_png`` (locate per-angle residual diagnostics PNG).
  - ``CreateConfigDialog`` / ``ConfigTextEditorDialog`` File-menu dialogs.
  - ``MainWindow.create_project`` / ``create_config`` workflow methods.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from xpcsjax.gui.views.plots_view import PhiResultsGrid, find_diagnostics_png  # noqa: E402
from xpcsjax.gui.viz_bundle import VizBundle  # noqa: E402

_MODES = ("static_anisotropic", "static_isotropic", "laminar_flow", "two_component")


def _bundle(n_phi: int, *, with_fit: bool = True) -> VizBundle:
    rng = np.random.default_rng(0)
    exp = rng.random((n_phi, 12, 12))
    if with_fit:
        model = rng.random((n_phi, 12, 12))
        return VizBundle(
            exp_c2=exp,
            model_c2=model,
            residuals=exp - model,
            phi_angles=np.linspace(0.0, 90.0, n_phi),
        )
    return VizBundle(exp_c2=exp, phi_angles=np.linspace(0.0, 90.0, n_phi))


# ---------------------------------------------------------------------------
# PhiResultsGrid
# ---------------------------------------------------------------------------
def test_phi_grid_section_count_matches_n_phi(qtbot):
    """The grid builds exactly one section per phi angle (matches n_phi)."""
    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    for n_phi in (1, 3, 5):
        grid.set_bundle(_bundle(n_phi))
        assert grid.section_count() == n_phi
        assert grid.phi_count() == n_phi


def test_phi_grid_handles_missing_fit_surfaces(qtbot):
    """An exp-only bundle (no model/residuals) still builds n_phi sections, no crash."""
    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    grid.set_bundle(_bundle(4, with_fit=False))
    assert grid.section_count() == 4
    # Every section reports its fitted/residual surfaces absent.
    assert all(not s._has_fitted and not s._has_residual for s in grid._sections)


def test_phi_grid_set_bundle_none_clears(qtbot):
    """set_bundle(None) tears down all sections (no lingering prior-run grid)."""
    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    grid.set_bundle(_bundle(3))
    assert grid.section_count() == 3
    grid.set_bundle(None)
    assert grid.section_count() == 0
    assert grid.phi_count() == 0


def test_phi_grid_missing_png_is_placeholder(qtbot, tmp_path):
    """A result dir with no diagnostics PNGs degrades to placeholders (no PNG)."""
    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    grid.set_bundle(_bundle(2), result_dir=str(tmp_path))
    assert grid.section_count() == 2
    assert all(not s._has_png for s in grid._sections)


def test_phi_grid_embeds_present_png(qtbot, tmp_path):
    """When residuals_phi_NNN.png exists, the matching section embeds it."""
    from PySide6.QtGui import QPixmap

    plots = tmp_path / "plots"
    plots.mkdir()
    png = plots / "residuals_phi_000_0.000deg.png"
    pm = QPixmap(16, 16)
    pm.fill()
    assert pm.save(str(png), "PNG")

    grid = PhiResultsGrid()
    qtbot.addWidget(grid)
    grid.set_bundle(_bundle(2), result_dir=str(tmp_path))
    assert grid._sections[0]._has_png  # section 0 found its PNG
    assert not grid._sections[1]._has_png  # section 1 has none


# ---------------------------------------------------------------------------
# find_diagnostics_png
# ---------------------------------------------------------------------------
def test_find_diagnostics_png_locates_by_index(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    target = plots / "residuals_phi_001_4.878deg.png"
    target.write_bytes(b"not really a png but the locator only globs by name")
    found = find_diagnostics_png(str(tmp_path), 1)
    assert found == target
    assert find_diagnostics_png(str(tmp_path), 2) is None  # no slice-2 file
    assert find_diagnostics_png(None, 0) is None  # no result dir


# ---------------------------------------------------------------------------
# CreateConfigDialog / ConfigTextEditorDialog
# ---------------------------------------------------------------------------
def test_create_config_dialog_collects_inputs(qtbot):
    from xpcsjax.gui.views.config_dialogs import CreateConfigDialog

    dlg = CreateConfigDialog(default_dir="/tmp/proj")
    qtbot.addWidget(dlg)
    assert dlg.selected_mode() == "static_anisotropic"  # first mode
    assert dlg.output_path().endswith("xpcsjax_config.yaml")

    dlg._mode_combo.setCurrentText("laminar_flow")
    dlg._data_edit.setText("/data/run.h5")
    dlg._q_edit.setText("0.05")
    dlg._dt_edit.setText("0.2")
    dlg._time_edit.setText("2000")
    assert dlg.selected_mode() == "laminar_flow"
    assert dlg.generation_kwargs() == {
        "data_path": "/data/run.h5",
        "q": 0.05,
        "dt": 0.2,
        "time_length": 2000,
    }


def test_create_config_dialog_omits_blank_optionals(qtbot):
    from xpcsjax.gui.views.config_dialogs import CreateConfigDialog

    dlg = CreateConfigDialog()
    qtbot.addWidget(dlg)
    assert dlg.generation_kwargs() == {}  # nothing filled in → no injections


def test_config_text_editor_round_trip(qtbot, tmp_path):
    from xpcsjax.gui.views.config_dialogs import ConfigTextEditorDialog

    cfg = tmp_path / "c.yaml"
    cfg.write_text("analysis_mode: static_isotropic\n", encoding="utf-8")
    dlg = ConfigTextEditorDialog(cfg)
    qtbot.addWidget(dlg)
    assert "static_isotropic" in dlg.text()
    assert dlg.load_error() is None
    dlg.set_text("analysis_mode: laminar_flow\n")
    dlg.save()
    assert cfg.read_text(encoding="utf-8") == "analysis_mode: laminar_flow\n"


# ---------------------------------------------------------------------------
# MainWindow workflow methods
# ---------------------------------------------------------------------------
def _window(qtbot):
    from xpcsjax.gui.views.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_create_project_sets_working_dir(qtbot, tmp_path):
    win = _window(qtbot)
    proj = tmp_path / "myproject"
    win.create_project(proj)
    assert win._project_dir == proj
    assert win._output_dir == proj  # default per-run output base
    assert proj.is_dir()


@pytest.mark.parametrize("mode", _MODES)
def test_create_config_writes_template_for_mode(qtbot, tmp_path, mode):
    """create_config writes the mode's template; the written config's mode matches."""
    win = _window(qtbot)
    out = tmp_path / f"{mode}.yaml"
    written = win.create_config(mode, out)
    assert written == out and out.is_file()
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert parsed["analysis_mode"] == mode


def test_file_menu_has_workflow_actions(qtbot):
    from PySide6.QtGui import QAction

    win = _window(qtbot)
    names = {a.objectName() for a in win.findChildren(QAction)}
    assert {
        "action_create_project",
        "action_create_config",
        "action_edit_config",
        "action_load_config",
    } <= names
