"""Tests for Plan H Task 4: Save/Open project wiring in MainWindow."""

import pytest

pytest.importorskip("PySide6")

from xpcsjax.gui.app import build_workbench  # noqa: E402


def test_save_then_open_round_trips_through_window(qtbot, tmp_path):
    window, _ = build_workbench()
    qtbot.addWidget(window)
    window.add_dataset(str(tmp_path / "cfg.yaml"))  # the Open-Config slot, factored to be callable
    proj = tmp_path / "s.xpcsproj"
    window.save_project_to(proj)
    assert proj.is_file()

    window2, _ = build_workbench()
    qtbot.addWidget(window2)
    window2.open_project_from(proj)
    assert window2.sidebar_dataset_count() == 1


def test_open_tolerates_deleted_result_dir(qtbot, tmp_path):
    # Spec §8 dead-path: a restored run whose result_dir is gone must open
    # without raising, leave summary None, AND be flagged result_missing so the
    # sidebar surfaces it as "missing" rather than a normal "done" run.
    from xpcsjax.gui.project.model import DONE, Project
    from xpcsjax.gui.project.persist import save_project

    p = Project()
    d = p.add_dataset(str(tmp_path / "cfg.yaml"), label="DS-A")  # config never created either
    r = p.add_run(d.dataset_id)
    p.set_run_status(r.run_id, DONE, result_dir=str(tmp_path / "gone"))  # never created
    proj = tmp_path / "s.xpcsproj"
    save_project(p, proj)

    window, _ = build_workbench()
    qtbot.addWidget(window)
    window.open_project_from(proj)  # must not raise
    assert window.sidebar_dataset_count() == 1
    dataset, run = window._project.run_by_id(r.run_id)
    assert run.summary is None
    assert run.result_missing is True  # eagerly flagged at load (spec §8)
    assert dataset.config_missing is True  # gone config_path is flagged too
