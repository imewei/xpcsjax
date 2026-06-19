"""Round-trip tests for .xpcsproj save/load."""

import pytest

from xpcsjax.gui.project.model import DONE, Project
from xpcsjax.gui.project.persist import load_project, save_project


def test_round_trip_preserves_datasets_and_runs(tmp_path):
    p = Project()
    d = p.add_dataset(str(tmp_path / "cfg.yaml"), label="DS-A")
    r = p.add_run(d.dataset_id)
    p.set_run_status(r.run_id, DONE, result_dir=str(tmp_path / "out"))

    proj_file = tmp_path / "session.xpcsproj"
    save_project(p, proj_file)
    loaded = load_project(proj_file)

    assert [ds.label for ds in loaded.datasets] == ["DS-A"]
    ld = loaded.datasets[0]
    assert ld.dataset_id == d.dataset_id           # stable id preserved
    assert len(ld.runs) == 1
    lr = ld.runs[0]
    assert lr.run_id == r.run_id and lr.status == DONE
    assert lr.result_dir == str(tmp_path / "out")  # path resolved back
    assert lr.summary is None                       # summaries re-loaded lazily


def test_load_rejects_unknown_schema(tmp_path):
    bad = tmp_path / "bad.xpcsproj"
    bad.write_text('{"schema": "nope", "datasets": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_project(bad)
