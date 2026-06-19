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
    assert ld.dataset_id == d.dataset_id  # stable id preserved
    assert len(ld.runs) == 1
    lr = ld.runs[0]
    assert lr.run_id == r.run_id and lr.status == DONE
    assert lr.result_dir == str(tmp_path / "out")  # path resolved back
    assert lr.summary is None  # summaries re-loaded lazily


def test_save_is_atomic_failed_write_keeps_original_intact(tmp_path, monkeypatch):
    """A failure during the swap must leave the prior .xpcsproj intact (no torn write)."""
    import os

    proj_file = tmp_path / "session.xpcsproj"

    v1 = Project()
    v1.add_dataset(str(tmp_path / "cfg.yaml"), label="V1")
    save_project(v1, proj_file)
    original = proj_file.read_text(encoding="utf-8")

    # Mutate, then make the atomic swap fail — the original must survive untouched.
    v2 = Project()
    v2.add_dataset(str(tmp_path / "cfg.yaml"), label="V2-CORRUPT")
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError, match="boom"):
        save_project(v2, proj_file)

    assert proj_file.read_text(encoding="utf-8") == original  # untouched
    assert load_project(proj_file).datasets[0].label == "V1"  # still valid + V1
    # No temp turds left behind in the directory.
    assert list(tmp_path.glob("*.tmp")) == []
    assert [p.name for p in tmp_path.iterdir()] == ["session.xpcsproj"]


def test_load_rejects_unknown_schema(tmp_path):
    bad = tmp_path / "bad.xpcsproj"
    bad.write_text('{"schema": "nope", "datasets": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_project(bad)
