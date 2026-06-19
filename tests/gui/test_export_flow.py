"""Tests for the JAX-free figure-export helper (export.py).

Two required cases:
  (a) stage fake PNGs under plots/, call export_figures, assert they land in dest.
  (b) result_dir with no plots/ dir → returns [] and does NOT raise.
"""

from __future__ import annotations

from pathlib import Path


def test_export_copies_figures_to_dest(tmp_path: Path) -> None:
    """export_figures copies *.png/pdf from <result_dir>/plots recursively."""
    from xpcsjax.gui.export import export_figures

    result_dir = tmp_path / "run_001"
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(parents=True)

    # Top-level PNG
    (plots_dir / "fit_overview.png").write_bytes(b"\x89PNG\r\n")
    # Sub-directory PNG (simulated_data/)
    sub = plots_dir / "simulated_data"
    sub.mkdir()
    (sub / "c2_map.png").write_bytes(b"\x89PNG\r\n")
    # Also a PDF
    (plots_dir / "residuals.pdf").write_bytes(b"%PDF-1.4")

    dest = tmp_path / "exported"
    copied = export_figures(result_dir, dest)

    assert dest.is_dir(), "dest_dir must be created"
    assert len(copied) == 3, f"expected 3 files, got {len(copied)}: {copied}"
    dest_names = {p.name for p in copied}
    assert "fit_overview.png" in dest_names
    assert "c2_map.png" in dest_names
    assert "residuals.pdf" in dest_names
    for p in copied:
        assert p.exists(), f"expected {p} to exist in dest"


def test_export_disambiguates_same_name_files(tmp_path: Path) -> None:
    """Two PNGs with the same filename in different sub-dirs must both survive."""
    from xpcsjax.gui.export import export_figures

    result_dir = tmp_path / "run_002"
    plots_dir = result_dir / "plots"
    sub_a = plots_dir / "alpha"
    sub_b = plots_dir / "beta"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)

    # Both sub-dirs contain a file named "result.png" — would silently clobber.
    (sub_a / "result.png").write_bytes(b"\x89PNG_A")
    (sub_b / "result.png").write_bytes(b"\x89PNG_B")

    dest = tmp_path / "exported"
    copied = export_figures(result_dir, dest)

    assert len(copied) == 2, f"expected 2 distinct files, got {len(copied)}: {copied}"
    # All returned paths must be distinct.
    assert len(set(copied)) == 2, "returned list must not contain duplicate paths"
    # Both files must physically exist in dest.
    for p in copied:
        assert p.exists(), f"expected {p} to exist"
    # The dest directory must contain exactly 2 files — no silent overwrite.
    dest_files = list(dest.iterdir())
    assert len(dest_files) == 2, f"dest must contain 2 distinct files, got {dest_files}"
    # The two destination files must have different content (not overwritten).
    contents = {p.read_bytes() for p in dest_files}
    assert len(contents) == 2, "file contents must differ — one must not have clobbered the other"


def test_export_returns_empty_when_plots_dir_absent(tmp_path: Path) -> None:
    """export_figures must return [] and NOT raise when there is no plots/ dir."""
    from xpcsjax.gui.export import export_figures

    result_dir = tmp_path / "failed_run"
    result_dir.mkdir()
    # No plots/ subdirectory at all

    dest = tmp_path / "out"
    copied = export_figures(result_dir, dest)

    assert copied == [], f"expected [], got {copied}"
    # Must not raise — already asserted by getting here
