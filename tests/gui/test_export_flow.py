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
