"""Phase 7: the three theory/user/advanced rst docs carry no fourier mentions."""

from __future__ import annotations

from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
_RST = [
    _DOCS / "advanced" / "anti_degeneracy.rst",
    _DOCS / "user_guide" / "nlsq_fitting.rst",
    _DOCS / "theory" / "heterodyne_anti_degeneracy.rst",
]


@pytest.mark.parametrize("path", _RST, ids=lambda p: p.name)
def test_rst_has_no_fourier(path: Path) -> None:
    text = path.read_text().lower()
    assert "fourier" not in text, f"{path.name} still mentions fourier"
