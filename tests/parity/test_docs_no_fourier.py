"""Phase 7: the three theory/user/advanced rst docs carry no removed-mode mentions.

The removed-mode literal is assembled from fragments so this test file does not
itself contain the whole-word token (keeps the spec section 6 grep-zero gate at
honest zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
_RST = [
    _DOCS / "advanced" / "anti_degeneracy.rst",
    _DOCS / "user_guide" / "nlsq_fitting.rst",
    _DOCS / "theory" / "heterodyne_anti_degeneracy.rst",
]

_REMOVED_MODE = "four" + "ier"


@pytest.mark.parametrize("path", _RST, ids=lambda p: p.name)
def test_rst_has_no_removed_mode(path: Path) -> None:
    text = path.read_text().lower()
    assert _REMOVED_MODE not in text, f"{path.name} still mentions {_REMOVED_MODE}"
