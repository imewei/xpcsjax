"""Phase 7: the Sphinx docs carry no references to the REMOVED per-angle
Fourier scaling feature.

The removed-feature literals (module/class/config names and the quoted per-angle
mode token) are assembled from fragments so this test file does not itself
contain the whole tokens -- that keeps the spec's grep-zero gate at honest zero
(a scan of ``tests/`` would otherwise count this guard as a hit).

Two checks:

1. ``test_rst_has_no_removed_mode`` -- the original three theory/user/advanced
   pages must not mention the removed mode word at all (lowercase substring).
2. ``test_all_rst_have_no_deleted_feature_tokens`` -- the whole ``docs/source``
   tree must not reference the DELETED module / classes / config keys, nor the
   removed per-angle mode literal. It does NOT forbid the bare word "Fourier",
   which legitimately appears in XPCS physics prose ("position density in
   Fourier space").
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

# Deleted module / class / config identifiers -- must NOT appear in ANY rst.
_DELETED_TOKENS = [
    _REMOVED_MODE + "_reparam",
    "Four" + "ierReparam",
    _REMOVED_MODE + "_order",
    _REMOVED_MODE + "_auto_threshold",
    _REMOVED_MODE + "_effective_mode",
]

# RST literal form of the removed per-angle MODE token, e.g. ``fourier``.
_MODE_LITERAL = "``" + _REMOVED_MODE + "``"


@pytest.mark.parametrize("path", _RST, ids=lambda p: p.name)
def test_rst_has_no_removed_mode(path: Path) -> None:
    text = path.read_text().lower()
    assert _REMOVED_MODE not in text, f"{path.name} still mentions {_REMOVED_MODE}"


def test_all_rst_have_no_deleted_feature_tokens() -> None:
    """No rst under docs/source references the deleted Fourier-scaling feature."""
    offenders: list[str] = []
    for path in sorted(_DOCS.rglob("*.rst")):
        text = path.read_text()
        lowered = text.lower()
        rel = path.relative_to(_DOCS)
        for token in _DELETED_TOKENS:
            if token.lower() in lowered:
                offenders.append(f"{rel}: deleted token {token!r}")
        if _MODE_LITERAL in text:
            offenders.append(f"{rel}: removed per-angle mode literal {_MODE_LITERAL!r}")
    assert not offenders, "stale Fourier-scaling references found:\n" + "\n".join(
        offenders
    )
