"""Phase 7: L1 is renamed from 'Fourier/Constant Reparameterization' to
'Per-Angle Reparameterization'. No package source or CLAUDE.md may carry the old
L1 name, and the canonical name must appear in CLAUDE.md's _LAYER_GATES table row.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OLD_L1_NAME = "Fourier/Constant Reparameterization"

# Text-source suffixes scanned for the forbidden name. Pure-Python scanning is
# used instead of an `rg`/`grep` subprocess so the guard runs on every CI runner
# (Windows/macOS lack ripgrep) without an external-tool dependency.
_SCAN_SUFFIXES = {".py", ".pyi", ".yaml", ".yml", ".rst", ".md", ".txt", ".toml", ".cfg"}


def test_old_l1_name_absent_from_package_and_claude() -> None:
    targets = [
        p
        for p in (_REPO_ROOT / "xpcsjax").rglob("*")
        if p.is_file() and p.suffix in _SCAN_SUFFIXES
    ]
    targets.append(_REPO_ROOT / "CLAUDE.md")
    offenders: list[str] = []
    for path in targets:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _OLD_L1_NAME in line:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Old L1 name still present:\n" + "\n".join(offenders)


def test_claude_md_l1_row_uses_canonical_name() -> None:
    text = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Per-Angle Reparameterization" in text
    # The L1 table row no longer cites the deleted module.
    assert "| L1 | Per-Angle Reparameterization" in text
