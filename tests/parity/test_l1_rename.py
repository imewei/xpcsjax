"""Phase 7: L1 is renamed from 'Fourier/Constant Reparameterization' to
'Per-Angle Reparameterization'. No package source or CLAUDE.md may carry the old
L1 name, and the canonical name must appear in CLAUDE.md's _LAYER_GATES table row.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_old_l1_name_absent_from_package_and_claude() -> None:
    proc = subprocess.run(
        ["rg", "-n", "Fourier/Constant Reparameterization", "xpcsjax/", "CLAUDE.md"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1 and proc.stdout.strip() == "", (
        "Old L1 name still present:\n" + proc.stdout
    )


def test_claude_md_l1_row_uses_canonical_name() -> None:
    text = (_REPO_ROOT / "CLAUDE.md").read_text()
    assert "Per-Angle Reparameterization" in text
    # The L1 table row no longer cites the deleted module.
    assert "| L1 | Per-Angle Reparameterization" in text
