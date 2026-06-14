"""Phase 7 exit gate: the removed per-angle tokens must not survive anywhere.

This is the authoritative spec section 6 grep-zero checklist as an executable test.
It runs ``rg -n -w <tokens> tests/ xpcsjax/`` from the repo root and asserts zero
matching lines. It is RED until every Phase-7 task below has landed, GREEN once
the teardown is complete. Kept permanently as a regression tripwire against any
re-introduction of fourier/independent/old-token vocabulary.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Repo root = three parents up from tests/parity/<file>.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_TOKENS = (
    "fourier|use_fourier|fourier_order|fourier_auto_threshold|"
    "auto_averaged|fixed_constant|heterodyne_layout|independent"
)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep (rg) not installed")
def test_no_removed_per_angle_tokens_in_tests_or_package() -> None:
    """``rg -n -w <tokens> tests/ xpcsjax/`` returns zero lines after teardown."""
    proc = subprocess.run(
        ["rg", "-n", "-w", _TOKENS, "tests/", "xpcsjax/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # rg exit 1 == no matches (the success state); exit 0 == matches found (fail).
    matches = proc.stdout.strip()
    assert proc.returncode == 1 and matches == "", (
        "Removed per-angle tokens still present (spec section 6 grep-zero failed):\n"
        f"{matches}"
    )
